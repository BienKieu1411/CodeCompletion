"""Eval-only reproduction using the official RLCoder inference/evaluation code."""

from __future__ import annotations

import argparse
import atexit
import copy
import gc
import json
import os
import shutil
import signal
import sys
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import requests


REPO = Path(os.environ.get("RLCODER_REPO", "/content/RLCoder-official")).resolve()
if not REPO.is_dir():
    raise FileNotFoundError(f"Official RLCoder checkout not found: {REPO}")
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from datasets import load_test_dataset
from generator import CustomDataset as OfficialGeneratorDataset
from generator import Generator as OfficialGenerator
from main import retrieve_codeblocks
from retriever import Retriever
from utils.eval_metric import compute_metric_stmt
from transformers import AutoTokenizer


DATA_REVISION = "cb9639f20f2374d75e5b0b8e8650f1b20802bf9f"
RETRIEVER_ID = "nov3630/RLRetriever"
RETRIEVER_REVISION = "ec587f5d8635462fff3cfb06f9a946148acda08b"
GENERATOR_ID = "deepseek-ai/deepseek-coder-1.3b-base"
GENERATOR_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"

BENCHMARKS = (
    ("cceval_python", "cceval", "python", "data/cceval/python/test.jsonl", "python"),
    ("cceval_java", "cceval", "java", "data/cceval/java/test.jsonl", "java"),
    ("repoeval_line", "repoeval", "line_level", "data/repoeval/line_level/test.jsonl", "python"),
    ("repoeval_api", "repoeval", "api_level", "data/repoeval/api_level/test.jsonl", "python"),
)


def make_args(output_dir: Path, generator_batch_size: int) -> SimpleNamespace:
    # Use the official parser defaults for the requested 1.3B generator.
    # The README's 1536/2048 example is specifically shown with the 6.7B model.
    return SimpleNamespace(
        generator_model_path="/content/models/deepseek-coder-1.3b-base",
        generator_batch_size_per_gpu=generator_batch_size,
        generator_batch_size=generator_batch_size * torch.cuda.device_count(),
        generator_max_crossfile_length=1536,
        generator_max_context_length=2048,
        generator_max_generation_length=64,
        disable_generator=False,
        retriever_model_path="/content/models/RLRetriever",
        retriever_batch_size_per_gpu=64,
        retriever_batch_size=64 * torch.cuda.device_count(),
        disable_retriever=False,
        retriever_query_context_length=256,
        retriever_candidate_context_length=512,
        inference_type="unixcoder_with_rl",
        output_dir=str(output_dir),
        eval=True,
        enable_tqdm=True,
        enable_generation=True,
        debug=False,
        num_workers=4,
        weighted_keywords=True,
        enable_fixed_block=False,
        enable_sft=False,
        disable_stop_block=False,
        enable_repocoder=False,
        rlcoder_model_path="/content/models/RLRetriever",
        do_codereval=False,
        enable_forward_generation=False,
        forward_generation_times=1,
        sample_number=10,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


class VLLMGenerator:
    """vLLM server adapter using RLCoder's exact prompt construction."""

    def __init__(self, args, batch_size: int = 128, gpu_memory_utilization: float = 0.35):
        self.args = args
        self.batch_size = batch_size
        self.model_name = "deepseek-coder-1.3b"
        self.base_url = "http://127.0.0.1:8000"
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.generator_model_path, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        vllm_bin = shutil.which("vllm")
        if not vllm_bin:
            raise FileNotFoundError("vLLM is not installed in this Colab runtime")
        self.server_log_path = Path("/content/vllm-server.log")
        self.server_log = self.server_log_path.open("w", encoding="utf-8")
        command = [
            vllm_bin,
            "serve",
            args.generator_model_path,
            "--host", "127.0.0.1",
            "--port", "8000",
            "--served-model-name", self.model_name,
            "--dtype", "half",
            "--max-model-len", str(args.generator_max_context_length),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-num-seqs", "128",
            "--max-num-batched-tokens", "32768",
        ]
        self.server = subprocess.Popen(
            command,
            stdout=self.server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.closed = False
        atexit.register(self.close)
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        self._wait_until_ready()
        self.http = requests.Session()

    def _wait_until_ready(self) -> None:
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.server.poll() is not None:
                self.server_log.flush()
                tail = self.server_log_path.read_text(
                    encoding="utf-8", errors="replace"
                )[-8000:]
                raise RuntimeError("vLLM exited during startup:\n" + tail)
            try:
                response = requests.get(f"{self.base_url}/health", timeout=3)
                if response.status_code == 200:
                    print("vLLM generator ready on A100", flush=True)
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        self.server_log.flush()
        tail = self.server_log_path.read_text(
            encoding="utf-8", errors="replace"
        )[-8000:]
        raise TimeoutError("vLLM health check timed out:\n" + tail)

    def _handle_signal(self, signum, frame) -> None:
        self.close()
        raise SystemExit(128 + signum)

    def generate(self, examples, retrieved_codeblocks, max_generation_length):
        # Reuse RLCoder's prompt assembly and tokenizer boundary, then send
        # token IDs to vLLM so its server does not retokenize the prompt.
        dataset = OfficialGeneratorDataset(
            self.args, self.tokenizer, examples, retrieved_codeblocks, generation=True
        )
        prompt_ids = []
        for example, selected in zip(examples, retrieved_codeblocks):
            prompt = dataset.construct_prompts(example, selected)
            ids = self.tokenizer.encode(prompt)[-self.args.generator_max_context_length:]
            prompt_ids.append(ids)

        generated = []
        for start in range(0, len(prompt_ids), self.batch_size):
            batch = prompt_ids[start:start + self.batch_size]
            response = self.http.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model_name,
                    "prompt": batch,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": max_generation_length,
                    "n": 1,
                    "stream": False,
                    "skip_special_tokens": True,
                },
                timeout=900,
            )
            response.raise_for_status()
            choices = sorted(response.json()["choices"], key=lambda choice: choice["index"])
            if len(choices) != len(batch):
                raise RuntimeError(
                    f"vLLM returned {len(choices)} completions for {len(batch)} prompts"
                )
            generated.extend(choice["text"] for choice in choices)
            done = min(start + len(batch), len(prompt_ids))
            print(f"vLLM generation: {done}/{len(prompt_ids)}", flush=True)
        return generated

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if getattr(self, "server", None) is not None and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait(timeout=10)
        if getattr(self, "server_log", None) is not None:
            self.server_log.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="/content/rlcoder-results/deepseek-coder-1.3b-8004",
    )
    parser.add_argument("--generator-batch-size", type=int, default=32)
    parser.add_argument(
        "--generator-backend", choices=("vllm", "huggingface"), default="vllm"
    )
    parser.add_argument("--vllm-batch-size", type=int, default=128)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument(
        "--benchmarks",
        default=",".join(spec[0] for spec in BENCHMARKS),
        help="Comma-separated official benchmarks; defaults to all four.",
    )
    parser.add_argument(
        "--limit-per-benchmark",
        type=int,
        default=None,
        help="Optional smoke-run limit; leave unset for the full reproduction.",
    )
    args_cli = parser.parse_args()
    requested_benchmarks = {name.strip() for name in args_cli.benchmarks.split(",") if name.strip()}
    unknown_benchmarks = requested_benchmarks - {spec[0] for spec in BENCHMARKS}
    if unknown_benchmarks:
        raise ValueError(f"Unknown benchmarks: {sorted(unknown_benchmarks)}")

    if not torch.cuda.is_available():
        raise RuntimeError("Colab GPU is required for this reproduction")
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = make_args(output_dir, args_cli.generator_batch_size)
    import transformers
    import tree_sitter
    import importlib.metadata

    run_info = {
        "status": "running",
        "official_repo_commit": "164d8d88cde324a38f5da70c4f858cc4679ef08e",
        "dataset_repo": "nov3630/Data4RLCoder",
        "dataset_revision": DATA_REVISION,
        "retriever_repo": RETRIEVER_ID,
        "retriever_revision": RETRIEVER_REVISION,
        "generator_repo": GENERATOR_ID,
        "generator_revision": GENERATOR_REVISION,
        "generator_batch_size_per_gpu": args_cli.generator_batch_size,
        "generator_backend": args_cli.generator_backend,
        "vllm_batch_size": args_cli.vllm_batch_size,
        "vllm_version": (
            importlib.metadata.version("vllm")
            if args_cli.generator_backend == "vllm"
            else None
        ),
        "vllm_gpu_memory_utilization": args_cli.vllm_gpu_memory_utilization,
        "vllm_max_num_seqs": 128,
        "vllm_max_num_batched_tokens": 32768,
        "decoding": {"temperature": 0.0, "top_p": 1.0, "n": 1},
        "retriever_batch_size_per_gpu": 64,
        "bm25_pool_size": 100,
        "retrieved_snippets": 10,
        "stop_block_enabled": True,
        "crossfile_token_limit": 1536,
        "total_prompt_token_limit": 2048,
        "max_generation_tokens": 64,
        "fixed_block": False,
        "progressive_generation": False,
        "metrics_impl": "official RLCoder utils.eval_metric.compute_metric_stmt",
        "eval_only_entrypoint": True,
        "eval_loss_skipped": True,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tree_sitter": importlib.metadata.version("tree-sitter"),
        "python": sys.version,
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "benchmarks_requested": sorted(requested_benchmarks),
        "limit_per_benchmark": args_cli.limit_per_benchmark,
        "benchmarks": {},
    }
    write_json(output_dir / "run_info.json", run_info)

    print(
        f"RLCoder official inference: repo={run_info['official_repo_commit']} "
        f"generator={GENERATOR_ID}@{GENERATOR_REVISION} "
        f"retriever={RETRIEVER_ID}@{RETRIEVER_REVISION}",
        flush=True,
    )
    print(
        "Config: BM25 pool=100, RLRetriever slate=10 incl. STOP, "
        "crossfile=1536, prompt=2048, max_new=64, fixed_block=False",
        flush=True,
    )

    # Load released weights from the pinned HF snapshots. The official model
    # classes and their greedy generate() path are kept unchanged.
    if args_cli.generator_backend == "vllm":
        generator = VLLMGenerator(
            args,
            batch_size=args_cli.vllm_batch_size,
            gpu_memory_utilization=args_cli.vllm_gpu_memory_utilization,
        )
    else:
        generator = OfficialGenerator(args)
    retriever = Retriever(args)
    bm25 = {}
    all_started = time.time()

    for name, dataset_name, language, prompt_file, metric_language in BENCHMARKS:
        if name not in requested_benchmarks:
            continue
        started = time.time()
        print(f"\n=== {name}: loading official test split ===", flush=True)
        examples = load_test_dataset(args, dataset_name, language)
        if args_cli.limit_per_benchmark is not None:
            examples = examples[: args_cli.limit_per_benchmark]
        print(f"{name}: n={len(examples)}", flush=True)

        _, retrieved = retrieve_codeblocks(args, examples, bm25, retriever, name)
        selected_counts = [len(row) for row in retrieved]
        print(
            f"{name}: retrieved {len(retrieved)} slates; "
            f"mean slate size={sum(selected_counts)/max(1,len(selected_counts)):.3f}",
            flush=True,
        )

        predictions = generator.generate(
            examples, retrieved, args.generator_max_generation_length
        )
        output_dataset_dir = output_dir / name
        output_dataset_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = output_dataset_dir / "prediction.jsonl"
        with prediction_path.open("w", encoding="utf-8") as handle:
            for example, prediction in zip(examples, predictions):
                handle.write(
                    json.dumps(
                        {"task_id": example.task_id, "pred": prediction},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # Uses the paper repository's scoring implementation and its original
        # test.jsonl prompt/reference files (not a locally recomputed metric).
        scores = compute_metric_stmt(
            str(output_dataset_dir),
            prompt_file,
            language=metric_language,
            ts_lib=(
                "utils/build/java-lang-parser.so"
                if metric_language == "java"
                else "utils/build/python-lang-parser.so"
            ),
        )
        elapsed = time.time() - started
        run_info["benchmarks"][name] = {
            "n": len(examples),
            "mean_selected_count": sum(selected_counts) / max(1, len(selected_counts)),
            "scores": scores,
            "seconds": round(elapsed, 2),
            "peak_vram_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
        }
        write_json(output_dir / "run_info.json", run_info)
        print(
            f"RESULT {name}: n={len(examples)} scores={scores} "
            f"seconds={elapsed:.1f} peak_vram_gib="
            f"{run_info['benchmarks'][name]['peak_vram_gib']}",
            flush=True,
        )

        del examples, retrieved, predictions
        bm25.clear()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if isinstance(generator, VLLMGenerator):
        generator.close()
    run_info["status"] = "complete"
    run_info["total_seconds"] = round(time.time() - all_started, 2)
    write_json(output_dir / "run_info.json", run_info)
    print(f"\nRUN_COMPLETE {output_dir}", flush=True)
    print(json.dumps(run_info["benchmarks"], indent=2), flush=True)


if __name__ == "__main__":
    main()
