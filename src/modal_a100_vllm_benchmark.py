"""Bounded, one-shot Modal benchmark for the DeepSeek-Coder vLLM workload.

Run with:
    /Users/kieugiangbien/bienkieu_env/bin/modal run src/modal_a100_vllm_benchmark.py

The function requests one A100-80GB, measures prompt batches up to 108, and
terminates after completion or the 15-minute hard timeout. It does not deploy a
persistent server, create a Modal Volume, or write model outputs to storage.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Any

import modal


MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
INPUT_TOKENS = 3072
OUTPUT_TOKENS = 96
MAX_MODEL_LEN = INPUT_TOKENS + OUTPUT_TOKENS
MAX_NUM_SEQS = 108
BATCH_SIZES = (16, 32, 64, 96, 108)
GPU_MEMORY_UTILIZATION = 0.90
MAX_BATCHED_TOKENS = 16384
FUNCTION_TIMEOUT_SECONDS = 15 * 60


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers")
)

app = modal.App("bienkieu-deepseek-a100-throughput-test")


class GpuSampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, float]] = []

    @staticmethod
    def _read() -> dict[str, float] | None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        try:
            return {
                "gpu_util_pct": float(fields[0]),
                "memory_used_mib": float(fields[1]),
                "memory_total_mib": float(fields[2]),
                "power_w": float(fields[3]),
            }
        except (IndexError, ValueError):
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._read()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(1.0)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        sample = self._read()
        if sample is not None:
            self.samples.append(sample)


def _make_code_prompt_ids(tokenizer: Any, sample_id: int, target_tokens: int) -> list[int]:
    """Build distinct code-like token sequences of an exact input length."""
    prefix = f"""# Related repository snippets for sample {sample_id}
from typing import Optional, Sequence

class AccountService{sample_id}:
    def __init__(self, repository, cache=None):
        self.repository = repository
        self.cache = cache

    def find_account(self, account_id: str) -> Optional[dict]:
        if not account_id:
            return None
        cached = self.cache.get(account_id) if self.cache else None
        if cached is not None:
            return cached
        return self.repository.find_by_id(account_id)

# Current file: src/services/account_{sample_id}.py
class AccountController{sample_id}:
    def update_account(self, account_id, payload):
        account = self.service.find_account(account_id)
"""
    token_ids = tokenizer.encode(prefix, add_special_tokens=True)
    block_id = 0
    while len(token_ids) < target_tokens:
        filler = (
            f"\n    # repository helper block {sample_id}_{block_id}\n"
            f"    def normalize_{sample_id}_{block_id}(self, value):\n"
            "        if value is None:\n"
            "            return None\n"
            "        normalized = str(value).strip()\n"
            "        return normalized.lower() if normalized else None\n"
        )
        token_ids.extend(tokenizer.encode(filler, add_special_tokens=False))
        block_id += 1
    return token_ids[:target_tokens]


def _summarize_gpu(samples: list[dict[str, float]]) -> dict[str, float] | None:
    if not samples:
        return None
    return {
        "sample_count": len(samples),
        "mean_gpu_util_pct": round(
            sum(row["gpu_util_pct"] for row in samples) / len(samples), 1
        ),
        "peak_gpu_util_pct": round(max(row["gpu_util_pct"] for row in samples), 1),
        "peak_memory_gib": round(
            max(row["memory_used_mib"] for row in samples) / 1024, 2
        ),
        "gpu_memory_total_gib": round(
            max(row["memory_total_mib"] for row in samples) / 1024, 2
        ),
        "mean_power_w": round(sum(row["power_w"] for row in samples) / len(samples), 1),
    }


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32768,
    timeout=FUNCTION_TIMEOUT_SECONDS,
)
def benchmark() -> dict[str, Any]:
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(
        json.dumps(
            {
                "event": "benchmark_start",
                "model": MODEL_NAME,
                "vllm_version": vllm.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                "max_num_seqs": MAX_NUM_SEQS,
                "max_num_batched_tokens": MAX_BATCHED_TOKENS,
                "input_tokens": INPUT_TOKENS,
                "output_tokens": OUTPUT_TOKENS,
                "function_hard_timeout_seconds": FUNCTION_TIMEOUT_SECONDS,
            }
        ),
        flush=True,
    )
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    print("Loading vLLM engine; model download/cache is ephemeral for this test.", flush=True)
    engine = LLM(
        model=MODEL_NAME,
        dtype="float16",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=True,
        enable_prefix_caching=False,
        trust_remote_code=False,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
        seed=123,
    )

    sampler = GpuSampler()
    sampler.start()

    # Warm kernels/model once, outside measured batch rows.
    warmup = [
        {"prompt_token_ids": _make_code_prompt_ids(tokenizer, 9000 + i, 512)}
        for i in range(4)
    ]
    engine.generate(
        warmup,
        SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True, seed=123),
        use_tqdm=False,
    )
    print("Warmup complete; beginning measured sweeps.", flush=True)

    results: list[dict[str, Any]] = []
    next_sample_id = 0
    for batch_size in BATCH_SIZES:
        prompts = [
            {"prompt_token_ids": _make_code_prompt_ids(tokenizer, next_sample_id + i, INPUT_TOKENS)}
            for i in range(batch_size)
        ]
        next_sample_id += batch_size
        gpu_start = len(sampler.samples)
        started = time.perf_counter()
        try:
            outputs = engine.generate(prompts, sampling, use_tqdm=False)
        except Exception as exc:
            row = {
                "batch_size": batch_size,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:1200],
            }
            print(json.dumps({"event": "batch_result", **row}), flush=True)
            results.append(row)
            break
        elapsed = time.perf_counter() - started
        generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        gpu_stats = _summarize_gpu(sampler.samples[gpu_start:])
        row = {
            "batch_size": batch_size,
            "status": "ok",
            "elapsed_seconds": round(elapsed, 2),
            "prompts_per_second": round(batch_size / elapsed, 2),
            "input_tokens_per_second": round(prompt_tokens / elapsed, 1),
            "output_tokens_per_second": round(generated_tokens / elapsed, 1),
            "total_tokens_per_second": round((prompt_tokens + generated_tokens) / elapsed, 1),
            "mean_generated_tokens_per_prompt": round(generated_tokens / batch_size, 1),
            "gpu": gpu_stats,
        }
        print(json.dumps({"event": "batch_result", **row}), flush=True)
        results.append(row)

    sampler.stop()
    result = {
        "model": MODEL_NAME,
        "vllm_version": vllm.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "results": results,
        "overall_gpu_samples": _summarize_gpu(sampler.samples),
    }
    print(json.dumps({"event": "benchmark_complete", **result}), flush=True)
    return result


@app.local_entrypoint()
def main() -> None:
    result = benchmark.remote()
    print("FINAL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
