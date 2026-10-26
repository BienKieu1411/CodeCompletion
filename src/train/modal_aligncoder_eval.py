"""Run the unmodified AlignCoder RQ1 evaluation on Modal with retrieval traces.

The AlignCoder source remains the source of truth for chunking, BM25 filtering,
UniXcoder scoring/selection, prompt construction, generation postprocessing, and
benchmark metrics. Instrumentation observes those calls and writes compressed
JSONL events to a Modal Volume; it does not change candidate ordering or scores.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


REMOTE_ROOT = Path("/root/aligncoder")
_script_path = Path(__file__).resolve()
if len(_script_path.parents) > 2 and (_script_path.parents[2] / "RepoClone" / "AlignCoder").is_dir():
    ROOT = _script_path.parents[2]
    ALIGNCODER_DIR = ROOT / "RepoClone" / "AlignCoder"
else:
    # Modal imports this driver as /root/<filename>; the source directory is
    # already baked into the image at /root/aligncoder.
    ROOT = Path("/root")
    ALIGNCODER_DIR = REMOTE_ROOT
RUN_ID = "aligncoder-rq1-deepseek13b-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
REMOTE_OUTPUT = Path("/results") / RUN_ID
REMOTE_DATA = Path("/tmp/aligncoder-hf-data")
REMOTE_MODELS = Path("/tmp/aligncoder-models")

ALIGNCODER_REPO_COMMIT = "ab0808594febfa3903e6242833bb3c207b7956b4"
ALIGN_RETRIEVER_REVISION = "a179c0fdbc5a56d83ae142064ef1ea773a841753"
DATA4ALIGNCODER_REVISION = "ce72fe3ef4987e15a9219fea4174bb80700e742c"
DEEPSEEK_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"

OBSERVABILITY_VOLUME_NAME = "aligncoder-eval-observability"
APP_NAME = "aligncoder-rq1-deepseek13b-observability"
HARD_TIMEOUT_SECONDS = 8 * 60 * 60 + 30 * 60
BUDGET_USD = 30.0
CPU_PHYSICAL_CORES = 2
MAX_CPU_WORKERS = 2  # Match the 2 CPU slots visible inside this Modal container.
MEMORY_MIB = 48 * 1024
TRACE_COMMIT_RECORDS = 512
# Modal billing rates checked on 2026-10-14.
ESTIMATED_GPU_USD_PER_HOUR = 2.50  # A100 80 GB
ESTIMATED_CPU_USD_PER_HOUR = 0.0473 * CPU_PHYSICAL_CORES
ESTIMATED_MEMORY_USD_PER_HOUR = 0.008 * MEMORY_MIB / 1024
ESTIMATED_TOTAL_USD_PER_HOUR = (
    ESTIMATED_GPU_USD_PER_HOUR
    + ESTIMATED_CPU_USD_PER_HOUR
    + ESTIMATED_MEMORY_USD_PER_HOUR
)
ESTIMATED_MAX_COMPUTE_USD = ESTIMATED_TOTAL_USD_PER_HOUR * HARD_TIMEOUT_SECONDS / 3600


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


SOURCE_FILES = [
    "main.py",
    "retriever.py",
    "bm25.py",
    "datasets.py",
    "generator.py",
    "api.py",
    "utils/eval_metric.py",
    "utils/eval_utils.py",
    "utils/eval_repoeval.py",
    "utils/build/python-lang-parser.so",
    "utils/build/java-lang-parser.so",
]
LOCAL_SOURCE_SHA256 = {
    rel: _file_sha256(ALIGNCODER_DIR / rel)
    for rel in SOURCE_FILES
    if (ALIGNCODER_DIR / rel).is_file()
}


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.21.0",
        "tree-sitter==0.20.4",
        "pandas",
        "pyarrow",
        "rank-bm25",
        "editdistance",
        "prettytable",
        "nltk",
        "fuzzywuzzy",
        "sacrebleu",
        "timeout-decorator",
        "huggingface-hub",
    )
    .env(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": "/tmp/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    .add_local_dir(
        str(ALIGNCODER_DIR),
        remote_path=str(REMOTE_ROOT),
        ignore=[
            "results/**",
            "data/**",
            "**/__pycache__/**",
            "**/*.pyc",
            ".git/**",
        ],
    )
)

app = modal.App(APP_NAME)
observability_volume = modal.Volume.from_name(
    OBSERVABILITY_VOLUME_NAME, create_if_missing=True
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=(CPU_PHYSICAL_CORES, CPU_PHYSICAL_CORES),
    memory=(MEMORY_MIB, MEMORY_MIB),
    timeout=HARD_TIMEOUT_SECONDS,
    retries=0,
    max_containers=1,
    volumes={"/results": observability_volume},
)
def evaluate_aligncoder_rq1(
    run_id: str, source_commit: str, source_file_hashes: dict[str, str]
) -> dict[str, Any]:
    import sys
    import traceback

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download

    RUN_ID = run_id
    ALIGNCODER_REPO_COMMIT = source_commit
    LOCAL_SOURCE_SHA256 = source_file_hashes
    REMOTE_OUTPUT = Path("/results") / RUN_ID
    sys.path.insert(0, str(REMOTE_ROOT))
    os.chdir(REMOTE_ROOT)
    REMOTE_OUTPUT.mkdir(parents=True, exist_ok=True)
    trace_root = REMOTE_OUTPUT / "observability"
    trace_root.mkdir(parents=True, exist_ok=True)

    class TraceWriter:
        def __init__(self) -> None:
            self.count = 0

        def append(self, relative_path: str, records: list[dict[str, Any]]) -> None:
            if not records:
                return
            output_path = trace_root / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # gzip append writes valid concatenated members. Modal Volume commits
            # are comparatively expensive, so batch hundreds of trace records per
            # commit while keeping periodic crash-safe checkpoints.
            for offset in range(0, len(records), TRACE_COMMIT_RECORDS):
                batch = records[offset : offset + TRACE_COMMIT_RECORDS]
                with gzip.open(output_path, "at", encoding="utf-8") as handle:
                    for record in batch:
                        handle.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                default=_json_default,
                            )
                            + "\n"
                        )
                self.count += len(batch)
                observability_volume.commit()

    writer = TraceWriter()
    state: dict[str, Any] = {
        "dataset": None,
        "examples": {},
        "bm25_round": 0,
        "retrieval_round": 0,
        "active_bm25": None,
        "active_retriever": None,
        "active_generation": None,
        "source_hashes": {},
        "dataset_metrics": {},
        "dataset_counts": {},
    }

    def log(message: str) -> None:
        print(f"[{RUN_ID}] {message}", flush=True)

    def dataset_dir(dataset_name: str) -> str:
        return dataset_name

    def task_key(task_id: Any) -> str:
        return str(task_id)

    def _codeblock_record(block: Any, *, bm25_score: Any = None, bm25_rank: Any = None) -> dict[str, Any]:
        return {
            "file_path": block.file_path,
            "description": block.description,
            "code_content": block.code_content,
            "language": block.language,
            "type": getattr(block, "_type", None),
            "api_block": bool(getattr(block, "_api_block", False)),
            "bm25_score": None if bm25_score is None else float(bm25_score),
            "bm25_rank_in_full_index": bm25_rank,
        }

    def _install_instrumentation(main_module: Any) -> None:
        from multiprocessing import Pool as _BasePool

        import bm25 as bm25_module
        import generator as generator_module
        import retriever as retriever_module
        import utils.eval_metric as metric_module

        # The released AlignCoder snapshot hard-codes pools of 32/64 and its
        # metric uses host cpu_count(). Cap only parallel worker counts to the
        # requested Modal CPU; retrieval, chunking, ranking and metrics stay the
        # stock implementation.
        logged_pool_caps: set[int] = set()

        def bounded_pool(processes: int | None = None, *args: Any, **kwargs: Any) -> Any:
            requested = processes
            workers = MAX_CPU_WORKERS if requested is None else max(
                1, min(int(requested), MAX_CPU_WORKERS)
            )
            if requested != workers and requested not in logged_pool_caps:
                log(f"capping AlignCoder process pool {requested} -> {workers} workers")
                logged_pool_caps.add(requested)
            return _BasePool(processes=workers, *args, **kwargs)

        class _CappedMultiprocessingProxy:
            def __init__(self, original: Any, cap_cpu_count: bool = False) -> None:
                self._original = original
                self._cap_cpu_count = cap_cpu_count

            def __getattr__(self, name: str) -> Any:
                if name == "Pool":
                    return bounded_pool
                if name == "cpu_count" and self._cap_cpu_count:
                    return lambda: MAX_CPU_WORKERS + 1
                return getattr(self._original, name)

        for module in (bm25_module, generator_module, retriever_module, main_module):
            if hasattr(module, "Pool"):
                module.Pool = bounded_pool
            if hasattr(module, "mp") and hasattr(module.mp, "Pool"):
                module.mp = _CappedMultiprocessingProxy(module.mp)
        if hasattr(metric_module, "mp"):
            metric_module.mp = _CappedMultiprocessingProxy(
                metric_module.mp, cap_cpu_count=True
            )

        original_load_test = main_module.load_test_dataset

        def traced_load_test(args: Any, dataset_name: str, language: str) -> list[Any]:
            start = time.monotonic()
            log(f"loading original {dataset_name}/{language} test set")
            examples = original_load_test(args, dataset_name, language)
            name = "repoeval_line" if dataset_name == "repoeval" and language == "line_level" else (
                "repoeval_api" if dataset_name == "repoeval" and language == "api_level" else
                f"{dataset_name}_{language}"
            )
            state["dataset_counts"][name] = len(examples)
            log(f"loaded {name}: {len(examples)} tasks in {time.monotonic() - start:.1f}s")
            return examples

        main_module.load_test_dataset = traced_load_test
        # AlignCoder's eval path loads these files even though the values are not
        # used in evaluation. Avoid fetching train data; no eval input is affected.
        main_module.load_train_and_valid_dataset = lambda: ([], [])

        original_retrieve_codeblocks = main_module.retrieve_codeblocks

        def traced_retrieve_codeblocks(
            args: Any,
            examples: list[Any],
            bm25_models: dict[str, Any],
            retriever_model: Any,
            dataset_name: str,
            is_training: bool = False,
            inference_type: str | None = None,
        ) -> Any:
            state["dataset"] = dataset_name
            state["examples"] = {task_key(x.task_id): x for x in examples}
            state["current_examples"] = examples
            state["bm25_round"] = 0
            state["retrieval_round"] += 1
            log(
                f"{dataset_name}: AlignCoder retrieval pass "
                f"{state['retrieval_round']} for {len(examples)} tasks"
            )
            writer.append(
                f"{dataset_dir(dataset_name)}/retrieval_events.jsonl.gz",
                [
                    {
                        "schema": 1,
                        "run_id": RUN_ID,
                        "dataset": dataset_name,
                        "event": "task_input",
                        "task_id": x.task_id,
                        "language": x.language,
                        "file_path": x.file_path,
                        "left_context": x.left_context,
                        "right_context": x.right_context,
                        "target": x.target_code,
                        "related_file_count": len(x.related_files),
                        "related_file_paths": [f.file_path for f in x.related_files],
                        "related_file_char_counts": [len(f.code_content) for f in x.related_files],
                    }
                    for x in examples
                ],
            )
            return original_retrieve_codeblocks(
                args,
                examples,
                bm25_models,
                retriever_model,
                dataset_name,
                is_training,
                inference_type,
            )

        main_module.retrieve_codeblocks = traced_retrieve_codeblocks

        original_get_scores = bm25_module.BM25Okapi.get_scores

        def traced_get_scores(index: Any, query_tokens: list[str]) -> Any:
            scores = original_get_scores(index, query_tokens)
            active = state.get("active_bm25")
            if active is not None:
                task_id = active["index_to_task"].get(id(index))
                if task_id is not None:
                    active["scores"][task_key(task_id)] = np.asarray(scores)
            return scores

        bm25_module.BM25Okapi.get_scores = traced_get_scores

        def wrap_bm25_method(method_name: str, stage_prefix: str) -> None:
            original = getattr(bm25_module.TaskSpecificBM25, method_name)

            def wrapped(self: Any, task_ids: list[Any], queries: list[str], topk: int) -> Any:
                dataset_name = state.get("dataset") or "unknown"
                round_index = state["bm25_round"]
                state["bm25_round"] += 1
                stage = "seed_query_bm25" if round_index == 0 else "enhanced_query_bm25"
                index_dict = (
                    self.bm25_indices_all if method_name == "query_with_api" else self.bm25_indices
                )
                index_to_task = {id(index): task_id for task_id, index in index_dict.items() if index}
                capture = {"index_to_task": index_to_task, "scores": {}}
                previous = state.get("active_bm25")
                state["active_bm25"] = capture
                start = time.monotonic()
                try:
                    result = original(self, task_ids, queries, topk)
                finally:
                    state["active_bm25"] = previous
                records = []
                for task_id, query, blocks in zip(task_ids, queries, result):
                    key = task_key(task_id)
                    raw_scores = capture["scores"].get(key)
                    index = index_dict.get(task_id)
                    if method_name == "query_with_api":
                        all_blocks = self.code_blocks_all.get(task_id, [])
                    else:
                        all_blocks = self.code_blocks.get(task_id, [])
                    index_by_identity = {id(block): idx for idx, block in enumerate(all_blocks)}
                    score_order = (
                        sorted(range(len(raw_scores)), key=lambda idx: raw_scores[idx], reverse=True)
                        if raw_scores is not None
                        else []
                    )
                    rank_by_index = {idx: rank + 1 for rank, idx in enumerate(score_order)}
                    pool_records = []
                    for returned_rank, block in enumerate(blocks, start=1):
                        candidate_idx = index_by_identity.get(id(block))
                        score = (
                            raw_scores[candidate_idx]
                            if raw_scores is not None and candidate_idx is not None
                            else None
                        )
                        pool_records.append(
                            {
                                **_codeblock_record(
                                    block,
                                    bm25_score=score,
                                    bm25_rank=rank_by_index.get(candidate_idx),
                                ),
                                "returned_rank": returned_rank,
                                "candidate_index": candidate_idx,
                            }
                        )
                    records.append(
                        {
                            "schema": 1,
                            "run_id": RUN_ID,
                            "dataset": dataset_name,
                            "event": stage,
                            "task_id": task_id,
                            "query": query,
                            "topk_argument": topk,
                            "chunking_mode": "AlignCoder default blank-line mini-blocks; max 15 nonempty lines",
                            "bm25_index_size": len(all_blocks),
                            "returned_candidate_count": len(blocks),
                            "returned_candidates": pool_records,
                        }
                    )
                writer.append(f"{dataset_dir(dataset_name)}/retrieval_events.jsonl.gz", records)
                log(
                    f"{dataset_name}: {stage} returned {sum(len(x['returned_candidates']) for x in records)} "
                    f"snippets for {len(records)} tasks in {time.monotonic() - start:.1f}s"
                )
                return result

            setattr(bm25_module.TaskSpecificBM25, method_name, wrapped)

        wrap_bm25_method("query_with_api", "seed_query_bm25")
        wrap_bm25_method("query", "enhanced_query_bm25")

        original_retrieve = retriever_module.Retriever.retrieve
        original_torch_mm = torch.mm

        def traced_retrieve(self: Any, queries: list[str], candidate_codeblocks: list[list[Any]], topk: int, extracted_imports: Any = None) -> Any:
            dataset_name = state.get("dataset") or "unknown"
            examples_by_id = state.get("examples", {})
            query_task_ids = [task_key(x.task_id) for x in state.get("current_examples", [])]
            if not query_task_ids or len(query_task_ids) != len(queries):
                query_task_ids = list(examples_by_id.keys())
            total_candidates = sum(len(items) for items in candidate_codeblocks)
            capture = {
                "num_queries": len(queries),
                "num_candidates": total_candidates,
                "scores": None,
            }
            previous = state.get("active_retriever")
            state["active_retriever"] = capture

            def capture_mm(left: Any, right: Any) -> Any:
                output = original_torch_mm(left, right)
                active = state.get("active_retriever")
                if (
                    active is not None
                    and left.ndim == 2
                    and right.ndim == 2
                    and left.shape[0] == active["num_queries"]
                    and right.shape[1] == active["num_candidates"]
                    and left.shape[1] == right.shape[0]
                    and output.shape == (active["num_queries"], active["num_candidates"])
                ):
                    active["scores"] = output.detach().float().cpu().numpy()
                return output

            torch.mm = capture_mm
            start = time.monotonic()
            try:
                selected = original_retrieve(
                    self, queries, candidate_codeblocks, topk, extracted_imports
                )
            finally:
                torch.mm = original_torch_mm
                state["active_retriever"] = previous

            scores = capture["scores"]
            if scores is None:
                raise RuntimeError(
                    "Could not observe AlignCoder's in-path torch.mm similarity matrix; "
                    "refusing to produce an incomplete retrieval trace."
                )
            events = []
            flat_start = 0
            for row_idx, (query, blocks, chosen) in enumerate(
                zip(queries, candidate_codeblocks, selected)
            ):
                task_id = query_task_ids[row_idx] if row_idx < len(query_task_ids) else str(row_idx)
                query_scores = scores[row_idx, flat_start : flat_start + len(blocks)]
                selected_indices = query_scores.argsort()[-topk:][::-1].tolist() if blocks else []
                selected_rank_by_index = {
                    int(candidate_idx): rank + 1
                    for rank, candidate_idx in enumerate(selected_indices)
                }
                full_rank = query_scores.argsort()[::-1].tolist() if blocks else []
                full_rank_by_index = {
                    int(candidate_idx): rank + 1
                    for rank, candidate_idx in enumerate(full_rank)
                }
                all_records = []
                for candidate_idx, block in enumerate(blocks):
                    all_records.append(
                        {
                            **_codeblock_record(block),
                            "candidate_index": candidate_idx,
                            "cosine_score": float(query_scores[candidate_idx]),
                            "retriever_rank": full_rank_by_index[candidate_idx],
                            "selected": candidate_idx in selected_rank_by_index,
                            "selected_rank": selected_rank_by_index.get(candidate_idx),
                            "outside_top10": candidate_idx not in selected_rank_by_index,
                        }
                    )
                events.append(
                    {
                        "schema": 1,
                        "run_id": RUN_ID,
                        "dataset": dataset_name,
                        "event": "unixcoder_rerank_pool",
                        "task_id": task_id,
                        "enhanced_query": query,
                        "topk_argument": topk,
                        "candidate_count": len(blocks),
                        "selected_count": len(chosen),
                        "candidate_pool_after_bm25_and_api": all_records,
                        "selected_order": [
                            {
                                "file_path": block.file_path,
                                "description": block.description,
                                "code_content": block.code_content,
                                "api_block": bool(getattr(block, "_api_block", False)),
                                "stop_block": block.file_path == "",
                            }
                            for block in chosen
                        ],
                    }
                )
                if blocks:
                    flat_start += len(blocks)
            writer.append(f"{dataset_dir(dataset_name)}/retrieval_events.jsonl.gz", events)
            log(
                f"{dataset_name}: saved UniXcoder score/rank for {total_candidates} candidates; "
                f"topk={topk}; elapsed={time.monotonic() - start:.1f}s"
            )
            return selected

        retriever_module.Retriever.retrieve = traced_retrieve

        original_generator_class = main_module.vLLM_offline_Generator
        original_llm_class = generator_module.LLM
        original_llm_generate = original_llm_class.generate

        def create_bounded_llm(*args: Any, **kwargs: Any) -> Any:
            # AlignCoder's pinned source passes `task="generate"`, which newer
            # vLLM versions no longer accept. LLM defaults to text generation;
            # dropping this obsolete argument preserves the original behavior.
            kwargs.pop("task", None)
            kwargs.setdefault("gpu_memory_utilization", 0.50)
            kwargs.setdefault("max_model_len", 2112)
            kwargs.setdefault("max_num_seqs", 256)
            kwargs.setdefault("dtype", "float16")
            kwargs.setdefault("seed", 123)
            log(
                "starting vLLM DeepSeek-Coder-1.3B with "
            f"gpu_memory_utilization={kwargs['gpu_memory_utilization']}, "
                f"max_model_len={kwargs['max_model_len']}, "
                f"max_num_seqs={kwargs['max_num_seqs']}"
            )
            return original_llm_class(*args, **kwargs)

        generator_module.LLM = create_bounded_llm

        def traced_llm_generate(
            llm: Any, prompts: list[str], sampling_params: Any = None, *args: Any, **kwargs: Any
        ) -> Any:
            outputs = original_llm_generate(llm, prompts, sampling_params, *args, **kwargs)
            active = state.get("active_generation")
            if active is not None:
                active["prompts"] = list(prompts)
                active["sampling_params"] = {
                    key: getattr(sampling_params, key, None)
                    for key in ("temperature", "top_p", "top_k", "max_tokens", "n", "seed")
                }
                active["raw_outputs"] = [
                    [choice.text for choice in getattr(output, "outputs", [])]
                    for output in outputs
                ]
                raw_records = []
                for idx, example in enumerate(active["examples"]):
                    raw_records.append(
                        {
                            "schema": 1,
                            "run_id": RUN_ID,
                            "dataset": active["dataset"],
                            "event": f"{active['stage']}_raw",
                            "generation_call_id": active["generation_call_id"],
                            "task_id": example.task_id,
                            "prompt": active["prompts"][idx]
                            if idx < len(active["prompts"])
                            else None,
                            "sampling_params": active["sampling_params"],
                            "raw_completions": active["raw_outputs"][idx]
                            if idx < len(active["raw_outputs"])
                            else [],
                        }
                    )
                writer.append(
                    f"{active['dataset']}/generation_raw_events.jsonl.gz",
                    raw_records,
                )
                active["raw_trace_committed"] = True
                log(
                    f"{active['dataset']}: committed raw {active['stage']} outputs "
                    f"for {len(raw_records)} tasks before AlignCoder postprocessing"
                )
            return outputs

        original_llm_class.generate = traced_llm_generate

        class InstrumentedVLLMGenerator(original_generator_class):
            def generate(
                self,
                examples: list[Any],
                retrieved_codeblocks: list[list[Any]],
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                sample_number = kwargs.get("sample_number", args[2] if len(args) > 2 else 0)
                stage = "query_enhancement" if sample_number else "final_completion"
                dataset_name = state.get("dataset") or "unknown"
                generation_call_id = (
                    f"{RUN_ID}:{dataset_name}:{state['retrieval_round']}:{stage}"
                )
                active = {
                    "stage": stage,
                    "dataset": dataset_name,
                    "generation_call_id": generation_call_id,
                    "examples": examples,
                    "prompts": [],
                    "raw_outputs": [],
                    "sampling_params": {},
                    "raw_trace_committed": False,
                }
                previous = state.get("active_generation")
                state["active_generation"] = active
                start = time.monotonic()
                try:
                    generated = super().generate(
                        examples, retrieved_codeblocks, *args, **kwargs
                    )
                finally:
                    state["active_generation"] = previous

                prompts = active["prompts"]
                raw_outputs = active["raw_outputs"]
                records = []
                if stage == "query_enhancement":
                    unique_outputs, counts = generated
                    cursor = 0
                    for idx, (example, count) in enumerate(zip(examples, counts)):
                        records.append(
                            {
                                "schema": 1,
                                "run_id": RUN_ID,
                                "dataset": dataset_name,
                                "event": f"{stage}_postprocessed",
                                "generation_call_id": generation_call_id,
                                "task_id": example.task_id,
                                "deduplicated_postprocessed_completions": unique_outputs[cursor : cursor + count],
                                "deduplicated_completion_count": count,
                                "left_context_before_enhancement": example.left_context,
                            }
                        )
                        cursor += count
                else:
                    for idx, (example, prediction) in enumerate(zip(examples, generated)):
                        records.append(
                            {
                                "schema": 1,
                                "run_id": RUN_ID,
                                "dataset": dataset_name,
                                "event": f"{stage}_postprocessed",
                                "generation_call_id": generation_call_id,
                                "task_id": example.task_id,
                                "prediction": prediction,
                                "left_context_used": example.left_context,
                            }
                        )
                writer.append(f"{dataset_dir(dataset_name)}/generation_events.jsonl.gz", records)
                log(
                    f"{dataset_name}: {stage} wrote {len(records)} prompt/output records "
                    f"in {time.monotonic() - start:.1f}s"
                )
                return generated

        main_module.HF_Generator = InstrumentedVLLMGenerator

        original_compute_metrics = main_module.compute_metric_stmt

        def traced_compute_metrics(output_dir: str, prompt_file: str, language: str = "python", ts_lib: str = "utils/build/python-lang-parser.so") -> Any:
            start = time.monotonic()
            result = original_compute_metrics(output_dir, prompt_file, language, ts_lib)
            dataset_name = Path(output_dir).name
            detailed_path = Path(output_dir) / "detailed_results.json"
            records = []
            if detailed_path.exists():
                with detailed_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        item = json.loads(line)
                        records.append(
                            {
                                "schema": 1,
                                "run_id": RUN_ID,
                                "dataset": dataset_name,
                                "event": "official_aligncoder_metric",
                                **item,
                            }
                        )
            writer.append(f"{dataset_dir(dataset_name)}/metric_events.jsonl.gz", records)
            state["dataset_metrics"][dataset_name] = result
            observability_volume.commit()
            log(
                f"{dataset_name}: official AlignCoder metrics {result}; "
                f"metric time={time.monotonic() - start:.1f}s"
            )
            return result

        main_module.compute_metric_stmt = traced_compute_metrics

    manifest: dict[str, Any] = {
        "run_id": RUN_ID,
        "status": "starting",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "workspace_git_commit": ALIGNCODER_REPO_COMMIT,
            "snapshot_path": str(REMOTE_ROOT),
            "source_file_sha256": LOCAL_SOURCE_SHA256,
            "note": "AlignCoder source files were not edited; observer wraps calls and vLLM backend only.",
        },
        "huggingface": {
            "alignretriever_repo": "AlignCoder/AlignRetriever",
            "alignretriever_revision": ALIGN_RETRIEVER_REVISION,
            "dataset_repo": "AlignCoder/Data4AlignCoder",
            "dataset_revision": DATA4ALIGNCODER_REVISION,
            "generator_repo": "deepseek-ai/deepseek-coder-1.3b-base",
            "generator_revision": DEEPSEEK_REVISION,
        },
        "protocol": {
            "paper_section": "AlignCoder README RQ1",
            "datasets": ["cceval_python", "cceval_java", "repoeval_line", "repoeval_api"],
            "generation_rounds": 1,
            "enhancement_samples": 8,
            "bm25_seed_topk": 100,
            "bm25_enhanced_topk": 100,
            "retriever_topk": 10,
            "chunking": "AlignCoder default: blank-line mini-blocks, split/merge to <=15 nonempty lines; fixed-block flag OFF",
            "crossfile_token_budget": 1536,
            "generator_context_length": 2048,
            "generator_max_new_tokens": 64,
            "candidate_token_length": 512,
            "query_token_length": 256,
            "api_blocks": True,
            "stop_block": True,
            "generator_backend": "vLLM offline using AlignCoder's vLLM_offline_Generator logic",
            "generator_precision": "float16",
            "vllm_gpu_memory_utilization": 0.50,
            "vllm_max_model_len": 2112,
            "vllm_max_num_seqs": 256,
            "outputs": [
                "official AlignCoder prediction and metrics files",
                "all returned seed and enhanced BM25 candidates with BM25 score/rank",
                "every candidate in the final UniXcoder rerank pool with exact cosine score/rank and top-10 flag",
                "query-enhancement and final prompts plus raw/postprocessed generations",
                "per-task official EM/ES/identifier metrics",
            ],
        },
        "runtime_cap": {
            "gpu": "A100-80GB x1",
            "cpu_physical_cores": CPU_PHYSICAL_CORES,
            "cpu_vcpus_equivalent": CPU_PHYSICAL_CORES * 2,
            "max_cpu_process_workers": MAX_CPU_WORKERS,
            "cpu_limit_physical_cores": CPU_PHYSICAL_CORES,
            "memory_mib_request_and_limit": MEMORY_MIB,
            "timeout_seconds": HARD_TIMEOUT_SECONDS,
            "budget_usd": BUDGET_USD,
            "estimated_compute_usd_per_hour": ESTIMATED_TOTAL_USD_PER_HOUR,
            "estimated_max_compute_usd": ESTIMATED_MAX_COMPUTE_USD,
            "estimate_excludes": ["network transfer outside included allowances"],
            "max_containers": 1,
            "retries": 0,
            "modal_volume": OBSERVABILITY_VOLUME_NAME,
        },
    }
    manifest_path = REMOTE_OUTPUT / "run_manifest.json"

    try:
        log(
            f"runtime: torch={torch.__version__}, cuda={torch.version.cuda}, "
            f"gpu={torch.cuda.get_device_name(0)}, gpu_count={torch.cuda.device_count()}"
        )
        log("downloading pinned original test data and model revisions from Hugging Face")
        REMOTE_DATA.mkdir(parents=True, exist_ok=True)
        REMOTE_MODELS.mkdir(parents=True, exist_ok=True)
        data_patterns = [
            "data/cceval/python/test.parquet",
            "data/cceval/python/test.jsonl",
            "data/cceval/java/test.parquet",
            "data/cceval/java/test.jsonl",
            "data/repoeval/line_level/test_0.parquet",
            "data/repoeval/line_level/test_1.parquet",
            "data/repoeval/line_level/test.jsonl",
            "data/repoeval/api_level/test_0.parquet",
            "data/repoeval/api_level/test_1.parquet",
            "data/repoeval/api_level/test.jsonl",
        ]
        data_path = Path(
            snapshot_download(
                repo_id="AlignCoder/Data4AlignCoder",
                repo_type="dataset",
                revision=DATA4ALIGNCODER_REVISION,
                local_dir=str(REMOTE_DATA),
                allow_patterns=data_patterns,
            )
        )
        retriever_path = Path(
            snapshot_download(
                repo_id="AlignCoder/AlignRetriever",
                repo_type="model",
                revision=ALIGN_RETRIEVER_REVISION,
                local_dir=str(REMOTE_MODELS / "alignretriever"),
            )
        )
        generator_path = Path(
            snapshot_download(
                repo_id="deepseek-ai/deepseek-coder-1.3b-base",
                repo_type="model",
                revision=DEEPSEEK_REVISION,
                local_dir=str(REMOTE_MODELS / "deepseek-coder-1.3b-base"),
            )
        )
        data_link = REMOTE_ROOT / "data"
        if data_link.is_symlink():
            if data_link.resolve() != (data_path / "data").resolve():
                raise RuntimeError(f"Unexpected existing data symlink: {data_link}")
        elif data_link.exists():
            raise RuntimeError(f"Refusing to overwrite existing AlignCoder data path: {data_link}")
        else:
            os.symlink(data_path / "data", data_link, target_is_directory=True)
        manifest["resolved_paths"] = {
            "test_data": str(data_path),
            "alignretriever": str(retriever_path),
            "generator": str(generator_path),
        }

        import main as aligncoder_main

        _install_instrumentation(aligncoder_main)
        args = type("AlignCoderEvalArgs", (), {})()
        values = {
            "generator_model_path": str(generator_path),
            "generator_batch_size_per_gpu": 16,
            "generator_max_crossfile_length": 1536,
            "generator_max_context_length": 2048,
            "generator_max_generation_length": 64,
            "disable_generator": False,
            "retriever_model_path": str(retriever_path),
            "retriever_batch_size_per_gpu": 128,
            "disable_retriever": False,
            "retriever_query_context_length": 256,
            "retriever_candidate_context_length": 512,
            "inference_type": "unixcoder_with_rl",
            "output_dir": str(REMOTE_OUTPUT / "official_results"),
            "eval": True,
            "enable_tqdm": False,
            "enable_generation": True,
            "debug": False,
            "num_workers": MAX_CPU_WORKERS,
            "weighted_keywords": True,
            "enable_fixed_block": False,
            "enable_sft": False,
            "disable_stop_block": False,
            "enable_repocoder": False,
            "enable_prediction": True,
            "enable_oracle": False,
            "number_sample": 8,
            "vllm_generator_batch_size_per_gpu": 10000000,
            "feedback_signal": "ppl",
            "add_api_blocks": True,
            "temperature1": 0.8,
            "top_p1": 0.95,
            "rlcoder_model_path": str(retriever_path),
            "do_codereval": False,
            "enable_forward_generation": False,
            "forward_generation_times": 1,
            "epoch": 1,
            "inner_epoch": 1,
            "batch_size": 16,
            "sample_number": 10,
            "data_per_epoch": 2000,
            "lr": 5e-5,
        }
        for key, value in values.items():
            setattr(args, key, value)
        args.generator_batch_size = values["generator_batch_size_per_gpu"]
        args.retriever_batch_size = values["retriever_batch_size_per_gpu"]
        args.vllm_generator_batch_size = values["vllm_generator_batch_size_per_gpu"]

        log("starting stock AlignCoder main.run(args) with RQ1 settings")
        aligncoder_main.run(args)
        manifest["status"] = "completed"
        manifest["dataset_counts"] = state["dataset_counts"]
        manifest["metrics"] = state["dataset_metrics"]
        return {"run_id": RUN_ID, "status": "completed", "metrics": state["dataset_metrics"]}
    except BaseException as exc:
        manifest["status"] = "failed_or_timed_out"
        manifest["error"] = repr(exc)
        manifest["traceback"] = traceback.format_exc()
        manifest["dataset_counts"] = state["dataset_counts"]
        manifest["metrics_so_far"] = state["dataset_metrics"]
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["trace_records_written"] = writer.count
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        observability_volume.commit()


@app.local_entrypoint()
def main() -> None:
    if not ALIGNCODER_DIR.is_dir():
        raise FileNotFoundError(f"AlignCoder source snapshot not found: {ALIGNCODER_DIR}")
    print(f"Run ID: {RUN_ID}")
    print(f"AlignCoder source: {ALIGNCODER_DIR}")
    print(f"Source git commit: {ALIGNCODER_REPO_COMMIT}")
    print(f"Observability volume: {OBSERVABILITY_VOLUME_NAME}")
    print(
        f"Hard cap: {HARD_TIMEOUT_SECONDS / 3600:.1f}h; one A100-80GB, "
        f"{CPU_PHYSICAL_CORES} physical CPU cores ({CPU_PHYSICAL_CORES * 2} vCPUs, "
        f"max {MAX_CPU_WORKERS} process workers), "
        f"{MEMORY_MIB / 1024:.0f} GiB RAM; one container, no retries. "
        f"Estimated compute cap=${ESTIMATED_MAX_COMPUTE_USD:.2f} of ${BUDGET_USD:.2f} budget; "
        "estimate uses current published rates and excludes any non-compute charges."
    )
    function_call = evaluate_aligncoder_rq1.spawn(
        RUN_ID, ALIGNCODER_REPO_COMMIT, LOCAL_SOURCE_SHA256
    )
    print(f"FunctionCall ID: {function_call.object_id}")
    print(f"Submitted detached run: {RUN_ID}")

