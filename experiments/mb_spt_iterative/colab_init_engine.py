#!/usr/bin/env python3
"""Initialize persistent vLLM engine (8k context) and full CrossCodeEval dataset (4,804 tasks) in Colab kernel."""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
import tarfile
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import Any

ROOT_DIR = Path("/content/mb_spt")
BUNDLE_PATH_FULL = Path("/content/full_benchmark_8004_bundle.tar.gz")
BUNDLE_PATH_FAST = Path("/content/fast_eval_bundle.tar.gz")
GENERATOR = "deepseek-ai/deepseek-coder-1.3b-base"
GENERATOR_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
CROSSFILE_TOKENS = 1536
GENERATOR_CONTEXT = 2048
MAX_MODEL_LEN = 8192
GENERATOR_MAX_NEW = 64
SEED = 20261024


def ensure_deps() -> None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        version("vllm")
    except PackageNotFoundError:
        print("Installing vLLM...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv_path = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
        subprocess.check_call([uv_path, "pip", "install", "--system", "vllm", "--torch-backend=auto"])
    try:
        import torchaudio  # noqa: F401
    except ImportError:
        pass
    except RuntimeError as exc:
        if "different CUDA versions" in str(exc):
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"])
    missing = []
    for mod, pkg in (("fuzzywuzzy", "fuzzywuzzy"), ("Levenshtein", "python-Levenshtein"), ("rank_bm25", "rank_bm25"), ("nltk", "nltk")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    try:
        if version("tree-sitter") != "0.20.4":
            missing.append("tree-sitter==0.20.4")
    except PackageNotFoundError:
        missing.append("tree-sitter==0.20.4")
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def init_globals(global_ns: dict[str, Any]) -> None:
    ensure_deps()
    import torch
    from transformers import AutoTokenizer
    from tree_sitter import Language, Parser
    from vllm import LLM

    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    if not (ROOT_DIR / "tasks.jsonl.gz").exists():
        target_bundle = BUNDLE_PATH_FULL if BUNDLE_PATH_FULL.exists() else BUNDLE_PATH_FAST
        if target_bundle.exists():
            print(f"Extracting {target_bundle.name}...", flush=True)
            try:
                with tarfile.open(target_bundle, "r:gz") as tar:
                    tar.extractall(ROOT_DIR, filter="data")
            except TypeError:
                with tarfile.open(target_bundle, "r:gz") as tar:
                    tar.extractall(ROOT_DIR)
        else:
            print("Warning: Neither bundle tar found on disk!", flush=True)

    tok = AutoTokenizer.from_pretrained(GENERATOR, revision=GENERATOR_REVISION, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if "VLLM_ENGINE" not in global_ns or global_ns["VLLM_ENGINE"] is None:
        print(f"Loading persistent vLLM engine: {GENERATOR} (max_model_len={MAX_MODEL_LEN}, eager=True)...", flush=True)
        global_ns["VLLM_ENGINE"] = LLM(
            model=GENERATOR,
            tokenizer=GENERATOR,
            revision=GENERATOR_REVISION,
            tokenizer_revision=GENERATOR_REVISION,
            dtype="float16",
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=128,
            max_num_batched_tokens=16384,
            gpu_memory_utilization=0.85,
            seed=SEED,
            enforce_eager=True,
            disable_log_stats=True,
        )

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if torch.cuda.is_available() else 0.0
    print(f"Colab GPU ready: {gpu_name} ({gpu_gib:.1f} GiB)", flush=True)

    with gzip.open(ROOT_DIR / "tasks.jsonl.gz", "rt", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f]
    with gzip.open(ROOT_DIR / "retrieval_trace.jsonl.gz", "rt", encoding="utf-8") as f:
        traces = {(r["benchmark"], r["task_id"]): r for r in map(json.loads, f)}
    with gzip.open(ROOT_DIR / "files.jsonl.gz", "rt", encoding="utf-8") as f:
        files = {r["file_id"]: {"path": r["path"], "text": r["text"]} for r in map(json.loads, f)}
    
    pools = {}
    if (ROOT_DIR / "pools.jsonl.gz").exists():
        with gzip.open(ROOT_DIR / "pools.jsonl.gz", "rt", encoding="utf-8") as f:
            pools = {r["pool_id"]: r["file_ids"] for r in map(json.loads, f)}

    parser_map = {}
    for lang in ("python", "java"):
        so_path = ROOT_DIR / "parsers" / f"{lang}-lang-parser.so"
        language = Language(str(so_path), lang)
        p = Parser()
        p.set_language(language)
        parser_map[lang] = p

    global_ns["MB_DATA"] = {
        "tasks": tasks,
        "traces": traces,
        "files": files,
        "pools": pools,
        "parser_map": parser_map,
        "tokenizer": tok,
        "gpu_name": gpu_name,
    }
    print(f"Initialized persistent MB_DATA ({len(tasks)} tasks, {len(files)} files) and VLLM_ENGINE.", flush=True)


if __name__ == "__main__":
    init_globals(globals())
