#!/usr/bin/env python3
"""
colab_cadre_setup.py - Remote setup for CADRE Consolidated on Colab VM.
"""

import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

print("=== 1. Extracting CADRE and Benchmark Bundles ===", flush=True)
if Path("/content/cadre_official.tar.gz").is_file():
    print("Extracting cadre_official.tar.gz...", flush=True)
    with tarfile.open("/content/cadre_official.tar.gz", "r:gz") as tar:
        tar.extractall("/content")

if Path("/content/full_benchmark_8004_bundle.tar.gz").is_file():
    if not Path("/content/tasks.jsonl.gz").is_file():
        print("Extracting full_benchmark_8004_bundle.tar.gz...", flush=True)
        with tarfile.open("/content/full_benchmark_8004_bundle.tar.gz", "r:gz") as tar:
            tar.extractall("/content")

print("=== 2. Installing Dependencies ===", flush=True)
subprocess.check_call([
    "pip", "install", "-q",
    "vllm", "rank_bm25", "fuzzywuzzy", "python-Levenshtein", "editdistance",
    "tree-sitter==0.20.4", "huggingface_hub", "transformers", "accelerate"
])

print("=== 3. Setting up Tree-Sitter Parsers ===", flush=True)
parsers_dir = Path("/content/parsers")
parsers_dir.mkdir(parents=True, exist_ok=True)

try:
    from tree_sitter import Language
    if not (parsers_dir / "python-lang-parser.so").is_file():
        py_ts = Path("/content/tree-sitter-python")
        if not py_ts.is_dir():
            subprocess.check_call(["git", "clone", "https://github.com/tree-sitter/tree-sitter-python", str(py_ts)])
        Language.build_library(str(parsers_dir / "python-lang-parser.so"), [str(py_ts)])
        print("Built python-lang-parser.so", flush=True)

    if not (parsers_dir / "java-lang-parser.so").is_file():
        java_ts = Path("/content/tree-sitter-java")
        if not java_ts.is_dir():
            subprocess.check_call(["git", "clone", "https://github.com/tree-sitter/tree-sitter-java", str(java_ts)])
        Language.build_library(str(parsers_dir / "java-lang-parser.so"), [str(java_ts)])
        print("Built java-lang-parser.so", flush=True)
except Exception as e:
    print(f"Tree-sitter build note: {e}", flush=True)

print("=== 4. Downloading and Patching DeepSeek-Coder-1.3B Tokenizer ===", flush=True)
from huggingface_hub import snapshot_download

generator_dir = Path("/content/models/deepseek-coder-1.3b-base")
generator_dir.mkdir(parents=True, exist_ok=True)
snapshot_download(
    "deepseek-ai/deepseek-coder-1.3b-base",
    local_dir=str(generator_dir),
    revision="c919139c3a9b4070729c8b2cca4847ab29ca8d94",
    allow_patterns=["*.json", "*.model", "*.txt"]
)

cfg_path = generator_dir / "tokenizer_config.json"
if cfg_path.is_file():
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("tokenizer_class") != "PreTrainedTokenizerFast":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print("Patched tokenizer_config.json -> PreTrainedTokenizerFast", flush=True)

print("=== Remote Environment Setup Complete! ===", flush=True)
