import os
import subprocess
import shutil
import json
from pathlib import Path

print("=== 1. Installing required packages ===", flush=True)
subprocess.check_call([
    "pip", "install", "-q",
    "vllm", "fuzzywuzzy", "python-Levenshtein", "editdistance",
    "tree-sitter==0.20.4", "huggingface_hub", "datasets", "transformers",
    "timeout-decorator", "func-timeout", "sacrebleu", "rank_bm25", "prettytable", "nltk"
])

print("=== 2. Setting up RLCoder repository ===", flush=True)
repo_path = Path("/content/RLCoder-official")
if not repo_path.is_dir():
    subprocess.check_call(["git", "clone", "https://github.com/DeepSoftwareAnalytics/RLCoder", str(repo_path)])
    subprocess.check_call(["git", "checkout", "164d8d88cde324a38f5da70c4f858cc4679ef08e"], cwd=repo_path)

print("=== 3. Building Tree-sitter parsers ===", flush=True)
build_dir = repo_path / "utils" / "build"
build_dir.mkdir(parents=True, exist_ok=True)

from tree_sitter import Language
if not (build_dir / "python-lang-parser.so").is_file():
    py_ts = repo_path / "tree-sitter-python"
    if not py_ts.is_dir():
        subprocess.check_call(["git", "clone", "https://github.com/tree-sitter/tree-sitter-python", str(py_ts)])
    Language.build_library(str(build_dir / "python-lang-parser.so"), [str(py_ts)])
    print("Built python-lang-parser.so", flush=True)

if not (build_dir / "java-lang-parser.so").is_file():
    java_ts = repo_path / "tree-sitter-java"
    if not java_ts.is_dir():
        subprocess.check_call(["git", "clone", "https://github.com/tree-sitter/tree-sitter-java", str(java_ts)])
    Language.build_library(str(build_dir / "java-lang-parser.so"), [str(java_ts)])
    print("Built java-lang-parser.so", flush=True)

print("=== 4. Downloading Data and Models from Hugging Face Hub ===", flush=True)
from huggingface_hub import snapshot_download

data_dir = repo_path / "data"
if not (data_dir / "cceval" / "python").is_dir():
    print("Downloading test data nov3630/Data4RLCoder...", flush=True)
    snapshot_download("nov3630/Data4RLCoder", repo_type="dataset", local_dir=str(data_dir), revision="cb9639f20f2374d75e5b0b8e8650f1b20802bf9f")

models_dir = Path("/content/models")
models_dir.mkdir(parents=True, exist_ok=True)

retriever_dir = models_dir / "RLRetriever"
if not retriever_dir.is_dir():
    print("Downloading retriever nov3630/RLRetriever...", flush=True)
    snapshot_download("nov3630/RLRetriever", local_dir=str(retriever_dir), revision="ec587f5d8635462fff3cfb06f9a946148acda08b")

generator_dir = models_dir / "deepseek-coder-1.3b-base"
if not generator_dir.is_dir():
    print("Downloading generator deepseek-ai/deepseek-coder-1.3b-base...", flush=True)
    snapshot_download("deepseek-ai/deepseek-coder-1.3b-base", local_dir=str(generator_dir), revision="c919139c3a9b4070729c8b2cca4847ab29ca8d94")

print("=== 5. Applying Tokenizer Patch ===", flush=True)
cfg_path = generator_dir / "tokenizer_config.json"
if cfg_path.is_file():
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("tokenizer_class") != "PreTrainedTokenizerFast":
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print("Patched generator tokenizer_config.json -> PreTrainedTokenizerFast", flush=True)

print("=== Remote Environment Setup Complete! ===", flush=True)
