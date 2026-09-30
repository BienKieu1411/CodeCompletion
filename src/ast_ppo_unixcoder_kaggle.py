# %% [markdown]
# # AST retrieval + custom RRPO (true PPO-Clip) — Kaggle T4×2 + Modal A100
# Run cells top to bottom. This notebook is self-contained: upload only the .ipynb.
# Modal A100-80GB serves frozen DeepSeek-Coder-1.3B via authenticated vLLM;
# both Kaggle T4s train UniXcoder-base plus the action head with DataParallel.
# UniXcoder encodes 128 candidate sequences per pass (64 sequences per T4).
# The inactive value head is retained for
# checkpoint/evaluation architecture compatibility. No pairwise or CE winner loss.
# Defaults to `RUN_MODE="full"`; change to `"smoke"` for a quick pipeline test.
# The deployed Modal URL is configured below. Add only its Proxy Token to Kaggle
# Secrets as `MODAL_PROXY_TOKEN`; the notebook waits for vLLM,
# checks model/context, probes generation, and then starts RRPO automatically.
# Attach the `code-completion-train-dataset` Kaggle dataset. This notebook reads
# `/kaggle/input/datasets/bienkieu/code-completion-train-dataset/train.parquet`,
# containing fixed gold targets and prebuilt Python+Java AST/BM25 candidate pools.
# Each epoch visits every train row once.
# The training data path does not parse AST or mine candidates on Kaggle.
# The generator accepts 2,048 prompt tokens plus 96 completion tokens. It sees an
# exact cursor suffix plus relevant pre-cursor AST anchors when the prefix is long.
# Custom RRPO is the primary objective: final completion utility (ES + identifier
# F1) dominates prefix utility; STOP repeats its final utility through five slots.
# A frozen BM25-AST policy completes each actual pre-action state as the GAE
# reference baseline. PPO ratios use the rollout policy (not RRPO's pi_ref ratio).
# Full mode repeatedly visits the entire fixed train split, reshuffling between
# epochs, until the 7-hour wall-clock guard; there is no epoch-count cap. Each
# outer RRPO rollout/update batch contains 64 examples.
# `latest.pt` is saved after every completed epoch, before validation, and at
# the time guard before optional final validation. The
# last Modal generation request is cut off 5 minutes before the 7-hour hard stop;
# the server then scales its A100 to zero. The endpoint stays deployed but idle.
# The coarse BM25 AST pool is fixed within every PPO rollout/update cycle, so the
# learned component is a full-encoder retriever *selector*, not a global ANN policy.
# Pool=64 and slate<=5; AST snippets retain the existing 384-token cap. The
# generator input cap is 2,048 tokens; at most 1,344 is reserved for retrieved
# cross-file snippets, leaving up to 704 for the current-file path/left context.
# vLLM accepts up to 108 prompts per API request. This is generation batch size,
# not Modal HTTP concurrency. Startup PPO invariants use one prepared train row.
# This does not recover snippets missed by top-24 file selection or pool mining.
# Resume: the notebook accepts `latest.pt` or `latest` (some downloads lose the
# suffix) in the `resume-checkpoint` dataset. If neither is attached, it uses the
# Kaggle CLI to download the prior kernel output into `/kaggle/working`. A sibling
# `generation_cache.sqlite3` is copied into this run's output directory.
# The optimizer, AMP scaler, RNG, epoch order and cursor resume from a matching
# checkpoint. If resuming a checkpoint scored on the old 128-row validation
# subset, its weights/optimizer are kept but its old best score is reset.
# Attach the training dataset. The checkpoint dataset is optional if the Kaggle
# API credentials are available to download the prior kernel output.
# This training configuration skips CCEval/RepoEval test data; use the separate eval
# notebook after training. Data/prompt schema v5 uses one prebuilt row
# per eligible repository; checkpoints from a different dataset/objective cannot resume.

# %%
import importlib.util
import importlib.metadata
import faulthandler
import hashlib
import json
import keyword
import math
import os
import random
import re
import sqlite3
import shutil
import subprocess
import sys
import time
import zlib
from datetime import datetime, timedelta
from pathlib import Path

faulthandler.enable(all_threads=True)

RUN_MODE = "full"  # Set to "smoke" for a quick pipeline test.
SEED = 17
TRAIN_DATA_PATH = Path(
    "/kaggle/input/datasets/bienkieu/code-completion-train-dataset/train.parquet"
)
ARTIFACT_SCHEMA = "rrpo_full_repo_one_example_prebuilt_v1"
WORK_DIR = Path("/kaggle/working/ast_rrpo_unixcoder_v6_k5_ctx2048_xfb1344")
WORK_DIR.mkdir(parents=True, exist_ok=True)
FATAL_LOG_PATH = WORK_DIR / "python_fatal.log"
FATAL_LOG_STREAM = FATAL_LOG_PATH.open("a", encoding="utf-8", buffering=1)
FATAL_LOG_STREAM.write(f"Kernel run started: pid={os.getpid()} time={time.time()}\n")
faulthandler.enable(file=FATAL_LOG_STREAM, all_threads=True)
print("Fatal traceback log:", FATAL_LOG_PATH, flush=True)
CACHE_DIR = Path("/kaggle/working/hf-cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
START_MONOTONIC = time.monotonic()
RESUME_CHECKPOINT_PATH = None  # Optional explicit path; accepts `latest` too.
RESUME_CHECKPOINT_DIR = Path(
    "/kaggle/input/datasets/bienkieu/resume-checkpoint"
)
AUTO_DISCOVER_RESUME = True
DOWNLOAD_PREVIOUS_KAGGLE_OUTPUT = True
PREVIOUS_KERNEL_SLUG = "bienkieu/codecompletion"

def find_resume_checkpoint():
    if RESUME_CHECKPOINT_PATH is not None:
        explicit = Path(RESUME_CHECKPOINT_PATH)
        if explicit.is_file():
            return explicit
        extensionless = explicit.with_suffix("") if explicit.suffix == ".pt" else None
        if extensionless is not None and extensionless.is_file():
            return extensionless
        raise FileNotFoundError(f"Configured resume checkpoint not found: {explicit}")
    if not AUTO_DISCOVER_RESUME:
        return None
    # Prefer the explicitly attached resume dataset over leftovers from a
    # previous attempt in this Kaggle working directory.
    for directory in (RESUME_CHECKPOINT_DIR, WORK_DIR):
        for filename in ("latest.pt", "latest"):
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    input_root = Path("/kaggle/input")
    candidates = (
        sorted(path for filename in ("latest.pt", "latest")
               for path in input_root.rglob(filename)
               if WORK_DIR.name in path.parts)
        if input_root.exists() else []
    )
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple attached latest.pt checkpoints found. Set "
            "RESUME_CHECKPOINT_PATH to the intended absolute path: " +
            ", ".join(str(path) for path in candidates)
        )
    return candidates[0] if candidates else None

def prepare_kaggle_cli_auth():
    if shutil.which("kaggle") is None:
        print("Kaggle CLI missing; installing the lightweight kaggle package.", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "kaggle"],
            check=True,
        )
    if (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return
    if (Path.home() / ".kaggle" / "kaggle.json").is_file():
        return
    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
    except Exception as exc:
        raise RuntimeError(
            "To download the previous Kaggle output, add Kaggle Secrets named "
            "KAGGLE_USERNAME and KAGGLE_KEY (Kaggle API credentials). The Modal "
            "proxy token does not authenticate the Kaggle CLI."
        ) from exc

def download_previous_kaggle_output():
    prepare_kaggle_cli_auth()
    file_pattern = (
        rf"{re.escape(WORK_DIR.name)}/"
        r"(?:latest(?:\.pt)?|best\.pt|generation_cache\.sqlite3|"
        r"ast_bm25_token_ids_rrpo_v2\.pt)$"
    )
    command = [
        shutil.which("kaggle"), "kernels", "output", PREVIOUS_KERNEL_SLUG,
        "-p", "/kaggle/working", "--file-pattern", file_pattern,
    ]
    print("Downloading prior Kaggle checkpoint with:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    expected_checkpoints = (WORK_DIR / "latest.pt", WORK_DIR / "latest")
    if not any(path.is_file() for path in expected_checkpoints):
        raise FileNotFoundError(
            f"Kaggle output download completed but neither {expected_checkpoints[0]} "
            f"nor {expected_checkpoints[1]} exists. Check that the source kernel "
            "has a saved output containing its latest checkpoint."
        )

RESUME_SOURCE = find_resume_checkpoint()
if (RESUME_SOURCE is None and AUTO_DISCOVER_RESUME
        and DOWNLOAD_PREVIOUS_KAGGLE_OUTPUT):
    download_previous_kaggle_output()
    RESUME_SOURCE = find_resume_checkpoint()
if RESUME_SOURCE is not None:
    print("Resume source:", RESUME_SOURCE)

RETRIEVER_MODEL = "microsoft/unixcoder-base"
GENERATOR_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
SERVED_MODEL_NAME = "deepseek-coder-1.3b-base"
API_BASE = ""  # Derived from the configured Modal endpoint URL below.
MODAL_ENDPOINT_ROOT = (
    "https://bien14112005--bienkieu-ast-rrpo-deepseek-vllm-deepseekvl-3f59be.us-east.modal.direct"
)
VLLM_MAX_NUM_SEQS = 108
VLLM_GPU_MEMORY_UTILIZATION = 0.90
VLLM_MAX_BATCHED_TOKENS = 16384
VLLM_SERVER_VERSION = "0.21.0"
ENDPOINT_READY_TIMEOUT_SECONDS = 900
ENDPOINT_RECOVERY_TIMEOUT_SECONDS = 300
GENERATOR_REQUEST_TIMEOUT_SECONDS = 600
GENERATOR_RETRY_BACKOFF_SECONDS = 3
GENERATOR_INPUT_TOKENS = 2048
GENERATOR_OUTPUT_TOKENS = 96
GENERATOR_MAX_MODEL_LEN = GENERATOR_INPUT_TOKENS + GENERATOR_OUTPUT_TOKENS
MODAL_SCALEDOWN_WINDOW_SECONDS = 5 * 60
MAX_RELATED_FILES = 24
AST_CHUNK_TOKENS = 384
DATA_PIPELINE_SCHEMA = "phong_ast_boundary_prebuilt_repo_rows_v5"
SCORE_SCHEMA = "rlcoder_stmt_postprocess_em_es_idf1_v2"
RETRIEVER_QUERY_LENGTH = 256
RETRIEVER_CANDIDATE_LENGTH = 512
CANDIDATE_POOL_SIZE = 64
ENCODE_BATCH_SIZE = 128
CROSSFILE_TOKEN_BUDGET = 1344
GENERATOR_CURRENT_FILE_BUDGET = GENERATOR_INPUT_TOKENS - CROSSFILE_TOKEN_BUDGET
MAX_SLATE_STEPS = 5
RRPO_FINAL_WEIGHT = 0.7
RRPO_IDENTIFIER_WEIGHT = 0.2
RRPO_GAMMA = 1.0  # Undiscounted final-plus-prefix utility.
RRPO_GAE_LAMBDA = 0.95
RRPO_REFERENCE = "bm25_ast_from_actual_state_v1"
PPO_CLIP = 0.2
PPO_PASSES = 2
ROLLOUT_QUERIES = 64
ACCUMULATION_STEPS = 8
ENCODER_LR = 5e-5
HEAD_LR = 1e-4
ENTROPY_COEF = 0.001
TARGET_KL = 0.02
MAX_GRAD_NORM = 5.0
SAVE_INTERVAL_SECONDS = 1200
# Modal meter: $10.90 used this month; with the $30 cap, reserve about $1 and
# allow at most 7h at ~$2.588/h (A100-80GB + 0.5 CPU + 8 GiB memory).
TRAIN_STOP_HOURS_FROM_KERNEL_START = 7.0
STOP_NEW_BATCH_RESERVE_SECONDS = 600
FINAL_VALIDATION_MIN_REMAINING_SECONDS = 1200


class TrainingDeadlineReached(RuntimeError):
    """Graceful RRPO stop that leaves time for checkpointing and Modal scale-down."""


class GeneratorRequestError(RuntimeError):
    """A transient or exhausted Modal/vLLM request failure, safe to retry later."""


def modal_last_request_deadline():
    hard_deadline = (START_MONOTONIC +
                     TRAIN_STOP_HOURS_FROM_KERNEL_START * 3600)
    return hard_deadline - MODAL_SCALEDOWN_WINDOW_SECONDS

if RUN_MODE == "smoke":
    TRAIN_EXAMPLES = 24
    VALID_EXAMPLES = 8
    TRAIN_EPISODES_CAP = 32
    VALIDATE_EVERY_EPISODES = 32
elif RUN_MODE == "full":
    TRAIN_EXAMPLES = None  # Filled from the attached Parquet train split.
    VALID_EXAMPLES = None  # Use the complete fixed validation split.
    TRAIN_EPISODES_CAP = None  # No epoch/episode cap; stop only at the time guard.
    VALIDATE_EVERY_EPISODES = None  # Resolved to about twice per train epoch.
else:
    raise ValueError("RUN_MODE must be 'smoke' or 'full'")
print("Mode:", RUN_MODE, "Output:", WORK_DIR,
      "Train rows per epoch:", TRAIN_EXAMPLES or "all attached train rows",
      "Max hours from kernel start:", TRAIN_STOP_HOURS_FROM_KERNEL_START)
print(f"Generator allocation: max input {GENERATOR_INPUT_TOKENS}; cross-file "
      f"snippets <= {CROSSFILE_TOKEN_BUDGET}; current-file path + left context "
      f"share up to {GENERATOR_CURRENT_FILE_BUDGET} tokens; output "
      f"{GENERATOR_OUTPUT_TOKENS} tokens. Prompt packing enforces the total cap.",
      flush=True)

# %%
LIGHT_PACKAGES = [
    ("tree-sitter", "tree_sitter"),
    ("tree-sitter-python", "tree_sitter_python"),
    ("tree-sitter-java", "tree_sitter_java"),
    ("rank-bm25", "rank_bm25"),
    ("fuzzywuzzy", "fuzzywuzzy"),
    ("editdistance", "editdistance"),
]
missing = [package for package, module in LIGHT_PACKAGES
           if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests

try:
    from kaggle_secrets import UserSecretsClient
    _kaggle_secrets = UserSecretsClient()
    MODAL_PROXY_TOKEN = _kaggle_secrets.get_secret("MODAL_PROXY_TOKEN").strip()
except Exception as exc:
    raise RuntimeError(
        "Add Kaggle Secret MODAL_PROXY_TOKEN (combined token ID.secret) before running."
    ) from exc

if not MODAL_ENDPOINT_ROOT.startswith("https://") or not MODAL_PROXY_TOKEN:
    raise ValueError("Modal URL must be HTTPS and the Proxy Token must be non-empty.")
API_BASE = (MODAL_ENDPOINT_ROOT if MODAL_ENDPOINT_ROOT.endswith("/v1")
            else MODAL_ENDPOINT_ROOT + "/v1")
MODAL_HEALTH_URL = API_BASE[:-3] + "/health"
MODAL_AUTH_HEADERS = {"Authorization": f"Bearer {MODAL_PROXY_TOKEN}"}
del MODAL_PROXY_TOKEN, _kaggle_secrets

import torch
import editdistance
from fuzzywuzzy import fuzz
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
import tree_sitter as ts
import tree_sitter_java as ts_java
import tree_sitter_python as ts_python

print("AST runtime:", {name: importlib.metadata.version(name)
                       for name in ("tree-sitter", "tree-sitter-python",
                                    "tree-sitter-java", "transformers", "tokenizers")},
      flush=True)

if torch.cuda.device_count() != 2:
    raise RuntimeError("Select Kaggle GPU T4×2 before running this notebook.")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GiB")
torch.cuda.set_device(0)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
GEN_TOKENIZER = AutoTokenizer.from_pretrained(GENERATOR_MODEL, cache_dir=str(CACHE_DIR))
print("Generator tokenizer:", GEN_TOKENIZER.__class__.__name__, flush=True)
RET_TOKENIZER = AutoTokenizer.from_pretrained(
    RETRIEVER_MODEL, cache_dir=str(CACHE_DIR), use_fast=False
)
if RET_TOKENIZER.pad_token_id is None:
    raise RuntimeError("UniXcoder tokenizer must have a pad token.")
if RET_TOKENIZER.convert_tokens_to_ids("<encoder-only>") == RET_TOKENIZER.unk_token_id:
    raise RuntimeError("UniXcoder tokenizer lacks <encoder-only> mode token.")

# Persist exact, deterministic generator responses across epochs and Kaggle
# sessions. The namespace makes old entries unusable if the model/tokenizer or
# decoding settings change; only completed responses are cached.
GENERATION_CACHE_PATH = WORK_DIR / "generation_cache.sqlite3"
attached_generation_cache = (
    RESUME_SOURCE.with_name(GENERATION_CACHE_PATH.name)
    if RESUME_SOURCE is not None else None
)
if (not GENERATION_CACHE_PATH.is_file() and attached_generation_cache is not None
        and attached_generation_cache.is_file()
        and attached_generation_cache.resolve() != GENERATION_CACHE_PATH.resolve()):
    shutil.copy2(attached_generation_cache, GENERATION_CACHE_PATH)
    print("Carried deterministic-generation cache from", attached_generation_cache,
          flush=True)
GENERATION_CACHE_SIGNATURE = {
    "schema": "exact_prompt_deterministic_completion_v1",
    "model": GENERATOR_MODEL,
    "served_model": SERVED_MODEL_NAME,
    "engine": f"vllm=={VLLM_SERVER_VERSION}",
    "tokenizer_revision": (
        getattr(GEN_TOKENIZER, "_commit_hash", None)
        or getattr(GEN_TOKENIZER, "init_kwargs", {}).get("_commit_hash")
        or "unresolved"
    ),
    "input_tokens": GENERATOR_INPUT_TOKENS,
    "output_tokens": GENERATOR_OUTPUT_TOKENS,
    "temperature": 0.0,
    "top_p": 1.0,
    "prompt_encoding": "tokenizer_encode_add_special_tokens_v1",
}
GENERATION_CACHE_CONNECTION = sqlite3.connect(
    GENERATION_CACHE_PATH, timeout=60
)
GENERATION_CACHE_CONNECTION.execute("PRAGMA synchronous=FULL")
GENERATION_CACHE_CONNECTION.execute(
    "CREATE TABLE IF NOT EXISTS generation_cache ("
    "cache_key TEXT PRIMARY KEY, completion TEXT NOT NULL)"
)
if GENERATION_CACHE_CONNECTION.execute("PRAGMA quick_check").fetchone()[0] != "ok":
    raise RuntimeError(f"Generation response cache failed integrity check: "
                       f"{GENERATION_CACHE_PATH}")
GENERATION_CACHE_CONNECTION.commit()
print("Deterministic-generation cache:", GENERATION_CACHE_PATH,
      "entries:", GENERATION_CACHE_CONNECTION.execute(
          "SELECT COUNT(*) FROM generation_cache").fetchone()[0], flush=True)

# %%
if not TRAIN_DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Training Parquet not found at {TRAIN_DATA_PATH}. Attach the Kaggle "
        "dataset bienkieu/code-completion-train-dataset as an Input."
    )
TRAIN_DATA_PATH = TRAIN_DATA_PATH.resolve()

print("Using prebuilt train Parquet:", TRAIN_DATA_PATH, flush=True)

# %%
RETRIEVAL_LINE_NODE_TYPES = {
    "python": {"expression_statement", "assignment", "augmented_assignment",
               "return_statement", "raise_statement", "assert_statement",
               "pass_statement", "break_statement", "continue_statement"},
    "java": {"expression_statement", "local_variable_declaration",
             "field_declaration", "return_statement", "throw_statement",
             "break_statement", "continue_statement", "assert_statement"},
}
RETRIEVAL_BLOCK_NODE_TYPES = {
    "python": {"if_statement", "for_statement", "while_statement",
               "with_statement", "try_statement", "function_definition",
               "class_definition"},
    "java": {"if_statement", "for_statement", "enhanced_for_statement",
             "while_statement", "try_statement", "switch_expression",
             "method_declaration", "constructor_declaration", "class_declaration"},
}
RETRIEVAL_NODE_TYPES = {
    "python": RETRIEVAL_LINE_NODE_TYPES["python"] | RETRIEVAL_BLOCK_NODE_TYPES["python"] |
              {"import_statement", "import_from_statement", "comment",
               "decorated_definition"},
    "java": RETRIEVAL_LINE_NODE_TYPES["java"] | RETRIEVAL_BLOCK_NODE_TYPES["java"] |
            {"package_declaration", "import_declaration", "interface_declaration",
             "enum_declaration", "record_declaration", "comment"},
}
SCOPE_NODE_TYPES = {"function_definition", "class_definition", "method_declaration",
                    "constructor_declaration", "class_declaration", "interface_declaration",
                    "enum_declaration", "record_declaration"}
IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
PARSERS = {
    "python": ts.Parser(ts.Language(ts_python.language())),
    "java": ts.Parser(ts.Language(ts_java.language())),
}

def token_count(text):
    return len(GEN_TOKENIZER.encode(text, add_special_tokens=False))

def ast_chunks(path, code, language, max_tokens=AST_CHUNK_TOKENS):
    """Entity/statement chunks throughout a file; line fallback is explicitly tagged."""
    source = code.encode("utf-8", errors="replace")
    tree = PARSERS[language].parse(source)
    pieces = []

    def emit(start, end, kind):
        if start >= end:
            return
        body = source[start:end].decode("utf-8", errors="replace")
        if body.strip() and token_count(body) <= max_tokens:
            pieces.append({"path": path, "text": body, "type": kind,
                           "start": int(start), "end": int(end)})

    if tree.root_node.has_error:
        # Bad/missing syntax is visible in telemetry, never mislabeled as AST.
        start = 0
        for line in source.splitlines(keepends=True):
            end = start + len(line)
            emit(start, end, "fallback_line")
            start = end
        return pieces, True

    def visit(node):
        if node.type == "ERROR":
            return
        if node.type in RETRIEVAL_NODE_TYPES[language]:
            body = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            if token_count(body) <= max_tokens:
                emit(node.start_byte, node.end_byte, node.type)
                return
            if node.type in SCOPE_NODE_TYPES or node.type in RETRIEVAL_BLOCK_NODE_TYPES[language]:
                first_body = node.child_by_field_name("body")
                if first_body is None:
                    first_body = next((child for child in node.named_children
                                       if child.type in {"block", "class_body"}), None)
                if first_body is not None:
                    emit(node.start_byte, first_body.start_byte, node.type + ":header")
        for child in node.named_children:
            visit(child)

    visit(tree.root_node)
    unique = {(p["start"], p["end"]): p for p in pieces}
    return list(unique.values()), False

TRAIN_PARQUET = pq.ParquetFile(TRAIN_DATA_PATH)
ARTIFACT_INFO = TRAIN_PARQUET.schema_arrow.metadata or {}
if ARTIFACT_INFO.get(b"artifact_schema") != ARTIFACT_SCHEMA.encode("ascii"):
    raise RuntimeError("Attached train Parquet has an incompatible artifact schema")
if (ARTIFACT_INFO.get(b"generator_tokenizer") != GENERATOR_MODEL.encode("ascii") or
        ARTIFACT_INFO.get(b"retriever_tokenizer") != RETRIEVER_MODEL.encode("ascii")):
    raise RuntimeError("Attached train Parquet was encoded with different tokenizers")
if any(TRAIN_PARQUET.metadata.row_group(i).num_rows != 1
       for i in range(TRAIN_PARQUET.num_row_groups)):
    raise RuntimeError("Train Parquet needs one row per row group for bounded random access")
TRAIN_INDEX = TRAIN_PARQUET.read(
    columns=["split", "language", "repo_id", "task_id"]
).to_pylist()
if len(TRAIN_INDEX) != TRAIN_PARQUET.num_row_groups:
    raise RuntimeError("Train Parquet row-group index is inconsistent")
if len({item["task_id"] for item in TRAIN_INDEX}) != len(TRAIN_INDEX):
    raise RuntimeError("Train Parquet contains duplicate repository tasks")
TRAIN_ROW_GROUPS = [i for i, item in enumerate(TRAIN_INDEX) if item["split"] == "train"]
all_valid_group_ids = [i for i, item in enumerate(TRAIN_INDEX)
                       if item["split"] == "valid"]
if len(TRAIN_ROW_GROUPS) + len(all_valid_group_ids) != len(TRAIN_INDEX):
    raise RuntimeError("Train Parquet contains an unknown split")
if RUN_MODE == "smoke":
    TRAIN_ROW_GROUPS = TRAIN_ROW_GROUPS[:TRAIN_EXAMPLES]
TRAIN_EXAMPLES = len(TRAIN_ROW_GROUPS)
if RUN_MODE == "full":
    VALIDATE_EVERY_EPISODES = max(ROLLOUT_QUERIES, math.ceil(TRAIN_EXAMPLES / 2))
if VALID_EXAMPLES is None:
    VALID_ROW_GROUPS = list(all_valid_group_ids)
else:
    VALID_ROW_GROUPS = random.Random(SEED + 2).sample(
        all_valid_group_ids, min(VALID_EXAMPLES, len(all_valid_group_ids)))
VALID_EXAMPLES = len(VALID_ROW_GROUPS)
if TRAIN_EXAMPLES < 1 or VALID_EXAMPLES < 1:
    raise RuntimeError("Attached train Parquet needs non-empty train and validation splits")
VALID_SHA256 = hashlib.sha256(json.dumps(
    [(i, TRAIN_INDEX[i]["task_id"]) for i in VALID_ROW_GROUPS]
).encode("utf-8")).hexdigest()
with TRAIN_DATA_PATH.open("rb") as handle:
    DATASET_SHA256 = hashlib.file_digest(handle, "sha256").hexdigest()
print("Prebuilt repository examples:", TRAIN_EXAMPLES, "train /",
      len(all_valid_group_ids), "validation; validation rows used:", VALID_EXAMPLES,
      "train language counts:", {language: sum(
          TRAIN_INDEX[i]["language"] == language for i in TRAIN_ROW_GROUPS)
          for language in ("python", "java")},
      "single-file SHA256:", DATASET_SHA256, flush=True)
print("Periodic validation cadence:", VALIDATE_EVERY_EPISODES,
      "episodes (about every half train epoch); using", VALID_EXAMPLES,
      "validation rows.", flush=True)

# %%
def lexical_tokens(text):
    return re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*|\d+|\S", text.lower())

def left_anchors(left_context, language):
    """Read only pre-cursor AST nodes; return scope headers and useful imports."""
    source = left_context.encode("utf-8", errors="replace")
    root = PARSERS[language].parse(source).root_node
    probe = len(source.rstrip())
    imports, scopes, bindings = [], [], []
    import_types = ({"import_statement", "import_from_statement"}
                    if language == "python" else {"import_declaration", "package_declaration"})

    def visit(node):
        if node.start_byte >= probe:
            return
        if node.type in import_types and node.end_byte <= probe:
            imports.append(source[node.start_byte:node.end_byte].decode("utf-8", errors="replace"))
        if node.type in {"assignment", "local_variable_declaration", "field_declaration"} \
                and node.end_byte <= probe:
            binding = source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            if len(binding) <= 200 and "\n" not in binding:
                bindings.append(binding)
        if node.type in SCOPE_NODE_TYPES and node.start_byte <= probe <= node.end_byte:
            body = node.child_by_field_name("body")
            if body is None:
                body = next((child for child in node.named_children
                             if child.type in {"block", "class_body"}), None)
            end = body.start_byte if body is not None else min(node.end_byte, node.start_byte + 400)
            header = source[node.start_byte:end].decode("utf-8", errors="replace").strip()
            if header:
                scopes.append(header)
        for child in node.named_children:
            visit(child)

    visit(root)
    # Only recent definitions are eligible; actual usefulness is checked at pack time.
    return ([("scope", text) for text in scopes[-4:]] +
            [("binding", text) for text in reversed(bindings[-16:])] +
            [("import", text) for text in reversed(imports[-8:])])

def suffix_with_token_budget(text, budget, tokenizer):
    """Keep an exact character suffix, preferably from a complete line boundary."""
    if budget <= 0 or not text:
        return "", len(text)
    count = lambda value: len(tokenizer.encode(value, add_special_tokens=False))
    width = min(len(text), max(256, budget * 8))
    while width < len(text) and count(text[-width:]) <= budget:
        width = min(len(text), width * 2)
    lower = len(text) - width
    if count(text[lower:]) <= budget:
        start = lower
    else:
        upper = len(text)
        while lower < upper:
            middle = (lower + upper) // 2
            if count(text[middle:]) <= budget:
                upper = middle
            else:
                lower = middle + 1
        start = lower
    while start < len(text) and count(text[start:]) > budget:
        start += 1  # Guard against rare tokenizer boundary non-monotonicity.
    if start:
        boundary = text.find("\n", start)
        if boundary >= 0 and boundary + 1 < len(text):
            aligned = boundary + 1
            if 0.75 * budget <= count(text[aligned:]) <= budget:
                start = aligned
    return text[start:], start

def pack_left_context(left_context, language, anchors, tokenizer, budget):
    """Keep the cursor suffix, then spend only useful tokens on older AST hints."""
    full_tail, full_start = suffix_with_token_budget(left_context, budget, tokenizer)
    if full_start == 0 or budget < 80:
        return full_tail
    anchor_budget = min(512, budget // 5)
    tail, _ = suffix_with_token_budget(
        left_context, budget - anchor_budget, tokenizer)
    marker = "#" if language == "python" else "//"
    focus, _ = suffix_with_token_budget(left_context, min(512, budget // 2), tokenizer)
    focus_terms = set(IDENTIFIER_RE.findall(focus)) - {
        "self", "this", "return", "def", "class", "import", "from", "new"}
    selected = []
    selected_snippets = []
    for kind, snippet in anchors:
        if snippet in tail:
            continue
        identifiers = set(IDENTIFIER_RE.findall(snippet))
        if kind != "scope" and not (identifiers & focus_terms):
            continue
        lines = snippet.splitlines()[:4]
        hint = "".join(f"{marker} earlier {kind}: {line.strip()}\n" for line in lines)
        proposed = "".join(selected) + hint
        if len(tokenizer.encode(proposed, add_special_tokens=False)) <= anchor_budget:
            selected.append(hint)
            selected_snippets.append(snippet)
            if kind == "binding":
                focus_terms.update(identifiers)
        if len(selected) >= (12 if budget >= 1024 else 4):
            break
    if not selected:
        return full_tail
    selected = sorted(zip(selected_snippets, selected),
                      key=lambda item: left_context.rfind(item[0]))
    for _ in range(len(selected)):
        hints = "".join(hint for _, hint in selected)
        room = budget - len(tokenizer.encode(hints, add_special_tokens=False))
        tail, _ = suffix_with_token_budget(left_context, room, tokenizer)
        distinct = [(snippet, hint) for snippet, hint in selected if snippet not in tail]
        if len(distinct) == len(selected):
            break
        selected = distinct
        if not selected:
            return full_tail
    hints = "".join(hint for _, hint in selected)
    room = budget - len(tokenizer.encode(hints, add_special_tokens=False))
    while room >= 0:
        tail, _ = suffix_with_token_budget(left_context, room, tokenizer)
        packed = hints + tail
        excess = len(tokenizer.encode(packed, add_special_tokens=False)) - budget
        if excess <= 0:
            return packed
        room -= max(1, excess)
    return full_tail

def retrieval_query(example, anchors):
    marker = "#" if example["language"] == "python" else "//"
    path = f"{marker} current path: {example['file_path']}\n"
    room = RETRIEVER_QUERY_LENGTH - 4
    path_cost = len(RET_TOKENIZER.encode(path, add_special_tokens=False))
    left = pack_left_context(example["left_context"], example["language"],
                             anchors, RET_TOKENIZER, max(0, room - path_cost - 2))
    query = path + left
    while left and len(RET_TOKENIZER.tokenize(query)) > room:
        left = left[1:]
        query = path + left
    return query

def render_chunk(unit, language):
    marker = "#" if language == "python" else "//"
    header = f"{marker} file path: {unit['path']}\n{marker} AST node: {unit['type']}"
    body = "\n".join(f"{marker} {line}" for line in unit["text"].splitlines())
    return header + "\n" + body

def unixcoder_ids(text, max_length, is_query):
    tokens = RET_TOKENIZER.tokenize(text)
    room = max_length - 4
    tokens = tokens[-room:] if is_query else tokens[:room]
    sequence = [RET_TOKENIZER.cls_token, "<encoder-only>", RET_TOKENIZER.sep_token]
    sequence += tokens + [RET_TOKENIZER.sep_token]
    ids = RET_TOKENIZER.convert_tokens_to_ids(sequence)
    return ids + [RET_TOKENIZER.pad_token_id] * (max_length - len(ids))

def build_row(example):
    anchors = left_anchors(example["left_context"], example["language"])
    query = retrieval_query(example, anchors)
    related_files = []
    target_path_key = os.path.normpath(str(example["file_path"]).replace("\\", "/"))
    for related in example["related_files"]:
        if isinstance(related, dict):
            path, code = str(related.get("path", "")), str(related.get("text", ""))
        else:
            path, code = str(related[0]), str(related[1])
        path_key = os.path.normpath(path.replace("\\", "/"))
        if path and code.strip() and path_key != target_path_key:
            related_files.append((path, code))
    related_files_total = len(related_files)
    if len(related_files) > MAX_RELATED_FILES:
        file_corpus = [IDENTIFIER_RE.findall((path + "\n" + code).lower())
                       for path, code in related_files]
        file_scores = BM25Okapi(file_corpus).get_scores(lexical_tokens(query))
        file_order = sorted(range(len(related_files)),
                            key=lambda i: (-float(file_scores[i]), i))
        related_files = [related_files[i] for i in file_order[:MAX_RELATED_FILES]]
    units = []
    errors = 0
    for path, code in related_files:
        chunks, had_error = ast_chunks(path, code, example["language"])
        units.extend(chunks)
        errors += int(had_error)
    unique = {(u["path"], u["start"], u["end"]): u for u in units}
    units = list(unique.values())
    texts = [render_chunk(u, example["language"]) for u in units]
    keep = [i for i, text in enumerate(texts)
            if token_count(text) + 8 <= CROSSFILE_TOKEN_BUDGET
            and len(RET_TOKENIZER.tokenize(text)) + 4 <= RETRIEVER_CANDIDATE_LENGTH]
    units = [units[i] for i in keep]
    texts = [texts[i] for i in keep]
    if units:
        corpus = [lexical_tokens(u["text"]) for u in units]
        bm25 = BM25Okapi(corpus).get_scores(lexical_tokens(query))
        order = sorted(range(len(units)), key=lambda j: (-float(bm25[j]), j))
        units = [units[j] for j in order[:CANDIDATE_POOL_SIZE]]
        texts = [texts[j] for j in order[:CANDIDATE_POOL_SIZE]]
    # Eight tokens cover separators/tokenizer boundary effects for any slate order.
    costs = [token_count(t) + 8 for t in texts]
    return {**{key: example[key] for key in (
                "task_id", "language", "file_path", "left_context", "target_code")},
            "query_ids": unixcoder_ids(query, RETRIEVER_QUERY_LENGTH, True),
            "candidate_ids": [unixcoder_ids(t, RETRIEVER_CANDIDATE_LENGTH, False)
                              for t in texts],
            "candidate_texts": texts,
            "candidate_raw_texts": [u["text"] for u in units],
            "candidate_costs": costs,
            "candidate_paths": [u["path"] for u in units],
            "candidate_types": [u["type"] for u in units],
            "left_anchors": anchors,
            "target_kind": example.get("target_kind", "benchmark"),
            "requested_kind": example.get("requested_kind", "benchmark"),
            "target_node_type": example.get("target_node_type", "benchmark"),
            "parse_errors": errors, "all_ast_chunks": len(unique),
            "related_files_count": related_files_total,
            "related_files_used": len(related_files),
            "related_file_paths_used": [path for path, _ in related_files]}

def prepared_row(row_group_id):
    payload = TRAIN_PARQUET.read_row_group(
        row_group_id, columns=["payload"]
    ).column(0)[0].as_py()
    return json.loads(zlib.decompress(payload))

def build_train_row(epoch, slot):
    """Read one fixed row; shuffled epochs cover the full train split."""
    if not 0 <= slot < TRAIN_EXAMPLES:
        raise IndexError(slot)
    return prepared_row(TRAIN_ROW_GROUPS[slot])

DATA_SIGNATURE = {"mode": RUN_MODE, "seed": SEED, "train": TRAIN_EXAMPLES,
                  "valid": VALID_EXAMPLES, "pool_schema": 6,
                  "data_pipeline_schema": DATA_PIPELINE_SCHEMA,
                  "score_schema": SCORE_SCHEMA,
                  "file_selection_policy": "all_repo_files_then_query_bm25",
                  "sampling": "full_fixed_repo_rows_v1",
                  "dataset_sha256": DATASET_SHA256,
                  "valid_examples_sha256": VALID_SHA256,
                  "max_related_files": MAX_RELATED_FILES,
                  "pool": CANDIDATE_POOL_SIZE, "chunk": AST_CHUNK_TOKENS,
                  "encode_batch_size": ENCODE_BATCH_SIZE,
                  "retriever": RETRIEVER_MODEL, "generator": GENERATOR_MODEL,
                  "retriever_query_length": RETRIEVER_QUERY_LENGTH,
                  "retriever_candidate_length": RETRIEVER_CANDIDATE_LENGTH,
                  "crossfile_budget": CROSSFILE_TOKEN_BUDGET,
                  "max_slate_steps": MAX_SLATE_STEPS,
                  "generator_output_tokens": GENERATOR_OUTPUT_TOKENS,
                  "generator_input_tokens": GENERATOR_INPUT_TOKENS,
                  "generator_model_len": GENERATOR_MAX_MODEL_LEN,
                  "prompt_packer": "ast_relevant_hints_cursor_suffix_v2",
                  "objective": "rrpo_final_plus_prefix_es_idf1_v2",
                  "rrpo_final_weight": RRPO_FINAL_WEIGHT,
                  "rrpo_identifier_weight": RRPO_IDENTIFIER_WEIGHT,
                  "rrpo_reference": RRPO_REFERENCE,
                  "rrpo_gamma": RRPO_GAMMA,
                  "rrpo_gae_lambda": RRPO_GAE_LAMBDA,
                  "ppo_clip": PPO_CLIP, "ppo_passes": PPO_PASSES,
                  "accumulation_steps": ACCUMULATION_STEPS,
                  "target_kl": TARGET_KL,
                  "encoder_lr": ENCODER_LR, "head_lr": HEAD_LR,
                  "max_grad_norm": MAX_GRAD_NORM,
                  "entropy_coef": ENTROPY_COEF,
                  "reward": "rrpo_rlcoder_es_identifier_f1_v2"}

VALIDATION_SIGNATURE_KEYS = {
    "valid", "valid_examples_sha256", "target_chunk_tokens",
    "min_target_tokens", "min_prefix_tokens", "min_left_context_lines",
    "preferred_file_lines", "preferred_file_chars", "target_cut_distribution",
}
RESUMABLE_TUNING_KEYS = {"encode_batch_size", "accumulation_steps", "encoder_lr"}

def validation_signature_changed(saved_signature):
    if not isinstance(saved_signature, dict):
        return True
    return any(saved_signature.get(key) != DATA_SIGNATURE.get(key)
               for key in VALIDATION_SIGNATURE_KEYS)

def resume_signature_compatible(saved_signature):
    """Allow resume with a changed encoder LR or training microbatch."""
    if not isinstance(saved_signature, dict):
        return False
    ignored = VALIDATION_SIGNATURE_KEYS | RESUMABLE_TUNING_KEYS
    current = {key: value for key, value in DATA_SIGNATURE.items()
               if key not in ignored}
    saved = {key: value for key, value in saved_signature.items()
             if key not in ignored}
    return saved == current

def pool_cache_compatible(saved_signature):
    """Reuse fixed Parquet rows when only optimizer/microbatch settings changed."""
    if not isinstance(saved_signature, dict):
        return False
    current = {key: value for key, value in DATA_SIGNATURE.items()
               if key not in RESUMABLE_TUNING_KEYS}
    saved = {key: value for key, value in saved_signature.items()
             if key not in RESUMABLE_TUNING_KEYS}
    return saved == current

pool_cache = WORK_DIR / "ast_bm25_token_ids_rrpo_v2.pt"
attached_pool_cache = (RESUME_SOURCE.with_name(pool_cache.name)
                       if RESUME_SOURCE is not None else None)
cache_source = (pool_cache if pool_cache.is_file() else attached_pool_cache
                if attached_pool_cache is not None and attached_pool_cache.is_file()
                else None)
cached = (torch.load(cache_source, map_location="cpu", weights_only=False)
          if cache_source is not None else None)
if cached is not None and pool_cache_compatible(cached.get("signature")):
    valid_rows = cached["valid"]
    print("Reused validation AST cache:", cache_source)
    if cache_source != pool_cache and not pool_cache.exists():
        shutil.copy2(cache_source, pool_cache)
        print("Carried AST cache into this session's saved outputs:", pool_cache)
else:
    valid_rows = [prepared_row(i) for i in tqdm(VALID_ROW_GROUPS,
                                               desc="Prebuilt valid")]
    torch.save({"signature": DATA_SIGNATURE, "valid": valid_rows}, pool_cache)
invalid_valid_rows = [r["task_id"] for r in valid_rows
                      if not any(cost <= CROSSFILE_TOKEN_BUDGET
                                 for cost in r["candidate_costs"])]
if invalid_valid_rows:
    raise RuntimeError(
        f"The fixed validation split has {len(invalid_valid_rows)} rows without "
        "a candidate fitting the context budget; refusing to silently drop them."
    )
if len(valid_rows) != VALID_EXAMPLES:
    raise RuntimeError("Expected every fixed validation row.")
print("Loaded all", len(valid_rows), "fixed validation rows.", flush=True)

# %%
def verify_served_model(cards, require_context_metadata):
    matching = [card for card in cards if card.get("id") == SERVED_MODEL_NAME]
    if len(matching) != 1:
        raise RuntimeError(f"Modal endpoint does not serve {SERVED_MODEL_NAME}: "
                           f"{[card.get('id') for card in cards]}")
    advertised = matching[0].get("max_model_len")
    if advertised is None:
        if require_context_metadata:
            raise RuntimeError("Modal vLLM endpoint does not advertise max_model_len; "
                               f"verify the deployed server before using "
                               f"{GENERATOR_INPUT_TOKENS} input tokens.")
        print("WARNING: vLLM did not advertise max_model_len; configured endpoint limit is",
              GENERATOR_MAX_MODEL_LEN)
    elif int(advertised) < GENERATOR_MAX_MODEL_LEN:
        raise RuntimeError(f"Modal vLLM context limit {advertised} is below "
                           f"{GENERATOR_MAX_MODEL_LEN}; redeploy the endpoint.")
    else:
        print("vLLM advertised context length:", advertised)

def wait_for_modal_vllm(timeout_seconds=ENDPOINT_READY_TIMEOUT_SECONDS):
    deadline = time.monotonic() + timeout_seconds
    last_report = 0.0
    last_error = "endpoint is still starting"
    while time.monotonic() < deadline:
        try:
            health = requests.get(MODAL_HEALTH_URL, headers=MODAL_AUTH_HEADERS,
                                  timeout=10)
            if health.status_code in (401, 403):
                raise RuntimeError(
                    f"Modal authentication failed (HTTP {health.status_code}); "
                    "check Kaggle Secret MODAL_PROXY_TOKEN and token environment scope."
                )
            if health.ok:
                models = requests.get(API_BASE + "/models", headers=MODAL_AUTH_HEADERS,
                                      timeout=15)
                if models.status_code in (401, 403):
                    raise RuntimeError(
                        f"Modal authentication failed (HTTP {models.status_code}); "
                        "check the Kaggle Proxy Token."
                    )
                if models.ok:
                    verify_served_model(models.json().get("data", []), True)
                    print("Authenticated Modal vLLM endpoint ready:", MODAL_ENDPOINT_ROOT,
                          "max_num_seqs:", VLLM_MAX_NUM_SEQS, flush=True)
                    return
                last_error = f"/v1/models returned HTTP {models.status_code}"
            else:
                last_error = f"/health returned HTTP {health.status_code}"
        except RuntimeError:
            raise
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        now = time.monotonic()
        if now - last_report >= 30:
            print("Waiting for Modal vLLM endpoint:", last_error,
                  f"{max(0, int(deadline - now))}s remaining", flush=True)
            last_report = now
        time.sleep(10)
    raise TimeoutError(
        f"Modal vLLM endpoint was not ready within {timeout_seconds}s: {last_error}. "
        "No RRPO update has started; verify Modal deployment and URL."
    )

wait_for_modal_vllm()

def compose_prompt(row, selected):
    context = "\n\n".join(row["candidate_texts"][i] for i in selected)
    if token_count(context) > CROSSFILE_TOKEN_BUDGET:
        raise ValueError("Selected context exceeds budget; never silently truncate a chunk")
    marker = "#" if row["language"] == "python" else "//"
    sections = ([context] if context else [])
    sections.append(f"{marker} file path: {row['file_path']}")
    prefix = "\n\n".join(sections) + "\n\n"
    prompt_limit = GENERATOR_INPUT_TOKENS
    # After retrieved snippets and the current-file path, spend the remaining
    # actual tokenizer budget on AST hints plus the exact cursor-adjacent suffix.
    room = prompt_limit - len(GEN_TOKENIZER.encode(prefix, add_special_tokens=True)) - 2
    while room >= 0:
        left = pack_left_context(row["left_context"], row["language"],
                                 row.get("left_anchors", []), GEN_TOKENIZER, room)
        prompt = prefix + left
        excess = len(GEN_TOKENIZER.encode(prompt, add_special_tokens=True)) - prompt_limit
        if excess <= 0:
            return prompt
        room -= max(1, excess)
    raise ValueError("Prompt header alone exceeds the generator context window")

def completion_cache_key(prompt):
    namespace = json.dumps(GENERATION_CACHE_SIGNATURE, sort_keys=True,
                           separators=(",", ":"))
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt.encode("utf-8", errors="replace"))
    return digest.hexdigest()

def generate_completions(prompts, progress_label=None):
    if not prompts:
        return []

    # Deduplicate exact prompts globally for this call as well as across calls.
    prompt_by_key = {}
    ordered_keys = []
    for prompt in prompts:
        key = completion_cache_key(prompt)
        ordered_keys.append(key)
        prompt_by_key.setdefault(key, prompt)

    unique_keys = list(prompt_by_key)
    cached = {}
    # Stay below SQLite's bind-parameter limit on larger reward batches.
    for offset in range(0, len(unique_keys), 400):
        keys = unique_keys[offset:offset + 400]
        placeholders = ",".join("?" for _ in keys)
        cached.update(GENERATION_CACHE_CONNECTION.execute(
            f"SELECT cache_key, completion FROM generation_cache "
            f"WHERE cache_key IN ({placeholders})", keys
        ).fetchall())

    missing = [(key, prompt_by_key[key]) for key in unique_keys if key not in cached]
    persistent_hits = sum(key in cached for key in ordered_keys)
    duplicate_reuse = len(prompts) - len(unique_keys)
    batch_size = max(1, VLLM_MAX_NUM_SEQS)
    total_batches = math.ceil(len(missing) / batch_size)
    if progress_label or persistent_hits or duplicate_reuse:
        label = f"{progress_label}: " if progress_label else ""
        print(f"{label}response cache: {persistent_hits}/{len(prompts)} persistent "
              f"hits, {duplicate_reuse} in-call duplicate reuses, "
              f"{len(missing)} new generations", flush=True)

    generated = {}
    for start in range(0, len(missing), batch_size):
        batch_number = start // batch_size + 1
        batch = missing[start:start + batch_size]
        if progress_label and (batch_number == 1 or batch_number % 10 == 0):
            print(f"{progress_label}: vLLM batch {batch_number}/{total_batches} starting; "
                  f"generated={start}/{len(missing)} new prompts", flush=True)
        token_ids = [GEN_TOKENIZER.encode(prompt, add_special_tokens=True)
                     for _, prompt in batch]
        if any(len(ids) > GENERATOR_INPUT_TOKENS for ids in token_ids):
            raise ValueError(
                f"Generator prompt exceeds {GENERATOR_INPUT_TOKENS} input tokens; "
                "refusing to truncate retrieved context or cursor suffix"
            )
        try:
            request_payload = {
                "model": SERVED_MODEL_NAME, "prompt": token_ids,
                "max_tokens": GENERATOR_OUTPUT_TOKENS,
                "temperature": 0.0, "top_p": 1.0,
            }
            last_request_deadline = modal_last_request_deadline()
            request_deadline = min(
                time.monotonic() + ENDPOINT_RECOVERY_TIMEOUT_SECONDS,
                last_request_deadline,
            )
            response = None
            retry_count = 0
            retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
            while time.monotonic() < request_deadline:
                remaining_modal_window = last_request_deadline - time.monotonic()
                if remaining_modal_window <= 0.05:
                    raise TrainingDeadlineReached(
                        "Modal request cutoff reached; no more generator calls will be sent."
                    )
                request_timeout = min(
                    GENERATOR_REQUEST_TIMEOUT_SECONDS,
                    remaining_modal_window - 0.05,
                    request_deadline - time.monotonic(),
                )
                if request_timeout <= 0.05:
                    break
                try:
                    response = requests.post(
                        API_BASE + "/completions", headers=MODAL_AUTH_HEADERS,
                        json=request_payload, timeout=request_timeout,
                    )
                except requests.RequestException as exc:
                    if time.monotonic() >= last_request_deadline - 1.0:
                        raise TrainingDeadlineReached(
                            "Generator request reached the Modal idle-shutdown cutoff."
                        ) from exc
                    retry_count += 1
                    delay = min(GENERATOR_RETRY_BACKOFF_SECONDS *
                                (2 ** min(retry_count - 1, 4)), 30.0,
                                request_deadline - time.monotonic())
                    if delay <= 0:
                        break
                    print(f"Modal request interrupted ({type(exc).__name__}: {exc}); "
                          f"retry {retry_count} for the same deterministic batch "
                          f"in {delay:.1f}s.", flush=True)
                    time.sleep(delay)
                    response = None
                    continue
                if response.status_code in (401, 403):
                    raise RuntimeError(
                        f"Modal authentication failed (HTTP {response.status_code}); "
                        "check Kaggle Secret MODAL_PROXY_TOKEN."
                    )
                if response.status_code not in retryable_statuses:
                    break
                retry_count += 1
                delay = min(GENERATOR_RETRY_BACKOFF_SECONDS *
                            (2 ** min(retry_count - 1, 4)), 30.0,
                            request_deadline - time.monotonic())
                if delay <= 0:
                    break
                print(f"Modal temporarily returned HTTP {response.status_code}; "
                      f"retry {retry_count} for the same deterministic batch "
                      f"in {delay:.1f}s.", flush=True)
                time.sleep(delay)
            if time.monotonic() >= last_request_deadline:
                raise TrainingDeadlineReached(
                    "Modal request cutoff reached while recovering the endpoint."
                )
            if response is None or response.status_code in retryable_statuses:
                raise GeneratorRequestError(
                    f"Modal vLLM request failed after {retry_count} retries "
                    "within the recovery window; the current RRPO state is checkpointed."
                )
        except requests.RequestException as exc:
            raise GeneratorRequestError(
                f"Modal vLLM batch {start}:{start + len(batch)} failed: {exc}"
            ) from exc
        if not response.ok:
            raise RuntimeError(f"vLLM HTTP {response.status_code}: {response.text[:2000]}")
        choices = sorted(response.json()["choices"], key=lambda c: c.get("index", 0))
        if len(choices) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} completions, received {len(choices)}")
        batch_outputs = [str(choice.get("text", "")) for choice in choices]
        generated.update((key, output)
                         for (key, _prompt), output in zip(batch, batch_outputs))
        GENERATION_CACHE_CONNECTION.executemany(
            "INSERT INTO generation_cache (cache_key, completion) VALUES (?, ?) "
            "ON CONFLICT(cache_key) DO UPDATE SET completion=excluded.completion",
            list(zip((key for key, _prompt in batch), batch_outputs)),
        )
        # Commit every vLLM batch so an interrupted Kaggle session still leaves
        # all completed deterministic generations available for the next run.
        GENERATION_CACHE_CONNECTION.commit()
        if progress_label and (batch_number == 1 or batch_number % 10 == 0 or
                               batch_number == total_batches):
            print(f"{progress_label}: vLLM batch {batch_number}/{total_batches} done; "
                  f"generated={start + len(batch)}/{len(missing)} new prompts; "
                  f"cache entries={GENERATION_CACHE_CONNECTION.execute(
                      'SELECT COUNT(*) FROM generation_cache').fetchone()[0]}",
                  flush=True)

    outputs_by_key = {**cached, **generated}
    return [outputs_by_key[key] for key in ordered_keys]

probe = generate_completions(["def add(a, b):\n    "])
print("Text-generation probe passed:", repr(probe[0][:100]))

def rlcoder_postprocess_completion(left_context, prediction, language):
    """Match RLCoder's statement truncation using this notebook's Tree-sitter ABI."""
    if language in {"java", "csharp", "typescript"}:
        end = next((i for i, char in enumerate(prediction) if char in ";}{"), None)
        return prediction[:end + 1] if end else prediction
    if language != "python":
        return prediction
    parser = PARSERS[language]
    for i in range(len(prediction) - 1):
        if prediction[i + 1] != "\n":
            continue
        try:
            root = parser.parse((left_context + prediction[:i + 1]).encode("utf-8")).root_node
        except Exception:
            continue
        pending = [root]
        while pending and pending[-1].type != "ERROR":
            pending.extend(pending.pop().children)
        if not pending:
            return prediction[:i + 1].rstrip()
    return prediction

def identifier_match_f1(prediction, reference, language):
    """AlignCoder-style identifier F1: unique names, no strings/keywords."""
    java_keywords = set("""abstract assert boolean break byte case catch char class
        continue default do double else enum extends final finally float for if
        implements import instanceof int interface long native new package private
        protected public return short static strictfp super switch synchronized this
        throw throws transient try void volatile while var const goto""".split())
    excluded = (set(keyword.kwlist) - {"True", "False"}
                if language == "python" else java_keywords)
    string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''

    def names(code):
        without_strings = re.sub(string_pattern, "", code)
        return {token for token in re.findall(r"\w+", without_strings)
                if re.match(r"[_a-zA-Z][_a-zA-Z0-9]*", token)
                and token not in excluded}

    predicted, target = names(prediction), names(reference)
    overlap = len(predicted & target)
    denominator = len(predicted) + len(target)
    return (2 * overlap / denominator if denominator else 0.0), bool(target)

def rlcoder_score(prediction, reference, left_context, language):
    """RLCoder compute_metric_stmt primary scores and parenthesized RepoEval scores."""
    processed = rlcoder_postprocess_completion(left_context, str(prediction), language)
    # Deliberately mirror RLCoder's regex, including its behavior inside strings.
    pred = re.sub(r"//.*", "", re.sub(r"#.*", "", processed))
    target = re.sub(r"//.*", "", re.sub(r"#.*", "", str(reference)))
    pred_lines = [line.strip() for line in pred.split("\n") if line.strip()]
    target_lines = [line.strip() for line in target.split("\n") if line.strip()]
    repo_pred = "\n".join(pred_lines[:len(target_lines)])
    repo_target = "\n".join(target_lines)
    width = max(len(repo_pred), len(repo_target))
    id_f1, has_target_ids = identifier_match_f1(pred, target, language)
    return {
        "prediction_postprocessed": pred,
        "reference_postprocessed": target,
        "exact_match": int(pred_lines == target_lines),
        "edit_similarity": fuzz.ratio(pred.strip(), target.strip()) / 100.0,
        "identifier_f1": id_f1,
        "target_has_identifiers": has_target_ids,
        "repoeval_exact_match": int(pred_lines[:len(target_lines)] == target_lines),
        "repoeval_edit_similarity": (1.0 - editdistance.eval(repo_target, repo_pred) / width
                                       if width else 1.0),
    }

def rlcoder_repo_macro(records, field):
    by_repo = {}
    for record in records:
        repo = str(record["task_id"]).split("/")[0]
        by_repo.setdefault(repo, []).append(record[field])
    return (sum(round(sum(values) / len(values), 4) for values in by_repo.values())
            / len(by_repo)) if by_repo else float("nan")

def rlcoder_summary(records):
    groups = {}
    for record in records:
        keys = [(record["dataset"], record["method"])]
        if record["dataset"] in {"repoeval_line_0", "repoeval_line_1"}:
            keys.append(("repoeval_line", record["method"]))
        for key in keys:
            groups.setdefault(key, []).append(record)
    result = []
    for (dataset, method), rows in sorted(groups.items()):
        em = sum(row["exact_match"] for row in rows) / len(rows)
        es = sum(row["edit_similarity"] for row in rows) / len(rows)
        result.append({"dataset": dataset, "method": method, "n": len(rows),
                       "exact_match": em, "edit_similarity": es,
                       "identifier_f1": sum(row["identifier_f1"] for row in rows) / len(rows),
                       "rlcoder_em_pct": round(em * 100, 4),
                       "rlcoder_es_0to100": round(es * 100, 4),
                       "repoeval_em_pct": round(rlcoder_repo_macro(
                           rows, "repoeval_exact_match") * 100, 4),
                       "repoeval_es_pct": round(rlcoder_repo_macro(
                           rows, "repoeval_edit_similarity") * 100, 4)})
    return result

def synthetic_reward(prediction, row):
    score = rlcoder_score(prediction, row["target_code"],
                          row["left_context"], row["language"])
    es = score["edit_similarity"]
    utility = (es if not score["target_has_identifiers"] else
               (1 - RRPO_IDENTIFIER_WEIGHT) * es +
               RRPO_IDENTIFIER_WEIGHT * score["identifier_f1"])
    return {"utility": utility, "es": es, "id_f1": score["identifier_f1"]}

# %%
import torch.nn as nn
import torch.nn.functional as F

GPU_IDS = list(range(torch.cuda.device_count()))
DEVICE = torch.device(f"cuda:{GPU_IDS[0]}")
encoder_core = AutoModel.from_pretrained(RETRIEVER_MODEL, cache_dir=str(CACHE_DIR)).to(DEVICE)
# Activation checkpointing is disabled to avoid recomputation during PPO updates.
for module in encoder_core.modules():
    if isinstance(module, nn.Dropout):
        module.p = 0.0  # Same deterministic policy in rollout and PPO update.
class TensorOutputEncoder(nn.Module):
    """Make HF model outputs DataParallel-gatherable by returning only a tensor."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          return_dict=False)[0]

encoder = nn.DataParallel(TensorOutputEncoder(encoder_core), device_ids=GPU_IDS,
                          output_device=GPU_IDS[0], dim=0)
hidden = int(encoder_core.config.hidden_size)
print("UniXcoder DataParallel devices:", GPU_IDS,
      "encoder microbatch:", ENCODE_BATCH_SIZE, flush=True)

class SlateHeads(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.candidate = nn.Sequential(nn.Linear(5 * dim + 2, 128), nn.Tanh(),
                                       nn.Linear(128, 1))
        self.stop = nn.Sequential(nn.Linear(2 * dim + 1, 128), nn.Tanh(),
                                  nn.Linear(128, 1))
        self.value = nn.Sequential(nn.Linear(2 * dim + 1, 128), nn.Tanh(),
                                   nn.Linear(128, 1))

heads = SlateHeads(hidden).to(DEVICE)
optimizer = torch.optim.AdamW([
    {"params": list(encoder.parameters()), "lr": ENCODER_LR},
    {"params": list(heads.parameters()), "lr": HEAD_LR},
], weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")

def encode_row(row):
    sequences = [row["query_ids"]] + row["candidate_ids"]
    batches = []
    for start in range(0, len(sequences), ENCODE_BATCH_SIZE):
        current = sequences[start:start + ENCODE_BATCH_SIZE]
        ids = torch.full((len(current), max(map(len, current))),
                         RET_TOKENIZER.pad_token_id, dtype=torch.long, device=DEVICE)
        for index, sequence in enumerate(current):
            ids[index, :len(sequence)] = torch.as_tensor(sequence, device=DEVICE)
        mask = ids.ne(RET_TOKENIZER.pad_token_id)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=DEVICE.type == "cuda"):
            encoded = encoder(input_ids=ids, attention_mask=mask)
            hidden_states = (encoded if torch.is_tensor(encoded)
                             else encoded.last_hidden_state)
            weights = mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        batches.append(F.normalize(pooled.float(), p=2, dim=-1))
    pooled = torch.cat(batches, dim=0)
    return pooled[0], pooled[1:]

def state_distribution(row, query, candidates, selected, remaining):
    n = len(row["candidate_ids"])
    selected_set = set(selected)
    selected_mean = (candidates[list(selected)].mean(0) if selected else
                     torch.zeros_like(query))
    budget_fraction = float(remaining) / CROSSFILE_TOKEN_BUDGET
    budget_feature = query.new_full((n, 1), budget_fraction)
    cost_feature = query.new_tensor(row["candidate_costs"]).unsqueeze(1)
    cost_feature = cost_feature / CROSSFILE_TOKEN_BUDGET
    features = torch.cat([
        query.expand(n, -1), candidates, selected_mean.expand(n, -1),
        query * candidates, selected_mean * candidates, cost_feature,
        budget_feature,
    ], dim=-1)
    candidate_logits = heads.candidate(features).squeeze(-1).float()
    state_features = torch.cat([query, selected_mean,
                                query.new_tensor([budget_fraction])])
    stop_logit = heads.stop(state_features).float().view(1)
    value = heads.value(state_features).float().squeeze()
    valid = [i not in selected_set and row["candidate_costs"][i] <= remaining
             for i in range(n)] + [True]
    logits = torch.cat([candidate_logits, stop_logit]).masked_fill(
        ~torch.tensor(valid, dtype=torch.bool, device=DEVICE), -1e9
    )
    return torch.log_softmax(logits, dim=0), value, valid

def rollout(row, greedy=False):
    encoder.train()  # Dropout is zero; keep policy scoring deterministic.
    heads.eval()
    with torch.no_grad():
        query, candidates = encode_row(row)
        selected, steps = [], []
        remaining = CROSSFILE_TOKEN_BUDGET
        for _ in range(MAX_SLATE_STEPS + 1):
            log_probs, _, valid = state_distribution(
                row, query, candidates, selected, remaining
            )
            action = int(torch.argmax(log_probs).item()) if greedy else int(
                torch.multinomial(log_probs.exp(), 1).item()
            )
            steps.append({"selected": tuple(selected), "remaining": remaining,
                          "valid": valid, "action": action,
                          "old_logp": float(log_probs[action].item()),
                          "old_probs": log_probs.exp().cpu().tolist()})
            if action == len(row["candidate_ids"]):
                break
            selected.append(action)
            remaining -= row["candidate_costs"][action]
            if len(selected) >= MAX_SLATE_STEPS:
                break
    return {"row": row, "selected": selected, "steps": steps}

def bm25_reference_completion(row, selected, remaining):
    """Frozen BM25-order policy, starting from the actor's actual pre-action state."""
    chosen = list(selected)
    used = set(chosen)
    for index, cost in enumerate(row["candidate_costs"]):
        if len(chosen) >= MAX_SLATE_STEPS:
            break
        if index in used or cost > remaining:
            continue
        chosen.append(index)
        used.add(index)
        remaining -= cost
    return chosen

def rrpo_step_rewards(prefix_utility, stopped, final_utility):
    """Final-dominant utility plus prefix shaping over K fixed slots."""
    count = len(prefix_utility)
    if count > MAX_SLATE_STEPS or (stopped and count >= MAX_SLATE_STEPS) or (
            not stopped and count != MAX_SLATE_STEPS):
        raise ValueError("RRPO selected-prefix count disagrees with STOP/max-K")
    prefix_weight = 1 - RRPO_FINAL_WEIGHT
    rewards = [prefix_weight * value / MAX_SLATE_STEPS
               for value in prefix_utility]
    if stopped:
        rewards.append((RRPO_FINAL_WEIGHT +
                        prefix_weight * (MAX_SLATE_STEPS - count) / MAX_SLATE_STEPS)
                       * final_utility)
    else:
        rewards[-1] += RRPO_FINAL_WEIGHT * final_utility
    return rewards

def score_rrpo_episodes(episodes):
    """One batched generator pass for policy/reference prefixes, deduped per row."""
    if not episodes:
        return 0
    prompt_index, prompts = {}, []

    def register(episode_index, prefix):
        key = episode_index, tuple(prefix)
        if key not in prompt_index:
            prompt_index[key] = len(prompts)
            prompts.append(compose_prompt(episodes[episode_index]["row"], prefix))

    reference_sequences = []
    for episode_index, episode in enumerate(episodes):
        chosen = episode["selected"]
        for length in range(1, len(chosen) + 1):
            register(episode_index, chosen[:length])
        if not chosen:
            register(episode_index, ())
        references = []
        for step in episode["steps"]:
            selected = step["selected"]
            reference = bm25_reference_completion(
                episode["row"], selected, step["remaining"]
            )
            references.append(reference)
            for length in range(len(selected) + 1, len(reference) + 1):
                register(episode_index, reference[:length])
            if len(reference) < MAX_SLATE_STEPS:
                register(episode_index, reference)
        reference_sequences.append(references)

    api_batches = math.ceil(len(prompts) / max(1, VLLM_MAX_NUM_SEQS))
    print(f"RRPO reward scoring: {len(episodes)} episodes, {len(prompts)} unique "
          f"completion prompts, {api_batches} vLLM batches.", flush=True)
    outputs = generate_completions(prompts, progress_label="RRPO rewards")
    scores = {key: synthetic_reward(outputs[index], episodes[key[0]]["row"])
              for key, index in prompt_index.items()}
    for episode_index, episode in enumerate(episodes):
        chosen = episode["selected"]
        components = lambda prefix: scores[episode_index, tuple(prefix)]
        utility = lambda prefix: components(prefix)["utility"]
        prefix_utility = [utility(chosen[:length])
                          for length in range(1, len(chosen) + 1)]
        final_components = components(chosen)
        final_utility = final_components["utility"]
        stopped = episode["steps"][-1]["action"] == len(episode["row"]["candidate_ids"])
        step_rewards = rrpo_step_rewards(prefix_utility, stopped, final_utility)
        if len(step_rewards) != len(episode["steps"]):
            raise AssertionError("RRPO rewards and sampled actions have different lengths")
        reference_values = []
        for step, reference in zip(episode["steps"], reference_sequences[episode_index]):
            selected_count = len(step["selected"])
            future = [utility(reference[:length])
                      for length in range(selected_count + 1, len(reference) + 1)]
            if len(reference) < MAX_SLATE_STEPS:
                future.extend([utility(reference)] * (MAX_SLATE_STEPS - len(reference)))
            if len(future) != MAX_SLATE_STEPS - selected_count:
                raise AssertionError("Reference return has the wrong remaining horizon")
            reference_values.append((1 - RRPO_FINAL_WEIGHT) *
                                    sum(future) / MAX_SLATE_STEPS +
                                    RRPO_FINAL_WEIGHT * utility(reference))
        episode["prefix_utility"] = prefix_utility
        episode["final_utility"] = final_utility
        episode["final_es"] = final_components["es"]
        episode["final_id_f1"] = final_components["id_f1"]
        episode["step_rewards"] = step_rewards
        episode["reference_values"] = reference_values
        episode["reward"] = sum(step_rewards)
    return len(prompts)

def policy_invariants(row):
    episode = rollout(row, greedy=True)  # Do not perturb checkpoint RNG on resume.
    query, candidates = encode_row(row)
    with torch.no_grad():
        differences = []
        for step in episode["steps"]:
            current, _, valid = state_distribution(row, query, candidates,
                                                   step["selected"], step["remaining"])
            if valid != step["valid"]:
                raise AssertionError("Action support changed before PPO update")
            differences.append(abs(float(current[step["action"]]) - step["old_logp"]))
        if max(differences) > 1e-4:
            raise AssertionError(f"Old/new logp mismatch before update: {max(differences)}")
    encoder.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    query, candidates = encode_row(row)
    first, _, _ = state_distribution(row, query, candidates, (), CROSSFILE_TOKEN_BUDGET)
    valid_candidate = next((i for i, c in enumerate(row["candidate_costs"])
                            if c <= CROSSFILE_TOKEN_BUDGET), None)
    if valid_candidate is None:
        raise AssertionError("No valid candidate for encoder-gradient test")
    check_parameter = next((p for p in encoder.parameters()
                            if p.requires_grad and 1_000 < p.numel() <= 1_000_000), None)
    if check_parameter is None:
        raise AssertionError("Cannot locate a UniXcoder matrix for gradient test")
    grad = torch.autograd.grad(-first[valid_candidate],
                               check_parameter, allow_unused=True)[0]
    if grad is None or not torch.isfinite(grad).all() or grad.abs().sum().item() == 0:
        raise AssertionError("Actor loss has no finite nonzero UniXcoder gradient")
    encoder.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    a = torch.tensor([1.0, -1.0])
    ratios = torch.tensor([1.5, 0.5])
    clipped = torch.minimum(ratios * a, ratios.clamp(0.8, 1.2) * a)
    if not torch.allclose(clipped, torch.tensor([1.2, -0.8])):
        raise AssertionError("PPO positive/negative clipping test failed")
    print("PPO invariants passed: ratio=1 before update, actor→encoder gradient, two clip sides")

def prepare_advantages(episodes):
    all_advantages = []
    for episode in episodes:
        rewards = episode["step_rewards"]
        references = episode["reference_values"]
        if len(rewards) != len(episode["steps"]) or len(references) != len(rewards):
            raise ValueError("RRPO rewards/reference values must match action steps")
        advantages = [0.0] * len(rewards)
        running = 0.0
        for index in reversed(range(len(rewards))):
            next_value = references[index + 1] if index + 1 < len(rewards) else 0.0
            delta = rewards[index] + RRPO_GAMMA * next_value - references[index]
            running = delta + RRPO_GAMMA * RRPO_GAE_LAMBDA * running
            advantages[index] = running
        episode["raw_advantages"] = advantages
        all_advantages.extend(advantages)
    mean = float(np.mean(all_advantages))
    std = float(np.std(all_advantages))
    for episode in episodes:
        episode["advantages"] = [(value - mean) / max(std, 1e-6)
                                 for value in episode["raw_advantages"]]
    return mean, std

def ppo_episode_loss(episode):
    row = episode["row"]
    query, candidates = encode_row(row)
    actor_terms, entropy_terms, clips = [], [], []
    for step, advantage in zip(episode["steps"], episode["advantages"]):
        log_probs, _, valid = state_distribution(
            row, query, candidates, step["selected"], step["remaining"]
        )
        if valid != step["valid"]:
            raise AssertionError("PPO action support changed between rollout and update")
        old_logp = log_probs.new_tensor(step["old_logp"])
        log_ratio = log_probs[step["action"]] - old_logp
        if abs(float(log_ratio.detach())) > 20:
            raise FloatingPointError("PPO ratio diverged; lower LR or KL limit")
        ratio = log_ratio.exp()
        advantage_tensor = log_probs.new_tensor(advantage)
        unclipped = ratio * advantage_tensor
        clipped = ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP) * advantage_tensor
        actor_terms.append(torch.minimum(unclipped, clipped))
        entropy_terms.append(-(log_probs.exp() * log_probs).sum())
        clips.append(((ratio - 1).abs() > PPO_CLIP).float())
    horizon = MAX_SLATE_STEPS + 1
    actor_loss = -torch.stack(actor_terms).sum() / horizon
    entropy = torch.stack(entropy_terms).sum() / horizon
    loss = actor_loss - ENTROPY_COEF * entropy
    diagnostics = {"actor": float(actor_loss.detach()),
                   "entropy": float(entropy.detach()),
                   "clip_fraction": float(torch.stack(clips).mean().detach())}
    return loss, diagnostics

def measure_rollout_kl(episodes):
    """Exact categorical KL(old || updated) after the optimizer pass."""
    values = []
    with torch.no_grad():
        for episode in episodes:
            row = episode["row"]
            query, candidates = encode_row(row)
            for step in episode["steps"]:
                log_probs, _, valid = state_distribution(
                    row, query, candidates, step["selected"], step["remaining"]
                )
                if valid != step["valid"]:
                    raise AssertionError("PPO action support changed after update")
                old_probs = log_probs.new_tensor(step["old_probs"])
                kl = (old_probs * (old_probs.clamp_min(1e-12).log() - log_probs)).sum()
                values.append(float(kl))
    return float(np.mean(values))

def ppo_update(episodes):
    prepare_advantages(episodes)
    encoder.train()
    heads.train()
    totals = {k: [] for k in ("actor", "entropy",
                             "clip_fraction", "grad_norm", "amp_skipped")}
    passes_completed = 0
    pass_kl = 0.0
    for update_pass in range(PPO_PASSES):
        order = list(range(len(episodes)))
        random.shuffle(order)
        for start in range(0, len(order), ACCUMULATION_STEPS):
            batch = order[start:start + ACCUMULATION_STEPS]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                loss, diagnostics = ppo_episode_loss(episodes[index])
                scaler.scale(loss / len(batch)).backward()
                for key, value in diagnostics.items():
                    totals[key].append(value)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(heads.parameters()), MAX_GRAD_NORM
            )
            if not torch.isfinite(grad_norm):
                # GradScaler detects the overflow and skips this optimizer step.
                scaler.step(optimizer)
                scaler.update()
                totals["amp_skipped"].append(1.0)
                optimizer.zero_grad(set_to_none=True)
                continue
            totals["grad_norm"].append(float(grad_norm))
            totals["amp_skipped"].append(0.0)
            scaler.step(optimizer)
            scaler.update()
        passes_completed += 1
        pass_kl = measure_rollout_kl(episodes)
        if pass_kl > TARGET_KL:
            print(f"PPO early stop after pass {passes_completed}: KL={pass_kl:.5f}")
            break
    return {key: float(np.mean(values)) if values else float("nan")
            for key, values in totals.items()} | {
        "kl": pass_kl,
        "passes": passes_completed,
        "amp_scale": float(scaler.get_scale()),
    }

def validate_policy(rows):
    episodes = [rollout(row, greedy=True) for row in rows]
    prompts = [compose_prompt(ep["row"], ep["selected"]) for ep in episodes]
    outputs = generate_completions(prompts)
    scores = [rlcoder_score(out, ep["row"]["target_code"],
                            ep["row"]["left_context"], ep["row"]["language"])
              for out, ep in zip(outputs, episodes)]
    return {"es": float(np.mean([score["edit_similarity"] for score in scores])),
            "em": float(np.mean([score["exact_match"] for score in scores])),
            "id_f1": float(np.mean([score["identifier_f1"] for score in scores])),
            "n": len(episodes)}

CHECKPOINT_PATH = WORK_DIR / "latest.pt"
BEST_PATH = WORK_DIR / "best.pt"
attached_best = (RESUME_SOURCE.with_name("best.pt")
                 if RESUME_SOURCE is not None else None)
if (attached_best is not None and attached_best.is_file()
        and not BEST_PATH.is_file()):
    prior_best = torch.load(attached_best, map_location="cpu", weights_only=False)
    if prior_best.get("signature") != DATA_SIGNATURE:
        if not resume_signature_compatible(prior_best.get("signature")):
            raise RuntimeError(f"Attached best checkpoint has a different signature: {attached_best}")
        if validation_signature_changed(prior_best.get("signature")):
            print("Not carrying best.pt scored on a different validation split.")
        else:
            shutil.copy2(attached_best, BEST_PATH)
            print("Carried compatible best checkpoint into this session:", BEST_PATH)
    else:
        shutil.copy2(attached_best, BEST_PATH)
        print("Carried best checkpoint into this session's saved outputs:", BEST_PATH)
STATE = {"episodes_seen": 0, "epoch": 0, "cursor": 0,
         "order": [], "best_es": -1.0, "last_save_elapsed": 0.0}

def checkpoint_payload():
    return {"signature": DATA_SIGNATURE, "encoder": encoder_core.state_dict(),
            "heads": heads.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "state": STATE.copy(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng_all": (torch.cuda.get_rng_state_all()
                             if DEVICE.type == "cuda" else None),
            "cuda_rng": (torch.cuda.get_rng_state(DEVICE)
                         if DEVICE.type == "cuda" else None)}

def save_checkpoint(path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_payload(), temp)
    os.replace(temp, path)
    print("Saved", path, "episodes", STATE["episodes_seen"], flush=True)

def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved_signature = saved.get("signature")
    validation_changed = validation_signature_changed(saved_signature)
    if (saved_signature != DATA_SIGNATURE and
            not resume_signature_compatible(saved.get("signature"))):
        raise RuntimeError(
            "Resume checkpoint is not from this RRPO objective/schema run. "
            "For a new run, set AUTO_DISCOVER_RESUME=False; re-enable it only after "
            "saving a checkpoint from this notebook/configuration."
        )
    source = path
    best_path = path.with_name("best.pt")
    if best_path != path and best_path.is_file():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if best.get("signature") != DATA_SIGNATURE:
            if not resume_signature_compatible(best.get("signature")):
                raise RuntimeError(f"Attached best checkpoint has a different signature: {best_path}")
            if validation_signature_changed(best.get("signature")):
                print("Ignoring best.pt scored on a different validation split.")
            else:
                latest_state, best_state = saved["state"], best["state"]
                if (best_state["episodes_seen"] > latest_state["episodes_seen"] or
                        best_state["best_es"] > latest_state["best_es"]):
                    saved, source = best, best_path
                    validation_changed = False
                    print("Recovering newer best.pt after an interrupted latest.pt save")
        else:
            latest_state, best_state = saved["state"], best["state"]
            if (best_state["episodes_seen"] > latest_state["episodes_seen"] or
                    best_state["best_es"] > latest_state["best_es"]):
                saved, source = best, best_path
                validation_changed = False
                print("Recovering newer best.pt after an interrupted latest.pt save")
    encoder_core.load_state_dict(saved["encoder"])
    heads.load_state_dict(saved["heads"])
    optimizer.load_state_dict(saved["optimizer"])
    for group, learning_rate in zip(optimizer.param_groups, (ENCODER_LR, HEAD_LR)):
        group["lr"] = learning_rate
    scaler.load_state_dict(saved["scaler"])
    STATE.update(saved["state"])
    if validation_changed:
        STATE["best_es"] = -1.0  # Old best score used only a validation subset.
        print("Validation split changed; resumed model/optimizer state but reset "
              "best_es for scoring against the complete validation split.")
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    if saved.get("cuda_rng_all") is not None and len(saved["cuda_rng_all"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(saved["cuda_rng_all"])
    elif saved["cuda_rng"] is not None:
        torch.cuda.set_rng_state(saved["cuda_rng"], DEVICE)
    STATE["last_save_elapsed"] = 0.0  # New Kaggle kernel has a new monotonic clock.
    print("Resumed", source, "at episode", STATE["episodes_seen"],
          "epoch", STATE["epoch"], "cursor", STATE["cursor"])
    return saved

if RESUME_SOURCE is not None:
    load_checkpoint(RESUME_SOURCE)

policy_invariants(prepared_row(TRAIN_ROW_GROUPS[0]))
print("RRPO startup invariants passed; starting RRPO training.", flush=True)

# %%
train_deadline = START_MONOTONIC + TRAIN_STOP_HOURS_FROM_KERNEL_START * 3600
ETA_LAST_TIME = time.monotonic()
ETA_LAST_EPISODES = STATE["episodes_seen"]
ETA_EPISODES_PER_HOUR = None
STOP_REASON = "unknown"
while True:
    if time.monotonic() >= train_deadline - STOP_NEW_BATCH_RESERVE_SECONDS:
        STOP_REASON = "time_guard_reserve"
        print("Training deadline guard reached; no new RRPO batch will start.", flush=True)
        break
    if TRAIN_EPISODES_CAP is not None and STATE["episodes_seen"] >= TRAIN_EPISODES_CAP:
        STOP_REASON = "episode_cap"
        break
    if not STATE["order"] or STATE["cursor"] >= len(STATE["order"]):
        STATE["order"] = list(range(TRAIN_EXAMPLES))
        random.shuffle(STATE["order"])
        STATE["cursor"] = 0
        STATE["epoch"] += 1
        print("Full prebuilt train epoch:", STATE["epoch"],
              "rows:", TRAIN_EXAMPLES, flush=True)
    batch_ids = STATE["order"][STATE["cursor"]:STATE["cursor"] + ROLLOUT_QUERIES]
    if TRAIN_EPISODES_CAP is not None:
        batch_ids = batch_ids[:TRAIN_EPISODES_CAP - STATE["episodes_seen"]]
    if not batch_ids:
        break
    episodes_before_batch = STATE["episodes_seen"]
    batch_start = time.monotonic()
    for gpu_id in GPU_IDS:
        torch.cuda.reset_peak_memory_stats(gpu_id)
    try:
        load_start = time.monotonic()
        rows = [build_train_row(STATE["epoch"], slot) for slot in batch_ids]
        load_seconds = time.monotonic() - load_start
        print(f"Batch {episodes_before_batch + 1}-{episodes_before_batch + len(rows)}: "
              f"loaded {len(rows)} rows in {load_seconds:.1f}s; generating slates.",
              flush=True)
        rollout_start = time.monotonic()
        episodes = []
        for index, row in enumerate(rows, 1):
            episodes.append(rollout(row))
            if index == 1 or index % 8 == 0 or index == len(rows):
                print(f"Slate rollout progress: {index}/{len(rows)} episodes "
                      f"({time.monotonic() - rollout_start:.1f}s).", flush=True)
        rollout_seconds = time.monotonic() - rollout_start
        score_start = time.monotonic()
        generator_scores = score_rrpo_episodes(episodes)
        reward_seconds = time.monotonic() - score_start
        print(f"RRPO reward scoring finished in {reward_seconds:.1f}s; "
              "running PPO updates.", flush=True)
        update_start = time.monotonic()
        diagnostics = ppo_update(episodes)
        ppo_update_seconds = time.monotonic() - update_start
        print(f"PPO update finished in {ppo_update_seconds:.1f}s.", flush=True)
    except TrainingDeadlineReached as exc:
        STOP_REASON = "modal_idle_shutdown_deadline"
        print("Graceful time stop during RRPO batch:", exc, flush=True)
        save_checkpoint(CHECKPOINT_PATH)
        break
    except Exception:
        save_checkpoint(CHECKPOINT_PATH)
        raise
    STATE["cursor"] += len(episodes)
    STATE["episodes_seen"] += len(episodes)
    if STATE["cursor"] >= len(STATE["order"]):
        STATE["last_save_elapsed"] = time.monotonic() - START_MONOTONIC
        print(f"Epoch {STATE['epoch']} complete; saving resumable state.", flush=True)
        save_checkpoint(CHECKPOINT_PATH)
    batch_seconds = time.monotonic() - batch_start
    peak_vram_by_gpu_gb = {
        str(gpu_id): round(torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 3), 2)
        for gpu_id in GPU_IDS
    }
    peak_vram_gb = max(peak_vram_by_gpu_gb.values(), default=0.0)
    episodes_per_hour = len(episodes) * 3600 / max(batch_seconds, 1e-6)
    eta_now = time.monotonic()
    eta_interval_seconds = max(eta_now - ETA_LAST_TIME, 1e-6)
    eta_interval_episodes = STATE["episodes_seen"] - ETA_LAST_EPISODES
    measured_rate = eta_interval_episodes * 3600 / eta_interval_seconds
    ETA_EPISODES_PER_HOUR = (measured_rate if ETA_EPISODES_PER_HOUR is None else
                             0.3 * measured_rate + 0.7 * ETA_EPISODES_PER_HOUR)
    ETA_LAST_TIME = eta_now
    ETA_LAST_EPISODES = STATE["episodes_seen"]
    epoch_episodes_remaining = max(TRAIN_EXAMPLES - STATE["cursor"], 0)
    epoch_eta_seconds = epoch_episodes_remaining * 3600 / max(ETA_EPISODES_PER_HOUR, 1e-6)
    train_seconds_remaining = max(train_deadline - eta_now, 0.0)
    estimated_examples_before_deadline = int(
        train_seconds_remaining * ETA_EPISODES_PER_HOUR / 3600
    )
    epoch_eta_local = datetime.now().astimezone() + timedelta(seconds=epoch_eta_seconds)
    hard_stop_eta_local = datetime.now().astimezone() + timedelta(
        seconds=train_seconds_remaining
    )
    print(json.dumps({"episode": STATE["episodes_seen"],
                      "epoch": STATE["epoch"],
                      "epoch_progress_pct": round(100 * STATE["cursor"] /
                                                   TRAIN_EXAMPLES, 1),
                      "epoch_eta_local": epoch_eta_local.strftime("%Y-%m-%d %H:%M:%S %Z"),
                      "train_hard_stop_eta_local": hard_stop_eta_local.strftime(
                          "%Y-%m-%d %H:%M:%S %Z"),
                      "train_hours_remaining": round(train_seconds_remaining / 3600, 2),
                      "ema_episodes_per_hour": round(ETA_EPISODES_PER_HOUR, 1),
                      "estimated_examples_before_hard_stop": estimated_examples_before_deadline,
                      "languages": {language: sum(ep["row"]["language"] == language
                                                   for ep in episodes)
                                    for language in ("python", "java")},
                      "target_kinds": {kind: sum(ep["row"]["target_kind"] == kind
                                                 for ep in episodes)
                                       for kind in ("line", "block")},
                      "mean_candidates": float(np.mean([
                          len(ep["row"]["candidate_ids"]) for ep in episodes])),
                      "mean_selected_chunks": float(np.mean([
                          len(ep["selected"]) for ep in episodes])),
                      "mean_selected_token_cost": float(np.mean([
                          sum(ep["row"]["candidate_costs"][i] for i in ep["selected"])
                          for ep in episodes])),
                      "mean_reward": float(np.mean([ep["reward"] for ep in episodes])),
                      "generator_scores": generator_scores,
                      "mean_final_es": float(np.mean([ep["final_es"] for ep in episodes])),
                      "mean_final_id_f1": float(np.mean([
                          ep["final_id_f1"] for ep in episodes])),
                      "mean_reference_value": float(np.mean([
                          value for ep in episodes for value in ep["reference_values"]])),
                      "stop_rate": float(np.mean([
                          ep["steps"][-1]["action"] == len(ep["row"]["candidate_ids"])
                          for ep in episodes])),
                      "load_seconds": round(load_seconds, 2),
                      "rollout_seconds": round(rollout_seconds, 2),
                      "reward_seconds": round(reward_seconds, 2),
                      "ppo_update_seconds": round(ppo_update_seconds, 2),
                      "batch_seconds": round(batch_seconds, 2),
                      "peak_vram_gb": round(peak_vram_gb, 2),
                      "peak_vram_by_gpu_gb": peak_vram_by_gpu_gb,
                      "episodes_per_hour": round(episodes_per_hour, 1),
                      **diagnostics}), flush=True)
    due = (STATE["episodes_seen"] // VALIDATE_EVERY_EPISODES >
           episodes_before_batch // VALIDATE_EVERY_EPISODES)
    if due or (TRAIN_EPISODES_CAP is not None and
               STATE["episodes_seen"] >= TRAIN_EPISODES_CAP):
        # Validation uses the remote generator too. Persist the just-completed
        # optimizer update first so an endpoint disconnect cannot roll back a batch.
        STATE["last_save_elapsed"] = time.monotonic() - START_MONOTONIC
        save_checkpoint(CHECKPOINT_PATH)
        try:
            validation = validate_policy(valid_rows)
        except TrainingDeadlineReached as exc:
            STOP_REASON = "modal_idle_shutdown_deadline"
            print("Skipping validation at the Modal shutdown cutoff:", exc, flush=True)
            save_checkpoint(CHECKPOINT_PATH)
            break
        except GeneratorRequestError as exc:
            print("Validation skipped after Modal/vLLM transport failure; "
                  "checkpoint is safe and training will continue:", exc, flush=True)
            continue
        print("Validation:", validation, flush=True)
        if validation["es"] > STATE["best_es"]:
            STATE["best_es"] = validation["es"]
            save_checkpoint(BEST_PATH)
            save_checkpoint(CHECKPOINT_PATH)
    elapsed = time.monotonic() - START_MONOTONIC
    if elapsed - STATE["last_save_elapsed"] >= SAVE_INTERVAL_SECONDS:
        STATE["last_save_elapsed"] = elapsed
        save_checkpoint(CHECKPOINT_PATH)
# Save the exact final training state before doing any optional validation/eval.
save_checkpoint(CHECKPOINT_PATH)
remaining_before_deadline = train_deadline - time.monotonic()
if remaining_before_deadline >= FINAL_VALIDATION_MIN_REMAINING_SECONDS:
    try:
        final_validation = validate_policy(valid_rows)
        print("Final validation:", final_validation, flush=True)
        if final_validation["es"] > STATE["best_es"] or not BEST_PATH.is_file():
            STATE["best_es"] = final_validation["es"]
            save_checkpoint(BEST_PATH)
    except TrainingDeadlineReached as exc:
        print("Skipping final validation at the Modal shutdown cutoff:", exc,
              flush=True)
    except GeneratorRequestError as exc:
        print("Skipping final validation after Modal/vLLM transport failure; "
              "latest.pt contains the completed training state:", exc, flush=True)
else:
    print("Skipping final validation to preserve the Kaggle shutdown margin; "
          "latest.pt already contains the final RRPO state.", flush=True)
save_checkpoint(CHECKPOINT_PATH)
print("Training stop reason:", STOP_REASON)
print("Training stopped; elapsed hours:",
      round((time.monotonic() - START_MONOTONIC) / 3600, 2))


# %%
# Training-only finalization. Run held-out test benchmarks from the separate
# evaluation notebook after this training session.
GENERATION_CACHE_CONNECTION.commit()
GENERATION_CACHE_CONNECTION.close()
print("Training artifacts:", flush=True)
print("  latest checkpoint:", CHECKPOINT_PATH, CHECKPOINT_PATH.exists(), flush=True)
print("  best checkpoint:", BEST_PATH, BEST_PATH.exists(), flush=True)
print("  generation cache:", GENERATION_CACHE_PATH, flush=True)
print("Kaggle output directory:", WORK_DIR, flush=True)
