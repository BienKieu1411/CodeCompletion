# %% [markdown]
# # Evaluate AST/BM25/learned-critic-PPO retrieval — Kaggle T4×2
#
# Standalone CCEval evaluation notebook for terminal-reward PPO. Attach the
# prepared `Code_Completion_Train_Dataset` and the matching PPO checkpoint pair.
# Only CCEval Python and Java are evaluated in this first pass.
# GPU 0 serves frozen DeepSeek-Coder-1.3B; GPU 1 runs best and latest UniXcoder
# weights in turn. Both checkpoints see the same tasks and frozen generator.
# Prompts use up to 3,072 tokens plus 96 completion tokens, preserving an exact
# cursor suffix and adding only relevant pre-cursor AST hints for long files.
# No training, optimizer restore, or training-data preparation occurs here.
#
# Set both checkpoint paths below if auto-discovery finds multiple runs. The
# resolver only accepts the terminal-PPO run directory, never the RRPO checkpoint.
# Outputs are written to a new timestamped `/kaggle/working/ast_ppo_terminal_eval_v1_k10_ctx3072_xfb2344/...` folder.
# Requires data schema v5 + identifier-aware score schema + K=10 PPO checkpoints; old checkpoints
# remain untouched. EM/ES use RLCoder's statement truncation and comment removal.
# The two prepared flat files (`cceval_python.parquet`, `cceval_java.parquet`)
# are auto-discovered; raw nested Data4AlignCoder CCEval files remain a fallback.
# With sampled tasks and a different prompt/retriever, these are not directly
# comparable with a paper's full official benchmark.
# pool_identifier_coverage.jsonl reports an eval-only lexical recall proxy across
# all crossfile sources, selected files, and the final pool of 64 AST chunks.
# The gold target is used only for this diagnostic after candidates are fixed.

# %%
import hashlib
import importlib.util
import json
import keyword
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BEST_CHECKPOINT_PATH = None  # Example: "/kaggle/input/my-ppo-checkpoint/best.pt"
LATEST_CHECKPOINT_PATH = None  # Example: "/kaggle/input/my-ppo-checkpoint/latest.pt"
EVAL_LIMIT_PER_FILE = 100  # Set to None for all rows in each benchmark parquet.
SEED = 17
DOWNLOAD_DATA_IF_MISSING = True
DATASET_REPO = "AlignCoder/Data4AlignCoder"
RETRIEVER_MODEL = "microsoft/unixcoder-base"
GENERATOR_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
SERVED_MODEL_NAME = "deepseek-coder-1.3b-base"
API_BASE = "http://127.0.0.1:8000/v1"
VLLM_PORT = 8000
VLLM_MAX_NUM_SEQS = 16
GENERATOR_INPUT_TOKENS = 3072
GENERATOR_OUTPUT_TOKENS = 96
GENERATOR_MAX_MODEL_LEN = GENERATOR_INPUT_TOKENS + GENERATOR_OUTPUT_TOKENS
MAX_RELATED_FILES = 24
AST_CHUNK_TOKENS = 384
DATA_PIPELINE_SCHEMA = "phong_ast_boundary_prebuilt_repo_rows_v5"
SCORE_SCHEMA = "rlcoder_stmt_postprocess_em_es_idf1_v2"
RETRIEVER_QUERY_LENGTH = 256
RETRIEVER_CANDIDATE_LENGTH = 512
CANDIDATE_POOL_SIZE = 64
ENCODE_BATCH_SIZE = 16  # Eval microbatch only; not part of checkpoint compatibility.
CROSSFILE_TOKEN_BUDGET = 2344
GENERATOR_CURRENT_FILE_BUDGET = GENERATOR_INPUT_TOKENS - CROSSFILE_TOKEN_BUDGET
MAX_SLATE_STEPS = 10
OUTPUT_ROOT = Path("/kaggle/working/ast_ppo_terminal_eval_v1_k10_ctx3072_xfb2344")
CACHE_DIR = Path("/kaggle/working/hf-cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

import editdistance
import numpy as np
import pandas as pd
import pyarrow as pa
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import tree_sitter as ts
import tree_sitter_java as ts_java
import tree_sitter_python as ts_python
import pyarrow.parquet as pq
from fuzzywuzzy import fuzz
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

if torch.cuda.device_count() != 2:
    raise RuntimeError("Select Kaggle T4×2 GPU; the generator and retriever use separate cards.")
DEVICE = torch.device("cuda:1")
torch.cuda.set_device(DEVICE)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(2)])
print(f"Generator allocation: max input {GENERATOR_INPUT_TOKENS}; cross-file "
      f"snippets <= {CROSSFILE_TOKEN_BUDGET}; current-file path + left context "
      f"share up to {GENERATOR_CURRENT_FILE_BUDGET} tokens; output "
      f"{GENERATOR_OUTPUT_TOKENS} tokens.", flush=True)

def find_checkpoints():
    if BEST_CHECKPOINT_PATH is not None or LATEST_CHECKPOINT_PATH is not None:
        if BEST_CHECKPOINT_PATH is None or LATEST_CHECKPOINT_PATH is None:
            raise ValueError("Set both BEST_CHECKPOINT_PATH and LATEST_CHECKPOINT_PATH.")
        paths = {"best": Path(BEST_CHECKPOINT_PATH),
                 "latest": Path(LATEST_CHECKPOINT_PATH)}
        if not paths["latest"].is_file() and paths["latest"].name == "latest.pt":
            extensionless = paths["latest"].with_name("latest")
            if extensionless.is_file():
                paths["latest"] = extensionless
        for label, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{label} checkpoint not found: {path}")
        return paths

    run_dirname = "ast_ppo_terminal_v1_k10_ctx3072_xfb2344"
    roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    pairs = set()
    for root in roots:
        if not root.exists():
            continue
        for best_path in root.rglob("best.pt"):
            latest_path = best_path.with_name("latest.pt")
            if not latest_path.is_file():
                latest_path = best_path.with_name("latest")
            if latest_path.is_file() and run_dirname in best_path.parts:
                pairs.add(best_path.parent.resolve())
    pairs = sorted(pairs)
    if len(pairs) != 1:
        raise RuntimeError(
            f"Expected one terminal-PPO best.pt/latest checkpoint pair, found {pairs}. "
            "Attach the matching files from ast_ppo_terminal_v1_k10_ctx3072_xfb2344 "
            "or set both paths explicitly; do not use an RRPO checkpoint."
        )
    latest = pairs[0] / "latest.pt"
    if not latest.is_file():
        latest = pairs[0] / "latest"
    return {"best": pairs[0] / "best.pt", "latest": latest}

CKPT_PATHS = find_checkpoints()
# Only load checkpoints that you created/trust: torch.load uses Python pickle.
policies = {}
expected = {
    "retriever": RETRIEVER_MODEL, "generator": GENERATOR_MODEL,
    "data_pipeline_schema": DATA_PIPELINE_SCHEMA, "pool_schema": 6,
    "score_schema": SCORE_SCHEMA,
    "file_selection_policy": "all_repo_files_then_query_bm25",
    "max_related_files": MAX_RELATED_FILES,
    "chunk": AST_CHUNK_TOKENS, "retriever_query_length": RETRIEVER_QUERY_LENGTH,
    "retriever_candidate_length": RETRIEVER_CANDIDATE_LENGTH,
    "pool": CANDIDATE_POOL_SIZE, "crossfile_budget": CROSSFILE_TOKEN_BUDGET,
    "max_slate_steps": MAX_SLATE_STEPS,
    "objective": "ppo_clip_terminal_reward_learned_critic_gae_v1",
    "advantage_estimator": "learned_critic_gae_v1",
    "ppo_gamma": 1.0, "ppo_gae_lambda": 0.95,
    "value_coef": 0.5, "value_clip": 0.2,
    "identifier_f1_weight": 0.2,
    "reward": "terminal_rlcoder_es_identifier_f1_v1",
    "generator_output_tokens": GENERATOR_OUTPUT_TOKENS,
    "generator_input_tokens": GENERATOR_INPUT_TOKENS,
    "generator_model_len": GENERATOR_MAX_MODEL_LEN,
    "prompt_packer": "ast_relevant_hints_cursor_suffix_v2",
}
LEGACY_UNRECORDED_KEYS = set()

def validate_signature(signature, label):
    if not isinstance(signature, dict):
        raise ValueError(f"{label} checkpoint has no configuration signature.")
    mismatches = {k: (signature.get(k), v) for k, v in expected.items()
                  if (k in signature or k not in LEGACY_UNRECORDED_KEYS)
                  and signature.get(k) != v}
    if mismatches:
        raise ValueError(f"{label} checkpoint/eval configuration differs: {mismatches}")
    missing = sorted(LEGACY_UNRECORDED_KEYS - signature.keys())
    if missing:
        assumed = {k: expected[k] for k in missing}
        print(f"WARNING: {label} checkpoint did not record {missing}; "
              f"assuming eval settings {assumed}. "
              "Verify these against the original training notebook.")

for label, path in CKPT_PATHS.items():
    payload = torch.load(path, map_location="cpu", weights_only=False)
    signature = payload.get("signature")
    validate_signature(signature, label)
    if "encoder" not in payload or "heads" not in payload:
        raise ValueError(f"{label} checkpoint needs encoder and heads state dicts.")
    policies[label] = {key: payload[key] for key in ("signature", "encoder", "heads", "state")}
    print(label, "checkpoint:", path, "training state:", policies[label]["state"])
    del payload
if policies["best"]["signature"] != policies["latest"]["signature"]:
    raise ValueError("best.pt and latest.pt come from different training configurations.")
if policies["best"]["state"]["episodes_seen"] > policies["latest"]["state"]["episodes_seen"]:
    print("WARNING: best.pt is newer than latest.pt, likely from an interrupted "
          "session. Evaluating both saved snapshots; resume training will recover best.pt.")
RUN_DIR = OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=False)
with (RUN_DIR / "run_manifest.json").open("w", encoding="utf-8") as handle:
    json.dump({"checkpoints": {label: {"path": str(path), "size": path.stat().st_size,
                                       "episodes_seen": policies[label]["state"]["episodes_seen"]}
                               for label, path in CKPT_PATHS.items()},
               "seed": SEED, "limit_per_file": EVAL_LIMIT_PER_FILE,
               "generator": GENERATOR_MODEL, "retriever": RETRIEVER_MODEL,
               "objective": expected["objective"],
               "reward_mode": "terminal_only",
               "data_pipeline_schema": DATA_PIPELINE_SCHEMA,
               "score_schema": SCORE_SCHEMA},
              handle, indent=2)

# %%
BENCHMARKS = [
    ("cceval_python", "cceval_python.parquet", "python"),
    ("cceval_java", "cceval_java.parquet", "java"),
]

def locate_data_root():
    mounts = [Path("/kaggle/working/data4aligncoder"), Path("/kaggle/working")]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        mounts = sorted(p for p in input_root.iterdir() if p.is_dir()) + mounts
    candidates = []
    for mount in mounts:
        candidates.extend((mount, mount / "data", mount / "Data4AlignCoder",
                           mount / "Data4AlignCoder" / "data"))
        if mount not in {Path("/kaggle/working"),
                         Path("/kaggle/working/data4aligncoder")}:
            candidates.extend(path.parent
                              for path in mount.glob("**/cceval_python.parquet"))
            candidates.extend(path.parents[2]
                              for path in mount.glob("**/cceval/python/test.parquet"))
    unique_candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    def complete(root):
        flat = all((root / rel).is_file() for _, rel, _ in BENCHMARKS)
        nested = all((root / "cceval" / language / "test.parquet").is_file()
                     for _, _, language in BENCHMARKS)
        return flat or nested
    complete_candidates = [root for root in unique_candidates if complete(root)]
    for root in complete_candidates:
        flat_python = root / "cceval_python.parquet"
        nested_python = root / "cceval" / "python" / "test.parquet"
        schema = pq.ParquetFile(
            flat_python if flat_python.is_file() else nested_python
        ).schema_arrow
        if "crossfile_ast_chunks" in schema.names:
            return root
    if complete_candidates:
        return complete_candidates[0]
    if DOWNLOAD_DATA_IF_MISSING:
        from huggingface_hub import snapshot_download
        destination = Path("/kaggle/working/data4aligncoder")
        snapshot_download(repo_id=DATASET_REPO, repo_type="dataset",
                          local_dir=str(destination), cache_dir=str(CACHE_DIR / "hub"),
                          allow_patterns=["data/cceval/python/test.parquet",
                                          "data/cceval/java/test.parquet"])
        if complete(destination / "data"):
            return (destination / "data").resolve()
    raise FileNotFoundError(
        "Missing cceval_python.parquet/cceval_java.parquet in Kaggle inputs "
        "or raw CCEval files from Data4AlignCoder."
    )

DATA_ROOT = locate_data_root()
print("Benchmark data:", DATA_ROOT)

def benchmark_data_path(root, relative_path, language):
    """Resolve the prepared flat CCEval file or raw nested benchmark path."""
    flat_path = root / relative_path
    if flat_path.is_file():
        return flat_path
    return root / "cceval" / language / "test.parquet"

GEN_TOKENIZER = AutoTokenizer.from_pretrained(GENERATOR_MODEL, cache_dir=str(CACHE_DIR))
RET_TOKENIZER = AutoTokenizer.from_pretrained(
    RETRIEVER_MODEL, cache_dir=str(CACHE_DIR), use_fast=False
)
if RET_TOKENIZER.pad_token_id is None:
    raise RuntimeError("UniXcoder tokenizer needs a pad token.")
if RET_TOKENIZER.convert_tokens_to_ids("<encoder-only>") == RET_TOKENIZER.unk_token_id:
    raise RuntimeError("UniXcoder tokenizer lacks <encoder-only> mode token.")

# %%
PARSERS = {
    "python": ts.Parser(ts.Language(ts_python.language())),
    "java": ts.Parser(ts.Language(ts_java.language())),
}
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
            hidden_states = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            weights = mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        batches.append(F.normalize(pooled.float(), p=2, dim=-1))
    pooled = torch.cat(batches, dim=0)
    return pooled[0], pooled[1:]

def verify_served_model(cards, require_context_metadata):
    matching = [card for card in cards if card.get("id") == SERVED_MODEL_NAME]
    if len(matching) != 1:
        raise RuntimeError(f"Port {VLLM_PORT} does not serve {SERVED_MODEL_NAME}: "
                           f"{[card.get('id') for card in cards]}")
    advertised = matching[0].get("max_model_len")
    if advertised is None:
        if require_context_metadata:
            raise RuntimeError("Existing vLLM server does not advertise max_model_len; "
                               f"restart the Kaggle session before using "
                               f"{GENERATOR_INPUT_TOKENS} input tokens.")
        print("WARNING: new vLLM server did not advertise max_model_len; "
              "using its configured launch limit", GENERATOR_MAX_MODEL_LEN)
    elif int(advertised) < GENERATOR_MAX_MODEL_LEN:
        raise RuntimeError(f"Existing vLLM context limit {advertised} is below "
                           f"{GENERATOR_MAX_MODEL_LEN}; restart the Kaggle session.")
    else:
        print("vLLM advertised context length:", advertised)

def decode_crossfile_context(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Unexpected crossfile_context type: {type(value).__name__}")
    if any(not isinstance(item, dict) or "path" not in item or "text" not in item
           for item in value):
        raise ValueError("crossfile_context must contain {path, text} objects")
    return list(value)

probe = pa.table({"crossfile_context": pa.array(
    [[{"path": "helper.py", "text": "def helper(): return 1"}]],
    type=pa.list_(pa.struct([("path", pa.string()), ("text", pa.string())]))
)}).to_pandas()["crossfile_context"].iloc[0]
assert isinstance(probe, np.ndarray) and len(decode_crossfile_context(probe)) == 1
print("Arrow/Pandas crossfile_context decoder preflight passed")

def lexical_tokens(text):
    return re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*|\d+|\S", text.lower())

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

def bm25_budget_selection(row):
    selected, remaining = [], CROSSFILE_TOKEN_BUDGET
    for i, cost in enumerate(row["candidate_costs"]):
        if cost <= remaining:
            selected.append(i)
            remaining -= cost
        if len(selected) >= MAX_SLATE_STEPS:
            break
    return selected

# %%
encoder = AutoModel.from_pretrained(RETRIEVER_MODEL, cache_dir=str(CACHE_DIR)).to(DEVICE)
hidden = int(encoder.config.hidden_size)

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
def activate_policy(label):
    encoder.load_state_dict(policies[label]["encoder"], strict=True)
    heads.load_state_dict(policies[label]["heads"], strict=True)
    encoder.eval()
    heads.eval()

activate_policy("best")
print("Loaded PPO retriever on GPU 1; encoder hidden size:", hidden)

@torch.inference_mode()
def ppo_select(row):
    query, candidates = encode_row(row)
    selected, remaining = [], CROSSFILE_TOKEN_BUDGET
    for _ in range(MAX_SLATE_STEPS):
        n = len(row["candidate_ids"])
        selected_mean = (candidates[selected].mean(0) if selected else
                         torch.zeros_like(query))
        fraction = remaining / CROSSFILE_TOKEN_BUDGET
        features = torch.cat([
            query.expand(n, -1), candidates, selected_mean.expand(n, -1),
            query * candidates, selected_mean * candidates,
            query.new_tensor(row["candidate_costs"]).unsqueeze(1) / CROSSFILE_TOKEN_BUDGET,
            query.new_full((n, 1), fraction),
        ], dim=-1)
        candidate_logits = heads.candidate(features).squeeze(-1).float()
        stop_features = torch.cat([query, selected_mean, query.new_tensor([fraction])])
        stop_logit = heads.stop(stop_features).float().view(1)
        valid = [i not in selected and row["candidate_costs"][i] <= remaining
                 for i in range(n)] + [True]
        logits = torch.cat([candidate_logits, stop_logit]).masked_fill(
            ~torch.tensor(valid, dtype=torch.bool, device=DEVICE), -1e9
        )
        action = int(torch.argmax(logits).item())
        if action == n:
            break
        selected.append(action)
        remaining -= row["candidate_costs"][action]
    return selected

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

# %%
def served_models():
    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=3)
        if not health.ok:
            return None
        response = requests.get(API_BASE + "/models", timeout=3)
        response.raise_for_status()
        return response.json()["data"]
    except requests.RequestException:
        return None

models = served_models()
if models is not None:
    verify_served_model(models, require_context_metadata=True)
if models is None:
    if importlib.util.find_spec("kaggle_vllm") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "kaggle-vllm[hub]==0.2.0"])
    runtime_root = OUTPUT_ROOT / "kaggle-vllm-runtime"
    manifest = runtime_root / "runtime.json"
    runtime_root.mkdir(parents=True, exist_ok=True)
    if not manifest.is_file():
        subprocess.run([
            "kaggle-vllm", "bootstrap", "--strict",
            "--cache", str(OUTPUT_ROOT / "kaggle-vllm-cache"),
            "--staged", str(runtime_root / "vllm-staged"),
            "--overlay", str(runtime_root / "vllm-overlay"),
            "--manifest", str(manifest),
        ], check=True)
    log_path = RUN_DIR / "vllm-server.log"
    log_handle = log_path.open("w", encoding="utf-8")
    server_env = os.environ.copy()
    server_env.update({"CUDA_VISIBLE_DEVICES": "0", "HF_HOME": str(CACHE_DIR),
                       "GENERATOR_MODEL": GENERATOR_MODEL,
                       "SERVED_MODEL_NAME": SERVED_MODEL_NAME,
                       "VLLM_MAX_MODEL_LEN": str(GENERATOR_MAX_MODEL_LEN),
                       "KAGGLE_VLLM_MANIFEST_PATH": str(manifest)})
    serve_shell = (
        'VLLM_ENV="$(kaggle-vllm env --manifest "$KAGGLE_VLLM_MANIFEST_PATH")" && '
        'eval "$VLLM_ENV" && '
        'exec vllm serve "$GENERATOR_MODEL" '
        '--host 127.0.0.1 --port 8000 '
        '--served-model-name "$SERVED_MODEL_NAME" '
        '--dtype float16 --tensor-parallel-size 1 '
        '--max-model-len "$VLLM_MAX_MODEL_LEN" --gpu-memory-utilization 0.78 '
        f'--max-num-seqs {VLLM_MAX_NUM_SEQS} --enforce-eager'
    )
    process = subprocess.Popen(["bash", "-lc", serve_shell], env=server_env,
                               stdout=log_handle, stderr=subprocess.STDOUT,
                               start_new_session=True)
    log_handle.close()
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError("vLLM exited during startup:\n" +
                               log_path.read_text(errors="replace")[-6000:])
        current_models = served_models()
        if current_models is not None and any(
                card.get("id") == SERVED_MODEL_NAME for card in current_models):
            break
        time.sleep(10)
    else:
        raise TimeoutError("vLLM startup timed out:\n" +
                           log_path.read_text(errors="replace")[-6000:])
    print("Started DeepSeek-Coder vLLM on GPU 0; PID:", process.pid)
    verify_served_model(served_models(), require_context_metadata=False)
else:
    print("Reusing DeepSeek-Coder vLLM server on GPU 0")

OUTPUT_CACHE = {}

def generate_completions(prompts):
    fresh = list(dict.fromkeys(p for p in prompts if p not in OUTPUT_CACHE))
    for start in range(0, len(fresh), VLLM_MAX_NUM_SEQS):
        batch = fresh[start:start + VLLM_MAX_NUM_SEQS]
        token_ids = [GEN_TOKENIZER.encode(p, add_special_tokens=True) for p in batch]
        max_prompt = GENERATOR_INPUT_TOKENS
        if any(len(ids) > max_prompt for ids in token_ids):
            raise ValueError(f"Generator prompt exceeds {max_prompt} input tokens; "
                             "refusing to truncate retrieved context or cursor suffix")
        response = requests.post(API_BASE + "/completions", json={
            "model": SERVED_MODEL_NAME, "prompt": token_ids,
            "max_tokens": GENERATOR_OUTPUT_TOKENS,
            "temperature": 0.0, "top_p": 1.0,
        }, timeout=900)
        if not response.ok:
            raise RuntimeError(f"vLLM HTTP {response.status_code}: {response.text[:2000]}")
        choices = sorted(response.json()["choices"], key=lambda item: item.get("index", 0))
        if len(choices) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} completions, got {len(choices)}")
        OUTPUT_CACHE.update({prompt: str(choice.get("text", ""))
                             for prompt, choice in zip(batch, choices)})
    return [OUTPUT_CACHE[prompt] for prompt in prompts]

print("Generation probe:", repr(generate_completions(["def add(a, b):\n    "])[0][:80]))

# %%
def choose_rows(frame, name):
    if EVAL_LIMIT_PER_FILE is None or len(frame) <= EVAL_LIMIT_PER_FILE:
        return frame
    if not name.startswith("repoeval"):
        return frame.sample(n=EVAL_LIMIT_PER_FILE, random_state=SEED).sort_index()
    # RepoEval is grouped by repository; head(100) may cover only one repo.
    groups = [group.sample(frac=1, random_state=SEED).index.tolist()
              for _, group in frame.groupby(frame["task_id"].str.split("/").str[0],
                                            sort=True)]
    indices = []
    for position in range(max(map(len, groups))):
        for group in groups:
            if position < len(group):
                indices.append(group[position])
                if len(indices) == EVAL_LIMIT_PER_FILE:
                    return frame.loc[indices]
    return frame.loc[indices]

def benchmark_examples(name, relative_path, language):
    benchmark_path = benchmark_data_path(DATA_ROOT, relative_path, language)
    if not benchmark_path.is_file():
        raise FileNotFoundError(benchmark_path)
    frame = pd.read_parquet(benchmark_path)
    required = {"task_id", "path", "left_context", "crossfile_context", "groundtruth"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{name} missing {sorted(required - set(frame.columns))}")
    frame = choose_rows(frame, name)
    examples = []
    def text_or_empty(value):
        return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    for item in frame.to_dict("records"):
        related = decode_crossfile_context(item["crossfile_context"])
        examples.append({"task_id": str(item["task_id"]), "language": language,
                         "file_path": str(item["path"]),
                         "left_context": text_or_empty(item["left_context"]),
                         "target_code": text_or_empty(item["groundtruth"]),
                         "related_files": related})
    return examples

def pool_identifier_probe(example, row):
    """Eval-only lexical proxy for where target identifiers leave the candidate funnel.

    This uses the gold completion only after build_row has fixed its candidates.
    Identifier overlap is not proof that a snippet is useful for generation.
    """
    java_keywords = set("""abstract assert boolean break byte case catch char class
        const continue default do double else enum extends final finally float for
        goto if implements import instanceof int interface long native new package
        private protected public return short static strictfp super switch synchronized
        this throw throws transient try void volatile while true false null var""".split())
    excluded = set(keyword.kwlist) | java_keywords | {"self"}
    target_names = set(IDENTIFIER_RE.findall(example["target_code"]))
    left_names = set(IDENTIFIER_RE.findall(example["left_context"]))
    novel = target_names - left_names - excluded
    normalize_path = lambda path: os.path.normpath(str(path).replace("\\", "/"))
    target_path = normalize_path(example["file_path"])
    selected_paths = {normalize_path(path) for path in row["related_file_paths_used"]}
    repo_hits, selected_file_hits = set(), set()
    for related in example["related_files"]:
        if isinstance(related, dict):
            path, code = related.get("path", ""), related.get("text", "")
        else:
            path, code = related[0], related[1]
        path = normalize_path(path)
        if path == target_path or not code:
            continue
        hits = novel & set(IDENTIFIER_RE.findall(str(code)))
        repo_hits.update(hits)
        if path in selected_paths:
            selected_file_hits.update(hits)
    pool_hits = novel & set(IDENTIFIER_RE.findall(
        "\n".join(row["candidate_raw_texts"])))
    return {"task_id": example["task_id"],
            "novel_target_identifiers": len(novel),
            "repo_identifier_hits": len(repo_hits),
            "selected_file_identifier_hits": len(selected_file_hits),
            "pool_identifier_hits": len(pool_hits),
            "missed_at_file_selection": sorted(repo_hits - selected_file_hits),
            "missed_after_file_selection": sorted(selected_file_hits - pool_hits)}

def summarize(records):
    frame = pd.DataFrame(records)
    line_frame = frame.loc[frame["dataset"].isin(["repoeval_line_0", "repoeval_line_1"])].copy()
    line_frame.loc[:, "dataset"] = "repoeval_line"
    diagnostics = pd.concat([frame, line_frame]).groupby(["dataset", "method"], sort=True).agg(
        mean_candidates=("candidate_count", "mean"),
        mean_fallback_candidates=("fallback_candidates", "mean"),
        eligible_rate=("eligible_candidates", lambda x: float((x > 0).mean())),
        selection_rate=("selected_count", lambda x: float((x > 0).mean())),
        mean_selected=("selected_count", "mean"),
    ).reset_index()
    score_rows = rlcoder_summary(records)
    for row in score_rows:
        row.pop("repoeval_em_pct", None)
        row.pop("repoeval_es_pct", None)
    summary = pd.DataFrame(score_rows).merge(
        diagnostics, on=["dataset", "method"], how="left")
    summary.to_csv(RUN_DIR / "benchmark_summary.csv", index=False)
    print(summary.to_string(index=False))
    paired = []
    for name, group in frame.groupby("dataset"):
        table = group.pivot(index="task_id", columns="method",
                            values=["edit_similarity", "prediction", "eligible_candidates"])
        for policy, baseline in (("ppo_best_ast", "no_retrieval"),
                                 ("ppo_best_ast", "bm25_ast_budget"),
                                 ("ppo_latest_ast", "no_retrieval"),
                                 ("ppo_latest_ast", "bm25_ast_budget"),
                                 ("ppo_best_ast", "ppo_latest_ast")):
            for eligible_only in (False, True):
                selected = table[table[("eligible_candidates", policy)] > 0] \
                    if eligible_only else table
                if selected.empty:
                    continue
                delta = selected[("edit_similarity", policy)].astype(float) - \
                    selected[("edit_similarity", baseline)].astype(float)
                paired.append({"dataset": name, "policy": policy, "baseline": baseline,
                               "eligible_only": eligible_only, "n": len(delta),
                               "mean_delta_es": float(delta.mean()),
                               "wins": int((delta > 0).sum()),
                               "ties": int((delta == 0).sum()),
                               "losses": int((delta < 0).sum()),
                               "same_prediction_rate": float((
                                   selected[("prediction", policy)] ==
                                   selected[("prediction", baseline)]).mean())})
    pair_frame = pd.DataFrame(paired)
    pair_frame.to_csv(RUN_DIR / "paired_deltas.csv", index=False)
    print("\nPaired differences (PPO minus baseline):")
    print(pair_frame.to_string(index=False))

prediction_path = RUN_DIR / "benchmark_predictions.jsonl"
coverage_path = RUN_DIR / "pool_identifier_coverage.jsonl"
records = []
with prediction_path.open("w", encoding="utf-8") as output, \
        coverage_path.open("w", encoding="utf-8") as coverage_output:
    for name, relative_path, language in BENCHMARKS:
        examples = benchmark_examples(name, relative_path, language)
        print(f"\n{name}: {len(examples)} fixed tasks")
        rows = [build_row(example) for example in tqdm(examples, desc=f"AST {name}")]
        candidate_counts = [len(row["candidate_ids"]) for row in rows]
        if candidate_counts:
            print(f"{name}: pool size min/median/max = "
                  f"{min(candidate_counts)}/{np.median(candidate_counts):.0f}/"
                  f"{max(candidate_counts)}; full-64 = "
                  f"{sum(count == CANDIDATE_POOL_SIZE for count in candidate_counts)}/"
                  f"{len(candidate_counts)}")
        coverage = [pool_identifier_probe(example, row)
                    for example, row in zip(examples, rows)]
        for probe in coverage:
            coverage_output.write(json.dumps({"dataset": name, **probe},
                                             ensure_ascii=False) + "\n")
        coverage_output.flush()
        with_repo_signal = [probe for probe in coverage if probe["repo_identifier_hits"]]
        if with_repo_signal:
            selected_rate = sum(bool(probe["selected_file_identifier_hits"])
                                for probe in with_repo_signal) / len(with_repo_signal)
            pool_rate = sum(bool(probe["pool_identifier_hits"])
                            for probe in with_repo_signal) / len(with_repo_signal)
            print(f"{name}: target-identifier proxy (gold available in repo, "
                  f"n={len(with_repo_signal)}): selected-file={selected_rate:.1%}, "
                  f"pool-64={pool_rate:.1%}; details: {coverage_path}")
        else:
            print(f"{name}: no novel target identifiers found in crossfile sources; "
                  "coverage proxy unavailable")
        available = sum(any(cost <= CROSSFILE_TOKEN_BUDGET
                            for cost in row["candidate_costs"]) for row in rows)
        print(f"{name}: rows with usable candidates = {available}/{len(rows)}")
        if not available:
            raise RuntimeError(
                f"{name}: no usable AST candidates. Inspect crossfile_context and "
                "candidate costs; refusing to report no-retrieval-only baselines."
            )
        selections = {}
        for label in ("best", "latest"):
            activate_policy(label)
            selections[label] = [ppo_select(row) for row in tqdm(rows, desc=f"PPO {label}")]
        dataset_records = []
        for start in tqdm(range(0, len(rows), 8), desc=f"Generate {name}"):
            prompts, metadata = [], []
            for offset, row in enumerate(rows[start:start + 8], start):
                eligible = sum(cost <= CROSSFILE_TOKEN_BUDGET
                               for cost in row["candidate_costs"])
                for method, chosen in (("no_retrieval", []),
                                       ("bm25_ast_budget", bm25_budget_selection(row)),
                                       ("ppo_best_ast", selections["best"][offset]),
                                       ("ppo_latest_ast", selections["latest"][offset])):
                    prompt = compose_prompt(row, chosen)
                    prompts.append(prompt)
                    metadata.append((row, method, chosen, eligible, prompt))
            predictions = generate_completions(prompts)
            for (row, method, chosen, eligible, prompt), prediction in zip(
                    metadata, predictions):
                score = rlcoder_score(prediction, row["target_code"],
                                      row["left_context"], row["language"])
                record = {"dataset": name, "task_id": row["task_id"],
                          "method": method, "prediction": prediction,
                          "reference": row["target_code"],
                          **score, "score_schema": SCORE_SCHEMA,
                          "related_files": row["related_files_count"],
                          "related_files_used": row["related_files_used"],
                          "ast_chunks": row["all_ast_chunks"],
                          "candidate_count": len(row["candidate_ids"]),
                          "fallback_candidates": sum(
                              kind == "fallback_line" for kind in row["candidate_types"]),
                          "eligible_candidates": eligible,
                          "selected_count": len(chosen), "selected_chunks": chosen,
                          "selected_paths": [row["candidate_paths"][i] for i in chosen],
                          "selected_costs": [row["candidate_costs"][i] for i in chosen],
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                dataset_records.append(record)
            output.flush()
        records.extend(dataset_records)
        summarize(records)

print("Saved evaluation outputs to", RUN_DIR)
