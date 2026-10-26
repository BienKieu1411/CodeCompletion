#!/usr/bin/env python3
"""Paired RLCoder retrieval/generation experiment: raw prefix vs dependency graph."""

from __future__ import annotations

import gzip
import hashlib
import json
import keyword
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RLCODER_DIR = ROOT / "RLCoder"
OUTPUT_DIR = ROOT / "outputs"
RLR_RETRIEVER = "nov3630/RLRetriever"
RLR_REVISION = "ec587f5d8635462fff3cfb06f9a946148acda08b"
GENERATOR = "deepseek-ai/deepseek-coder-1.3b-base"
GENERATOR_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
BM25_TOPK = 100
RETRIEVER_TOPK = 10
QUERY_LENGTH = 256
CANDIDATE_LENGTH = 512
CROSSFILE_TOKENS = 1536
GENERATOR_CONTEXT = 2048
GENERATOR_MAX_NEW = 64
RETRIEVER_BATCH = 64
SEED = 20261026
ARMS = ("raw_left_context", "dependency_graph")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {message}", flush=True)


def ensure_dependencies() -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        version("vllm")
    except PackageNotFoundError:
        log("Installing vLLM 0.32.0 with the Colab CUDA/PyTorch backend resolver")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv_path = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
        subprocess.check_call([uv_path, "pip", "install", "--system", "vllm==0.32.0",
                               "--torch-backend=auto"])

    try:
        import torchaudio  # noqa: F401
    except ImportError:
        pass
    except RuntimeError as exc:
        if "different CUDA versions" not in str(exc):
            raise
        log("Removing incompatible optional torchaudio wheel")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"])

    missing: list[str] = []
    for module, package in (("torch", "torch"), ("transformers", "transformers"),
                            ("rank_bm25", "rank-bm25"), ("fuzzywuzzy", "fuzzywuzzy"),
                            ("editdistance", "editdistance"), ("tree_sitter", "tree-sitter==0.20.4")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    try:
        if version("tree-sitter") != "0.20.4":
            missing.append("tree-sitter==0.20.4")
    except PackageNotFoundError:
        missing.append("tree-sitter==0.20.4")
    if missing:
        log(f"Installing small evaluation dependencies: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *sorted(set(missing))])


class CodeBlock:
    """Faithful minimal port of RLCoder/datasets.py::CodeBlock."""

    __slots__ = ("file_path", "description", "code_content", "language", "_type")

    def __init__(self, file_path: str, description: str, code_content: str,
                 language: str, kind: str):
        self.file_path = file_path
        self.description = description
        self.code_content = code_content
        self.language = language
        self._type = kind

    def __str__(self) -> str:
        comment_label = "#" if self.language == "python" else "//"
        context = "\n".join(
            f"{comment_label} {line}" for line in self.description.strip().split("\n") if line
        ) + "\n"
        context += "\n".join(
            f"{comment_label} {line}" for line in self.code_content.split("\n") if line
        )
        return context.strip()


def rl_coder_default_chunks(source_block: CodeBlock) -> list[CodeBlock]:
    """Port of RLCoder/bm25.py::split_into_smaller_blocks(enable_fixed_block=False)."""
    smaller_blocks: list[CodeBlock] = []
    mini_blocks: list[list[str]] = []
    current_block: list[str] = []
    for line in source_block.code_content.splitlines():
        if line.strip() == "":
            if current_block:
                mini_blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line)
    if current_block:
        mini_blocks.append(current_block)

    split_mini: list[list[str]] = []
    for mini_block in mini_blocks:
        if len(mini_block) > 15:
            split_mini.extend(mini_block[index:index + 15]
                              for index in range(0, len(mini_block), 15))
        else:
            split_mini.append(mini_block)

    merged: list[list[str]] = []
    current: list[str] = []
    for block in split_mini:
        if current and len(current) + len(block) > 15:
            merged.append(current)
            current = []
        current.extend(block)
    if current:
        merged.append(current)

    total_lines = 0
    for block in merged:
        if total_lines >= 5000:
            break
        if len(block) <= 15:
            start = total_lines + 1
            total_lines += len(block)
            end = total_lines
            smaller_blocks.append(CodeBlock(
                source_block.file_path,
                f"file path: {source_block.file_path}\nlines: {start}-{end}",
                "\n".join(block), source_block.language, "mini_block"))
    return smaller_blocks


def rlcoder_tokenize(text: str, tokenizer: Any, max_length: int, is_query: bool) -> list[int]:
    """Exact token slicing and special-token pattern from RLCoder/retriever.py."""
    import_tokens: list[str] = []
    tokens = tokenizer.tokenize(text)
    if is_query:
        tokens = tokens[-(max_length - len(import_tokens)) + 4:]
    else:
        tokens = tokens[:(max_length - len(import_tokens)) - 4]
    tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + tokens + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(tokens)[:max_length]
    return ids + [tokenizer.pad_token_id] * (max_length - len(ids))


def read_gzip_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def graph_serialization(task: dict[str, Any], tokenizer: Any) -> tuple[str, str, dict[str, int]]:
    """Serialize prefix-visible dependencies and fit the query's actual token window.

    Edges sharing import/relation/file are grouped so wildcard imports do not
    repeat long paths. The complete graph is retained in the trace; the input
    graph is greedily reduced to RLCoder's 252-content-token query capacity.
    """
    raw_edges = task.get("dependency_edges") or []
    full = json.dumps({"type": "prefix_dependency_graph", "target": task["path"],
                       "edges": raw_edges}, ensure_ascii=False, separators=(",", ":"))
    grouped: dict[str, set[str]] = defaultdict(set)
    for edge in raw_edges:
        path = str(edge.get("path", ""))
        symbol = str(edge.get("symbol", ""))
        if path and symbol:
            grouped[path].add(symbol)
    rows = [[path, sorted(symbols)] for path, symbols in sorted(grouped.items())]

    prefix = task.get("left_context", "")
    def recency(row: list[Any]) -> tuple[int, str]:
        mentions = [prefix.rfind(symbol) for symbol in row[1]]
        mentions.append(prefix.rfind(row[0].rsplit("/", 1)[-1].split(".", 1)[0]))
        return (max(mentions), row[0])

    ordered = sorted(rows, key=recency, reverse=True)
    included: list[list[Any]] = []
    def render(current: list[list[Any]]) -> str:
        lines = [f"TARGET={task['path']}"]
        lines.extend(f"DEPENDS_ON={path} SYMBOLS={','.join(symbols)}" for path, symbols in current)
        return "\n".join(lines)

    token_capacity = QUERY_LENGTH - 4
    for row in ordered:
        trial = included + [row]
        if len(tokenizer.tokenize(render(trial))) <= token_capacity:
            included = trial
            continue
        # A file with many symbols can still contribute a bounded, prefix-relevant subset.
        if row[1]:
            partial = []
            prioritized_symbols = sorted(row[1],
                                         key=lambda symbol: (prefix.rfind(symbol), symbol),
                                         reverse=True)
            for symbol in prioritized_symbols:
                candidate = [row[0], partial + [symbol]]
                if len(tokenizer.tokenize(render(included + [candidate]))) <= token_capacity:
                    partial.append(symbol)
            if partial:
                included.append([row[0], partial])
    text = render(included)
    stats = {"full_edge_groups": len(rows), "input_edge_groups": len(included),
             "full_resolved_edges": sum(len(row[1]) for row in rows),
             "input_resolved_edges": sum(len(row[1]) for row in included),
             "input_query_tokens": len(tokenizer.tokenize(text)),
             "token_capacity": token_capacity}
    return text, full, stats


def last_nonempty_lines(text: str, count: int = 20) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-count:])


def bm25_top_candidates(task: dict[str, Any], files: dict[str, dict[str, str]],
                        pool_rows: dict[str, dict[str, Any]], index_cache: dict[str, Any],
                        query: str) -> tuple[list[CodeBlock], list[float], int]:
    from rank_bm25 import BM25Okapi

    pool_id = task["pool_id"]
    if pool_id not in index_cache:
        source_blocks: list[CodeBlock] = []
        language = task["language"]
        for file_id in pool_rows[pool_id]["file_ids"]:
            item = files[file_id]
            source = item["text"]
            original = CodeBlock(item["path"],
                                 f"file path: {item['path']}\nlines: 0-{len(source.splitlines())}",
                                 source, language, "")
            source_blocks.extend(rl_coder_default_chunks(original))
        bm25 = BM25Okapi([block.code_content.lower().split() for block in source_blocks]) if source_blocks else None
        index_cache[pool_id] = (source_blocks, bm25)

    blocks, bm25 = index_cache[pool_id]
    if bm25 is None:
        return [], [], len(blocks)
    # RLCoder keeps query token case as-is and sorts with Python's stable reverse sort.
    scores = bm25.get_scores(query.split())
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    mini_indices = [idx for idx in order if blocks[idx]._type == "mini_block"][:BM25_TOPK]
    return [blocks[idx] for idx in mini_indices], [float(scores[idx]) for idx in mini_indices], len(blocks)


def embed_batches(texts: list[str], *, tokenizer: Any, model: Any, torch: Any,
                  max_length: int, is_query: bool, batch_size: int = RETRIEVER_BATCH) -> list[Any]:
    vectors: list[Any] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            part = texts[start:start + batch_size]
            source_ids = torch.tensor(
                [rlcoder_tokenize(text, tokenizer, max_length, is_query) for text in part],
                dtype=torch.long, device="cuda")
            mask = source_ids.ne(tokenizer.pad_token_id)
            token_embeddings = model(source_ids, attention_mask=mask)[0]
            pooled = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(pooled.float().cpu().numpy())
            if start == 0 or (start // batch_size + 1) % 50 == 0:
                log(f"Retriever encoding {'queries' if is_query else 'candidate chunks'}: "
                    f"{min(start + len(part), len(texts))}/{len(texts)}")
    return vectors


def chunk_trace(block: CodeBlock, *, score: float | None, rank: int,
                bm25_score: float | None = None) -> dict[str, Any]:
    return {"rank": rank, "file_path": block.file_path, "description": block.description,
            "code_content": block.code_content, "rendered_text": str(block), "kind": block._type,
            "retriever_score": score, "bm25_score": bm25_score}


def edge_hits(block: CodeBlock, edges: list[dict[str, Any]]) -> set[tuple[str, str, str, str]]:
    hits = set()
    for edge in edges:
        if edge.get("path") != block.file_path:
            continue
        symbol = str(edge.get("symbol", ""))
        if edge.get("relation") == "imports_module" or (
                symbol and symbol != "*" and re.search(rf"\b{re.escape(symbol)}\b", block.code_content)):
            hits.add((str(edge.get("import", "")), str(edge.get("path", "")),
                      str(edge.get("relation", "")), symbol))
    return hits


def retrieval_metrics(top10: list[CodeBlock], effective: list[CodeBlock],
                      edges: list[dict[str, Any]]) -> dict[str, Any]:
    required = {(str(e.get("import", "")), str(e.get("path", "")),
                 str(e.get("relation", "")), str(e.get("symbol", ""))) for e in edges}
    top_hits = set().union(*(edge_hits(block, edges) for block in top10)) if top10 else set()
    effective_hits = set().union(*(edge_hits(block, edges) for block in effective)) if effective else set()
    needed = max(1, len(required))
    return {"resolved_edge_count": len(required),
            "dependency_edge_recall_at10": len(required & top_hits) / needed,
            "dependency_edge_recall_used": len(required & effective_hits) / needed,
            "dependency_file_recall_at10": len({e[1] for e in required} & {b.file_path for b in top10}) /
                max(1, len({e[1] for e in required})),
            "dependency_file_recall_used": len({e[1] for e in required} & {b.file_path for b in effective}) /
                max(1, len({e[1] for e in required}))}


def is_stop(block: CodeBlock) -> bool:
    return block.file_path == ""


def build_prompt(task: dict[str, Any], retrieved: list[CodeBlock], left_context: str,
                 tokenizer: Any) -> tuple[str, list[int], int, int]:
    # Same order, clipping, token budgets, and path header as RLCoder/generator.py.
    valid_blocks: list[CodeBlock] = []
    for block in retrieved:
        if block.file_path != "":
            valid_blocks.append(block)
        else:
            break
    crossfile_text = "\n\n".join(str(block) for block in valid_blocks)
    cross_ids = tokenizer.encode(crossfile_text[:CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]
    path_text = f"\n\n# file path: {task['path']}\n\n"
    path_ids = tokenizer.encode(path_text, add_special_tokens=False)
    allowed_left = GENERATOR_CONTEXT - (len(cross_ids) + len(path_ids) + 10)
    left_ids = tokenizer.encode(left_context, add_special_tokens=False)[-allowed_left:]
    prompt = tokenizer.decode(cross_ids + path_ids + left_ids)
    prompt_ids = tokenizer.encode(prompt)[-GENERATOR_CONTEXT:]
    return prompt, prompt_ids, len(prompt_ids), len(cross_ids)


def load_parsers() -> dict[str, Any]:
    from tree_sitter import Language, Parser

    parser_map = {}
    for language in ("python", "java"):
        grammar = Language(str(RLCODER_DIR / "utils" / "build" / f"{language}-lang-parser.so"), language)
        parser = Parser()
        parser.set_language(grammar)
        parser_map[language] = parser
    return parser_map


def has_syntax_error(node: Any) -> bool:
    if node.type == "ERROR":
        return True
    try:
        return any(has_syntax_error(child) for child in node.children)
    except RecursionError:
        return True


def is_parse_valid(parser: Any, code: str) -> bool:
    try:
        tree = parser.parse(code.encode("utf-8"))
        return tree is not None and not has_syntax_error(tree.root_node)
    except Exception:
        return False


def official_postprocess(original_prefix: str, completion: str, parser: Any, language: str) -> str:
    # RLCoder/utils/eval_utils.py policy: first bracket delimiter for Java;
    # first syntactically complete Python statement boundary for Python.
    try:
        if language in ("java", "csharp", "typescript"):
            end_idx = next((i for i, char in enumerate(completion) if char in (";", "}", "{")), None)
            return completion[:end_idx + 1] if end_idx else completion
        if language == "python":
            for index in range(len(completion)):
                if not is_parse_valid(parser, original_prefix + completion[:index + 1]):
                    continue
                if completion[index + 1] == "\n":
                    return completion[:index + 1].rstrip()
            return completion
    except Exception:
        return completion
    return completion


def remove_comments(code: str) -> str:
    # Keep RLCoder's evaluation behavior unchanged.
    return re.sub(r"//.*|#.*", "", code)


STRING_PATTERN = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''
IDENTIFIER_REGEX = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")


def extract_identifiers(code: str, language: str, java_keywords: set[str]) -> list[str]:
    code = re.sub(STRING_PATTERN, "", code)
    ignored = {word for word in keyword.kwlist if word not in ("True", "False")} if language == "python" else java_keywords
    return [token for token in re.findall(r"\w+", code)
            if IDENTIFIER_REGEX.match(token) and token not in ignored]


def score_prediction(task: dict[str, Any], completion: str, parser: Any,
                     java_keywords: set[str]) -> dict[str, Any]:
    from fuzzywuzzy import fuzz

    # Scoring always uses the original source prefix, never the graph prompt.
    prediction = remove_comments(official_postprocess(task["left_context"], completion,
                                                       parser, task["language"])).strip()
    target = remove_comments(task["groundtruth"]).strip()
    pred_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    em = int(pred_lines == target_lines)
    es = float(fuzz.ratio(target, prediction))
    pred_ids = set(extract_identifiers(prediction, task["language"], java_keywords))
    target_ids = set(extract_identifiers(target, task["language"], java_keywords))
    tp = len(pred_ids & target_ids)
    fp, fn = len(pred_ids - target_ids), len(target_ids - pred_ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    result = {"processed_prediction": prediction, "target_for_scoring": target,
              "em": em, "es": es, "id_em": float(pred_ids == target_ids),
              "id_precision": precision, "id_recall": recall, "id_f1": f1}
    if task["benchmark"].startswith("repoeval"):
        import editdistance
        pred_repo_lines = pred_lines[:len(target_lines)]
        pred_repo = "\n".join(pred_repo_lines)
        target_repo = "\n".join(target_lines)
        denom = max(len(pred_repo), len(target_repo))
        result["repo_em"] = int(pred_repo_lines == target_lines)
        result["repo_es"] = 100.0 * (1.0 - editdistance.eval(target_repo, pred_repo) / denom) if denom else 100.0
    else:
        result["repo_em"] = None
        result["repo_es"] = None
    return result


def bootstrap_delta(values: list[tuple[float, float]], seed: int = SEED,
                    repeats: int = 10000) -> dict[str, float]:
    import numpy as np
    delta = np.asarray([right - left for left, right in values], dtype=np.float64)
    if len(delta) == 0:
        return {"mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0,
                "p_delta_gt_zero": 0.0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(repeats, len(delta)))
    boot = delta[indices].mean(axis=1)
    return {"mean_delta": float(delta.mean()),
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "p_delta_gt_zero": float((boot > 0).mean())}


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def run() -> None:
    ensure_dependencies()
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("Colab GPU is unavailable")
    if not (DATA_DIR / "tasks.jsonl.gz").is_file():
        raise FileNotFoundError(f"Missing prepared task bundle under {DATA_DIR}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = read_gzip_jsonl(DATA_DIR / "tasks.jsonl.gz")
    pool_rows = {row["pool_id"]: row for row in read_gzip_jsonl(DATA_DIR / "pools.jsonl.gz")}
    files = {row["file_id"]: {"path": row["path"], "text": row["text"]}
             for row in read_gzip_jsonl(DATA_DIR / "files.jsonl.gz")}
    selection = json.loads((DATA_DIR / "selection.json").read_text(encoding="utf-8"))
    log(f"Dataset loaded: {len(tasks)} paired tasks; benchmarks="
        f"{dict((b, sum(t['benchmark']==b for t in tasks)) for b in sorted({t['benchmark'] for t in tasks}))}")

    try:
        import pynvml
        pynvml.nvmlInit()
        device = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
    except Exception:
        device = torch.cuda.get_device_name(0)

    log(f"Loading pinned RLRetriever tokenizer {RLR_RETRIEVER}@{RLR_REVISION}")
    retriever_tokenizer = AutoTokenizer.from_pretrained(RLR_RETRIEVER, revision=RLR_REVISION)

    # Build the exact RLCoder blank-line / <=15-nonempty-line mini-block pool.
    index_cache: dict[str, Any] = {}
    arms_by_task: dict[tuple[str, str], dict[str, Any]] = {}
    query_records: list[str] = []
    query_keys: list[tuple[str, str, str]] = []
    candidate_text_to_idx: dict[str, int] = {}
    candidate_texts: list[str] = []
    log("Preparing paired BM25 top-100 candidate sets using RLCoder default chunking")
    for task in tasks:
        graph, graph_full, graph_stats = graph_serialization(task, retriever_tokenizer)
        for arm in ARMS:
            query = last_nonempty_lines(task["left_context"], 20) if arm == "raw_left_context" else graph
            blocks, bm25_scores, full_chunk_count = bm25_top_candidates(
                task, files, pool_rows, index_cache, query)
            stop = CodeBlock("", "Don't need cross file context for completion", "",
                             task["language"], "stop")
            ranked_pool = blocks + [stop]
            ranked_pool_scores = bm25_scores + [None]
            record = {"task": task, "arm": arm, "graph": graph, "graph_full": graph_full,
                      "graph_stats": graph_stats, "query": query,
                      "bm25_blocks": blocks, "bm25_scores": bm25_scores,
                      "ranked_pool": ranked_pool, "ranked_pool_scores": ranked_pool_scores,
                      "full_chunk_count": full_chunk_count}
            arms_by_task.setdefault((task["benchmark"], task["task_id"]), {})[arm] = record
            query_records.append(query)
            query_keys.append((task["benchmark"], task["task_id"], arm))
            for block in ranked_pool:
                rendered = str(block)
                if rendered not in candidate_text_to_idx:
                    candidate_text_to_idx[rendered] = len(candidate_texts)
                    candidate_texts.append(rendered)
    log(f"Prepared {len(index_cache)} repository BM25 indexes; unique chunks="
        f"{sum(len(index_cache[key][0]) for key in index_cache):,}; "
        f"unique encoded candidate texts={len(candidate_texts):,}")

    log(f"Loading pinned RLRetriever weights {RLR_RETRIEVER}@{RLR_REVISION}")
    retriever_model = AutoModel.from_pretrained(RLR_RETRIEVER, revision=RLR_REVISION).cuda().eval()
    query_vectors = embed_batches(query_records, tokenizer=retriever_tokenizer, model=retriever_model,
                                  torch=torch, max_length=QUERY_LENGTH, is_query=True)
    candidate_vectors = embed_batches(candidate_texts, tokenizer=retriever_tokenizer,
                                      model=retriever_model, torch=torch,
                                      max_length=CANDIDATE_LENGTH, is_query=False)
    query_vector_by_key = {key: query_vectors[i] for i, key in enumerate(query_keys)}
    del query_vectors, query_records, retriever_model, retriever_tokenizer
    torch.cuda.empty_cache()
    log("RLRetriever embeddings complete; scoring all BM25 top-100 candidates plus stop token")

    retrieval_trace: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    prompt_token_ids: list[list[int]] = []
    prompt_keys: list[tuple[str, str, str]] = []
    generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR, revision=GENERATOR_REVISION)
    generator_tokenizer.model_max_length = 1_000_000_000
    if generator_tokenizer.pad_token_id is None:
        generator_tokenizer.pad_token_id = generator_tokenizer.eos_token_id

    for task in tasks:
        key_base = (task["benchmark"], task["task_id"])
        for arm in ARMS:
            record = arms_by_task[key_base][arm]
            query_key = (task["benchmark"], task["task_id"], arm)
            qvec = query_vector_by_key[query_key]
            blocks = record["ranked_pool"]
            doc_ids = [candidate_text_to_idx[str(block)] for block in blocks]
            matrix = np.asarray([candidate_vectors[doc_id] for doc_id in doc_ids], dtype=np.float32)
            scores = np.dot(matrix, np.asarray(qvec, dtype=np.float32))
            top_indices = np.argsort(scores)[-RETRIEVER_TOPK:][::-1].tolist()
            retrieved = [blocks[idx] for idx in top_indices]
            retrieved_scores: list[float | None] = [float(scores[idx]) for idx in top_indices]
            while len(retrieved) < RETRIEVER_TOPK:
                retrieved.append(CodeBlock("", "Don't need cross file context for completion", "",
                                           task["language"], "stop"))
                retrieved_scores.append(None)
            effective: list[CodeBlock] = []
            for block in retrieved:
                if is_stop(block):
                    break
                effective.append(block)
            graph = record["graph"]
            generator_left = task["left_context"] if arm == "raw_left_context" else graph
            prompt_text, ids, prompt_len, cross_len = build_prompt(
                task, retrieved, generator_left, generator_tokenizer)
            record["retrieved"] = retrieved
            record["retrieved_scores"] = retrieved_scores
            record["effective"] = effective
            record["prompt_text"] = prompt_text
            record["prompt_ids"] = ids
            record["generator_left"] = generator_left
            metric = retrieval_metrics(retrieved, effective, task.get("dependency_edges") or [])
            retrieval_rows.append({"benchmark": task["benchmark"], "task_id": task["task_id"],
                                   "arm": arm, "candidate_chunk_count": record["full_chunk_count"],
                                   "bm25_returned": len(record["bm25_blocks"]),
                                   "rlretriever_top10": len(retrieved),
                                   "generator_snippet_count": len(effective), **metric})
            trace = {"benchmark": task["benchmark"], "task_id": task["task_id"], "arm": arm,
                     "query": record["query"], "generator_left_context": generator_left,
                     "dependency_graph_input": graph, "dependency_graph_full": record["graph_full"],
                     "dependency_graph_stats": record["graph_stats"],
                     "resolved_dependency_edges": task.get("dependency_edges", []),
                     "full_chunk_count": record["full_chunk_count"],
                     "bm25_top100": [chunk_trace(block, score=None, rank=rank,
                                                  bm25_score=record["bm25_scores"][rank - 1])
                                     for rank, block in enumerate(record["bm25_blocks"], 1)],
                     "rlretriever_ranked_candidates": [chunk_trace(block, score=retrieved_scores[idx], rank=idx + 1)
                                                        for idx, block in enumerate(retrieved)],
                     "generator_effective_snippets": [chunk_trace(block, score=retrieved_scores[idx], rank=idx + 1)
                                                       for idx, block in enumerate(retrieved[:len(effective)])],
                     "retrieval_metrics": metric}
            retrieval_trace.append(trace)
            prompt_rows.append({"benchmark": task["benchmark"], "task_id": task["task_id"],
                                "arm": arm, "prompt": prompt_text,
                                "prompt_token_ids": ids, "prompt_tokens": prompt_len,
                                "crossfile_tokens": cross_len,
                                "generator_left_context": generator_left})
            prompt_token_ids.append(ids)
            prompt_keys.append(query_key)

    write_jsonl_gz(OUTPUT_DIR / "retrieval_trace.jsonl.gz", retrieval_trace)
    write_jsonl_gz(OUTPUT_DIR / "prompts.jsonl.gz", prompt_rows)
    write_jsonl_gz(OUTPUT_DIR / "retrieval_metrics.jsonl.gz", retrieval_rows)
    log(f"Retrieval outputs saved. Mean snippets used: "
        f"{np.mean([r['generator_snippet_count'] for r in retrieval_rows]):.2f}; "
        f"dependency recall@10: "
        f"{np.mean([r['dependency_edge_recall_at10'] for r in retrieval_rows]):.3f}")

    # Let vLLM use the A100 after releasing all retriever memory.
    del candidate_vectors, query_vector_by_key, arms_by_task, index_cache, files, pool_rows
    torch.cuda.empty_cache()
    log(f"Loading greedy vLLM generator {GENERATOR}@{GENERATOR_REVISION} on {device}")
    engine = LLM(model=GENERATOR, tokenizer=GENERATOR, revision=GENERATOR_REVISION,
                 tokenizer_revision=GENERATOR_REVISION, dtype="float16",
                 max_model_len=GENERATOR_CONTEXT + GENERATOR_MAX_NEW,
                 max_num_seqs=64, max_num_batched_tokens=8192,
                 gpu_memory_utilization=0.90, seed=SEED, disable_log_stats=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=GENERATOR_MAX_NEW, seed=SEED)
    llm_prompts = [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids]
    log(f"Generating {len(llm_prompts)} completions (two paired arms per task)")
    started = time.time()
    generated = engine.generate(llm_prompts, sampling, use_tqdm=True)
    raw_by_key = {key: outputs.outputs[0].text for key, outputs in zip(prompt_keys, generated)}
    log(f"Generation complete in {time.time() - started:.1f}s")
    del engine
    torch.cuda.empty_cache()

    parser_map = load_parsers()
    java_keywords = {line.strip() for line in
                     (RLCODER_DIR / "utils" / "keywords" / "java.txt").read_text(encoding="utf-8").splitlines()
                     if line.strip()}
    task_map = {(task["benchmark"], task["task_id"]): task for task in tasks}
    prediction_rows: list[dict[str, Any]] = []
    by_benchmark: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for key, raw in raw_by_key.items():
        task = task_map[(key[0], key[1])]
        arm = key[2]
        scored = score_prediction(task, raw, parser_map[task["language"]], java_keywords)
        row = {"benchmark": task["benchmark"], "task_id": task["task_id"],
               "repo_id": task["task_id"].split("/")[0], "arm": arm,
               "raw_prediction": raw, **scored}
        prediction_rows.append(row)
        by_benchmark[task["benchmark"]][arm].append(row)
    write_jsonl_gz(OUTPUT_DIR / "predictions.jsonl.gz", prediction_rows)

    summaries: dict[str, Any] = {}
    for benchmark in sorted(by_benchmark):
        per_arm = by_benchmark[benchmark]
        left_rows = {row["task_id"]: row for row in per_arm[ARMS[0]]}
        graph_rows = {row["task_id"]: row for row in per_arm[ARMS[1]]}
        paired = [(left_rows[task_id], graph_rows[task_id]) for task_id in sorted(left_rows.keys() & graph_rows.keys())]
        result: dict[str, Any] = {"n": len(paired)}
        for metric in ("em", "es", "id_em", "id_precision", "id_recall", "id_f1"):
            result[f"{ARMS[0]}_{metric}"] = float(np.mean([left[metric] for left, _ in paired]))
            result[f"{ARMS[1]}_{metric}"] = float(np.mean([right[metric] for _, right in paired]))
        es_delta = bootstrap_delta([(left["es"], right["es"]) for left, right in paired],
                                   seed=SEED + len(summaries))
        result["paired_es_delta_graph_minus_raw"] = es_delta
        if benchmark.startswith("repoeval"):
            for arm in ARMS:
                repo_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in per_arm[arm]:
                    repo_rows[row["repo_id"]].append(row)
                result[f"{arm}_repo_macro_em"] = float(np.mean([
                    np.mean([item["repo_em"] for item in group]) for group in repo_rows.values()]))
                result[f"{arm}_repo_macro_es"] = float(np.mean([
                    np.mean([item["repo_es"] for item in group]) for group in repo_rows.values()]))
            # The dependency-filtered RepoEval subset contains only two repos;
            # report paired task CI for fuzzy ES and raw repo-macro deltas descriptively.
            result["paired_repo_macro_es_delta_graph_minus_raw"] = (
                result[f"{ARMS[1]}_repo_macro_es"] - result[f"{ARMS[0]}_repo_macro_es"])
            result["repo_cluster_count"] = len({row["repo_id"] for row in per_arm[ARMS[0]]})
        summaries[benchmark] = result

    retrieval_summary: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in retrieval_rows if row["arm"] == arm]
        retrieval_summary[arm] = {
            "mean_candidate_chunk_count": float(np.mean([r["candidate_chunk_count"] for r in rows])),
            "mean_bm25_returned": float(np.mean([r["bm25_returned"] for r in rows])),
            "mean_generator_snippet_count": float(np.mean([r["generator_snippet_count"] for r in rows])),
            "mean_dependency_edge_recall_at10": float(np.mean([r["dependency_edge_recall_at10"] for r in rows])),
            "mean_dependency_edge_recall_used": float(np.mean([r["dependency_edge_recall_used"] for r in rows])),
            "mean_dependency_file_recall_at10": float(np.mean([r["dependency_file_recall_at10"] for r in rows])),
        }
    retrieval_by_benchmark: dict[str, Any] = {}
    for benchmark in sorted({row["benchmark"] for row in retrieval_rows}):
        retrieval_by_benchmark[benchmark] = {}
        for arm in ARMS:
            rows = [row for row in retrieval_rows
                    if row["benchmark"] == benchmark and row["arm"] == arm]
            retrieval_by_benchmark[benchmark][arm] = {
                "mean_candidate_chunk_count": float(np.mean([r["candidate_chunk_count"] for r in rows])),
                "mean_bm25_returned": float(np.mean([r["bm25_returned"] for r in rows])),
                "mean_generator_snippet_count": float(np.mean([r["generator_snippet_count"] for r in rows])),
                "mean_dependency_edge_recall_at10": float(np.mean([r["dependency_edge_recall_at10"] for r in rows])),
                "mean_dependency_edge_recall_used": float(np.mean([r["dependency_edge_recall_used"] for r in rows])),
                "mean_dependency_file_recall_at10": float(np.mean([r["dependency_file_recall_at10"] for r in rows])),
            }
        raw_rows = [row for row in retrieval_rows
                    if row["benchmark"] == benchmark and row["arm"] == ARMS[0]]
        graph_rows = [row for row in retrieval_rows
                      if row["benchmark"] == benchmark and row["arm"] == ARMS[1]]
        raw_by_id = {row["task_id"]: row for row in raw_rows}
        graph_by_id = {row["task_id"]: row for row in graph_rows}
        paired_rows = [(raw_by_id[task_id], graph_by_id[task_id])
                       for task_id in sorted(raw_by_id.keys() & graph_by_id.keys())]
        retrieval_by_benchmark[benchmark]["paired_edge_recall_at10_delta_graph_minus_raw"] = bootstrap_delta(
            [(left["dependency_edge_recall_at10"], right["dependency_edge_recall_at10"])
             for left, right in paired_rows], seed=SEED + 100 + len(retrieval_by_benchmark))
    config = {
        "experiment": "RLCoder RLRetriever raw-left-context vs dependency-graph left-context",
        "sample_count": len(tasks), "selection": selection,
        "arms": {"raw_left_context": "RLCoder default; last 20 non-empty left-context lines as query; raw left context to generator",
                 "dependency_graph": "serialized prefix-visible dependency graph as BM25+RLRetriever query and as generator left context; no raw prefix, query enhancement, rerank bonus, or per-snippet notes"},
        "retriever": RLR_RETRIEVER, "retriever_revision": RLR_REVISION,
        "bm25_topk": BM25_TOPK, "rlretriever_topk": RETRIEVER_TOPK,
        "rlcoder_chunking": "RLCoder default mini-block: blank-line groups, split/merge to <=15 non-empty lines; no fixed block",
        "rlcoder_query_context_lines": 20, "retriever_query_tokens": QUERY_LENGTH,
        "graph_query_token_capacity": QUERY_LENGTH - 4,
        "retriever_candidate_tokens": CANDIDATE_LENGTH,
        "generator": GENERATOR, "generator_revision": GENERATOR_REVISION,
        "generator_prompt": "RLCoder generator.py construction order, budgets, and path header; graph arm replaces only left_context; retrieved blocks remain unannotated",
        "generator_crossfile_tokens": CROSSFILE_TOKENS, "generator_context_tokens": GENERATOR_CONTEXT,
        "generator_max_new_tokens": GENERATOR_MAX_NEW, "decoding": "greedy; temperature=0",
        "scoring": "RLCoder eval_metric.py postprocess, comment removal, fuzzywuzzy ES, EM, ID metrics; RepoEval official repo-macro edit ES",
        "python_postprocess_prefix": "original benchmark left_context for scoring only; never supplied to graph-arm generator",
        "device": device, "torch": torch.__version__, "seed": SEED,
        "retrieval_summary": retrieval_summary,
        "retrieval_by_benchmark": retrieval_by_benchmark, "benchmarks": summaries,
        "limits": ["256 high-dependency tasks selected from the same four benchmark subsets; not a full-benchmark estimate",
                   "RepoEval selection covers two repository clusters; repo-level confidence is limited"],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    with (OUTPUT_DIR / "metrics_by_benchmark.csv").open("w", encoding="utf-8") as handle:
        columns = ["benchmark", "n", "raw_left_context_es", "dependency_graph_es",
                   "es_delta_graph_minus_raw", "es_ci95_low", "es_ci95_high",
                   "raw_left_context_em", "dependency_graph_em",
                   "raw_left_context_id_f1", "dependency_graph_id_f1",
                   "raw_repo_macro_es", "graph_repo_macro_es"]
        handle.write(",".join(columns) + "\n")
        for benchmark, result in summaries.items():
            delta = result["paired_es_delta_graph_minus_raw"]
            values = [benchmark, result["n"], result[f"{ARMS[0]}_es"], result[f"{ARMS[1]}_es"],
                      delta["mean_delta"], delta["ci95_low"], delta["ci95_high"],
                      result[f"{ARMS[0]}_em"], result[f"{ARMS[1]}_em"],
                      result[f"{ARMS[0]}_id_f1"], result[f"{ARMS[1]}_id_f1"],
                      result.get(f"{ARMS[0]}_repo_macro_es", ""),
                      result.get(f"{ARMS[1]}_repo_macro_es", "")]
            handle.write(",".join(str(value) for value in values) + "\n")

    result_archive = ROOT / "rlcoder_dependency_graph_results.tar.gz"
    with tarfile.open(result_archive, "w:gz") as archive:
        for path in sorted(OUTPUT_DIR.iterdir()):
            archive.add(path, arcname=f"outputs/{path.name}")
    sha = hashlib.sha256(result_archive.read_bytes()).hexdigest()
    (ROOT / "rlcoder_dependency_graph_results.sha256").write_text(
        f"{sha}  {result_archive.name}\n", encoding="utf-8")
    log("EXPERIMENT COMPLETE")
    log(json.dumps({"summary": str(OUTPUT_DIR / 'summary.json'),
                    "archive": str(result_archive), "sha256": sha,
                    "benchmarks": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        run()
    except Exception:
        log("EXPERIMENT FAILED\n" + traceback.format_exc())
        raise
