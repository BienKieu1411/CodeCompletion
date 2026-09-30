"""Build one self-contained Python+Java RRPO train Parquet from the local AST cache.

Each quality-filtered repository contributes one fixed AST-boundary completion
example. The row includes the gold target, exact left context, BM25-mined AST
candidate pool, and UniXcoder token IDs; Kaggle reads only this file.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import time
import zlib

from rank_bm25 import BM25Okapi
import pyarrow as pa
import pyarrow.parquet as pq
import tree_sitter as ts
import tree_sitter_java as ts_java
import tree_sitter_python as ts_python
from transformers import AutoTokenizer

from src.prepare_ast_rrpo_data import (EXPECTED_SOURCE_SHA256,
                                       NOTEBOOK_SOURCE, PROJECT_ROOT,
                                       SOURCE_REVISION, pipeline_functions,
                                       verify_artifact)


ARTIFACT_SCHEMA = "rrpo_full_repo_one_example_prebuilt_v1"
QUALITY_RULE = {"min_files": 3, "min_target_files": 2,
                "min_retrieval_files": 2, "min_ast_chunks": 16,
                "min_budget_eligible_candidates": 2}
ROW_FUNCTIONS = {"prepared_ast_lookup", "lexical_tokens", "left_anchors",
                 "suffix_with_token_budget", "pack_left_context",
                 "retrieval_query", "render_chunk", "unixcoder_ids",
                 "build_row", "choose_target_kind"}
ROW_SCALARS = {"SEED", "TARGET_CUT_DISTRIBUTION", "PREFERRED_FILE_LINES",
               "PREFERRED_FILE_CHARS", "MAX_RELATED_FILES",
               "CANDIDATE_POOL_SIZE", "RETRIEVER_QUERY_LENGTH",
               "RETRIEVER_CANDIDATE_LENGTH", "CROSSFILE_TOKEN_BUDGET"}


def repository_groups(path):
    group = []
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=128, columns=["path", "content", "first"]):
        for row in batch.to_pylist():
            if row["first"] and group:
                yield group
                group = []
            group.append((row["path"], row["content"]))
    if group:
        yield group


def cached_file_info(connection, language, source):
    raw_digest = hashlib.sha256(source.encode("utf-8", errors="replace")).hexdigest()
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    norm_digest = hashlib.sha256(
        normalized.encode("utf-8", errors="replace")
    ).hexdigest()
    raw = connection.execute(
        "SELECT chunks FROM ast_entries WHERE language=? AND source_sha256=?",
        (language, raw_digest),
    ).fetchone()
    norm = connection.execute(
        "SELECT spans FROM ast_entries WHERE language=? AND source_sha256=?",
        (language, norm_digest),
    ).fetchone()
    if raw is None or raw[0] is None or norm is None or norm[0] is None:
        raise RuntimeError("Missing offline AST cache entry")
    chunks = json.loads(zlib.decompress(raw[0]))
    spans = json.loads(zlib.decompress(norm[0]))
    return normalized, spans, len(chunks)


def quality_decision(connection, language, group):
    if len(group) < QUALITY_RULE["min_files"]:
        return "too_few_files"
    target_files = retrieval_files = chunks_total = 0
    for _path, code in group:
        _normalized, spans, chunks = cached_file_info(connection, language, code)
        target_files += bool(spans["line"] or spans["block"])
        retrieval_files += chunks > 0
        chunks_total += chunks
    if target_files < QUALITY_RULE["min_target_files"]:
        return "too_few_target_files"
    if retrieval_files < QUALITY_RULE["min_retrieval_files"]:
        return "too_few_retrieval_files"
    if chunks_total < QUALITY_RULE["min_ast_chunks"]:
        return "too_few_ast_chunks"
    return None


def row_functions(generator_tokenizer, retriever_tokenizer, connection):
    env = pipeline_functions(generator_tokenizer)
    env.update({"PREPARED_AST": connection, "RET_TOKENIZER": retriever_tokenizer,
                "BM25Okapi": BM25Okapi, "IDENTIFIER_RE": re.compile(
                    r"[A-Za-z_$][A-Za-z0-9_$]*"),
                "hashlib": hashlib, "json": json, "zlib": zlib,
                "os": os, "re": re, "random": random})
    tree = ast.parse(NOTEBOOK_SOURCE.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ROW_SCALARS:
                env[target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.FunctionDef) and node.name in ROW_FUNCTIONS:
            selected.append(node)
    found = {node.name for node in selected}
    if found != ROW_FUNCTIONS:
        raise RuntimeError(f"Missing row functions: {ROW_FUNCTIONS - found}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(NOTEBOOK_SOURCE),
                 "exec"), env)
    return env


def example_for_repo(connection, env, language, repo_id, group):
    seed = env["SEED"]
    rng = random.Random(f"{seed}:{language}:repo:{repo_id}")
    candidates = []
    for index, (_path, code) in enumerate(group):
        normalized, spans, _chunks = cached_file_info(connection, language, code)
        if spans["line"] or spans["block"]:
            preferred = (normalized.count("\n") + 1 >= env["PREFERRED_FILE_LINES"]
                         and len(normalized) >= env["PREFERRED_FILE_CHARS"])
            candidates.append((index, normalized, spans, preferred))
    preferred = [item for item in candidates if item[3]]
    others = [item for item in candidates if not item[3]]
    rng.shuffle(preferred)
    rng.shuffle(others)
    for target_index, normalized, spans, _preferred in preferred + others:
        desired = env["choose_target_kind"](rng)
        first = ([(kind, *span) for kind in ("line", "block")
                  for span in spans[kind]] if desired == "mixed" else
                 [(desired, *span) for span in spans[desired]])
        fallback = ([] if desired == "mixed" else
                    [(kind, *span) for kind in ("line", "block")
                     if kind != desired for span in spans[kind]])
        rng.shuffle(first)
        rng.shuffle(fallback)
        for kind, start, end, node_type in (first + fallback)[:4]:
            source_bytes = normalized.encode("utf-8", errors="replace")
            prefix = source_bytes[:start].decode("utf-8", errors="replace")
            target = source_bytes[start:end].decode("utf-8", errors="replace")
            if prefix + target + source_bytes[end:].decode("utf-8", errors="replace") != normalized:
                continue
            example = {"task_id": f"{language}-repo-{repo_id}",
                       "language": language, "file_path": group[target_index][0],
                       "left_context": prefix, "target_code": target,
                       "related_files": [item for i, item in enumerate(group)
                                         if i != target_index],
                       "target_kind": kind, "requested_kind": desired,
                       "target_node_type": node_type}
            row = env["build_row"](example)
            if sum(cost <= env["CROSSFILE_TOKEN_BUDGET"]
                   for cost in row["candidate_costs"]) >= QUALITY_RULE[
                       "min_budget_eligible_candidates"]:
                return row
    return None


def build(source_root, output_path):
    verification = verify_artifact(source_root)
    connection = sqlite3.connect(
        f"{(source_root / 'ast_prepared_v1.sqlite').as_uri()}?mode=ro", uri=True)
    generator = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    retriever = AutoTokenizer.from_pretrained("microsoft/unixcoder-base", use_fast=False)
    env = row_functions(generator, retriever, connection)
    paths = {language: source_root / "data" / "github_repos" / language / "train.parquet"
             for language in ("python", "java")}
    eligible = {}
    quality_counts = {}
    for language, path in paths.items():
        kept = []
        quality_counts[language] = Counter()
        for repo_id, group in enumerate(repository_groups(path)):
            reason = quality_decision(connection, language, group)
            quality_counts[language][reason or "kept"] += 1
            if reason is None:
                kept.append(repo_id)
        eligible[language] = kept
        print("Quality", language, dict(quality_counts[language]), flush=True)
    split_ids = {}
    for language, ids in eligible.items():
        shuffled = ids[:]
        random.Random(f"{env['SEED']}:{language}:split").shuffle(shuffled)
        n_valid = max(1, int(0.15 * len(shuffled)))
        validation_ids = set(shuffled[:n_valid])
        split_ids[language] = {repo_id: ("valid" if repo_id in validation_ids
                                         else "train") for repo_id in ids}
    metadata = {
        b"artifact_schema": ARTIFACT_SCHEMA.encode(),
        b"source_revision": SOURCE_REVISION.encode(),
        b"source_sha256": json.dumps(EXPECTED_SOURCE_SHA256, sort_keys=True).encode(),
        b"prepared_dataset_sha256": verification["dataset_sha256"].encode(),
        b"quality_rule": json.dumps(QUALITY_RULE, sort_keys=True).encode(),
        b"generator_tokenizer": b"deepseek-ai/deepseek-coder-1.3b-base",
        b"retriever_tokenizer": b"microsoft/unixcoder-base",
    }
    schema = pa.schema([("split", pa.string()), ("language", pa.string()),
                        ("repo_id", pa.int64()), ("task_id", pa.string()),
                        ("payload", pa.binary())], metadata=metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".incomplete")
    stats = Counter()
    started = time.monotonic()
    with pq.ParquetWriter(temporary_path, schema, compression="zstd") as writer:
        for language, path in paths.items():
            for repo_id, group in enumerate(repository_groups(path)):
                split = split_ids[language].get(repo_id)
                if split is None:
                    continue
                row = example_for_repo(connection, env, language, repo_id, group)
                if row is None:
                    stats[(language, "pool_rejected")] += 1
                    continue
                payload = zlib.compress(json.dumps(row, ensure_ascii=False,
                                                   separators=(",", ":")).encode(), 1)
                writer.write_table(pa.Table.from_pylist([
                    {"split": split, "language": language, "repo_id": repo_id,
                     "task_id": row["task_id"], "payload": payload}
                ], schema=schema), row_group_size=1)
                stats[(language, split)] += 1
                if sum(stats.values()) % 100 == 0:
                    print("Built", sum(stats.values()), "repos;",
                          round((time.monotonic() - started) / 60, 1), "min", flush=True)
    connection.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_path, output_path)
    with output_path.open("rb") as handle:
        output_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(output_path), "bytes": output_path.stat().st_size,
            "sha256": output_sha256,
            "quality": {language: dict(counts) for language, counts in quality_counts.items()},
            "rows": {f"{language}_{split}": count for (language, split), count in stats.items()}}


def verify_train_file(path):
    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata or {}
    if metadata.get(b"artifact_schema") != ARTIFACT_SCHEMA.encode("ascii"):
        raise RuntimeError("Unexpected train artifact schema")
    counts = Counter()
    seen = set()
    for row_group_id in range(parquet.num_row_groups):
        if parquet.metadata.row_group(row_group_id).num_rows != 1:
            raise RuntimeError("Expected one training example per Parquet row group")
        record = parquet.read_row_group(row_group_id).to_pylist()[0]
        split, language, repo_id = (record[key] for key in
                                    ("split", "language", "repo_id"))
        if split not in {"train", "valid"} or language not in {"python", "java"}:
            raise RuntimeError("Unknown train split or language")
        key = (language, repo_id)
        if key in seen:
            raise RuntimeError("Repository appears more than once")
        seen.add(key)
        row = json.loads(zlib.decompress(record["payload"]))
        candidates = row["candidate_ids"]
        count = len(candidates)
        if (record["task_id"] != row["task_id"] or
                row["language"] != language or
                not row["target_code"].strip() or
                row["target_kind"] not in {"line", "block"} or
                len(row["query_ids"]) != 256 or
                count < QUALITY_RULE["min_budget_eligible_candidates"] or
                count > 64 or
                sum(cost <= 1536 for cost in row["candidate_costs"]) <
                QUALITY_RULE["min_budget_eligible_candidates"] or
                any(len(ids) != 512 for ids in candidates) or
                any(len(row[name]) != count for name in
                    ("candidate_texts", "candidate_costs", "candidate_paths",
                     "candidate_types")) or
                any(os.path.normpath(p) == os.path.normpath(row["file_path"])
                    for p in row["candidate_paths"])):
            raise RuntimeError(f"Invalid prepared training row: {record['task_id']}")
        counts[f"{language}_{split}"] += 1
    if not counts["python_train"] or not counts["java_train"] or not counts["python_valid"] or not counts["java_valid"]:
        raise RuntimeError("One language or split is empty")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size,
            "row_groups": parquet.num_row_groups, "rows": dict(counts),
            "quality_rule": json.loads(metadata[b"quality_rule"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=PROJECT_ROOT / "local_server_results/data4aligncoder_prepared")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "local_server_results/rrpo_train/train_rrpo_ast.parquet")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_train_file(args.output) if args.verify_only else build(
        args.source_root, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
