"""Precompute only cross-file AST chunks for the fixed Data4AlignCoder tests.

Original Parquet fields (including groundtruth labels) are copied byte-for-value;
the output adds AST chunks and parse-error paths in a new, parallel directory.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
from itertools import chain
import json
import os
from pathlib import Path
import sqlite3
import time
import zlib

import pyarrow as pa
import pyarrow.parquet as pq
import tree_sitter as ts
import tree_sitter_java as ts_java
import tree_sitter_python as ts_python
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SOURCE = Path(__file__).with_name("ast_ppo_unixcoder_kaggle.py")
DATASET_REPO = "AlignCoder/Data4AlignCoder"
DATASET_REVISION = "ce72fe3ef4987e15a9219fea4174bb80700e742c"
GENERATOR_TOKENIZER = "deepseek-ai/deepseek-coder-1.3b-base"
BENCHMARK_FILES = (
    ("data/cceval/python/test.parquet", "python"),
    ("data/cceval/java/test.parquet", "java"),
    ("data/repoeval/line_level/test_0.parquet", "python"),
    ("data/repoeval/line_level/test_1.parquet", "python"),
    ("data/repoeval/api_level/test_0.parquet", "python"),
    ("data/repoeval/api_level/test_1.parquet", "python"),
)
AST_SCHEMA = "rrpo_crossfile_ast_chunks_v1"
AST_CHUNK_TYPE = pa.list_(pa.struct([
    pa.field("path", pa.string()),
    pa.field("text", pa.string()),
    pa.field("type", pa.string()),
    pa.field("start", pa.int64()),
    pa.field("end", pa.int64()),
]))
ERROR_PATHS_TYPE = pa.list_(pa.string())
PIPELINE_ASSIGNMENTS = {
    "AST_CHUNK_TOKENS", "RETRIEVAL_LINE_NODE_TYPES",
    "RETRIEVAL_BLOCK_NODE_TYPES", "RETRIEVAL_NODE_TYPES",
    "SCOPE_NODE_TYPES", "PARSERS",
}
PIPELINE_FUNCTIONS = {"token_count", "ast_chunks"}


def source_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_chunker(tokenizer):
    """Load the exact AST chunker/constants from the RRPO training source."""
    tree = ast.parse(TRAIN_SOURCE.read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets
                     if isinstance(target, ast.Name)}
            if names & PIPELINE_ASSIGNMENTS:
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in PIPELINE_FUNCTIONS:
            nodes.append(node)
    found = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    if found != PIPELINE_FUNCTIONS:
        raise RuntimeError(f"Training source is missing chunker functions: "
                           f"{PIPELINE_FUNCTIONS - found}")
    namespace = {"ts": ts, "ts_java": ts_java, "ts_python": ts_python,
                 "GEN_TOKENIZER": tokenizer}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN_SOURCE), "exec"),
         namespace)
    return namespace


def related_file_entries(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Unexpected crossfile_context type: {type(value).__name__}")
    entries = []
    for item in value:
        if isinstance(item, dict):
            path, code = item.get("path", ""), item.get("text", "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            path, code = item[0], item[1]
        else:
            raise ValueError("crossfile_context entries must contain path and text")
        entries.append(("" if path is None else str(path),
                        "" if code is None else str(code)))
    return entries


def cached_chunks(connection, language, path, code, chunker, counters):
    source = code.encode("utf-8", errors="replace")
    digest = hashlib.sha256(language.encode() + b"\0" + source).hexdigest()
    cached = connection.execute(
        "SELECT chunks, had_error FROM ast_chunk_cache WHERE source_sha256=?",
        (digest,),
    ).fetchone()
    if cached is None:
        chunks, had_error = chunker["ast_chunks"]("", code, language)
        normalized_chunks = [{key: chunk[key] for key in
                             ("text", "type", "start", "end")}
                            for chunk in chunks]
        payload = zlib.compress(json.dumps(
            normalized_chunks, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8"), 1)
        connection.execute(
            "INSERT INTO ast_chunk_cache (source_sha256, chunks, had_error) "
            "VALUES (?, ?, ?)", (digest, payload, int(had_error)))
        counters["cache_misses"] += 1
        counters["unique_source_chunks"] += len(normalized_chunks)
        counters["parse_error_sources"] += int(had_error)
    else:
        normalized_chunks = json.loads(zlib.decompress(cached[0]))
        had_error = bool(cached[1])
        counters["cache_hits"] += 1
    counters["chunk_references"] += len(normalized_chunks)
    result = [{"path": path, **chunk} for chunk in normalized_chunks]
    return result, had_error


def prepare_parquet(source_path, output_path, language, chunker,
                    cache_connection, batch_size=16, progress_every=500):
    cache_connection.execute("CREATE TABLE IF NOT EXISTS ast_chunk_cache ("
                             "source_sha256 TEXT PRIMARY KEY, chunks BLOB NOT NULL, "
                             "had_error INTEGER NOT NULL)")
    source_file = pq.ParquetFile(source_path)
    source_schema = source_file.schema_arrow
    if "crossfile_context" not in source_schema.names:
        raise ValueError(f"{source_path} has no crossfile_context column")
    if "crossfile_ast_chunks" in source_schema.names:
        raise ValueError(f"Refusing to double-chunk an already prepared file: {source_path}")

    metadata = dict(source_schema.metadata or {})
    metadata.update({
        b"ast_chunk_schema": AST_SCHEMA.encode("ascii"),
        b"ast_chunk_tokens": str(chunker["AST_CHUNK_TOKENS"]).encode("ascii"),
        b"ast_chunk_tokenizer": GENERATOR_TOKENIZER.encode("utf-8"),
        b"ast_chunk_train_source_sha256": source_sha256(TRAIN_SOURCE).encode("ascii"),
        b"source_dataset_revision": DATASET_REVISION.encode("ascii"),
        b"ast_chunk_scope": b"crossfile_context_only; labels and all source fields retained",
    })
    output_schema = source_schema.append(
        pa.field("crossfile_ast_chunks", AST_CHUNK_TYPE, nullable=True)
    ).append(pa.field("crossfile_ast_parse_error_paths", ERROR_PATHS_TYPE,
                      nullable=True)).with_metadata(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".incomplete")
    counters = Counter()
    started = time.monotonic()

    with pq.ParquetWriter(temporary_path, output_schema, compression="zstd") as writer:
        for batch in source_file.iter_batches(batch_size=batch_size):
            rows = batch.to_pylist()
            row_chunks, row_error_paths = [], []
            for row in rows:
                chunks = []
                errors = []
                for path, code in related_file_entries(row["crossfile_context"]):
                    if not path or not code.strip():
                        continue
                    file_chunks, had_error = cached_chunks(
                        cache_connection, language, path, code, chunker, counters
                    )
                    chunks.extend(file_chunks)
                    if had_error:
                        errors.append(path)
                    counters["related_files_chunked"] += 1
                row_chunks.append(chunks)
                row_error_paths.append(errors)
                counters["rows"] += 1

            table = pa.Table.from_batches([batch])
            table = table.append_column(
                "crossfile_ast_chunks", pa.array(row_chunks, type=AST_CHUNK_TYPE)
            ).append_column(
                "crossfile_ast_parse_error_paths",
                pa.array(row_error_paths, type=ERROR_PATHS_TYPE),
            ).replace_schema_metadata(metadata)
            writer.write_table(table, row_group_size=table.num_rows)
            if counters["rows"] % progress_every < len(rows):
                cache_connection.commit()
                print(f"{source_path.name}: rows={counters['rows']} "
                      f"chunks={counters['chunk_references']} "
                      f"cache={counters['cache_hits']}/{counters['cache_misses']} "
                      f"elapsed_min={(time.monotonic() - started) / 60:.1f}",
                      flush=True)
    cache_connection.commit()

    os.replace(temporary_path, output_path)
    output_file = pq.ParquetFile(output_path)
    if output_file.metadata.num_rows != source_file.metadata.num_rows:
        raise RuntimeError(f"Row count changed in {output_path}")
    if output_file.schema_arrow.names != source_schema.names + [
            "crossfile_ast_chunks", "crossfile_ast_parse_error_paths"]:
        raise RuntimeError(f"Unexpected output columns in {output_path}")

    # Verify every original field batch-for-batch (including groundtruth labels).
    original_columns = source_schema.names
    source_batches = iter(source_file.iter_batches(batch_size=batch_size,
                                                  columns=original_columns))
    output_batches = iter(output_file.iter_batches(batch_size=batch_size,
                                                   columns=original_columns))
    verified_rows = 0
    while True:
        source_batch = next(source_batches, None)
        output_batch = next(output_batches, None)
        if source_batch is None or output_batch is None:
            if source_batch is not output_batch:
                raise RuntimeError(f"Original field rows differ in {output_path}")
            break
        source_table = pa.Table.from_batches([source_batch])
        output_table = pa.Table.from_batches([output_batch])
        if (source_batch.num_rows != output_batch.num_rows or
                not source_table.equals(output_table, check_metadata=False)):
            raise RuntimeError(
                f"An original field (including labels) changed in {output_path}, "
                f"starting at row {verified_rows}"
            )
        verified_rows += source_batch.num_rows
    if verified_rows != source_file.metadata.num_rows:
        raise RuntimeError(f"Not all original rows were verified in {output_path}")

    return {
        "source": str(source_path), "output": str(output_path),
        "rows": counters["rows"], "related_files_chunked": counters["related_files_chunked"],
        "chunk_references": counters["chunk_references"],
        "unique_source_chunks": counters["unique_source_chunks"],
        "parse_error_sources": counters["parse_error_sources"],
        "original_columns_preserved": original_columns,
        "original_values_verified": verified_rows,
        "source_sha256": source_sha256(source_path),
        "output_sha256": source_sha256(output_path),
        "output_bytes": output_path.stat().st_size,
    }


def merge_prepared_shards(shard_paths, merged_path, batch_size=16):
    """Concatenate prepared Parquet shards in order and verify every field."""
    shard_files = [pq.ParquetFile(path) for path in shard_paths]
    first_schema = shard_files[0].schema_arrow
    for path, parquet_file in zip(shard_paths[1:], shard_files[1:]):
        if not first_schema.remove_metadata().equals(
                parquet_file.schema_arrow.remove_metadata()):
            raise ValueError(f"Cannot merge Parquets with different schemas: {path}")

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = merged_path.with_name(merged_path.name + ".incomplete")
    temporary_path.unlink(missing_ok=True)
    with pq.ParquetWriter(temporary_path, first_schema, compression="zstd") as writer:
        for parquet_file in shard_files:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                writer.write_batch(batch)
    os.replace(temporary_path, merged_path)

    merged_file = pq.ParquetFile(merged_path)
    expected_rows = sum(parquet_file.metadata.num_rows for parquet_file in shard_files)
    if merged_file.metadata.num_rows != expected_rows:
        raise RuntimeError(f"Merged row count mismatch for {merged_path}")
    expected_batches = chain.from_iterable(
        parquet_file.iter_batches(batch_size=batch_size)
        for parquet_file in shard_files
    )
    merged_batches = iter(merged_file.iter_batches(batch_size=batch_size))
    verified_rows = 0
    while True:
        expected = next(expected_batches, None)
        actual = next(merged_batches, None)
        if expected is None or actual is None:
            if expected is not actual:
                raise RuntimeError(f"Merged batch count mismatch for {merged_path}")
            break
        expected_table = pa.Table.from_batches([expected])
        actual_table = pa.Table.from_batches([actual])
        if not expected_table.equals(actual_table, check_metadata=False):
            raise RuntimeError(
                f"Merged fields or row order changed in {merged_path}, "
                f"starting at row {verified_rows}"
            )
        verified_rows += expected.num_rows
    if verified_rows != expected_rows:
        raise RuntimeError(f"Not all merged rows were verified in {merged_path}")
    return {
        "output": str(merged_path), "rows": expected_rows,
        "original_values_verified": verified_rows,
        "original_columns_preserved": first_schema.names,
        "output_sha256": source_sha256(merged_path),
        "output_bytes": merged_path.stat().st_size,
    }


def download_sources(source_root):
    expected = [source_root / relative for relative, _ in BENCHMARK_FILES]
    missing = [path for path in expected if not path.is_file()]
    if not missing:
        return
    from huggingface_hub import snapshot_download
    print(f"Downloading {len(missing)} missing test Parquets from "
          f"{DATASET_REPO}@{DATASET_REVISION}.", flush=True)
    snapshot_download(
        repo_id=DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION,
        local_dir=str(source_root), allow_patterns=[relative for relative, _ in BENCHMARK_FILES],
    )
    still_missing = [path for path in expected if not path.is_file()]
    if still_missing:
        raise FileNotFoundError(f"Dataset download missed: {still_missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=PROJECT_ROOT / "local_server_results/data4aligncoder_raw")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "local_server_results/data4aligncoder_test_ast_chunks")
    parser.add_argument("--cache", type=Path, default=None,
                        help="SQLite chunk cache; defaults under output-root")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    download_sources(args.source_root)
    output_paths = [args.output_root / relative for relative, _ in BENCHMARK_FILES]
    output_paths.extend(args.output_root / relative for relative in (
        "data/repoeval/line_level/test.parquet",
        "data/repoeval/api_level/test.parquet",
    ))
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Prepared Parquets already exist; choose another --output-root to "
            f"preserve them: {existing}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    # Keep the reusable parser cache out of the upload-ready dataset directory.
    cache_path = args.cache or args.output_root.parent / (
        args.output_root.name + "_ast_chunk_cache.sqlite")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE IF NOT EXISTS ast_chunk_cache ("
                       "source_sha256 TEXT PRIMARY KEY, chunks BLOB NOT NULL, "
                       "had_error INTEGER NOT NULL)")
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_TOKENIZER)
    chunker = load_chunker(tokenizer)
    print("AST chunk tokens:", chunker["AST_CHUNK_TOKENS"],
          "tokenizer:", GENERATOR_TOKENIZER,
          "dataset revision:", DATASET_REVISION, flush=True)

    results = []
    for relative, language in BENCHMARK_FILES:
        source_path = args.source_root / relative
        output_path = args.output_root / relative
        if output_path.with_name(output_path.name + ".incomplete").exists():
            output_path.with_name(output_path.name + ".incomplete").unlink()
        print(f"Preparing {relative} ({language})", flush=True)
        result = prepare_parquet(source_path, output_path, language, chunker,
                                 connection, batch_size=args.batch_size)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    merge_specs = (
        ("line_level", ("data/repoeval/line_level/test_0.parquet",
                        "data/repoeval/line_level/test_1.parquet")),
        ("api_level", ("data/repoeval/api_level/test_0.parquet",
                        "data/repoeval/api_level/test_1.parquet")),
    )
    merged_results = []
    for level, shard_relatives in merge_specs:
        shard_outputs = [args.output_root / relative for relative in shard_relatives]
        merged_path = shard_outputs[0].parent / "test.parquet"
        merge_stats = merge_prepared_shards(shard_outputs, merged_path,
                                            batch_size=args.batch_size)
        shard_stats = [result for result in results
                       if Path(result["output"]) in shard_outputs]
        if len(shard_stats) != len(shard_outputs):
            raise RuntimeError(f"Missing preparation stats for RepoEval {level} shards")
        merged_results.append({
            "source": [result["source"] for result in shard_stats],
            "source_sha256s": [result["source_sha256"] for result in shard_stats],
            "source_shard_outputs": [result["output"] for result in shard_stats],
            "source_shard_output_sha256s": [result["output_sha256"]
                                             for result in shard_stats],
            "output": merge_stats["output"],
            "rows": merge_stats["rows"],
            "related_files_chunked": sum(result["related_files_chunked"]
                                          for result in shard_stats),
            "chunk_references": sum(result["chunk_references"]
                                     for result in shard_stats),
            "unique_source_chunks": sum(result["unique_source_chunks"]
                                         for result in shard_stats),
            "parse_error_sources": sum(result["parse_error_sources"]
                                        for result in shard_stats),
            "original_columns_preserved": merge_stats["original_columns_preserved"],
            "original_values_verified": merge_stats["original_values_verified"],
            "output_sha256": merge_stats["output_sha256"],
            "output_bytes": merge_stats["output_bytes"],
        })
        for shard_output in shard_outputs:
            shard_output.unlink()
        print(f"Merged RepoEval {level}: {merge_stats['rows']} rows -> "
              f"{merged_path}", flush=True)
    results = [result for result in results
               if Path(result["output"]) not in {
                   args.output_root / relative
                   for _, relatives in merge_specs for relative in relatives
               }]
    results.extend(merged_results)
    connection.close()

    manifest = {
        "schema": AST_SCHEMA, "dataset": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "training_chunker_source": str(TRAIN_SOURCE),
        "training_chunker_source_sha256": source_sha256(TRAIN_SOURCE),
        "generator_tokenizer": GENERATOR_TOKENIZER,
        "ast_chunk_tokens": chunker["AST_CHUNK_TOKENS"],
        "labels_and_all_original_fields": "copied unchanged and verified batch-for-batch",
        "files": results,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print("Prepared test dataset:", args.output_root.resolve(), flush=True)
    print("Manifest:", manifest_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
