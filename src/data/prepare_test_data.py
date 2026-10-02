"""Prepare CCEval/RepoEval retrieval inputs without changing benchmark labels.

The original benchmark columns are copied byte-for-byte through Parquet.  The
additional ``ast_payload`` column contains only model-input views: cleaned
left context, AST chunks from the supplied cross-file context, the ranked
retrieval pool, and the budgeted prompt view.  ``groundtruth`` and
``right_context`` are never read while building candidates.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import time
import zlib

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.ast_training_data import (DataConfig, SourceFile, encoder_text,
                                        key, mine_candidates)
from src.data.audit_completion_data import BENCHMARKS, TOKENIZER_REVISION, source_hash
from src.data.build_ast_repo_pool import RET_REVISION
from src.data.code_input_cleanup import clean_chunk
from src.data.cross_file_budget import select_cross_file_context
from src.data.left_context import pack_left_context

SCHEMA = "ast_completion_test_inputs_v1"
VALID_SCHEMA = "ast_completion_validation_v1"


def pack(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8"), 1)


def unpack(blob):
    return json.loads(zlib.decompress(blob))


def sampled_positions(root, source_paths, sample_size, seed):
    """Select deterministic random positions from merged benchmark shards."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    total = sum(pq.ParquetFile(Path(root) / source_path).metadata.num_rows
                for source_path in source_paths)
    if sample_size > total:
        raise ValueError(f"Cannot sample {sample_size} rows from {total}")
    return set(random.Random(seed).sample(range(total), sample_size)), total


class ChunkCache:
    """Disk-backed deduplication for repeated RepoEval cross-file sources."""

    def __init__(self, path, generator, retriever, config):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks (cache_key TEXT PRIMARY KEY, payload BLOB NOT NULL)")
        self.db.commit()
        self.generator = generator
        self.retriever = retriever
        self.config = config
        self.pending_writes = 0

    def close(self):
        self.db.commit()
        self.db.close()

    def get(self, path, code, language):
        digest = source_hash(code)
        cache_key = hashlib.sha256(
            f"{language}\n{key(path)}\n{digest}".encode("utf-8")).hexdigest()
        row = self.db.execute("SELECT payload FROM chunks WHERE cache_key = ?",
                              (cache_key,)).fetchone()
        if row is not None:
            value = unpack(row[0])
            return value["chunks"], value["stats"]

        source = SourceFile(path, code, language, self.generator,
                            self.retriever, self.config)
        chunks = []
        stats = Counter(parse_error=int(source.tree.root_node.has_error))
        for chunk in source.chunks():
            cleaned = clean_chunk(chunk, source.code, language)
            model_text = cleaned["text"]
            if len(self.generator.encode(model_text, add_special_tokens=False)) > self.config.chunk_tokens:
                model_text = chunk["text"]
                cleaned = {"text": model_text,
                           "diagnostics": {"raw_fallback_token_cap": 1},
                           "version": cleaned.get("version")}
            enriched = {
                **chunk,
                "model_text": model_text,
                "cleanup_diagnostics": cleaned["diagnostics"],
                "retrieval_text": encoder_text(
                    self.retriever, self.config.retriever_tokens,
                    "\n".join(h["text"] for h in chunk["scope_headers"]),
                    model_text),
            }
            chunks.append(enriched)
        stats.update(chunk_count=len(chunks))
        value = {"chunks": chunks, "stats": dict(stats)}
        self.db.execute("INSERT OR REPLACE INTO chunks(cache_key, payload) VALUES (?, ?)",
                        (cache_key, pack(value)))
        self.pending_writes += 1
        if self.pending_writes >= 32:
            self.db.commit()
            self.pending_writes = 0
        return chunks, dict(stats)


def prepare_row(row, language, benchmark, cache, generator, retriever, config):
    """Build only input-side fields; original row values are returned untouched."""
    path = key(row["path"])
    left = row.get("left_context") or ""
    packed_left = pack_left_context(
        left, len(left.encode("utf-8")), language, generator,
        max_tokens=config.max_left_context_tokens,
        tail_tokens=config.left_tail_tokens,
        hint_tokens=config.left_hint_tokens)

    all_chunks, source_stats = [], Counter()
    for item in row.get("crossfile_context") or []:
        item_path = item.get("path")
        code = item.get("text") or ""
        if not item_path or not code:
            continue
        chunks, stats = cache.get(item_path, code, language)
        # Preserve the benchmark's path-level exclusion contract even if the
        # source provider repeats the target path in crossfile_context.
        all_chunks.extend(chunk for chunk in chunks if key(chunk["path"]) != path)
        source_stats["cross_file_sources"] += 1
        source_stats.update(stats)

    candidates = mine_candidates(packed_left["text"], path, all_chunks,
                                 retriever, config)
    cross_file = select_cross_file_context(
        candidates, generator, language,
        budget_tokens=config.cross_file_budget_tokens,
        max_snippets=config.max_cross_file_snippets)
    payload = {
        "schema": SCHEMA,
        "benchmark": benchmark,
        "language": language,
        "target_path": path,
        "model_left_context": packed_left["text"],
        "left_context_metadata": packed_left,
        "candidate_pool": candidates,
        "cross_file_context": cross_file["items"],
        "cross_file_metadata": {key: value for key, value in cross_file.items()
                                if key != "items"},
        "stats": {
            **dict(source_stats),
            "candidate_count": len(candidates),
            "candidate_file_count": len({c["path"] for c in candidates}),
        },
        "label_policy": "benchmark labels are preserved in original columns; no label was used",
    }
    return dict(row), payload


def prepare_benchmark(root, output, benchmark, language, source_paths,
                      cache, generator, retriever, config, sample_size=100,
                      seed=123):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing output: {output}")
    first = Path(root) / source_paths[0]
    source_schema = pq.ParquetFile(first).schema_arrow
    output_metadata = dict(source_schema.metadata or {})
    output_metadata.update({
        b"artifact_schema": SCHEMA.encode(),
        b"benchmark": benchmark.encode(),
        b"language": language.encode(),
        b"config": json.dumps(config.__dict__, sort_keys=True).encode(),
        b"source_files": json.dumps(source_paths).encode(),
        b"validation_sample_size": str(sample_size).encode(),
        b"validation_sample_seed": str(seed).encode(),
    })
    output_schema = source_schema.append(pa.field("ast_payload", pa.binary()))
    output_schema = output_schema.with_metadata(output_metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.incomplete")
    selected_positions, source_rows = sampled_positions(
        root, source_paths, sample_size, seed)
    count = 0
    position = 0
    started = time.monotonic()
    with pq.ParquetWriter(temporary, output_schema, compression="zstd") as writer:
        for source_path in source_paths:
            path = Path(root) / source_path
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
                rows = batch.to_pylist()
                output_rows = []
                for row in rows:
                    if position not in selected_positions:
                        position += 1
                        continue
                    original, payload = prepare_row(
                        row, language, benchmark, cache, generator, retriever, config)
                    original["ast_payload"] = pack(payload)
                    output_rows.append(original)
                    count += 1
                    position += 1
                    if count % 25 == 0:
                        print(f"{benchmark}: prepared {count} rows", flush=True)
                if output_rows:
                    writer.write_table(pa.Table.from_pylist(output_rows,
                                                           schema=output_schema))
    if count != sample_size:
        raise RuntimeError(f"Prepared {count} rows, expected {sample_size}")
    temporary.replace(output)
    report = {"schema": SCHEMA, "benchmark": benchmark, "rows": count,
              "source_files": source_paths, "output": str(output),
              "source_rows": source_rows, "sample_size": sample_size,
              "sample_seed": seed,
              "seconds": time.monotonic() - started}
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def merge_validation_files(output_dir, reports, sample_size, seed):
    """Merge the four prepared benchmark files into one validation Parquet."""
    output = Path(output_dir) / "valid.parquet"
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing output: {output}")
    first = Path(reports[0]["output"])
    base_schema = pq.ParquetFile(first).schema_arrow
    metadata = dict(base_schema.metadata or {})
    metadata.update({
        b"artifact_schema": VALID_SCHEMA.encode(),
        b"sample_size_per_benchmark": str(sample_size).encode(),
        b"sample_seed": str(seed).encode(),
        b"source_benchmarks": json.dumps([report["benchmark"] for report in reports]).encode(),
    })
    output_schema = base_schema.append(pa.field("benchmark", pa.string()))
    output_schema = output_schema.append(pa.field("language", pa.string()))
    output_schema = output_schema.with_metadata(metadata)
    temporary = output.with_suffix(".parquet.incomplete")
    count = 0
    with pq.ParquetWriter(temporary, output_schema, compression="zstd") as writer:
        for report in reports:
            benchmark = report["benchmark"]
            language = "java" if benchmark == "cceval_java" else "python"
            for batch in pq.ParquetFile(report["output"]).iter_batches(batch_size=32):
                rows = batch.to_pylist()
                for row in rows:
                    row["benchmark"] = benchmark
                    row["language"] = language
                writer.write_table(pa.Table.from_pylist(rows, schema=output_schema))
                count += len(rows)
    expected = sample_size * len(reports)
    if count != expected:
        raise RuntimeError(f"Merged {count} validation rows, expected {expected}")
    temporary.replace(output)
    report = {"schema": VALID_SCHEMA, "output": str(output), "rows": count,
              "sample_size_per_benchmark": sample_size, "seed": seed,
              "benchmarks": [item["benchmark"] for item in reports]}
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="datasets/data4aligncoder")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--samples-per-benchmark", type=int, default=100,
                        help="Fixed random validation rows per benchmark (default: 100)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    sample_size = args.limit if args.limit is not None else args.samples_per_benchmark
    if sample_size < 1:
        raise ValueError("samples-per-benchmark must be positive")

    config = DataConfig()
    from transformers import AutoTokenizer
    generator = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base", revision=TOKENIZER_REVISION,
        local_files_only=True)
    retriever = AutoTokenizer.from_pretrained(
        "microsoft/unixcoder-base", revision=RET_REVISION, local_files_only=True)
    generator.model_max_length = retriever.model_max_length = 10**9
    chunk_cache = ChunkCache(args.cache, generator, retriever, config)
    try:
        reports = {}
        for benchmark_index, (benchmark, source_paths) in enumerate(BENCHMARKS.items()):
            language = "java" if benchmark == "cceval_java" else "python"
            reports[benchmark] = prepare_benchmark(
                args.root, args.output_dir / f"{benchmark}.parquet", benchmark,
                language, source_paths, chunk_cache, generator, retriever,
                config, sample_size, args.seed + benchmark_index * 1_000_003)
        benchmark_reports = list(reports.values())
        reports["valid"] = merge_validation_files(
            args.output_dir, benchmark_reports, sample_size, args.seed)
        manifest = {"schema": SCHEMA, "validation_schema": VALID_SCHEMA,
                    "sample_size_per_benchmark": sample_size,
                    "total_validation_rows": sample_size * len(BENCHMARKS),
                    "seed": args.seed, "benchmarks": reports}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    finally:
        chunk_cache.close()


if __name__ == "__main__":
    main()
