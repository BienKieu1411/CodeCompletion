"""Prepare every multi-file Python/Java train repository for the Kaggle RRPO notebook.

Run with bienkieu_env after downloading the two pinned Data4AlignCoder train
parquets. The notebook's actual AST functions are loaded without running its
GPU/top-level cells. Interrupted runs reuse completed SQLite AST entries.
"""

import argparse
import ast
import hashlib
import json
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
NOTEBOOK_SOURCE = Path(__file__).with_name("ast_ppo_unixcoder_kaggle.py")
SOURCE_REVISION = "ce72fe3ef4987e15a9219fea4174bb80700e742c"
AST_PREP_PIPELINE_SCHEMA = "phong_ast_boundary_semantic_chunks_prepared_all_train_v4"
EXPECTED_SOURCE_SHA256 = {
    "python": "c7ead86805619020ca881be05c15896bce6cc75df6496c54dc7c0014511bef61",
    "java": "1abc3546f19d83c2461a3683a6c06e1d2258a1d544700b6507b0c8b6d8bb59f7",
}
SCALAR_NAMES = {
    "GENERATOR_MODEL", "PREPARED_AST_SCHEMA", "DATA_PIPELINE_SCHEMA",
    "AST_CHUNK_TOKENS", "TARGET_CHUNK_TOKENS", "MIN_TARGET_TOKENS",
    "MIN_PREFIX_TOKENS", "MIN_LEFT_CONTEXT_LINES",
}
AST_NAMES = {"TARGET_LINE_TYPES", "TARGET_BLOCK_TYPES",
             "RETRIEVAL_NODE_TYPES", "SCOPE_NODE_TYPES"}
FUNCTION_NAMES = {"token_count", "ast_chunks", "target_spans", "prepared_ast_config"}


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pipeline_functions(tokenizer):
    source_tree = ast.parse(NOTEBOOK_SOURCE.read_text(encoding="utf-8"))
    namespace = {"PARSERS": {
        "python": ts.Parser(ts.Language(ts_python.language())),
        "java": ts.Parser(ts.Language(ts_java.language())),
    }, "GEN_TOKENIZER": tokenizer,
       "prepared_ast_lookup": lambda *_args: None,
       "AST_CACHE_REQUIRED": False}
    nodes = []
    for node in source_tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in SCALAR_NAMES:
                namespace[target.id] = ast.literal_eval(node.value)
            elif isinstance(target, ast.Name) and target.id in AST_NAMES:
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES:
            nodes.append(node)
    # The AST cache has its own stable schema; checkpoint/train-row revisions do
    # not change how spans or chunks were computed from the source parquet.
    namespace["DATA_PIPELINE_SCHEMA"] = AST_PREP_PIPELINE_SCHEMA
    functions = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    if functions != FUNCTION_NAMES:
        raise RuntimeError(f"Missing notebook AST functions: {FUNCTION_NAMES - functions}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK_SOURCE), "exec"),
         namespace)
    return namespace


def packed(value):
    return sqlite3.Binary(zlib.compress(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 1
    ))


def prepare_one(connection, language, code, pipeline, stats):
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    raw_digest = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()
    normalized_digest = hashlib.sha256(
        normalized.encode("utf-8", errors="replace")
    ).hexdigest()
    chunk_row = connection.execute(
        "SELECT chunks FROM ast_entries WHERE language=? AND source_sha256=?",
        (language, raw_digest),
    ).fetchone()
    if chunk_row is None or chunk_row[0] is None:
        chunks, had_error = pipeline["ast_chunks"]("", code, language)
        for chunk in chunks:
            chunk.pop("path")  # Path is assigned at retrieval time.
        connection.execute(
            "INSERT INTO ast_entries (language, source_sha256, chunks, had_error) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(language, source_sha256) DO UPDATE "
            "SET chunks=excluded.chunks, had_error=excluded.had_error",
            (language, raw_digest, packed(chunks), int(had_error)),
        )
        stats["new_chunk_sources"] += 1
        stats["chunks"] += len(chunks)
        stats["parse_error_files"] += int(had_error)
    span_row = connection.execute(
        "SELECT spans FROM ast_entries WHERE language=? AND source_sha256=?",
        (language, normalized_digest),
    ).fetchone()
    if span_row is None or span_row[0] is None:
        spans = pipeline["target_spans"](normalized, language)
        connection.execute(
            "INSERT INTO ast_entries (language, source_sha256, spans) "
            "VALUES (?, ?, ?) ON CONFLICT(language, source_sha256) DO UPDATE "
            "SET spans=excluded.spans",
            (language, normalized_digest, packed(spans)),
        )
        stats["new_span_sources"] += 1
        stats["line_spans"] += len(spans["line"])
        stats["block_spans"] += len(spans["block"])


def prepare_language(language, source_path, output_path, connection, pipeline):
    source = pq.ParquetFile(source_path)
    schema = pa.schema([("path", pa.string()), ("content", pa.string()),
                        ("first", pa.bool_())])
    stats = {"source_rows": source.metadata.num_rows, "eligible_repos": 0,
             "eligible_files": 0, "singletons_dropped": 0,
             "empty_repos_dropped": 0, "new_chunk_sources": 0,
             "new_span_sources": 0, "chunks": 0, "line_spans": 0,
             "block_spans": 0, "parse_error_files": 0}
    group = []
    output_rows = []
    seen_rows = 0
    started = time.monotonic()
    with pq.ParquetWriter(output_path, schema, compression="zstd") as writer:
        def write_buffer():
            if output_rows:
                writer.write_table(pa.Table.from_pylist(output_rows, schema=schema))
                output_rows.clear()

        def finish_group():
            if len(group) < 2:
                stats["singletons_dropped" if group else "empty_repos_dropped"] += 1
                group.clear()
                return
            stats["eligible_repos"] += 1
            for index, row in enumerate(group):
                prepare_one(connection, language, row["content"], pipeline, stats)
                output_rows.append({"path": row["path"], "content": row["content"],
                                    "first": index == 0})
                stats["eligible_files"] += 1
            group.clear()
            if len(output_rows) >= 2048:
                write_buffer()
                connection.commit()

        for batch in source.iter_batches(batch_size=128,
                                         columns=["path", "content", "first"]):
            for row in batch.to_pylist():
                seen_rows += 1
                if row["first"] and group:
                    finish_group()
                if row["path"] and row["content"]:
                    group.append(row)
            if seen_rows // 2000 > (seen_rows - len(batch)) // 2000:
                connection.commit()
                print(f"{language}: {seen_rows}/{source.metadata.num_rows} source rows, "
                      f"{stats['eligible_repos']} eligible repos, "
                      f"{stats['eligible_files']} eligible files, "
                      f"{(time.monotonic() - started) / 60:.1f} min", flush=True)
        finish_group()
        write_buffer()
        connection.commit()
    if seen_rows != source.metadata.num_rows:
        raise RuntimeError(f"{language} parquet row count changed while preparing")
    filtered = pq.ParquetFile(output_path)
    if filtered.metadata.num_rows != stats["eligible_files"]:
        raise RuntimeError(f"{language} filtered parquet file count is inconsistent")
    return stats


def verify_artifact(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    database = root / "ast_prepared_v1.sqlite"
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    info = dict(connection.execute("SELECT key, value FROM cache_info"))
    if info.get("complete") != "1":
        raise RuntimeError("AST cache is not marked complete")
    if json.loads(info["config_json"]) != manifest["config"]:
        raise RuntimeError("AST cache config differs from the manifest")
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("AST cache failed integrity_check")
    digest = hashlib.sha256()
    report = {}
    for language in ("python", "java"):
        path = root / "data" / "github_repos" / language / "train.parquet"
        digest.update(language.encode("ascii"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        parquet = pq.ParquetFile(path)
        count = 0
        repos = 0
        current_repo_files = 0
        checked_content = 0
        for batch in parquet.iter_batches(batch_size=128,
                                          columns=["path", "content", "first"]):
            for row in batch.to_pylist():
                if row["first"]:
                    if current_repo_files == 1:
                        raise RuntimeError(f"{language} filtered parquet contains a singleton")
                    current_repo_files = 0
                    repos += 1
                elif current_repo_files == 0:
                    raise RuntimeError(f"{language} filtered parquet has a missing repo boundary")
                current_repo_files += 1
                count += 1
                code = row["content"]
                normalized = code.replace("\r\n", "\n").replace("\r", "\n")
                raw_sha = hashlib.sha256(code.encode("utf-8", errors="replace")).hexdigest()
                norm_sha = hashlib.sha256(
                    normalized.encode("utf-8", errors="replace")
                ).hexdigest()
                chunks_row = connection.execute(
                    "SELECT chunks FROM ast_entries WHERE language=? AND source_sha256=?",
                    (language, raw_sha),
                ).fetchone()
                spans_row = connection.execute(
                    "SELECT spans FROM ast_entries WHERE language=? AND source_sha256=?",
                    (language, norm_sha),
                ).fetchone()
                if (chunks_row is None or chunks_row[0] is None or
                        spans_row is None or spans_row[0] is None):
                    raise RuntimeError(f"Missing {language} AST cache entry: {row['path']}")
                if count % 211 == 1:
                    raw_bytes = code.encode("utf-8", errors="replace")
                    chunks = json.loads(zlib.decompress(chunks_row[0]))
                    spans = json.loads(zlib.decompress(spans_row[0]))
                    for chunk in chunks:
                        start, end = chunk["start"], chunk["end"]
                        if not (0 <= start < end <= len(raw_bytes)):
                            raise RuntimeError("Invalid cached AST chunk byte range")
                        if raw_bytes[start:end].decode("utf-8", errors="replace") != chunk["text"]:
                            raise RuntimeError("Cached AST chunk text does not match source")
                    normalized_bytes = normalized.encode("utf-8", errors="replace")
                    for kind in ("line", "block"):
                        for start, end, _node_type in spans[kind]:
                            if not (0 <= start < end <= len(normalized_bytes)):
                                raise RuntimeError("Invalid cached target byte range")
                    checked_content += 1
        if current_repo_files == 1:
            raise RuntimeError(f"{language} filtered parquet ends with a singleton")
        expected = manifest["statistics"][language]
        if (count != expected["eligible_files"] or
                repos != expected["eligible_repos"] or
                count != parquet.metadata.num_rows):
            raise RuntimeError(f"{language} filtered parquet counts differ from manifest")
        report[language] = {"repositories": repos, "files": count,
                            "sampled_contents_checked": checked_content}
    if (digest.hexdigest() != manifest["dataset_sha256"] or
            digest.hexdigest() != info.get("dataset_sha256")):
        raise RuntimeError("Filtered parquet SHA-256 differs from AST cache/manifest")
    entries = connection.execute("SELECT COUNT(*) FROM ast_entries").fetchone()[0]
    if entries != manifest["ast_entries"]:
        raise RuntimeError("AST entry count differs from manifest")
    connection.close()
    return {"dataset_sha256": digest.hexdigest(), "ast_entries": entries,
            "languages": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=PROJECT_ROOT / "local_server_results/data4aligncoder_raw/data")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "local_server_results/data4aligncoder_prepared")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_artifact(args.output_root), indent=2))
        return
    source_paths = {language: args.source_root / "github_repos" / language / "train.parquet"
                    for language in ("python", "java")}
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    database_path = args.output_root / "ast_prepared_v1.sqlite"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE IF NOT EXISTS ast_entries ("
                       "language TEXT NOT NULL, source_sha256 TEXT NOT NULL, "
                       "spans BLOB, chunks BLOB, had_error INTEGER, "
                       "PRIMARY KEY (language, source_sha256))")
    connection.execute("CREATE TABLE IF NOT EXISTS cache_info ("
                       "key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    pipeline = pipeline_functions(tokenizer)
    config_json = json.dumps(pipeline["prepared_ast_config"](), sort_keys=True)
    source_hashes = {language: file_sha256(path)
                     for language, path in source_paths.items()}
    if source_hashes != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Train parquets differ from the pinned Data4AlignCoder revision")
    expected = {"source_revision": SOURCE_REVISION,
                "source_hashes_json": json.dumps(source_hashes, sort_keys=True),
                "config_json": config_json}
    existing = dict(connection.execute("SELECT key, value FROM cache_info"))
    for key, value in expected.items():
        if key in existing and existing[key] != value:
            raise RuntimeError(f"Existing AST cache {key} differs; choose a new output directory")
        connection.execute("INSERT OR REPLACE INTO cache_info VALUES (?, ?)", (key, value))
    connection.execute("INSERT OR REPLACE INTO cache_info VALUES ('complete', '0')")
    connection.commit()
    statistics = {}
    for language in ("python", "java"):
        destination = args.output_root / "data" / "github_repos" / language
        destination.mkdir(parents=True, exist_ok=True)
        statistics[language] = prepare_language(
            language, source_paths[language], destination / "train.parquet",
            connection, pipeline,
        )
        print(f"Finished {language}: {statistics[language]}", flush=True)
    digest = hashlib.sha256()
    for language in ("python", "java"):
        digest.update(language.encode("ascii"))
        path = args.output_root / "data" / "github_repos" / language / "train.parquet"
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    dataset_sha256 = digest.hexdigest()
    connection.execute("INSERT OR REPLACE INTO cache_info VALUES (?, ?)",
                       ("dataset_sha256", dataset_sha256))
    connection.execute("INSERT OR REPLACE INTO cache_info VALUES ('complete', '1')")
    connection.commit()
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("SQLite AST cache failed integrity_check")
    entry_count = connection.execute("SELECT COUNT(*) FROM ast_entries").fetchone()[0]
    connection.close()
    manifest = {"source_revision": SOURCE_REVISION, "source_sha256": source_hashes,
                "dataset_sha256": dataset_sha256, "config": json.loads(config_json),
                "statistics": statistics, "ast_entries": entry_count}
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print("Prepared artifact:", args.output_root, flush=True)
    print("Dataset SHA-256:", dataset_sha256, "AST entries:", entry_count, flush=True)


if __name__ == "__main__":
    main()
