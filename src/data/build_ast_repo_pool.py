"""Build the canonical mixed Python/Java AST repository pool.

This replaces the old fixed-five-tasks-per-repository artifact.  It stores one
row per quality-filtered repository; the epoch sampler chooses 2,000 repository
instances and one random eligible target per instance later.  There is no
repository-level validation split; benchmark test data is held out separately.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import zlib

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.ast_repo_pool import REPO_POOL_SCHEMA, build_repo_pool_record
from src.data.ast_training_data import DataConfig
from src.data.audit_completion_data import REVISION, TOKENIZER_REVISION
from src.data.repo_index import SOURCE_SHA256, digest_file, index_repositories

RET_REVISION = "5604afdc964f6c53782a6813140ade5216b99006"


def pack(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(), 1)


def unpack(blob):
    return json.loads(zlib.decompress(blob))


def init_worker():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    global GENERATOR, RETRIEVER
    GENERATOR = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base", revision=TOKENIZER_REVISION,
        local_files_only=True)
    RETRIEVER = AutoTokenizer.from_pretrained(
        "microsoft/unixcoder-base", revision=RET_REVISION, local_files_only=True)
    GENERATOR.model_max_length = RETRIEVER.model_max_length = 10**9


def worker(repo, config):
    return build_repo_pool_record(repo, GENERATOR, RETRIEVER, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True,
                        help="Benchmark exact-file exclusions")
    parser.add_argument("--output", type=Path, default=Path("train.parquet"),
                        help="Final prepared training pool; defaults to train.parquet")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-repos", type=int, help="Bounded smoke test")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repo-timeout", type=int, default=900)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite existing repository pool")
    if args.workers < 1 or (args.max_repos is not None and args.max_repos < 1):
        raise ValueError("Positive worker/repository counts required")

    config = DataConfig()
    exclusions = json.loads(args.exclusions.read_text())
    if exclusions["source_revision"] != REVISION:
        raise ValueError("Benchmark exclusions use a different source revision")
    repos, index_stats = index_repositories(
        args.root, set(exclusions["file_hashes"]), config)
    if args.max_repos:
        by_language = {language: [repo for repo in repos if repo["language"] == language]
                       for language in ("python", "java")}
        interleaved = []
        for index in range(max(map(len, by_language.values()))):
            for language in ("python", "java"):
                if index < len(by_language[language]):
                    interleaved.append(by_language[language][index])
        repos = interleaved[:args.max_repos]

    contract = {
        "schema": REPO_POOL_SCHEMA,
        "config": asdict(config),
        "source_revision": REVISION,
        "source_sha256": SOURCE_SHA256,
        "generator_tokenizer_revision": TOKENIZER_REVISION,
        "retriever_tokenizer_revision": RET_REVISION,
        "exclusion_sha256": digest_file(args.exclusions),
        "code_sha256": {
            name: digest_file(Path(__file__).with_name(name))
            for name in ("ast_training_data.py", "ast_repo_pool.py",
                         "code_input_cleanup.py", "repo_index.py",
                         "build_ast_repo_pool.py", "left_context.py",
                         "cross_file_budget.py")
        },
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.work_dir / "contract.json"
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError("Work directory contains another contract; use a new work directory")
    manifest.write_text(json.dumps(contract, indent=2) + "\n")

    def shard(repo):
        return args.work_dir / f"{repo['language']}-{repo['repo_id']}-{repo['uid'][:16]}.json.z"

    pending = [repo for repo in repos if not shard(repo).exists()]
    started = time.monotonic()
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker,
                                       maxtasksperchild=50) as pool:
        active, completed = [], 0
        iterator = iter(pending)
        while active or completed < len(pending):
            while len(active) < args.workers:
                repo = next(iterator, None)
                if repo is None:
                    break
                active.append((repo, pool.apply_async(worker, (repo, config)), time.monotonic()))
            progressed = False
            for repo, future, submitted in list(active):
                if not future.ready():
                    if time.monotonic() - submitted > args.repo_timeout:
                        raise TimeoutError(f"Repository timed out: {repo['language']}:{repo['repo_id']}")
                    continue
                result = future.get()
                destination = shard(repo)
                temporary = destination.with_suffix(".incomplete")
                temporary.write_bytes(pack(result))
                os.replace(temporary, destination)
                active.remove((repo, future, submitted))
                completed += 1
                progressed = True
                if completed % 20 == 0 or completed == len(pending):
                    print(f"Prepared {completed}/{len(pending)} pending repos "
                          f"in {time.monotonic()-started:.1f}s", flush=True)
            if not progressed:
                time.sleep(.05)

    schema = pa.schema(
        [("split", pa.string()), ("language", pa.string()), ("repo_id", pa.int64()),
         ("repo_uid", pa.string()), ("payload", pa.binary())],
        metadata={b"artifact_schema": REPO_POOL_SCHEMA.encode(),
                  b"preparation_contract": json.dumps(contract).encode()})
    counts = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".parquet.incomplete")
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for repo in repos:
            result = unpack(shard(repo).read_bytes())
            counts[f"repos:{repo['language']}:{result['status']}"] += 1
            if result["status"] != "kept":
                continue
            payload = result["payload"]
            counts[f"kept:{payload['split']}:{payload['language']}"] += 1
            counts["chunks"] += payload["stats"]["chunk_count"]
            counts["targets"] += payload["stats"]["target_count"]
            row = {"split": payload["split"], "language": payload["language"],
                   "repo_id": payload["repo_id"], "repo_uid": payload["repo_uid"],
                   "payload": pack(payload)}
            writer.write_table(pa.Table.from_pylist([row], schema=schema))
    if not counts["kept:train:python"] and not counts["kept:train:java"]:
        raise ValueError("No usable training repositories; refusing empty pool")
    os.replace(temporary, args.output)
    report = {"contract": contract, "index_counts": dict(index_stats),
              "counts": dict(counts), "smoke_test": bool(args.max_repos),
              "output_sha256": digest_file(args.output),
              "seconds": time.monotonic() - started}
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
