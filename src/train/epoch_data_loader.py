"""Materialize one stochastic training epoch from an AST repository pool."""

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import zlib

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.ast_repo_pool import TASK_SCHEMA, sample_repo_epoch
from src.data.ast_training_data import DataConfig
from src.data.audit_completion_data import TOKENIZER_REVISION
from src.data.build_ast_repo_pool import RET_REVISION
from src.data.repo_index import digest_file


def unpack(blob):
    return json.loads(zlib.decompress(blob))


def pack(value):
    return zlib.compress(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(), 1)


TRAIN_REPOSITORIES_PER_EPOCH = 2_000
PYTHON_REPOSITORIES_PER_EPOCH = 800


def _cycled_sample(indices, count, rng):
    """Sample ``count`` instances, exhausting each repository once per cycle."""
    if count <= 0 or not indices:
        return []
    pool = list(indices)
    selected = []
    while len(selected) < count:
        rng.shuffle(pool)
        take = min(count - len(selected), len(pool))
        selected.extend(pool[:take])
    return selected


def balanced_epoch_indices(repo_rows, candidate_indices, target_count, seed,
                            python_quota=PYTHON_REPOSITORIES_PER_EPOCH):
    """Return a deterministic Python-quota/Java-fill selection.

    The repository pool contains fewer Python repositories than Java
    repositories.  Each training epoch reserves ``python_quota`` instances for
    Python and fills the remainder with Java.  A language is oversampled only
    after every available repository has appeared once in the current cycle.
    The returned list is shuffled so language blocks do not reach the trainer
    in two large phases.
    """
    if target_count < 1:
        raise ValueError("target_count must be positive")
    if not 0 <= python_quota <= target_count:
        raise ValueError("python_quota must be between zero and target_count")
    if not candidate_indices:
        return []

    by_language = defaultdict(list)
    for index in candidate_indices:
        language = repo_rows[index]["language"]
        by_language[language].append(index)

    rng = random.Random(seed)
    languages = [language for language in ("python", "java")
                 if by_language.get(language)]

    if len(languages) == 2:
        python_count = python_quota
        java_count = target_count - python_count
        selected = (_cycled_sample(by_language["python"], python_count, rng)
                    + _cycled_sample(by_language["java"], java_count, rng))
    else:
        # Preserve the old behavior for a pool that is temporarily missing one
        # language instead of fabricating a second language quota.
        selected = list(candidate_indices)
        if len(selected) > target_count:
            selected = rng.sample(selected, target_count)

    rng.shuffle(selected)
    return selected


class EpochDataLoader:
    """Index metadata once; decompress only repositories selected this epoch."""

    def __init__(self, pool_path, generator, retriever, config=DataConfig(),
                 epoch_size=TRAIN_REPOSITORIES_PER_EPOCH,
                 python_quota=PYTHON_REPOSITORIES_PER_EPOCH):
        self.pool_path = Path(pool_path)
        self.generator = generator
        self.retriever = retriever
        self.config = config
        if epoch_size < 1 or not 0 <= python_quota <= epoch_size:
            raise ValueError("Invalid epoch size or Python quota")
        self.epoch_size = epoch_size
        self.python_quota = python_quota
        self.parquet = pq.ParquetFile(self.pool_path)
        # These four columns are tiny; load only metadata, never the compressed
        # repository payload column. Row-group sizes give exact payload
        # locations without opening every group during initialization.
        metadata = pq.read_table(
            self.pool_path, columns=["split", "language", "repo_id", "repo_uid"])
        self.repo_rows = metadata.to_pylist()
        self.locations = []
        for row_group in range(self.parquet.num_row_groups):
            row_count = self.parquet.metadata.row_group(row_group).num_rows
            self.locations.extend((row_group, local_row)
                                  for local_row in range(row_count))
        if len(self.locations) != len(self.repo_rows):
            raise ValueError("Repository pool metadata/payload row counts disagree")
        # The repository-level 15% split is intentionally not used anymore.
        # Benchmark test files are the only held-out data; every prepared
        # quality-filtered repository is eligible for training.
        self.train_indices = list(range(len(self.repo_rows)))
        self.train_repo_count = len(self.train_indices)
        self.last_selection = None

    def _load_payloads(self, indices):
        """Read selected rows only, preserving their random order."""
        by_group = defaultdict(list)
        for output_index, row_index in enumerate(indices):
            row_group, local_row = self.locations[row_index]
            by_group[row_group].append((local_row, output_index))
        payloads = [None] * len(indices)
        row_groups = sorted(by_group)
        table = self.parquet.read_row_groups(row_groups, columns=["payload"])
        group_offset = 0
        for row_group in row_groups:
            row_count = self.parquet.metadata.row_group(row_group).num_rows
            values = table["payload"][group_offset:group_offset + row_count].to_pylist()
            group_offset += row_count
            requests = by_group[row_group]
            for local_row, output_index in requests:
                payloads[output_index] = unpack(values[local_row])
        return payloads

    def select_epoch_indices(self, epoch, seed=123, split="train"):
        """Select one epoch from the complete prepared repository pool.

        ``split`` remains as a compatibility argument so old callers fail
        explicitly instead of silently training on a partial pool.
        """
        if split != "train":
            raise ValueError("Repository validation split is disabled; use benchmark test data")
        if epoch < 1:
            raise ValueError("epoch must be positive")
        selected = balanced_epoch_indices(
            self.repo_rows, self.train_indices, self.epoch_size,
            seed + epoch * 1_000_003, self.python_quota)
        self.last_selection = {
            "split": "train",
            "requested": self.epoch_size,
            "selected": len(selected),
            "unique_repositories": len(set(selected)),
            "python_quota": self.python_quota,
            "languages": {
                language: sum(self.repo_rows[index]["language"] == language
                              for index in selected)
                for language in ("python", "java")
            },
        }
        return selected

    def load_epoch(self, epoch, seed=123):
        """Create one language-balanced stochastic training epoch."""
        selected = self.select_epoch_indices(epoch, seed, split="train")
        payloads = self._load_payloads(selected)
        config = replace(self.config, repos_per_epoch=self.epoch_size)
        return sample_repo_epoch(payloads, self.generator, self.retriever,
                                 epoch, config, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.epoch < 1:
        raise ValueError("epoch must be positive")
    if args.output.exists():
        raise ValueError("Refusing to overwrite an epoch artifact")

    config = DataConfig()
    from transformers import AutoTokenizer
    generator = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base", revision=TOKENIZER_REVISION,
        local_files_only=True)
    retriever = AutoTokenizer.from_pretrained(
        "microsoft/unixcoder-base", revision=RET_REVISION, local_files_only=True)
    generator.model_max_length = retriever.model_max_length = 10**9
    loader = EpochDataLoader(args.pool, generator, retriever, config)
    tasks = loader.load_epoch(args.epoch, args.seed)
    if not tasks:
        raise ValueError("No tasks sampled from repository pool")
    schema = pa.schema(
        [("split", pa.string()), ("language", pa.string()), ("repo_id", pa.int64()),
         ("repo_uid", pa.string()), ("task_id", pa.string()), ("payload", pa.binary())],
        metadata={b"artifact_schema": TASK_SCHEMA.encode(),
                  b"source_pool_sha256": digest_file(args.pool).encode(),
                  b"config": json.dumps(asdict(config)).encode(),
                  b"epoch": str(args.epoch).encode(), b"seed": str(args.seed).encode()})
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.incomplete")
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for task in tasks:
            writer.write_table(pa.Table.from_pylist([{
                "split": "train", "language": task["language"],
                "repo_id": task["repo_id"], "repo_uid": task["repo_uid"],
                "task_id": task["task_id"], "payload": pack(task),
            }], schema=schema))
    temporary.replace(output)
    report = {
        "schema": TASK_SCHEMA, "epoch": args.epoch, "seed": args.seed,
        "requested_repositories": loader.epoch_size,
        "python_repositories_per_epoch": loader.python_quota,
        "available_repositories": loader.train_repo_count,
        "selection": loader.last_selection,
        "sampled_tasks": len(tasks),
        "unique_repositories": len({task["repo_uid"] for task in tasks}),
        "languages": {language: sum(task["language"] == language for task in tasks)
                      for language in ("python", "java")},
        "target_kinds": {kind: sum(task["target_kind"] == kind for task in tasks)
                         for kind in ("member_suffix", "line", "api_statement")},
        "dependent_targets": sum(task["label_metadata"]["dependent_target"] for task in tasks),
        "output_sha256": digest_file(output),
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
