#!/usr/bin/env python3
"""Package the frozen task subset, RLCoder source references, and runner."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE_DATA = ROOT / "results/graph-dependency-pilot-colab-20261024-v4"
BUNDLE = HERE / "rlcoder_dependency_graph_bundle.tar.gz"

DATA_FILES = ("tasks.jsonl.gz", "pools.jsonl.gz", "files.jsonl.gz", "selection.json")
RLCODER_FILES = (
    "datasets.py", "bm25.py", "retriever.py", "generator.py",
    "utils/build/python-lang-parser.so", "utils/build/java-lang-parser.so",
    "utils/keywords/java.txt", "utils/keywords/keywordlist.py",
    "utils/eval_metric.py", "utils/eval_utils.py", "utils/eval_repoeval.py",
)


def main() -> None:
    entries: list[tuple[Path, str]] = []
    for name in DATA_FILES:
        source = SOURCE_DATA / name
        if not source.is_file():
            raise FileNotFoundError(source)
        entries.append((source, f"data/{name}"))
    entries.append((HERE / "runner.py", "runner.py"))
    for name in RLCODER_FILES:
        source = ROOT / "RepoClone/RLCoder" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        entries.append((source, f"RLCoder/{name}"))

    with tarfile.open(BUNDLE, "w:gz") as archive:
        for source, arcname in entries:
            archive.add(source, arcname=arcname)
    digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    (HERE / "rlcoder_dependency_graph_bundle.sha256").write_text(
        f"{digest}  {BUNDLE.name}\n", encoding="utf-8")
    print(f"Bundle: {BUNDLE}")
    print(f"Bytes: {BUNDLE.stat().st_size}")
    print(f"SHA256: {digest}")
    print(f"Included entries: {len(entries)}")


if __name__ == "__main__":
    main()
