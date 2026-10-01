"""Streaming repository groups from the Data4AlignCoder parquet layout."""

from pathlib import Path

import pyarrow.parquet as pq


def groups(path):
    """Yield one list of ``(relative_path, source)`` rows per repository."""
    path = Path(path)
    group = []
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=32, columns=["path", "content", "first"]):
        for row in batch.to_pylist():
            if row["first"] and group:
                yield group
                group = []
            group.append((row["path"], row["content"]))
    if group:
        yield group
