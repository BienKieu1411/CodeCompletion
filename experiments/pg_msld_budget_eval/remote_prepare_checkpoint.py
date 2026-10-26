#!/usr/bin/env python3
"""Join uploaded checkpoint parts and verify all eval inputs on Colab."""

from __future__ import annotations

import hashlib
from pathlib import Path

PARTS_DIR = Path("/content")
PART_PREFIX = "part-"
OUTPUT = Path("/content/pg_msld_eval_checkpoint.pt")
EXPECTED_SHA256 = "30a6807a0b2727cf558d87308fe70a2bea6d40eb0893d002ae4a3105311686f7"
EXPECTED_BYTES = 1_506_667_019


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


parts = sorted(PARTS_DIR.glob(f"{PART_PREFIX}*"))
if len(parts) != 23:
    raise RuntimeError(f"Expected 23 checkpoint pieces, found {len(parts)}: {[p.name for p in parts]}")
if OUTPUT.exists():
    raise FileExistsError(f"Refusing to overwrite {OUTPUT}")

with OUTPUT.open("xb") as destination:
    for part in parts:
        with part.open("rb") as source:
            while block := source.read(8 * 1024 * 1024):
                destination.write(block)
        print(f"joined {part.name}", flush=True)

if OUTPUT.stat().st_size != EXPECTED_BYTES:
    raise RuntimeError(f"Checkpoint size mismatch: {OUTPUT.stat().st_size} != {EXPECTED_BYTES}")
actual = digest(OUTPUT)
print(f"checkpoint_bytes={OUTPUT.stat().st_size}", flush=True)
print(f"checkpoint_sha256={actual}", flush=True)
if actual != EXPECTED_SHA256:
    raise RuntimeError(f"Checkpoint hash mismatch: {actual}")

for required in (Path("/content/valid_400.parquet"), Path("/content/original_details.jsonl"), Path("/content/colab_eval.py")):
    if not required.is_file():
        raise FileNotFoundError(required)
    print(f"input_ok={required} bytes={required.stat().st_size}", flush=True)

print("READY_FOR_EVAL", flush=True)
