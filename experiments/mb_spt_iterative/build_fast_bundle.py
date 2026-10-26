#!/usr/bin/env python3
"""Package pilot_bundle.tar.gz + retrieval_trace.jsonl.gz + predictions.jsonl.gz into fast_eval_bundle.tar.gz."""

import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V4_DIR = ROOT / "results" / "graph-dependency-pilot-colab-20261024-v4"
OUT_BUNDLE = Path(__file__).resolve().parent / "fast_eval_bundle.tar.gz"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(V4_DIR / "pilot_bundle.tar.gz", "r:gz") as tar:
            tar.extractall(tmp_path, filter="data")
        shutil.copy2(V4_DIR / "outputs" / "retrieval_trace.jsonl.gz", tmp_path / "retrieval_trace.jsonl.gz")
        shutil.copy2(V4_DIR / "outputs" / "predictions.jsonl.gz", tmp_path / "baseline_predictions.jsonl.gz")
        with tarfile.open(OUT_BUNDLE, "w:gz") as out:
            for item in sorted(tmp_path.iterdir()):
                out.add(item, arcname=item.name)
    print(f"Created {OUT_BUNDLE} ({OUT_BUNDLE.stat().st_size / (1024*1024):.2f} MiB)")


if __name__ == "__main__":
    main()
