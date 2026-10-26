"""E1 in-domain diagnostic (e1_cv.py) on Modal: 5 folds x {bi_soft, cross_soft}, one L4 container each.

usage (from this folder, Modal venv, Git Bash: export MSYS_NO_PATHCONV=1):
    modal run --detach modal_e1cv.py                 # all 10 jobs
    modal volume get dep-oracle-probe e1cv ./modal_out/
"""
import os

import modal

from modal_probe import CANDS, HERE, REMOTE, _run_committed, image, vol  # noqa: F401

img = image
for f in ["e1_train.py", "e1_cv.py"]:
    img = img.add_local_file(os.path.join(HERE, f), f"{REMOTE}/{f}")
for ds in CANDS:
    img = img.add_local_file(os.path.join(HERE, "modal_out", f"pilot_{ds}", "chunks.jsonl"),
                             f"{REMOTE}/pilot_{ds}_chunks.jsonl")
img = img.add_local_python_source("modal_probe")
app = modal.App("dep-oracle-e1cv", image=img)


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=3 * 3600, cpu=4.0, memory=16384)
def fold(k: int, variant: str) -> int:
    cmd = ["python", f"{REMOTE}/e1_cv.py", "--fold", str(k), "--variant", variant, "--out", "/vol/e1cv",
           "--data_dir", "/vol/aligncoder_data"]
    for ds, cf in CANDS.items():
        cmd += ["--pilot", f"{ds}={REMOTE}/{cf}:{REMOTE}/pilot_{ds}_chunks.jsonl"]
    return _run_committed(cmd)


@app.local_entrypoint()
def main(variants: str = "bi_soft,cross_soft", folds: int = 5, skip: str = ""):
    skip_set = set(skip.split(",")) if skip else set()  # e.g. "bi_soft:0" when that job already ran
    calls = [fold.spawn(k, v) for v in variants.split(",") for k in range(folds) if f"{v}:{k}" not in skip_set]
    print([c.get() for c in calls])
