"""E2 hold-out sweep (budget_sweep.py --holdout) and E3 in-file compaction (e3_compact.py) on Modal L4.

usage (Git Bash: export MSYS_NO_PATHCONV=1):
    modal run --detach modal_e2e3.py --mode e3_smoke     # 4 samples per dataset
    modal run --detach modal_e2e3.py --mode all          # hold-out sweep + E3, 4 containers
    modal volume get dep-oracle-probe e2hold_repoeval ./modal_out/
"""
import os

import modal

from modal_probe import CANDS, HERE, REMOTE, _run_committed, image, vol  # noqa: F401

img = image.add_local_file(os.path.join(HERE, "e3_compact.py"), f"{REMOTE}/e3_compact.py")
img = img.add_local_python_source("modal_probe")
app = modal.App("dep-oracle-e2e3", image=img)
HOLD_BUDGETS = "256,512,1024,1536"


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=6 * 3600, cpu=4.0, memory=16384)
def sweep_holdout(dataset: str) -> int:
    return _run_committed(["python", f"{REMOTE}/budget_sweep.py", "sweep", "--holdout", "--dataset", dataset,
                           "--cands", f"{REMOTE}/{CANDS[dataset]}", "--data_dir", "/vol/aligncoder_data",
                           "--budgets", HOLD_BUDGETS, "--out", f"/vol/e2hold_{dataset}"])


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=6 * 3600, cpu=4.0, memory=16384)
def compact(dataset: str, tag: str = "e3", limit: int = 0) -> int:
    cmd = ["python", f"{REMOTE}/e3_compact.py", "run", "--dataset", dataset, "--cands", f"{REMOTE}/{CANDS[dataset]}",
           "--data_dir", "/vol/aligncoder_data", "--out", f"/vol/{tag}_{dataset}"]
    if limit:
        cmd += ["--limit", str(limit)]
    return _run_committed(cmd)


@app.local_entrypoint()
def main(mode: str = "e3_smoke"):
    if mode == "e3_smoke":
        print([c.get() for c in [compact.spawn(ds, "e3_smoke", 4) for ds in CANDS]])
    elif mode == "all":
        calls = [sweep_holdout.spawn(ds) for ds in CANDS] + [compact.spawn(ds) for ds in CANDS]
        print([c.get() for c in calls])
