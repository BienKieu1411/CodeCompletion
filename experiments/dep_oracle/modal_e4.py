"""E4 hidden-state extraction (e4_probe.py extract) on Modal L4, one container per dataset, full test.

usage (Git Bash: export MSYS_NO_PATHCONV=1):
    modal run modal_e4.py
    modal volume get dep-oracle-probe e4_repoeval ./modal_out/
"""
import os

import modal

from modal_probe import CANDS, HERE, REMOTE, _run_committed, image, vol  # noqa: F401

img = image.add_local_file(os.path.join(HERE, "e4_probe.py"), f"{REMOTE}/e4_probe.py")
img = img.add_local_python_source("modal_probe")
app = modal.App("dep-oracle-e4", image=img)


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=2 * 3600, cpu=4.0, memory=32768)
def extract(dataset: str, limit: int = 0) -> int:
    cmd = ["python", f"{REMOTE}/e4_probe.py", "extract", "--dataset", dataset, "--cands", f"{REMOTE}/{CANDS[dataset]}",
           "--data_dir", "/vol/aligncoder_data", "--out", f"/vol/e4_{dataset}"]
    if limit:
        cmd += ["--limit", str(limit)]
    return _run_committed(cmd)


@app.local_entrypoint()
def main(limit: int = 0):
    print([c.get() for c in [extract.spawn(ds, limit) for ds in CANDS]])
