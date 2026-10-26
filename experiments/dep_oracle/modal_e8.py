"""E8 (e8_cad.py run) on Modal L4: 2 datasets x NSHARDS containers.

usage (Git Bash: export MSYS_NO_PATHCONV=1):
    modal run modal_e8.py --limit 4                 # smoke, 1 shard per dataset
    modal run --detach modal_e8.py                  # 200-sample pilot per dataset
    modal volume get dep-oracle-probe e8_repoeval ./modal_out/
"""
import os

import modal

from modal_probe import CANDS, HERE, REMOTE, _run_committed, image, vol  # noqa: F401

img = image.add_local_file(os.path.join(HERE, "e8_cad.py"), f"{REMOTE}/e8_cad.py")
img = img.add_local_python_source("modal_probe")
app = modal.App("dep-oracle-e8", image=img)


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=6 * 3600, cpu=4.0, memory=32768)
def run(dataset: str, shard: int, nshards: int, tag: str, limit: int = 0) -> int:
    cmd = ["python", f"{REMOTE}/e8_cad.py", "run", "--dataset", dataset, "--cands", f"{REMOTE}/{CANDS[dataset]}",
           "--data_dir", "/vol/aligncoder_data", "--out", f"/vol/{tag}_{dataset}",
           "--shard", str(shard), "--nshards", str(nshards)]
    if limit:
        cmd += ["--limit", str(limit)]
    return _run_committed(cmd)


@app.local_entrypoint()
def main(limit: int = 0, nshards: int = 2):
    if limit:
        calls = [run.spawn(ds, 0, 1, "e8_smoke", limit) for ds in CANDS]
    else:
        calls = [run.spawn(ds, s, nshards, "e8") for ds in CANDS for s in range(nshards)]
    print([c.get() for c in calls])
