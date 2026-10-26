"""Run e0_order.py on Modal (L4), one container per dataset. Reuses the Phase-2 labels on the Volume
(/vol/pilot_<dataset>/chunks.jsonl) and writes to /vol/<tag>_<dataset>/.

usage (from this folder, with the Modal venv):
    modal run --detach modal_e0.py --mode smoke     # 4 samples, all conditions
    modal run --detach modal_e0.py --mode full      # the 200 pilot samples
    modal volume get dep-oracle-probe e0_repoeval ./modal_out/
"""
import os

import modal

from modal_probe import CANDS, REMOTE, _run_committed, image, vol  # noqa: F401  (same image + Volume)

image_e0 = (image.add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e0_order.py"),
                                 f"{REMOTE}/e0_order.py")
            .add_local_python_source("modal_probe"))  # this file imports it inside the container too
app = modal.App("dep-oracle-e0", image=image_e0)


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=4 * 3600, cpu=4.0, memory=16384)
def e0(dataset: str, tag: str, limit: int = 0) -> dict:
    out = f"/vol/{tag}_{dataset}"
    cmd = ["python", f"{REMOTE}/e0_order.py", "run", "--dataset", dataset, "--cands", f"{REMOTE}/{CANDS[dataset]}",
           "--labels", f"/vol/pilot_{dataset}/chunks.jsonl", "--data_dir", "/vol/aligncoder_data", "--out", out]
    if limit:
        cmd += ["--limit", str(limit)]
    rc = _run_committed(cmd)
    return {"dataset": dataset, "returncode": rc,
            "conditions_done": sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)))
            if os.path.isdir(out) else []}


@app.local_entrypoint()
def main(mode: str = "smoke", datasets: str = "repoeval,cceval"):
    kw = {"smoke": dict(tag="e0_smoke", limit=4), "full": dict(tag="e0", limit=0)}[mode]
    calls = [e0.spawn(ds, **kw) for ds in datasets.split(",")]
    for c in calls:
        print(c.get())
