"""Run likelihood_probe.py `score` on Modal (L4), one container per dataset, in parallel.

Outputs go to the Modal Volume `dep-oracle-probe` (/vol/<tag>_<dataset>/{gate.json,probe.jsonl}); the HF cache and
the AlignCoder data download live there too, so re-runs and resumes do not download again.

usage (from this folder, with the Modal venv):
    modal run --detach modal_probe.py --mode smoke                 # first 40 samples (gate on all 40)
    modal run --detach modal_probe.py --mode full                  # 1000 samples, both datasets
    modal volume get dep-oracle-probe full_repoeval ./modal_out/   # fetch results, then `analyze` locally
    modal run --detach modal_probe.py --mode pilot_smoke           # Phase 2 (chunk_pilot.py), 4 samples
    modal run --detach modal_probe.py --mode pilot                 # Phase 2, 200 samples per dataset
"""
import os
import subprocess
import threading

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE = "/root/dep"
CANDS = {"repoeval": "prediction_with_candidates.jsonl", "cceval": "cceval_python_prediction_with_candidates.jsonl"}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "huggingface_hub", "pandas", "pyarrow", "numpy")
    .env({"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1"})
)
for f in ["dep_oracle.py", "joint_retrieval.py", "joint_dense.py", "budget_sweep.py", "likelihood_probe.py",
          "chunk_pilot.py", *CANDS.values()]:
    image = image.add_local_file(os.path.join(HERE, f), f"{REMOTE}/{f}")
image = image.add_local_dir(os.path.join(HERE, "probe_inputs"), f"{REMOTE}/probe_inputs")

app = modal.App("dep-oracle-likelihood-probe", image=image)
vol = modal.Volume.from_name("dep-oracle-probe", create_if_missing=True)


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=4 * 3600, cpu=4.0, memory=16384)
def score(dataset: str, tag: str, limit: int = 0, gate_n: int = 50, gate_regen_n: int = 20) -> dict:
    out = f"/vol/{tag}_{dataset}"
    cmd = ["python", f"{REMOTE}/likelihood_probe.py", "score", "--dataset", dataset,
           "--cands", f"{REMOTE}/{CANDS[dataset]}", "--prev_dir", f"{REMOTE}/probe_inputs/{dataset}",
           "--data_dir", "/vol/aligncoder_data", "--out", out, "--gate_n", str(gate_n), "--gate_regen_n", str(gate_regen_n)]
    if limit:
        cmd += ["--limit", str(limit)]
    print(" ".join(cmd), flush=True)
    stop = threading.Event()

    def committer():  # keep partial probe.jsonl durable so a resumed run skips finished rows
        while not stop.wait(120):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()
    rc = subprocess.run(cmd, cwd=REMOTE).returncode
    stop.set()
    vol.commit()
    res = {"dataset": dataset, "returncode": rc}
    for fn in ("gate.json",):
        p = os.path.join(out, fn)
        if os.path.exists(p):
            res[fn] = open(p).read()
    p = os.path.join(out, "probe.jsonl")
    res["probe_rows"] = sum(1 for _ in open(p)) if os.path.exists(p) else 0
    return res


def _run_committed(cmd):
    """Run a command, committing the volume every 2 minutes so partial outputs survive and resume."""
    print(" ".join(cmd), flush=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(120):
            vol.commit()

    threading.Thread(target=committer, daemon=True).start()
    rc = subprocess.run(cmd, cwd=REMOTE).returncode
    stop.set()
    vol.commit()
    return rc


@app.function(gpu="L4", volumes={"/vol": vol}, timeout=6 * 3600, cpu=4.0, memory=16384)
def pilot(dataset: str, tag: str, limit: int = 0) -> dict:
    """Phase 2: chunk_pilot.py run (dLL labels + packed generations) on the 200-sample pilot subset."""
    out = f"/vol/{tag}_{dataset}"
    cmd = ["python", f"{REMOTE}/chunk_pilot.py", "run", "--dataset", dataset, "--cands", f"{REMOTE}/{CANDS[dataset]}",
           "--data_dir", "/vol/aligncoder_data", "--out", out]
    if limit:
        cmd += ["--limit", str(limit)]
    rc = _run_committed(cmd)
    p = os.path.join(out, "chunks.jsonl")
    return {"dataset": dataset, "returncode": rc, "label_rows": sum(1 for _ in open(p)) if os.path.exists(p) else 0,
            "conditions_done": sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)))
            if os.path.isdir(out) else []}


@app.local_entrypoint()
def main(mode: str = "smoke", datasets: str = "repoeval,cceval"):
    if mode.startswith("pilot"):
        kw = {"pilot_smoke": dict(tag="pilot_smoke", limit=4), "pilot": dict(tag="pilot", limit=0)}[mode]
        calls = [pilot.spawn(ds, **kw) for ds in datasets.split(",")]
        for c in calls:
            print(c.get())
        return
    kw = {"smoke": dict(tag="smoke", limit=40, gate_n=40, gate_regen_n=5), "full": dict(tag="full", limit=0, gate_n=50)}[mode]
    calls = [score.spawn(ds, **kw) for ds in datasets.split(",")]
    for c in calls:
        print(c.get())
