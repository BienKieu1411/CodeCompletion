"""E9c/E9d on Modal L4: CADRE v2.8 x budgets {384, 1536, 2304} at window 3072, storing conf of raw / main / final.
Runs `e9_cadre_budget.py run` then `analyze --ref <partner predictions>`; the analysis gives E9c (conf3 on the gated
final) and E9d (conf6 over {raw, main} x budgets) from the same generations. vLLM 0.32.0 = the Colab version that
reproduced the partner's predictions (397/400 exact at window 3072).

usage (Git Bash: export MSYS_NO_PATHCONV=1, PYTHONIOENCODING=utf-8):
    modal run modal_e9.py --limit 8             # smoke -> /vol/e9d_smoke
    modal run --detach modal_e9.py              # full 8004 tasks -> /vol/e9d
    modal volume get dep-oracle-probe e9d/e9_summary.json ./modal_out/e9d/
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL, REV = "deepseek-ai/deepseek-coder-1.3b-base", "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
DATA = os.path.join(HERE, "data_in", "e9data")
REF = os.path.join(HERE, "data_in", "cadre_v28", "cadre_v28_8004", "cadre_v28_8004_predictions.jsonl.gz")

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.32.0", "huggingface_hub", "fuzzywuzzy", "python-Levenshtein", "editdistance")
         .run_commands(f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL}', revision='{REV}')\"")
         .add_local_file(os.path.join(DATA, "tasks.jsonl.gz"), "/data/tasks.jsonl.gz")
         .add_local_file(os.path.join(DATA, "files.jsonl.gz"), "/data/files.jsonl.gz")
         .add_local_file(os.path.join(DATA, "rlcoder_full_8004_retrieval_trace.jsonl.gz"),
                         "/data/rlcoder_full_8004_retrieval_trace.jsonl.gz")
         .add_local_file(REF, "/data/cadre_v28_ref.jsonl.gz")
         .add_local_dir(os.path.join(HERE, "data_in", "cadre_pkg", "cadre"), "/root/cadre_official/cadre",
                        ignore=["__pycache__"])
         .add_local_file(os.path.join(HERE, "e9_cadre_budget.py"), "/root/e9/e9_cadre_budget.py")
         .add_local_file(os.path.join(HERE, "cadre_v28_unified.py"), "/root/e9/cadre_v28_unified.py")
         .add_local_file(os.path.join(HERE, "budget_sweep.py"), "/root/e9/budget_sweep.py")
         .add_local_file(os.path.join(HERE, "dep_oracle.py"), "/root/e9/dep_oracle.py"))
vol = modal.Volume.from_name("dep-oracle-probe")
app = modal.App("dep-oracle-e9", image=image)


@app.function(gpu=["L4", "A10G"], volumes={"/vol": vol}, timeout=5 * 60 * 60, cpu=8.0, memory=32768)
def run(limit: int = 0, tag: str = "e9d") -> str:
    import subprocess
    out = f"/vol/{tag}"
    common = "--window 3072 --budgets 384,1536,2304"
    cmd = (f"cd /root/e9 && python e9_cadre_budget.py run {common} --data-root /data --cadre /root/cadre_official "
           f"--out {out} --limit {limit} --gpu-mem 0.85 && python e9_cadre_budget.py analyze {common} --out {out} "
           f"--ref /data/cadre_v28_ref.jsonl.gz > {out}/analyze.log 2>&1")
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/run.log", "w") as f:
        rc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT).returncode
    vol.commit()
    return f"rc={rc} out={out}"


@app.local_entrypoint()
def main(limit: int = 0):
    if limit:
        print(run.remote(limit, "e9d_smoke"))
    else:  # spawn and return: with --detach the call survives the local client exiting
        print("spawned", run.spawn(0, "e9d").object_id)
