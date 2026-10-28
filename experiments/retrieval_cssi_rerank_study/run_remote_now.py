#!/usr/bin/env python3
"""
run_remote_now.py
Executes the BM25 Top 100 CSSI Rerank Study on the active Colab A100 GPU session.
"""

import subprocess
import sys
import time
from pathlib import Path

COLAB_BIN = "/Users/kieugiangbien/bienkieu_env/bin/colab"
SESSION_NAME = "cadre-rerank-study"
STUDY_DIR = Path("experiments/retrieval_cssi_rerank_study").resolve()
LOCAL_OUT = STUDY_DIR / "colab_results"
LOCAL_OUT.mkdir(parents=True, exist_ok=True)

# 1. Upload latest script
print("1. Uploading latest run_bm25_top100_cssi_rerank_study.py to Colab...", flush=True)
subprocess.check_call([
    COLAB_BIN, "upload", "-s", SESSION_NAME,
    str(STUDY_DIR / "run_bm25_top100_cssi_rerank_study.py"),
    "/content/run_bm25_top100_cssi_rerank_study.py"
])

# 2. Restart kernel for clean GPU memory
print("2. Restarting kernel for clean A100 VRAM...", flush=True)
subprocess.run([COLAB_BIN, "restart-kernel", "-s", SESSION_NAME], check=True)
time.sleep(8)

# 3. Create execution runner
print("3. Launching BM25 Top 100 CSSI Reranking & vLLM Generation on A100...", flush=True)
is_smoke = "--smoke" in sys.argv
smoke_flag = '        "--smoke",' if is_smoke else ''
runner_py = STUDY_DIR / "_exec_remote.py"
runner_py.write_text(f"""
import subprocess, os, sys
sys.path.insert(0, "/content/cadre_official")
cmd = [
    "python3", "-u", "/content/run_bm25_top100_cssi_rerank_study.py",
    "--data-root", "/content",
    "--output-dir", "/content/cadre-results/bm25_top100_rerank",
    "--max-model-len", "3072",
{smoke_flag}
]
proc = subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH="/content/cadre_official"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
if proc.returncode != 0:
    raise RuntimeError(f"Failed with code {{proc.returncode}}")
print("✓ Completed remote execution!", flush=True)
""")

cmd = [COLAB_BIN, "exec", "-s", SESSION_NAME, "-f", str(runner_py), "--timeout", "3600"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
assert proc.stdout is not None
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()

if proc.returncode != 0:
    print(f"Error: execution returned code {proc.returncode}", flush=True)
    sys.exit(proc.returncode)

# 4. Download results
print("4. Downloading results from Colab...", flush=True)
dl_script = STUDY_DIR / "_dl_script.py"
dl_script.write_text("""
import shutil
shutil.make_archive("/content/cadre-results/bm25_top100_rerank", 'tar', root_dir="/content/cadre-results", base_dir="bm25_top100_rerank")
print("Archive created.", flush=True)
""")
subprocess.run([COLAB_BIN, "exec", "-s", SESSION_NAME, "-f", str(dl_script), "--timeout", "120"], check=True)
subprocess.run([COLAB_BIN, "download", "-s", SESSION_NAME, "/content/cadre-results/bm25_top100_rerank/bm25_top100_rerank_metrics.json", str(LOCAL_OUT / "bm25_top100_rerank_metrics.json")], check=True)
subprocess.run([COLAB_BIN, "download", "-s", SESSION_NAME, "/content/cadre-results/bm25_top100_rerank.tar", str(LOCAL_OUT / "bm25_top100_rerank.tar")], check=True)

subprocess.run(["tar", "-xf", str(LOCAL_OUT / "bm25_top100_rerank.tar"), "-C", str(LOCAL_OUT)], check=True)
print(f"\n✓ Successfully downloaded results to {LOCAL_OUT}!", flush=True)
