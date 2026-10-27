
import os, sys, subprocess
sys.path.insert(0, "/content/cadre_official")
cmd = [
    "python3", "-u", "/content/run_cadre_v28_unified.py",
    "--data-root", "/content",
    "--trace-path", "/content/rlcoder_full_8004_retrieval_trace.jsonl.gz",
    "--output-dir", "/content/cadre-results/cadre_v28_8004",
    "--benchmarks", "cceval_python,cceval_java,repoeval_line,repoeval_api",
    "--generator-batch-size", "128",
    "--gpu-memory-utilization", "0.85",
    "--max-model-len", "3072",
]
limit_val = "".strip()
if limit_val:
    cmd.extend(limit_val.split())

print("Executing:", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH="/content/cadre_official"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
if proc.returncode != 0:
    raise RuntimeError(f"Child process failed with code {proc.returncode}")
print("Evaluation finished with code:", proc.returncode, flush=True)
