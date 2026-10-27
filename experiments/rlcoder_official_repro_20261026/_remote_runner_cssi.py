
import sys, subprocess
sys.path.insert(0, "/content/cadre_official")
cmd = [
    "python3", "/content/run_cadre_consolidated_full_8004.py",
    "--data-root", "/content",
    "--trace-path", "/content/rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz",
    "--output-dir", "/content/cadre-results/cadre_cssi_guided_8004",
    "--benchmarks", "cceval_python,cceval_java,repoeval_line,repoeval_api",
    "--generator-batch-size", "128",
    "--gpu-memory-utilization", "0.85",
    "--max-model-len", "4096",
]
limit_val = "".strip()
if limit_val:
    cmd.extend(limit_val.split())

print("Executing:", " ".join(cmd), flush=True)
res = subprocess.run(cmd, check=True)
print("Evaluation finished with code:", res.returncode, flush=True)
