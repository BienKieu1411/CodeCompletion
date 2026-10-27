import os
import subprocess
import sys
from pathlib import Path

output_dir = os.environ.get("RLCODER_OUTPUT_DIR", "/content/rlcoder-results/fixed-repro")
benchmarks = os.environ.get("RLCODER_BENCHMARKS", "cceval_python,cceval_java,repoeval_line,repoeval_api")
limit = os.environ.get("RLCODER_LIMIT", "")

cmd = [
    sys.executable, "/content/run_eval.py",
    "--output-dir", output_dir,
    "--benchmarks", benchmarks,
    "--generator-backend", "vllm",
    "--vllm-batch-size", "128",
    "--retriever-batch-size", "512",
    "--vllm-gpu-memory-utilization", "0.85",
]

if limit:
    cmd.extend(["--limit-per-benchmark", str(limit)])

print("=== Starting RLCoder Fixed Reproduction Evaluation ===", flush=True)
print("Command:", " ".join(cmd), flush=True)
res = subprocess.run(cmd)
if res.returncode != 0:
    print(f"FAILED with returncode {res.returncode}", flush=True)
    sys.exit(res.returncode)

print("=== Evaluation Finished Successfully! ===", flush=True)
