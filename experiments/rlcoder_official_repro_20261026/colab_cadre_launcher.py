#!/usr/bin/env python3
"""
colab_cadre_launcher.py
Automated End-to-End Launcher for CADRE Consolidated (BẢN HIỆN TẠI) on Google Colab A100.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

COLAB_BIN = "/Users/kieugiangbien/bienkieu_env/bin/colab"
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

SESSION_NAME = "cadre-consolidated-8004"
BUNDLE_PATH = PROJECT_ROOT / "experiments/mb_spt_iterative/full_benchmark_8004_bundle.tar.gz"
TRACE_PATH = PROJECT_ROOT / "experiments/mb_spt_iterative/rlcoder_full_8004_retrieval_trace.jsonl.gz"
RUN_SCRIPT = CURRENT_DIR / "run_cadre_consolidated_full_8004.py"
RUN_SCRIPT_V27 = CURRENT_DIR / "run_cadre_v27_optimized.py"
RUN_SCRIPT_V28 = CURRENT_DIR / "run_cadre_v28_unified.py"
SETUP_SCRIPT = CURRENT_DIR / "colab_cadre_setup.py"
CADRE_PKG = PROJECT_ROOT / "cadre_official"


def run_colab(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = [COLAB_BIN] + args
    print(f"[Colab CLI] {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip(), flush=True)
    if res.stderr:
        print(f"[stderr] {res.stderr.strip()}", flush=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(cmd)}\n{res.stderr}")
    return res


def stream_colab_file(session: str, script_path: Path, timeout: int = 7200, env_vars: dict[str, str] | None = None, max_retries: int = 2) -> int:
    cmd = [COLAB_BIN, "exec", "-s", session, "-f", str(script_path), "--timeout", str(timeout)]
    if env_vars:
        for k, v in env_vars.items():
            cmd.extend(["--env", f"{k}={v}"])
    
    for attempt in range(max_retries):
        print(f"[Colab CLI Exec Stream (Attempt {attempt+1}/{max_retries})] {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        if proc.returncode == 0:
            return 0
        if attempt < max_retries - 1:
            print(f"[Warning] Stream exited with code {proc.returncode}. Waiting 10s before retry...", flush=True)
            time.sleep(10)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run 50 tasks per benchmark smoke test")
    parser.add_argument("--benchmarks", default="cceval_python,cceval_java,repoeval_line,repoeval_api")
    parser.add_argument("--v27", action="store_true", help="Run CADRE v2.7 Optimized (Comment-shielded + Dynamic Left Context)")
    parser.add_argument("--v28", action="store_true", default=True, help="Run CADRE v2.8 Unified Harmony (Raw chunks 1.5k + Dynamic Left Context up to 3k)")
    parser.add_argument("--consolidated", action="store_true", help="Run CADRE v2.6 Consolidated")
    args = parser.parse_args()

    if args.consolidated:
        active_script = RUN_SCRIPT
        target_remote_script = "/content/run_cadre_consolidated_full_8004.py"
        res_dir_name = "cadre_consolidated_8004"
        local_out = CURRENT_DIR / "results_cadre_consolidated"
        max_len_str = "4096"
    elif args.v27:
        active_script = RUN_SCRIPT_V27
        target_remote_script = "/content/run_cadre_v27_optimized.py"
        res_dir_name = "cadre_v27_8004"
        local_out = CURRENT_DIR / "results_cadre_v27"
        max_len_str = "2560"
    else:
        active_script = RUN_SCRIPT_V28
        target_remote_script = "/content/run_cadre_v28_unified.py"
        res_dir_name = "cadre_v28_8004"
        local_out = CURRENT_DIR / "results_cadre_v28"
        max_len_str = "3072"

    local_out.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Checking Colab Active Sessions ===", flush=True)
    res = run_colab(["sessions"], check=False)
    session_exists = SESSION_NAME in res.stdout

    if not session_exists:
        print(f"Creating new Colab session '{SESSION_NAME}' with NVIDIA A100 GPU...", flush=True)
        run_colab(["new", "--session", SESSION_NAME, "--gpu", "A100"])
    else:
        print(f"=== Reusing active session '{SESSION_NAME}' ===", flush=True)

    try:
        print("=== Step 2: Archiving and Uploading CADRE Package and Scripts ===", flush=True)
        cadre_tar = Path("/tmp/cadre_official.tar.gz")
        subprocess.run(["tar", "-czf", str(cadre_tar), "-C", str(PROJECT_ROOT), "cadre_official"], check=True)
        run_colab(["upload", "-s", SESSION_NAME, str(cadre_tar), "/content/cadre_official.tar.gz"])
        run_colab(["upload", "-s", SESSION_NAME, str(active_script), target_remote_script])

        print("=== Step 3: Checking / Uploading Bundles ===", flush=True)
        check_py = CURRENT_DIR / "_check_remote.py"
        check_py.write_text("""
import os
print("bundle:", os.path.exists("/content/full_benchmark_8004_bundle.tar.gz"))
print("trace:", os.path.exists("/content/rlcoder_full_8004_retrieval_trace.jsonl.gz"))
try:
    import vllm
    print("setup_done: True")
except Exception:
    print("setup_done: False")
""")
        res_chk = run_colab(["exec", "-s", SESSION_NAME, "-f", str(check_py), "--timeout", "60"], check=False)
        has_bundle = "bundle: True" in res_chk.stdout
        has_trace = "trace: True" in res_chk.stdout
        setup_done = "setup_done: True" in res_chk.stdout

        if not has_bundle:
            print(f"Uploading {BUNDLE_PATH.name} ({BUNDLE_PATH.stat().st_size / 1024 / 1024:.1f} MB)...", flush=True)
            run_colab(["upload", "-s", SESSION_NAME, str(BUNDLE_PATH), "/content/full_benchmark_8004_bundle.tar.gz"])
        else:
            print("✓ Remote full_benchmark_8004_bundle.tar.gz already present.")

        if not has_trace:
            print(f"Uploading {TRACE_PATH.name} ({TRACE_PATH.stat().st_size / 1024 / 1024:.1f} MB)...", flush=True)
            run_colab(["upload", "-s", SESSION_NAME, str(TRACE_PATH), "/content/rlcoder_full_8004_retrieval_trace.jsonl.gz"])
        else:
            print("✓ Remote rlcoder_full_8004_retrieval_trace.jsonl.gz already present.")

        if not setup_done:
            print("=== Step 4: Setting Up Remote Dependencies ===", flush=True)
            ret = stream_colab_file(SESSION_NAME, SETUP_SCRIPT, timeout=600)
            if ret != 0:
                raise RuntimeError(f"Remote setup failed with code {ret}")
        else:
            print("✓ Remote environment & dependencies already verified and ready.")

        print("=== Step 5: Restarting Kernel for Clean VRAM ===", flush=True)
        run_colab(["restart-kernel", "-s", SESSION_NAME])
        print("Waiting 10s for kernel to fully initialize...", flush=True)
        time.sleep(10)

        print(f"=== Step 6: Running {active_script.name} Full 8,004 Evaluation ===", flush=True)
        # Create remote entrypoint runner script
        remote_runner_py = CURRENT_DIR / "_remote_runner.py"
        limit_str = f"--limit-per-benchmark 50" if args.smoke else ""
        remote_runner_py.write_text(f"""
import os, sys, subprocess
sys.path.insert(0, "/content/cadre_official")
cmd = [
    "python3", "-u", "{target_remote_script}",
    "--data-root", "/content",
    "--trace-path", "/content/rlcoder_full_8004_retrieval_trace.jsonl.gz",
    "--output-dir", "/content/cadre-results/{res_dir_name}",
    "--benchmarks", "{args.benchmarks}",
    "--generator-batch-size", "128",
    "--gpu-memory-utilization", "0.85",
    "--max-model-len", "{max_len_str}",
]
limit_val = "{limit_str}".strip()
if limit_val:
    cmd.extend(limit_val.split())

print("Executing:", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH="/content/cadre_official"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
if proc.returncode != 0:
    raise RuntimeError(f"Child process failed with code {{proc.returncode}}")
print("Evaluation finished with code:", proc.returncode, flush=True)
""")
        ret = stream_colab_file(SESSION_NAME, remote_runner_py, timeout=7200)
        if ret != 0:
            raise RuntimeError(f"Evaluation failed with code {ret}")

        print("=== Step 7: Archiving and Downloading Results ===", flush=True)
        zip_script = CURRENT_DIR / "_zip_results.py"
        zip_script.write_text(f"""
import shutil, os
src = "/content/cadre-results/{res_dir_name}"
dst = "/content/cadre-results/{res_dir_name}"
shutil.make_archive(dst, 'tar', root_dir="/content/cadre-results", base_dir="{res_dir_name}")
print("Archive created at", dst + ".tar", flush=True)
""")
        stream_colab_file(SESSION_NAME, zip_script, timeout=300)

        run_colab(["download", "-s", SESSION_NAME, f"/content/cadre-results/{res_dir_name}/{res_dir_name}_metrics.json", str(local_out / f"{res_dir_name}_metrics.json")])
        run_colab(["download", "-s", SESSION_NAME, f"/content/cadre-results/{res_dir_name}.tar", str(local_out / f"{res_dir_name}.tar")])

        # Extract locally
        subprocess.run(["tar", "-xf", str(local_out / f"{res_dir_name}.tar"), "-C", str(local_out)], check=True)
        print(f"\n✓ Successfully downloaded and extracted results to {local_out}!", flush=True)

    finally:
        print("=== Step 8: Terminating Colab Session to Save Compute Units ===", flush=True)
        run_colab(["stop", "-s", SESSION_NAME], check=False)
        print("Colab session terminated cleanly. Compute units saved.")


if __name__ == "__main__":
    main()
