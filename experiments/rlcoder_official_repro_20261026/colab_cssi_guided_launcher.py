#!/usr/bin/env python3
"""
colab_cssi_guided_launcher.py
Automated End-to-End Launcher for CADRE Consolidated with CSSI-Guided Retrieval on Google Colab A100.
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

SESSION_NAME = "cadre-cssi-guided-8004"
BUNDLE_PATH = PROJECT_ROOT / "experiments/mb_spt_iterative/full_benchmark_8004_bundle.tar.gz"
TRACE_PATH = PROJECT_ROOT / "experiments/mb_spt_iterative/rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz"
RUN_SCRIPT = CURRENT_DIR / "run_cadre_consolidated_full_8004.py"
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


def stream_colab_file(session: str, script_path: Path, timeout: int = 7200, env_vars: dict[str, str] | None = None) -> int:
    cmd = [COLAB_BIN, "exec", "-s", session, "-f", str(script_path), "--timeout", str(timeout)]
    if env_vars:
        for k, v in env_vars.items():
            cmd.extend(["--env", f"{k}={v}"])
    print(f"[Colab CLI Exec Stream] {' '.join(cmd)}", flush=True)
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
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run 50 tasks per benchmark smoke test")
    parser.add_argument("--benchmarks", default="cceval_python,cceval_java,repoeval_line,repoeval_api")
    args = parser.parse_args()

    local_out = CURRENT_DIR / "results_cadre_cssi_guided"
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
        run_colab(["upload", "-s", SESSION_NAME, str(RUN_SCRIPT), "/content/run_cadre_consolidated_full_8004.py"])

        print("=== Step 3: Checking / Uploading Bundles ===", flush=True)
        check_py = CURRENT_DIR / "_check_remote_cssi.py"
        check_py.write_text("""
import os
print("bundle:", os.path.exists("/content/full_benchmark_8004_bundle.tar.gz"))
print("trace:", os.path.exists("/content/rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz"))
""")
        res_chk = run_colab(["exec", "-s", SESSION_NAME, "-f", str(check_py), "--timeout", "60"], check=False)
        has_bundle = "bundle: True" in res_chk.stdout
        has_trace = "trace: True" in res_chk.stdout

        if not has_bundle:
            print(f"Uploading {BUNDLE_PATH.name} ({BUNDLE_PATH.stat().st_size / 1024 / 1024:.1f} MB)...", flush=True)
            run_colab(["upload", "-s", SESSION_NAME, str(BUNDLE_PATH), "/content/full_benchmark_8004_bundle.tar.gz"])
        else:
            print("✓ Remote full_benchmark_8004_bundle.tar.gz already present.")

        if not has_trace:
            print(f"Uploading {TRACE_PATH.name} ({TRACE_PATH.stat().st_size / 1024 / 1024:.1f} MB)...", flush=True)
            run_colab(["upload", "-s", SESSION_NAME, str(TRACE_PATH), "/content/rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz"])
        else:
            print("✓ Remote rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz already present.")

        print("=== Step 4: Setting Up Remote Dependencies ===", flush=True)
        ret = stream_colab_file(SESSION_NAME, SETUP_SCRIPT, timeout=600)
        if ret != 0:
            raise RuntimeError(f"Remote setup failed with code {ret}")

        print("=== Step 5: Restarting Kernel for Clean VRAM ===", flush=True)
        run_colab(["restart-kernel", "-s", SESSION_NAME])

        print("=== Step 6: Running CADRE Consolidated on CSSI-Guided Retrieval (8,004 Tasks) ===", flush=True)
        remote_runner_py = CURRENT_DIR / "_remote_runner_cssi.py"
        limit_str = f"--limit-per-benchmark 50" if args.smoke else ""
        remote_runner_py.write_text(f"""
import sys, subprocess
sys.path.insert(0, "/content/cadre_official")
cmd = [
    "python3", "/content/run_cadre_consolidated_full_8004.py",
    "--data-root", "/content",
    "--trace-path", "/content/rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz",
    "--output-dir", "/content/cadre-results/cadre_cssi_guided_8004",
    "--benchmarks", "{args.benchmarks}",
    "--generator-batch-size", "128",
    "--gpu-memory-utilization", "0.85",
    "--max-model-len", "4096",
]
limit_val = "{limit_str}".strip()
if limit_val:
    cmd.extend(limit_val.split())

print("Executing:", " ".join(cmd), flush=True)
res = subprocess.run(cmd, check=True)
print("Evaluation finished with code:", res.returncode, flush=True)
""")
        ret = stream_colab_file(SESSION_NAME, remote_runner_py, timeout=7200)
        if ret != 0:
            raise RuntimeError(f"Evaluation failed with code {ret}")

        print("=== Step 7: Archiving and Downloading Results ===", flush=True)
        zip_script = CURRENT_DIR / "_zip_results_cssi.py"
        zip_script.write_text("""
import shutil, os
import tarfile

src_dir = "/content/cadre-results/cadre_cssi_guided_8004"
out_tar = "/content/cadre_cssi_guided_8004.tar"
if os.path.exists(src_dir):
    print(f"Archiving {src_dir} to {out_tar}...")
    with tarfile.open(out_tar, "w") as tar:
        tar.add(src_dir, arcname="cadre_cssi_guided_8004")
    print("Archive complete, size:", os.path.getsize(out_tar))
""")
        stream_colab_file(SESSION_NAME, zip_script, timeout=300)

        remote_tar = "/content/cadre_cssi_guided_8004.tar"
        local_tar = local_out / "cadre_cssi_guided_8004.tar"
        run_colab(["download", "-s", SESSION_NAME, remote_tar, str(local_tar)])

        print("Extracting downloaded archive locally...")
        with subprocess.Popen(["tar", "-xf", str(local_tar), "-C", str(local_out)]) as proc:
            proc.wait()

        remote_metrics = "/content/cadre-results/cadre_cssi_guided_8004/cadre_consolidated_8004_metrics.json"
        local_metrics = local_out / "cadre_cssi_guided_8004_metrics.json"
        try:
            run_colab(["download", "-s", SESSION_NAME, remote_metrics, str(local_metrics)])
        except Exception:
            pass

        print("=== EVALUATION SUCCEEDED ===")
        if local_metrics.exists():
            print(f"Metrics saved to: {local_metrics}")
            print(local_metrics.read_text())

    finally:
        print("=== Step 8: Strict Compute Preserving - Stopping Colab Session ===", flush=True)
        run_colab(["stop", "-s", SESSION_NAME], check=False)
        print("Colab session stopped successfully. 0 Compute Units leaking.")


if __name__ == "__main__":
    main()
