"""Automated Colab runner to reproduce RLCoder with deepseek-coder-1.3b-base."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

COLAB_BIN = os.environ.get("COLAB_BIN") or shutil.which("colab") or "colab"
SESSION_NAME = "rlcoder-deepseek13b-repro"
CURRENT_DIR = Path(__file__).resolve().parent
LOCAL_RUN_EVAL = CURRENT_DIR / "run_eval.py"
LOCAL_SETUP_ENV = CURRENT_DIR / "colab_setup_env.py"
LOCAL_RUN_INFERENCE = CURRENT_DIR / "colab_run_inference.py"


def run_colab(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = [COLAB_BIN] + args
    print(f"[Colab CLI] {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_lines = []
    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            print(line, end="", flush=True)
            stdout_lines.append(line)
        process.stdout.close()
    returncode = process.wait()
    full_output = "".join(stdout_lines)
    if check and returncode != 0:
        print(f"Error: command failed with returncode {returncode}", flush=True)
        sys.exit(returncode)
    return subprocess.CompletedProcess(cmd, returncode, stdout=full_output, stderr="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run 50 tasks per benchmark for fast verification")
    parser.add_argument("--benchmarks", default="cceval_python,cceval_java,repoeval_line,repoeval_api", help="Comma-separated benchmarks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks per benchmark")
    parser.add_argument("--keep-alive", action="store_true", help="Keep Colab session running instead of stopping it")
    args = parser.parse_args()

    limit = 50 if args.smoke else args.limit

    print("=== Step 1: Checking Colab Active Sessions ===", flush=True)
    res = run_colab(["sessions"], check=False)
    session_exists = SESSION_NAME in res.stdout

    if not session_exists:
        print(f"=== Creating new Colab session '{SESSION_NAME}' with GPU A100 ===", flush=True)
        run_colab(["new", "--session", SESSION_NAME, "--gpu", "A100"])
    else:
        print(f"=== Reusing active session '{SESSION_NAME}' ===", flush=True)

    try:
        print("=== Step 2: Uploading Patched Scripts ===", flush=True)
        run_colab(["upload", "-s", SESSION_NAME, str(LOCAL_RUN_EVAL), "/content/run_eval.py"])

        print("=== Step 3: Setting Up Remote Environment and Models ===", flush=True)
        run_colab(["exec", "-s", SESSION_NAME, "-f", str(LOCAL_SETUP_ENV), "--timeout", "600.0"])

        print("=== Step 4: Restarting Kernel for Clean GPU VRAM ===", flush=True)
        run_colab(["restart-kernel", "-s", SESSION_NAME])

        print("=== Step 5: Running Fixed Reproduction Evaluation ===", flush=True)
        out_name = f"{'smoke' if limit else 'full'}-{int(time.time())}"
        output_dir = f"/content/rlcoder-results/{out_name}"
        env_args = [
            "--env", f"RLCODER_OUTPUT_DIR={output_dir}",
            "--env", f"RLCODER_BENCHMARKS={args.benchmarks}",
        ]
        if limit:
            env_args.extend(["--env", f"RLCODER_LIMIT={limit}"])

        timeout_sec = "600.0" if limit else "7200.0"
        exec_cmd = ["exec", "-s", SESSION_NAME, "-f", str(LOCAL_RUN_INFERENCE), "--timeout", timeout_sec] + env_args
        run_colab(exec_cmd)

        print("=== Step 6: Fetching Results and Metrics ===", flush=True)
        local_out = CURRENT_DIR / "results_fixed"
        local_out.mkdir(parents=True, exist_ok=True)

        # Archive results on remote
        zip_script = Path(__file__).resolve().parent / "make_archive.py"
        zip_script.write_text(f"""
import shutil, os
from pathlib import Path
out_path = Path("{output_dir}")
if out_path.is_dir():
    shutil.make_archive("{output_dir}", "zip", "{output_dir}")
    print("Created archive {output_dir}.zip", flush=True)
""")
        run_colab(["exec", "-s", SESSION_NAME, "-f", str(zip_script), "--timeout", "120.0"])

        run_colab(["download", "-s", SESSION_NAME, f"{output_dir}/run_info.json", str(local_out / "run_info.json")])
        run_colab(["download", "-s", SESSION_NAME, f"{output_dir}.zip", str(local_out / f"{out_name}.zip")])

        if (local_out / "run_info.json").is_file():
            info_obj = json.loads((local_out / "run_info.json").read_text(encoding="utf-8"))
            print("\n" + "="*50)
            print("🎯 FINAL REPRODUCTION SCORES (OFFICIAL SETTINGS)")
            print("="*50)
            for b_name, b_data in info_obj.get("benchmarks", {}).items():
                print(f"Benchmark: {b_name} (n={b_data.get('n')})")
                print(f"  Scores: {b_data.get('scores')}")
            print("="*50)

    finally:
        if not args.keep_alive:
            print("=== Step 7: Terminating Colab Session ===", flush=True)
            run_colab(["stop", "-s", SESSION_NAME], check=False)
            print("Colab session terminated cleanly. Compute units saved.")
        else:
            print(f"Colab session '{SESSION_NAME}' kept active as requested.")


if __name__ == "__main__":
    main()
