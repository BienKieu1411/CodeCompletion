#!/usr/bin/env python3
"""
launch_colab.py
Automated End-to-End Launcher to run the Retrieval & Rerank Study on Google Colab A100 GPU.
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
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
STUDY_DIR = ROOT_DIR / "experiments/retrieval_cssi_rerank_study"
MB_DIR = ROOT_DIR / "experiments/mb_spt_iterative"
REPRO_DIR = ROOT_DIR / "experiments/rlcoder_official_repro_20261026"

SESSION_NAME = "cadre-rerank-study"
BUNDLE_PATH = MB_DIR / "full_benchmark_8004_bundle.tar.gz"
TRACE_STD_PATH = MB_DIR / "rlcoder_full_8004_retrieval_trace.jsonl.gz"
TRACE_FULL_PATH = MB_DIR / "rlcoder_full_left_standard_8004_trace.jsonl.gz"
TRACE_CSSI_PATH = MB_DIR / "rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz"

RUN_SCRIPT = STUDY_DIR / "run_bm25_top100_cssi_rerank_study.py"
SETUP_SCRIPT = REPRO_DIR / "colab_cadre_setup.py"


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


def stream_colab_file(session: str, script_path: Path, timeout: int = 7200, max_retries: int = 2) -> int:
    cmd = [COLAB_BIN, "exec", "-s", session, "-f", str(script_path), "--timeout", str(timeout)]
    for attempt in range(max_retries):
        print(f"[Colab CLI Exec Stream (Attempt {attempt+1}/{max_retries})] {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        if proc.returncode == 0:
            return 0
        if attempt < max_retries - 1:
            print(f"[Warning] Stream exited with code {proc.returncode}. Retrying...", flush=True)
            time.sleep(10)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Colab A100 GPU Retrieval & Rerank Study")
    parser.add_argument("--smoke", action="store_true", help="Run 50 tasks per benchmark smoke test")
    parser.add_argument("--limit-per-bm", type=int, default=None, help="Limit number of tasks per benchmark")
    parser.add_argument("--gpu", default="A100", choices=["A100", "L4", "T4"], help="GPU type")
    parser.add_argument("--keep-alive", action="store_true", help="Keep Colab session alive after run")
    args = parser.parse_args()

    local_out = STUDY_DIR / "colab_results"
    local_out.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Checking Colab Active Sessions ===", flush=True)
    res = run_colab(["sessions"], check=False)
    session_exists = SESSION_NAME in res.stdout

    if not session_exists:
        print(f"Creating new Colab session '{SESSION_NAME}' with NVIDIA {args.gpu} GPU...", flush=True)
        run_colab(["new", "--session", SESSION_NAME, "--gpu", args.gpu])
    else:
        print(f"=== Reusing active session '{SESSION_NAME}' ===", flush=True)

    try:
        print("=== Step 2: Archiving and Uploading CADRE Package and Scripts ===", flush=True)
        cadre_tar = Path("/tmp/cadre_official.tar.gz")
        subprocess.run(["tar", "-czf", str(cadre_tar), "-C", str(ROOT_DIR), "cadre_official"], check=True)
        run_colab(["upload", "-s", SESSION_NAME, str(cadre_tar), "/content/cadre_official.tar.gz"])
        run_colab(["upload", "-s", SESSION_NAME, str(RUN_SCRIPT), "/content/run_bm25_top100_cssi_rerank_study.py"])

        print("=== Step 3: Checking / Uploading Bundles and Traces ===", flush=True)
        check_py = STUDY_DIR / "_check_remote.py"
        check_py.write_text("""
import os
print("bundle:", os.path.exists("/content/full_benchmark_8004_bundle.tar.gz"))
print("trace_std:", os.path.exists("/content/rlcoder_full_8004_retrieval_trace.jsonl.gz"))
try:
    import vllm
    print("setup_done: True")
except Exception:
    print("setup_done: False")
""")
        res_chk = run_colab(["exec", "-s", SESSION_NAME, "-f", str(check_py), "--timeout", "60"], check=False)
        has_bundle = "bundle: True" in res_chk.stdout
        has_trace_std = "trace_std: True" in res_chk.stdout
        setup_done = "setup_done: True" in res_chk.stdout

        if not has_bundle:
            print(f"Uploading {BUNDLE_PATH.name}...", flush=True)
            run_colab(["upload", "-s", SESSION_NAME, str(BUNDLE_PATH), "/content/full_benchmark_8004_bundle.tar.gz"])
        if not has_trace_std:
            print(f"Uploading {TRACE_STD_PATH.name}...", flush=True)
            run_colab(["upload", "-s", SESSION_NAME, str(TRACE_STD_PATH), "/content/rlcoder_full_8004_retrieval_trace.jsonl.gz"])

        if not setup_done:
            print("=== Step 4: Setting Up Remote Dependencies (vLLM, tree-sitter, etc.) ===", flush=True)
            ret = stream_colab_file(SESSION_NAME, SETUP_SCRIPT, timeout=600)
            if ret != 0:
                raise RuntimeError(f"Remote setup failed with code {ret}")
        else:
            print("✓ Remote environment & dependencies already verified and ready.")

        print("=== Step 5: Restarting Kernel for Clean GPU VRAM ===", flush=True)
        run_colab(["restart-kernel", "-s", SESSION_NAME])
        print("Waiting 10s for kernel initialization...", flush=True)
        time.sleep(10)

        print("=== Step 6: Running BM25 Top 100 CSSI Rerank & Live vLLM Generation on A100 GPU ===", flush=True)
        remote_runner_py = STUDY_DIR / "_remote_runner.py"
        limit_arg = ""
        if args.limit_per_bm:
            limit_arg = f"--limit-per-bm {args.limit_per_bm}"
        elif args.smoke:
            limit_arg = "--smoke"

        remote_runner_py.write_text(f"""
import os, sys, subprocess
sys.path.insert(0, "/content/cadre_official")
cmd = [
    "python3", "-u", "/content/run_bm25_top100_cssi_rerank_study.py",
    "--data-root", "/content",
    "--output-dir", "/content/cadre-results/bm25_top100_rerank",
    "--max-model-len", "3072",
]
lim = "{limit_arg}".strip()
if lim:
    cmd.extend(lim.split())

print("Executing:", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH="/content/cadre_official"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
if proc.returncode != 0:
    raise RuntimeError(f"Execution failed with code {{proc.returncode}}")
""")
        ret = stream_colab_file(SESSION_NAME, remote_runner_py, timeout=7200)
        if ret != 0:
            raise RuntimeError(f"Remote evaluation failed with code {ret}")

        print("=== Step 7: Archiving and Downloading Results ===", flush=True)
        zip_script = STUDY_DIR / "_zip_results.py"
        zip_script.write_text("""
import shutil
shutil.make_archive("/content/cadre-results/bm25_top100_rerank", 'tar', root_dir="/content/cadre-results", base_dir="bm25_top100_rerank")
print("Archive created successfully.", flush=True)
""")
        stream_colab_file(SESSION_NAME, zip_script, timeout=300)

        run_colab(["download", "-s", SESSION_NAME, "/content/cadre-results/bm25_top100_rerank/bm25_top100_rerank_metrics.json", str(local_out / "bm25_top100_rerank_metrics.json")])
        run_colab(["download", "-s", SESSION_NAME, "/content/cadre-results/bm25_top100_rerank.tar", str(local_out / "bm25_top100_rerank.tar")])

        subprocess.run(["tar", "-xf", str(local_out / "bm25_top100_rerank.tar"), "-C", str(local_out)], check=True)
        print(f"\n✓ Successfully downloaded and extracted results to {local_out}!", flush=True)
        run_colab(["download", "-s", SESSION_NAME, "/content/cadre-results/rerank_study.tar", str(local_out / "rerank_study.tar")])

        subprocess.run(["tar", "-xf", str(local_out / "rerank_study.tar"), "-C", str(local_out)], check=True)
        print(f"\n✓ Successfully downloaded and extracted results to {local_out}!", flush=True)

    finally:
        if not args.keep_alive:
            print("=== Step 8: Terminating Colab Session to Save Compute Units ===", flush=True)
            run_colab(["stop", "-s", SESSION_NAME], check=False)
            print("Colab session terminated cleanly.")
        else:
            print(f"Colab session '{SESSION_NAME}' kept active as requested.")


if __name__ == "__main__":
    main()
