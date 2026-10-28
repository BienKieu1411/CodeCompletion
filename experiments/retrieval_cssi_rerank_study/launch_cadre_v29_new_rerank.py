#!/usr/bin/env python3
"""
launch_cadre_v29_new_rerank.py
Automated End-to-End Launcher to run CADRE v2.9 with the new BM25 Top 100 CSSI Rerank trace
on Google Colab A100 GPU.
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

SESSION_NAME = "cadre-v29-rerank-full"
BUNDLE_PATH = MB_DIR / "full_benchmark_8004_bundle.tar.gz"
NEW_TRACE_PATH = STUDY_DIR / "colab_results/full_definition_20261028/retrieval_trace.jsonl.gz"
SETUP_SCRIPT = REPRO_DIR / "colab_cadre_setup.py"
RUN_SCRIPT = REPRO_DIR / "run_cadre_v29_adaptive_slack.py"
LOCAL_OUT = STUDY_DIR / "colab_results/cadre_v29_new_rerank"


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
        print(f"[Colab Exec Stream (Attempt {attempt+1}/{max_retries})] {' '.join(cmd)}", flush=True)
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


def upload_large_file(session: str, local_path: Path, remote_path: str, chunk_mb: int = 35) -> None:
    size = local_path.stat().st_size
    if size <= chunk_mb * 1024 * 1024:
        run_colab(["upload", "-s", session, str(local_path), remote_path])
        return

    print(f"Splitting {local_path.name} ({size / 1024 / 1024:.1f} MB) into {chunk_mb}MB chunks for stable Colab upload...", flush=True)
    temp_dir = Path("/tmp/colab_upload_chunks")
    temp_dir.mkdir(parents=True, exist_ok=True)
    for old in temp_dir.glob("part_*"):
        old.unlink()

    subprocess.run(["split", "-b", f"{chunk_mb}m", str(local_path), str(temp_dir / "part_")], check=True)
    parts = sorted(temp_dir.glob("part_*"))
    for idx, p in enumerate(parts):
        print(f"  Uploading chunk {idx+1}/{len(parts)}: {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)...", flush=True)
        run_colab(["upload", "-s", session, str(p), f"/content/{p.name}"])
        p.unlink()

    print(f"Reassembling chunks into {remote_path} on Colab...", flush=True)
    re_script = temp_dir / "reassemble.py"
    re_script.write_text(f"""
import subprocess, os
subprocess.run("cat /content/part_* > {remote_path} && rm -f /content/part_*", shell=True, check=True)
print("Reassembled {remote_path}, size:", os.path.getsize("{remote_path}"))
""")
    run_colab(["exec", "-s", session, "-f", str(re_script), "--timeout", "120"])
    print(f"✓ Reassembled {remote_path} successfully.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch CADRE v2.9 with new rerank trace on Colab A100")
    parser.add_argument("--smoke", action="store_true", help="Run 50 tasks per benchmark smoke test")
    parser.add_argument("--gpu", default="A100", choices=["A100", "L4", "T4"], help="GPU type")
    args = parser.parse_args()

    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Checking Colab Active Sessions ===", flush=True)
    res = run_colab(["sessions"], check=False)
    session_exists = SESSION_NAME in res.stdout

    if not session_exists:
        print(f"Creating new Colab session '{SESSION_NAME}' with NVIDIA {args.gpu} GPU...", flush=True)
        run_colab(["new", "--session", SESSION_NAME, "--gpu", args.gpu])
    else:
        print(f"=== Reusing active session '{SESSION_NAME}' ===", flush=True)

    print("=== Step 2: Archiving and Uploading CADRE Package and Scripts ===", flush=True)
    cadre_tar = Path("/tmp/cadre_official.tar.gz")
    subprocess.run(["tar", "-czf", str(cadre_tar), "-C", str(ROOT_DIR), "cadre_official"], check=True)
    run_colab(["upload", "-s", SESSION_NAME, str(cadre_tar), "/content/cadre_official.tar.gz"])
    run_colab(["upload", "-s", SESSION_NAME, str(RUN_SCRIPT), "/content/run_cadre_v29_adaptive_slack.py"])

    print("=== Step 3: Checking / Uploading Bundles and New Retrieval Trace ===", flush=True)
    check_py = STUDY_DIR / "_check_remote_v29.py"
    check_py.write_text("""
import os
print("bundle:", os.path.exists("/content/full_benchmark_8004_bundle.tar.gz"))
print("trace_rerank:", os.path.exists("/content/cadre_v29_rerank_trace.jsonl.gz"))
try:
    import vllm
    print("setup_done: True")
except Exception:
    print("setup_done: False")
""")
    res_chk = run_colab(["exec", "-s", SESSION_NAME, "-f", str(check_py), "--timeout", "60"], check=False)
    has_bundle = "bundle: True" in res_chk.stdout
    has_trace = "trace_rerank: True" in res_chk.stdout
    setup_done = "setup_done: True" in res_chk.stdout

    if not has_bundle:
        print(f"Uploading {BUNDLE_PATH.name} ({BUNDLE_PATH.stat().st_size / 1024 / 1024:.1f} MB)...", flush=True)
        upload_large_file(SESSION_NAME, BUNDLE_PATH, "/content/full_benchmark_8004_bundle.tar.gz")

    if not has_trace:
        print(f"Uploading new retrieval trace ({NEW_TRACE_PATH.stat().st_size / 1024 / 1024:.1f} MB)...", flush=True)
        upload_large_file(SESSION_NAME, NEW_TRACE_PATH, "/content/cadre_v29_rerank_trace.jsonl.gz")

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

    print("=== Step 6: Launching CADRE v2.9 Execution on A100 GPU ===", flush=True)
    smoke_arg = "--smoke" if args.smoke else ""
    remote_runner_py = STUDY_DIR / "_remote_runner_v29.py"
    remote_runner_py.write_text(f"""
import os, sys, subprocess
sys.path.insert(0, "/content/cadre_official")
cmd = [
    "python3", "-u", "/content/run_cadre_v29_adaptive_slack.py",
    "--data-root", "/content",
    "--trace-path", "/content/cadre_v29_rerank_trace.jsonl.gz",
    "--output-dir", "/content/cadre-results/cadre_v29_new_rerank",
    "--generator-batch-size", "128",
    "--gpu-memory-utilization", "0.80",
    "--max-model-len", "3072",
]
if "{smoke_arg}":
    cmd.append("{smoke_arg}")

print("Executing:", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH="/content/cadre_official"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()
if proc.returncode != 0:
    raise RuntimeError(f"CADRE v2.9 execution failed with code {{proc.returncode}}")
print("Evaluation finished with code:", proc.returncode, flush=True)
""")

    ret = stream_colab_file(SESSION_NAME, remote_runner_py, timeout=7200)
    if ret != 0:
        raise RuntimeError(f"Remote evaluation failed with code {ret}")

    print("=== Step 7: Archiving and Downloading Results ===", flush=True)
    dl_script = STUDY_DIR / "_dl_v29_results.py"
    dl_script.write_text("""
import shutil, os
shutil.make_archive("/content/cadre-results/cadre_v29_new_rerank", 'tar', root_dir="/content/cadre-results", base_dir="cadre_v29_new_rerank")
print("Archive created.", flush=True)
""")
    run_colab(["exec", "-s", SESSION_NAME, "-f", str(dl_script), "--timeout", "120"])
    
    # Download files
    run_colab(["download", "-s", SESSION_NAME, "/content/cadre-results/cadre_v29_new_rerank/cadre_v29_new_rerank_metrics.json", str(LOCAL_OUT / "cadre_v29_new_rerank_metrics.json")], check=False)
    run_colab(["download", "-s", SESSION_NAME, "/content/cadre-results/cadre_v29_new_rerank.tar", str(LOCAL_OUT / "cadre_v29_new_rerank.tar")])
    subprocess.run(["tar", "-xf", str(LOCAL_OUT / "cadre_v29_new_rerank.tar"), "-C", str(STUDY_DIR / "colab_results")], check=True)
    print(f"\n✓ Successfully downloaded and unpacked results to {LOCAL_OUT}!", flush=True)


if __name__ == "__main__":
    main()
