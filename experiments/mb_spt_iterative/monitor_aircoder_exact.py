#!/usr/bin/env python3
"""Start daemon, submit exact AIRCoder_r full benchmark, and monitor until completion."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path("/content/mb_spt")
CMD_FILE = ROOT_DIR / "next_cmd.json"
STATUS_FILE = ROOT_DIR / "daemon_status.json"
TAG = "aircoder_r_exact_8004_v2"
METRICS_FILE = ROOT_DIR / "outputs" / f"{TAG}_metrics.json"
RUNNING_METRICS_FILE = ROOT_DIR / "outputs" / f"{TAG}_running_metrics.json"

sys.path.insert(0, "/content")
import submit_job

print("=== STARTING EXACT AIRCODER_R BENCHMARK WORKFLOW ===", flush=True)

# 1. Start daemon if not running
submit_job.ensure_daemon()

# 2. Queue the job
cmd_payload = {
    "action": "run_script",
    "script": "/content/run_aircoder_exact_full.py",
    "tag": TAG,
    "subset_per_benchmark": 8004,
}
CMD_FILE.write_text(json.dumps(cmd_payload, indent=2) + "\n", encoding="utf-8")
print(f"Queued job tag={TAG} in {CMD_FILE}...", flush=True)

# 3. Monitor loop to keep kernel busy (prevents Colab idle timeout)
t0 = time.time()
last_count = 0
while True:
    time.sleep(15)
    elapsed = time.time() - t0

    # Check if final metrics file exists
    if METRICS_FILE.exists():
        print(f"\n[OK] Exact AIRCoder_r benchmark completed in {elapsed/60:.2f} minutes!", flush=True)
        try:
            m = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
            print("Final overall summary:\n", json.dumps(m, indent=2), flush=True)
        except Exception:
            pass
        break

    # Check running metrics
    if RUNNING_METRICS_FILE.exists():
        try:
            rm = json.loads(RUNNING_METRICS_FILE.read_text(encoding="utf-8"))
            cur_count = rm.get("processed_tasks", 0)
            if cur_count != last_count:
                last_count = cur_count
                strict_em = rm.get("aligncoder_strict_running_em", 0.0)
                official_em = rm.get("aircoder_official_running_em", 0.0)
                print(f"[{elapsed/60:.1f}m] Progress: {cur_count}/8004 tasks | Strict EM: {strict_em}% | AIRCoder Official EM: {official_em}%", flush=True)
        except Exception:
            pass

    # Check daemon status
    if STATUS_FILE.exists():
        try:
            st = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
            if st.get("last_status") == "error":
                print(f"\n[ERROR] Daemon reported error: {st.get('error')}", flush=True)
                print(st.get("traceback", ""), flush=True)
                sys.exit(1)
        except Exception:
            pass

print("=== WORKFLOW FINISHED SUCCESSFULLY ===", flush=True)
