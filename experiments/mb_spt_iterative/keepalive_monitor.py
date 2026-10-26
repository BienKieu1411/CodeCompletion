#!/usr/bin/env python3
"""
Keepalive and Progress Monitor for Colab session cadre-4k-ablation.
- Sends heartbeat execution every 30 seconds to prevent Colab idle timeout.
- Syncs run_4k_full_left.log locally.
- When metrics.json is generated:
    * Downloads metrics.json
    * Downloads predictions.jsonl.gz
    * Downloads traces
    * Stops session cadre-4k-ablation to conserve compute units.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

COLAB_BIN = "/Users/kieugiangbien/bienkieu_env/bin/colab"
SESSION = "cadre-4k-ablation"
LOCAL_DIR = Path("experiments/mb_spt_iterative")
LOG_PATH = LOCAL_DIR / "run_4k_full_left.log"
REMOTE_LOG = "/content/mb_spt/run_4k_full_left.log"
REMOTE_OUT = "/content/mb_spt/cadre_4k_eval_results"
REMOTE_METRICS = f"{REMOTE_OUT}/cadre_4k_full_left_context_eval_8004_metrics.json"

print(f"=== KEEPALIVE & PROGRESS MONITOR STARTED FOR {SESSION} ===", flush=True)

iter_count = 0
while True:
    iter_count += 1
    time.sleep(20)

    # 1. Download latest log
    try:
        subprocess.run(
            [COLAB_BIN, "download", "-s", SESSION, REMOTE_LOG, str(LOG_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Log download error: {e}", flush=True)

    # 2. Check if metrics file exists
    check_code = f"import os; print('EXISTS' if os.path.exists('{REMOTE_METRICS}') else 'RUNNING')"
    try:
        res = subprocess.run(
            [COLAB_BIN, "exec", "-s", SESSION, "--timeout", "20"],
            input=check_code,
            capture_output=True,
            text=True,
            timeout=25,
        )
        status = res.stdout.strip()
    except Exception as e:
        status = f"ERR: {e}"

    # Print log tail summary every minute (every 3 iterations)
    if iter_count % 3 == 0:
        tail = ""
        if LOG_PATH.exists():
            lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
            tail = lines[-1] if lines else ""
        print(f"[{time.strftime('%H:%M:%S')}] Heartbeat #{iter_count} | Status: {status} | Last log: {tail}", flush=True)

    if "EXISTS" in status:
        print(f"\n[SUCCESS] Final metrics file detected on Colab at {time.strftime('%H:%M:%S')}!", flush=True)
        # Download results
        files_to_download = [
            (REMOTE_METRICS, LOCAL_DIR / "cadre_4k_full_left_context_eval_8004_metrics.json"),
            (f"{REMOTE_OUT}/cadre_4k_full_left_context_eval_8004_predictions.jsonl.gz", LOCAL_DIR / "cadre_4k_full_left_context_eval_8004_predictions.jsonl.gz"),
            (f"{REMOTE_OUT}/rlcoder_full_left_standard_8004_trace.jsonl.gz", LOCAL_DIR / "rlcoder_full_left_standard_8004_trace.jsonl.gz"),
            (f"{REMOTE_OUT}/rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz", LOCAL_DIR / "rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz"),
            (REMOTE_LOG, LOCAL_DIR / "run_4k_full_left.log"),
        ]
        for rem, loc in files_to_download:
            try:
                subprocess.run([COLAB_BIN, "download", "-s", SESSION, rem, str(loc)], check=True)
                print(f"  ✓ Downloaded {loc.name}", flush=True)
            except Exception as e:
                print(f"  ✗ Failed to download {rem}: {e}", flush=True)

        # Stop VM immediately to conserve compute units
        print(f"\nStopping session {SESSION} to conserve compute units...", flush=True)
        subprocess.run([COLAB_BIN, "stop", "-s", SESSION])
        print("Session stopped. Monitor finished cleanly!", flush=True)
        sys.exit(0)
