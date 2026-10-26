#!/usr/bin/env python3
"""Ensure colab_daemon.py is running on Colab and submit/wait for an iteration job."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path("/content/mb_spt")
DAEMON_PATH = Path("/content/colab_daemon.py")
PID_FILE = ROOT_DIR / "daemon.pid"
LOG_FILE = ROOT_DIR / "daemon.log"
CMD_FILE = ROOT_DIR / "next_cmd.json"
STATUS_FILE = ROOT_DIR / "daemon_status.json"


def daemon_alive() -> bool:
    if not PID_FILE.exists():
        return False
    txt = PID_FILE.read_text(encoding="utf-8").strip()
    if not txt.isdigit():
        return False
    pid = int(txt)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    cmdline_file = Path(f"/proc/{pid}/cmdline")
    if cmdline_file.is_file():
        cmd = cmdline_file.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        return "colab_daemon.py" in cmd
    return False


def ensure_daemon() -> None:
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    if daemon_alive():
        return
    print("Starting colab_daemon.py...", flush=True)
    if STATUS_FILE.exists():
        STATUS_FILE.unlink()
    with LOG_FILE.open("ab", buffering=0) as log:
        proc = subprocess.Popen(
            [sys.executable, str(DAEMON_PATH)],
            cwd="/content",
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    PID_FILE.write_text(f"{proc.pid}\n", encoding="utf-8")
    print(f"Launched daemon PID={proc.pid}", flush=True)


def wait_until_ready_or_idle(target_tag: str | None = None, timeout: float = 420.0) -> dict:
    t0 = time.time()
    last_state = None
    while time.time() - t0 < timeout:
        if not daemon_alive():
            tail = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-40:] if LOG_FILE.exists() else []
            raise RuntimeError("Daemon died!\n" + "\n".join(tail))
        if STATUS_FILE.exists():
            try:
                st = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
            except Exception:
                st = {}
            state = st.get("state")
            if state != last_state:
                print(f"[{time.time() - t0:5.1f}s] Daemon state: {state}", flush=True)
                last_state = state
            if target_tag is None and state in ("ready", "idle"):
                return st
            if target_tag is not None and state == "idle" and st.get("last_tag") == target_tag:
                return st
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for daemon (target_tag={target_tag})")


def main() -> None:
    ensure_daemon()
    wait_until_ready_or_idle(target_tag=None, timeout=420.0)
    script = os.environ.get("JOB_SCRIPT", "/content/run_full_benchmark.py")
    tag = os.environ.get("JOB_TAG", "cadre_vs_aircoder_full_8004_v2")
    subset = int(os.environ.get("JOB_SUBSET", "8004"))
    CMD_FILE.write_text(
        json.dumps({"action": "run_script", "script": script, "tag": tag, "subset_per_benchmark": subset}),
        encoding="utf-8",
    )
    print(f"Submitted job tag={tag} (subset={subset}/benchmark)...", flush=True)
    st = wait_until_ready_or_idle(target_tag=tag, timeout=2400.0)
    if st.get("last_status") != "ok":
        print("Job failed:", st.get("traceback", st.get("error")), flush=True)
        sys.exit(1)
    metrics_path = ROOT_DIR / "outputs" / f"{tag}_metrics.json"
    print("\n=== JOB METRICS ===", flush=True)
    print(metrics_path.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
