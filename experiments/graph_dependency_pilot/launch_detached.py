#!/usr/bin/env python3
"""Launch the Colab pilot as a detached process and persist its PID/log path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


root = Path("/content/graph_dependency_pilot")
runner = Path("/content/colab_runner.py")
root.mkdir(parents=True, exist_ok=True)
if not runner.is_file():
    raise FileNotFoundError(f"Upload the runner first: {runner}")

log_path = root / "runner.log"
pid_path = root / "runner.pid"
if pid_path.exists():
    old_pid_text = pid_path.read_text(encoding="utf-8").strip()
    if old_pid_text.isdigit():
        proc_cmdline = Path(f"/proc/{old_pid_text}/cmdline")
        old_command = ""
        if proc_cmdline.is_file():
            old_command = proc_cmdline.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        if str(runner) in old_command:
            raise RuntimeError(f"A pilot process is already running with PID {old_pid_text}")
        try:
            os.kill(int(old_pid_text), 0)
        except ProcessLookupError:
            pass
        else:
            # A stale PID may have been reused by the notebook kernel or another
            # runtime process. Only treat it as this pilot when cmdline matches.
            pid_path.unlink()

with log_path.open("ab", buffering=0) as log:
    process = subprocess.Popen(
        [sys.executable, str(runner)],
        cwd=str(root),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )

pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
print(json.dumps({"status": "launched", "pid": process.pid,
                  "log": str(log_path), "runner": str(runner)}), flush=True)
