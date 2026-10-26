#!/usr/bin/env python3
"""Small dependency-free status probe for the detached Colab pilot."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


root = Path("/content/graph_dependency_pilot")
pid_file = root / "runner.pid"
log_file = root / "runner.log"

if pid_file.exists():
    pid_text = pid_file.read_text(encoding="utf-8").strip()
    if pid_text.isdigit():
        try:
            os.kill(int(pid_text), 0)
            alive = True
        except ProcessLookupError:
            alive = False
        proc_cmdline = Path(f"/proc/{pid_text}/cmdline")
        command = (proc_cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
                   if proc_cmdline.is_file() else "")
        is_runner = "/content/colab_runner.py" in command
        state_file = Path(f"/proc/{pid_text}/status")
        process_state = ""
        if state_file.is_file():
            for line in state_file.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("State:"):
                    process_state = line.split(":", 1)[1].strip()
                    break
        status = "running" if alive and is_runner and not process_state.startswith("Z") else "not running"
        print(f"Runner PID {pid_text}: {status}; state={process_state or 'unknown'}; command={command[:240]}",
              flush=True)
else:
    print("Runner PID file is missing.", flush=True)

print("\n--- GPU ---", flush=True)
gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
     "--format=csv,noheader"],
    check=False,
    capture_output=True,
    text=True,
)
print(gpu.stdout or gpu.stderr or f"nvidia-smi exit={gpu.returncode}", flush=True)

print("\n--- Recent runner log ---", flush=True)
if log_file.exists():
    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(lines[-80:]), flush=True)
else:
    print("Runner log is not present yet.", flush=True)

print("\n--- Output files ---", flush=True)
output_dir = root / "output"
if output_dir.exists():
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"{path.name}\t{path.stat().st_size} bytes", flush=True)
else:
    print("Output directory is not present yet.", flush=True)
