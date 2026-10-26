import json
from pathlib import Path
import os

STATUS_FILE = Path("/content/mb_spt/daemon_status.json")
LOG_FILE = Path("/content/mb_spt/daemon.log")
PID_FILE = Path("/content/mb_spt/daemon.pid")
OUT_DIR = Path("/content/mb_spt/outputs")

print("PID file exists:", PID_FILE.exists())
if PID_FILE.exists():
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, 0)
        print(f"Daemon PID {pid} is ALIVE!")
    except ProcessLookupError:
        print(f"Daemon PID {pid} is DEAD.")

print("\nStatus file exists:", STATUS_FILE.exists())
if STATUS_FILE.exists():
    print("Status content:", STATUS_FILE.read_text())

print("\nLog tail (last 30 lines):")
if LOG_FILE.exists():
    lines = LOG_FILE.read_text(errors="replace").splitlines()
    for l in lines[-30:]:
        print(l)

print("\nOutputs dir contents:")
if OUT_DIR.exists():
    for f in OUT_DIR.iterdir():
        print(f.name, f.stat().st_size)
