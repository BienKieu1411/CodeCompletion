from pathlib import Path
import json

OUT_DIR = Path("/content/mb_spt/outputs")
STATUS_FILE = Path("/content/mb_spt/daemon_status.json")
LOG_FILE = Path("/content/mb_spt/daemon.log")

if STATUS_FILE.exists():
    print("Status:", STATUS_FILE.read_text())

if OUT_DIR.exists():
    print("Outputs:")
    for f in OUT_DIR.iterdir():
        print(" ", f.name, f.stat().st_size)

if LOG_FILE.exists():
    lines = LOG_FILE.read_text(errors="replace").splitlines()
    print("\nLog tail (last 20 lines):")
    for l in lines[-20:]:
        print(" ", l)
