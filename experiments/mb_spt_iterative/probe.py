import subprocess
from pathlib import Path

status_file = Path("/content/mb_spt/daemon_status.json")
log_file = Path("/content/mb_spt/daemon.log")

if status_file.exists():
    print("STATUS:\n" + status_file.read_text(encoding="utf-8"))
else:
    print("STATUS: None")

if log_file.exists():
    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    print("LOG LAST 10 LINES:\n" + "\n".join(lines[-10:]))

ps = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
py_procs = [l for l in ps.stdout.splitlines() if any(k in l for k in ["python", "ninja", "c++", "nvcc", "colab_daemon"])]
print("\nCOMPILER / PYTHON PROCESSES:")
for p in py_procs:
    print(p[:120])
