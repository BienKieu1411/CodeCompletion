import subprocess
import sys

res = subprocess.run([sys.executable, "/content/run_cadre_unified_se_pilot.py"], capture_output=True, text=True)
print("=== STDOUT ===")
print(res.stdout[-3000:] if res.stdout else "EMPTY")
print("=== STDERR ===")
print(res.stderr[-3000:] if res.stderr else "EMPTY")
print("Returncode:", res.returncode)
