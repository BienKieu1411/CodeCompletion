import subprocess
import sys

print("=== Launching run_cadre_26_final_pilot.py on Colab GPU A100 ===", flush=True)
res = subprocess.run([sys.executable, "/content/run_cadre_26_final_pilot.py"])
if res.returncode != 0:
    print(f"FAILED with code {res.returncode}", flush=True)
    sys.exit(res.returncode)
print("=== Finished successfully! ===", flush=True)
