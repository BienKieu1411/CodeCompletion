import subprocess
import sys

print("=== Starting CADRE-SE Enhanced Pilot Execution on Colab GPU A100 ===", flush=True)
res = subprocess.run([sys.executable, "/content/run_cadre_se_enhanced_pilot.py"])
if res.returncode != 0:
    print(f"FAILED with returncode {res.returncode}", flush=True)
    sys.exit(res.returncode)
print("=== Execution Finished Successfully! ===", flush=True)
