import json
import subprocess
import time
from pathlib import Path

COLAB_BIN = "/Users/kieugiangbien/bienkieu_env/bin/colab"
cmd = {
    "action": "run_script",
    "script": "/content/run_full_benchmark.py",
    "tag": "cadre_vs_aircoder_full_8004",
}

tmp_cmd = Path("/tmp/next_cmd.json")
tmp_cmd.write_text(json.dumps(cmd), encoding="utf-8")

print("Uploading next_cmd.json via REST...", flush=True)
subprocess.check_call([COLAB_BIN, "upload", "-s", "cadre", str(tmp_cmd), "/content/mb_spt/next_cmd.json"])
print("Job submitted successfully via REST!", flush=True)
