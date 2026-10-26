import json
from pathlib import Path

cmd_file = Path("/content/mb_spt/next_cmd.json")
cmd_file.write_text(
    json.dumps({
        "action": "run_script",
        "script": "/content/run_cadre_pilot.py",
        "tag": "cadre_pilot_256",
        "subset_per_benchmark": 64,
    }),
    encoding="utf-8"
)
print("Submitted cadre_pilot_256 job to daemon!", flush=True)
