import subprocess
from pathlib import Path
for cmd in (["ps", "-o", "pid,etime,time,%cpu,%mem,stat,wchan:24,cmd", "-p", "3666,5132"], ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv"]):
    p = subprocess.run(cmd, text=True, capture_output=True)
    print("$", " ".join(cmd), "\n", p.stdout, p.stderr)
root = Path('/content/rlcoder_dependency_graph/outputs')
if root.exists():
    for p in sorted(root.iterdir()):
        print(p.name, p.stat().st_size if p.is_file() else 'dir')
