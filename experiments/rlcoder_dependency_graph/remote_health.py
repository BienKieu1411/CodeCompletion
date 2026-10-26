import subprocess
for cmd in (["nvidia-smi"], ["ps", "-o", "pid,etime,%cpu,%mem,cmd", "-p", "3666,5132"]):
    p = subprocess.run(cmd, text=True, capture_output=True)
    print("$", " ".join(cmd), "\n", p.stdout, p.stderr)
