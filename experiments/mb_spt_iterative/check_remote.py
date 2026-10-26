import subprocess
res = subprocess.run(["ls", "-lh", "/content"], capture_output=True, text=True)
print(res.stdout)
res2 = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(res2.stdout)
