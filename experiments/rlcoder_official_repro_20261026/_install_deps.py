
import subprocess, sys

print('Starting pip install vllm...', flush=True)
cmd = [sys.executable, '-m', 'pip', 'install', 'vllm', 'fuzzywuzzy', 'python-Levenshtein', 'editdistance', 'tree-sitter==0.20.4', 'huggingface_hub', 'transformers', 'accelerate']
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    if 'Successfully' in line or 'Installing' in line or 'Downloading' in line:
        print(line.strip(), flush=True)
proc.wait()
print('Pip return code:', proc.returncode, flush=True)
