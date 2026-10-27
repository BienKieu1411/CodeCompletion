import subprocess
import os

print("--- 1. Extracting cadre_official ---")
subprocess.check_call(["tar", "-xzf", "/content/cadre_official.tar.gz", "-C", "/content"])

print("--- 2. Installing uv & vLLM & Dependencies ---")
subprocess.check_call(["pip", "install", "-q", "uv"])
subprocess.check_call(["uv", "pip", "install", "--system", "vllm", "--torch-backend=auto"])
subprocess.call(["pip", "uninstall", "-y", "torchaudio"])
subprocess.check_call(["pip", "install", "-q", "fuzzywuzzy", "python-Levenshtein", "editdistance", "tree-sitter==0.20.4"])

print("--- Setup Complete! Ready for kernel restart ---")
