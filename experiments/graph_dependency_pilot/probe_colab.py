#!/usr/bin/env python3
"""Print Colab accelerator/runtime details before installing vLLM."""

import platform
import subprocess
import sys

import torch

print(f"Python: {sys.version}", flush=True)
print(f"Platform: {platform.platform()}", flush=True)
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {properties.name}; VRAM: {properties.total_memory / (1024 ** 3):.1f} GiB", flush=True)
subprocess.run(["nvidia-smi"], check=False)
