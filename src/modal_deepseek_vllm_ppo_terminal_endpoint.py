"""Modal vLLM endpoint dedicated to terminal-penalty PPO training.

Deploy with the project's environment:
    /Users/kieugiangbien/bienkieu_env/bin/modal deploy \
        src/modal_deepseek_vllm_ppo_terminal_endpoint.py

The Kaggle notebook sends authenticated OpenAI-compatible completion requests.
The A100 scales to zero after five idle minutes.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request

import modal


MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
SERVED_MODEL_NAME = "deepseek-coder-1.3b-base"
VLLM_VERSION = "0.21.0"
INPUT_TOKENS = 3072
OUTPUT_TOKENS = 96
MAX_MODEL_LEN = INPUT_TOKENS + OUTPUT_TOKENS
MAX_NUM_SEQS = 108
MAX_NUM_BATCHED_TOKENS = 16384
GPU_MEMORY_UTILIZATION = 0.90
PORT = 8000
STARTUP_TIMEOUT_SECONDS = 15 * 60
SCALEDOWN_WINDOW_SECONDS = 5 * 60


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM_VERSION}")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("bienkieu-ppo-terminal-badpick-deepseek-vllm")


@app.server(
    image=image,
    gpu="A100-80GB",
    cpu=0.5,
    memory=8192,
    port=PORT,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    routing_region="us-east",
    # Modal proxy authentication stays enabled; the Kaggle notebook sends the
    # user's MODAL_PROXY_TOKEN secret as an Authorization: Bearer credential.
)
class DeepSeekVLLMServer:
    @modal.enter()
    def start(self) -> None:
        command = [
            "vllm", "serve", MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--served-model-name", SERVED_MODEL_NAME,
            "--dtype", "float16",
            "--tensor-parallel-size", "1",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--max-num-seqs", str(MAX_NUM_SEQS),
            "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
            "--no-enforce-eager",
        ]
        print(
            "Launching terminal-PPO vLLM server:",
            {"model": MODEL_NAME, "max_model_len": MAX_MODEL_LEN,
             "max_num_seqs": MAX_NUM_SEQS,
             "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
             "gpu_memory_utilization": GPU_MEMORY_UTILIZATION},
            flush=True,
        )
        self.process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited during startup with code {self.process.returncode}."
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/health", timeout=3
                ) as response:
                    if response.status == 200:
                        print("vLLM health check passed; Modal server is ready.",
                              flush=True)
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(3)
        raise TimeoutError("vLLM did not become healthy before startup timeout.")

    @modal.exit()
    def stop(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
