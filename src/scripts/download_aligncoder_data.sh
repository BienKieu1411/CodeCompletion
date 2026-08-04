#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SRC_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORK_DIR="${WORK_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-${WORK_DIR}/data}"
HF_DATASET_REPO="${HF_DATASET_REPO:-AlignCoder/Data4AlignCoder}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${DATA_DIR}"

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

"${PYTHON_BIN}" - <<PY
from pathlib import Path
from huggingface_hub import snapshot_download

data_dir = Path("${DATA_DIR}").expanduser().resolve()
repo_id = "${HF_DATASET_REPO}"

local_dir = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(data_dir),
    local_dir_use_symlinks=False,
)
print(f"Dataset downloaded to: {local_dir}")
PY

find "${DATA_DIR}" -maxdepth 3 -type f | sort | sed -n '1,80p'
