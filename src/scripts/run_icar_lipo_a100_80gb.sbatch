#!/bin/bash

#SBATCH --job-name=icar-lipo
#SBATCH --comment="ICAR Co-Retrieval LiPO"
#SBATCH --partition=defq
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodelist=dgx02,dgx01
#SBATCH --time=12-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${REPO_DIR}"

# Site-specific environment setup is intentionally injected at submit time.
# Example: --export=ALL,ENV_SETUP_SCRIPT=/path/to/activate_icar.sh
if [[ -n "${ENV_SETUP_SCRIPT:-}" ]]; then
  if [[ ! -f "${ENV_SETUP_SCRIPT}" ]]; then
    echo "ENV_SETUP_SCRIPT does not exist: ${ENV_SETUP_SCRIPT}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${ENV_SETUP_SCRIPT}"
fi

if [[ -n "${MODULES:-}" ]]; then
  if ! type module >/dev/null 2>&1; then
    echo "MODULES was set but the environment has no 'module' command." >&2
    exit 1
  fi
  read -r -a module_list <<< "${MODULES}"
  for module_name in "${module_list[@]}"; do
    module load "${module_name}"
  done
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHON_BIN
export WORK_DIR="${WORK_DIR:-${REPO_DIR}}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing to start an A100 job." >&2
  exit 1
fi

echo "Host: $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Python: ${PYTHON_BIN}"
nvidia-smi -L

"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))'

mkdir -p "${REPO_DIR}/logs/slurm"
exec bash "${SCRIPT_DIR}/run_icar_lipo_a100_80gb.sh"
