#!/usr/bin/env bash

# Eval-only Slurm entrypoint.  Training is deliberately disabled here.
# Override paths/model-cache settings with --export=ALL,... when needed.
#SBATCH --job-name=icar-eval
#SBATCH --comment="ICAR Co-Retrieval evaluation"
#SBATCH --partition=defq
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" && -n "${SLURM_SUBMIT_DIR:-}" && \
      -f "${SLURM_SUBMIT_DIR}/src/scripts/run_icar_lipo_a100_80gb.sh" ]]; then
  REPO_DIR="${SLURM_SUBMIT_DIR}"
else
  REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
fi
SCRIPT_DIR="${REPO_DIR}/src/scripts"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${REPO_DIR}"

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

# Eval-only and OOM-safe defaults.
export RUN_DOWNLOAD=0
export RUN_TRAIN=0
export RUN_EVAL=1
export DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/data}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/checkpoints/icar_lipo_deepseek_6p7b_a100_80gb}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/results/icar_lipo_deepseek_6p7b_a100_80gb}"
export LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/icar_lipo_deepseek_6p7b_a100_80gb}"
export EVAL_INDEX_MODE="${EVAL_INDEX_MODE:-sharded}"
export BUILD_EVAL_INDEX="${BUILD_EVAL_INDEX:-1}"
export EVAL_RETRIEVER_DEVICE="${EVAL_RETRIEVER_DEVICE:-cpu}"
export EVAL_INDEX_SHARD_SIZE="${EVAL_INDEX_SHARD_SIZE:-50000}"
export EVAL_MAX_CONTEXT_TOKENS="${EVAL_MAX_CONTEXT_TOKENS:-3072}"
export EVAL_SKIP_NLL="${EVAL_SKIP_NLL:-1}"
export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
export BATCH_ENCODE_SIZE="${BATCH_ENCODE_SIZE:-32}"
export INCLUDE_ANALYSIS="${INCLUDE_ANALYSIS:-0}"
export INCLUDE_POLICY_VARIANTS="${INCLUDE_POLICY_VARIANTS:-0}"
export LEAVE_ONE_OUT_ANALYSIS_SAMPLES="${LEAVE_ONE_OUT_ANALYSIS_SAMPLES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing to start an A100 eval job." >&2
  exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/run_icar_lipo_a100_80gb.sh" ]]; then
  echo "Invalid REPO_DIR: ${REPO_DIR}" >&2
  echo "Expected ${SCRIPT_DIR}/run_icar_lipo_a100_80gb.sh" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
echo "Eval-only job on $(hostname)"
echo "Repo: ${REPO_DIR}"
echo "Data: ${DATA_DIR}"
echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Context: ${EVAL_MAX_CONTEXT_TOKENS}; index: ${EVAL_INDEX_MODE}; build: ${BUILD_EVAL_INDEX}"
nvidia-smi -L

"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))'

exec bash "${SCRIPT_DIR}/run_icar_lipo_a100_80gb.sh"
