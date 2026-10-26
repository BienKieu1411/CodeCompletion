#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$(pwd)}"

ABLATION_MODES_DEFAULT="retriever_only sequential_retriever_first"
ABLATION_MODES="${ABLATION_MODES:-${ABLATION_MODES_DEFAULT}}"

BASE_CHECKPOINT_ROOT="${BASE_CHECKPOINT_ROOT:-${WORK_DIR}/checkpoints/icar_lipo_ablation_a100_80gb}"
BASE_LOG_ROOT="${BASE_LOG_ROOT:-${WORK_DIR}/logs/icar_lipo_ablation_a100_80gb}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-${WORK_DIR}/results/icar_lipo_ablation_a100_80gb}"

RUN_DOWNLOAD_ONCE="${RUN_DOWNLOAD_ONCE:-0}"

first=1
for mode in ${ABLATION_MODES}; do
  echo "=== ICAR ablation mode: ${mode} ==="
  run_download=0
  if [[ "${first}" == "1" && "${RUN_DOWNLOAD_ONCE}" == "1" ]]; then
    run_download=1
  fi
  first=0

  adapter_type=none
  gate_mode=always_retrieve
  if [[ "${mode}" == "sequential_retriever_first" ]]; then
    # Second pass: add the soft prompt after the retriever-only result is
    # established, keeping the retriever protocol itself unchanged.
    adapter_type=soft_prompt
  fi

  RUN_DOWNLOAD="${run_download}" \
  RUN_TRAIN="${RUN_TRAIN:-1}" \
  RUN_EVAL="${RUN_EVAL:-1}" \
  EXPERIMENT_MODE="${mode}" \
  ADAPTER_TYPE="${adapter_type}" \
  GATE_MODE="${gate_mode}" \
  CHECKPOINT_DIR="${BASE_CHECKPOINT_ROOT}/${mode}" \
  LOG_DIR="${BASE_LOG_ROOT}/${mode}" \
  OUTPUT_ROOT="${BASE_OUTPUT_ROOT}/${mode}" \
    bash "${SCRIPT_DIR}/run_icar_lipo_a100_80gb.sh"
done

echo "Ablation outputs written under: ${BASE_OUTPUT_ROOT}"
