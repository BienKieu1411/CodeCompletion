#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SRC_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
WORK_DIR="${WORK_DIR:-$(pwd)}"

DATA_DIR="${DATA_DIR:-${WORK_DIR}/data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORK_DIR}/checkpoints/icar_lipo_deepseek_6p7b_a100_80gb}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs/icar_lipo_deepseek_6p7b_a100_80gb}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}/results/icar_lipo_deepseek_6p7b_a100_80gb}"

TRAIN_DATASETS_DEFAULT="${DATA_DIR}/github_repos/python/train.parquet,${DATA_DIR}/github_repos/java/train.parquet"
TRAIN_DATASETS="${TRAIN_DATASETS:-${TRAIN_DATASET:-${TRAIN_DATASETS_DEFAULT}}}"
GENERATOR_NAME="${GENERATOR_NAME:-deepseek-ai/deepseek-coder-6.7b-base}"
EXPERIMENT_MODE="${EXPERIMENT_MODE:-intent_main}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
COMPLETION_LEVEL="${COMPLETION_LEVEL:-mixed}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-${NUM_EPOCHS:-10}}"
EPOCH_BUDGET_MODE="${EPOCH_BUDGET_MODE:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
STEPS_PER_ROUND_PROMPT="${STEPS_PER_ROUND_PROMPT:-100}"
STEPS_PER_ROUND_RETRIEVER="${STEPS_PER_ROUND_RETRIEVER:-200}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_DOWNLOAD="${RUN_DOWNLOAD:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
INCLUDE_POLICY_VARIANTS="${INCLUDE_POLICY_VARIANTS:-1}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${OUTPUT_ROOT}"

if [[ "${RUN_DOWNLOAD}" == "1" ]]; then
  "${SCRIPT_DIR}/download_aligncoder_data.sh"
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  TRAIN_BUDGET_ARGS=()
  if [[ "${EPOCH_BUDGET_MODE}" == "1" ]]; then
    TRAIN_BUDGET_ARGS+=(--epoch-budget-mode)
  fi

  IFS=',' read -r -a TRAIN_DATASET_LIST <<< "${TRAIN_DATASETS}"
  for train_dataset_path in "${TRAIN_DATASET_LIST[@]}"; do
    if [[ ! -f "${train_dataset_path}" ]]; then
      echo "Missing train dataset: ${train_dataset_path}" >&2
      echo "Configured TRAIN_DATASETS=${TRAIN_DATASETS}" >&2
      echo "Run with RUN_DOWNLOAD=1 or run download_aligncoder_data.sh first." >&2
      exit 1
    fi
  done

  if [[ "${#TRAIN_DATASET_LIST[@]}" -eq 0 ]]; then
    echo "No train datasets configured." >&2
    echo "Run with RUN_DOWNLOAD=1 or run download_aligncoder_data.sh first." >&2
    exit 1
  fi

  echo "Training ${EXPERIMENT_MODE} for TRAIN_EPOCHS=${TRAIN_EPOCHS} (epoch_budget_mode=${EPOCH_BUDGET_MODE})"
  echo "Completion level: ${COMPLETION_LEVEL}"
  echo "Train datasets: ${TRAIN_DATASETS}"

  python3 -m co_retrieval.cli.co_retrieval_cli train \
    --use-neural \
    --skip-train-eval \
    --dataset-path "${TRAIN_DATASETS}" \
    --output-dir "${OUTPUT_ROOT}/train" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --log-dir "${LOG_DIR}/train" \
    --encoder-name jinaai/jina-code-embeddings-1.5b \
    --generator-name "${GENERATOR_NAME}" \
    --experiment-mode "${EXPERIMENT_MODE}" \
    --intent-mode static \
    --gate-mode learned \
    --adapter-type soft_prompt \
    --retriever-loss lipo \
    --lipo-tau 1.0 \
    --num-epochs "${TRAIN_EPOCHS}" \
    --train-epochs "${TRAIN_EPOCHS}" \
    "${TRAIN_BUDGET_ARGS[@]}" \
    --max-samples "${MAX_TRAIN_SAMPLES}" \
    --completion-level "${COMPLETION_LEVEL}" \
    --fixed-train-size "${MAX_TRAIN_SAMPLES}" \
    --max-train-samples "${MAX_TRAIN_SAMPLES}" \
    --eval-ratio 0 \
    --max-eval-samples 0 \
    --warmup-steps "${WARMUP_STEPS}" \
    --steps-per-round-prompt "${STEPS_PER_ROUND_PROMPT}" \
    --steps-per-round-retriever "${STEPS_PER_ROUND_RETRIEVER}" \
    --top-k 3 \
    --preference-pool-top-k 20 \
    --max-pairs-per-sample 4 \
    --num-hard-negatives 10 \
    --utility-margin 0.05 \
    --preference-margin 0.1 \
    --num-prompt-tokens 64 \
    --max-context-tokens 4096 \
    --encoder-max-length 512 \
    --gate-hidden-dim 384 \
    --gate-entropy-weight 0.01 \
    --gate-context-cost-weight 0.01 \
    --gate-context-cost-token-unit 512 \
    --gate-decision-threshold 0.5 \
    --gate-calibration-samples 256 \
    --gate-calibration-retrieval-penalty 0.05 \
    --leave-one-out-analysis-samples 0 \
    --retriever-lr 2e-5 \
    --gate-lr 1e-4 \
    --soft-prompt-lr 5e-3 \
    --grad-clip-norm 1.0 \
    --batch-encode-size 64 \
    --batch-size 4 \
    --max-new-tokens 128 \
    --generator-dtype bfloat16 \
    --device cuda \
    2>&1 | tee "${LOG_DIR}/train/icar_lipo_train.log"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  EVAL_EXTRA_ARGS=()
  if [[ "${INCLUDE_POLICY_VARIANTS}" == "1" ]]; then
    EVAL_EXTRA_ARGS+=(--include-policy-variants)
  fi

  declare -a EVAL_DATASETS_DEFAULT=(
    "cceval_python=${DATA_DIR}/cceval/python/test.parquet"
    "cceval_java=${DATA_DIR}/cceval/java/test.parquet"
    "repoeval_line=${DATA_DIR}/repoeval/line_level/test_0.parquet,${DATA_DIR}/repoeval/line_level/test_1.parquet"
    "repoeval_api=${DATA_DIR}/repoeval/api_level/test_0.parquet,${DATA_DIR}/repoeval/api_level/test_1.parquet"
  )

  if [[ -n "${EVAL_DATASETS:-}" ]]; then
    read -r -a EVAL_DATASETS_LIST <<< "${EVAL_DATASETS}"
  else
    EVAL_DATASETS_LIST=("${EVAL_DATASETS_DEFAULT[@]}")
  fi

  for eval_entry in "${EVAL_DATASETS_LIST[@]}"; do
    if [[ "${eval_entry}" == *"="* ]]; then
      dataset_name="${eval_entry%%=*}"
      dataset_path="${eval_entry#*=}"
    else
      dataset_path="${eval_entry}"
      dataset_name="${dataset_path#${DATA_DIR}/}"
      dataset_name="$(echo "${dataset_name}" | tr '/' '_' | sed 's/\.parquet$//')"
    fi

    IFS=',' read -r -a eval_path_list <<< "${dataset_path}"
    missing_eval=0
    for eval_path in "${eval_path_list[@]}"; do
      if [[ ! -f "${eval_path}" ]]; then
        echo "Skip ${dataset_name}: missing eval dataset ${eval_path}" >&2
        missing_eval=1
      fi
    done
    if [[ "${missing_eval}" == "1" ]]; then
      continue
    fi

    out_dir="${OUTPUT_ROOT}/eval/${dataset_name}"
    mkdir -p "${out_dir}" "${LOG_DIR}/eval"

    python3 -m co_retrieval.cli.co_retrieval_cli evaluate \
      --dataset-path "${dataset_path}" \
      --checkpoint-dir "${CHECKPOINT_DIR}" \
      --output-dir "${out_dir}" \
      --log-dir "${LOG_DIR}/eval/${dataset_name}" \
      --max-samples "${EVAL_MAX_SAMPLES}" \
      --top-k 3 \
      --batch-size 2 \
      --batch-encode-size 64 \
      --max-new-tokens 128 \
      --leave-one-out-analysis-samples 25 \
      "${EVAL_EXTRA_ARGS[@]}" \
      --generator-dtype bfloat16 \
      --device cuda \
      2>&1 | tee "${LOG_DIR}/eval/${dataset_name}.log"
  done
fi
