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
ENCODER_NAME="${ENCODER_NAME:-microsoft/unixcoder-base}"
# The primary paper baseline must isolate retrieval: DeepSeek-Coder is frozen,
# no soft prompt is trained, and retrieval is always enabled.  After this run,
# use EXPERIMENT_MODE=sequential_retriever_first ADAPTER_TYPE=soft_prompt for
# the optional generator-adaptation ablation.
EXPERIMENT_MODE="${EXPERIMENT_MODE:-retriever_only}"
ADAPTER_TYPE="${ADAPTER_TYPE:-none}"
GATE_MODE="${GATE_MODE:-learned}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
COMPLETION_LEVEL="${COMPLETION_LEVEL:-mixed}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-${NUM_EPOCHS:-10}}"
EPOCH_BUDGET_MODE="${EPOCH_BUDGET_MODE:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
STEPS_PER_ROUND_PROMPT="${STEPS_PER_ROUND_PROMPT:-100}"
STEPS_PER_ROUND_RETRIEVER="${STEPS_PER_ROUND_RETRIEVER:-200}"
PREFERENCE_POOL_TOP_K="${PREFERENCE_POOL_TOP_K:-5}"
MAX_PAIRS_PER_SAMPLE="${MAX_PAIRS_PER_SAMPLE:-4}"
UTILITY_SCORE_MICROBATCH_SIZE="${UTILITY_SCORE_MICROBATCH_SIZE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
BATCH_ENCODE_SIZE="${BATCH_ENCODE_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-64}"
BUILD_TRAIN_INDEX="${BUILD_TRAIN_INDEX:-0}"
REFRESH_TRAIN_INDEX="${REFRESH_TRAIN_INDEX:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
INCLUDE_ANALYSIS="${INCLUDE_ANALYSIS:-0}"
LEAVE_ONE_OUT_ANALYSIS_SAMPLES="${LEAVE_ONE_OUT_ANALYSIS_SAMPLES:-0}"
EVAL_INDEX_MODE="${EVAL_INDEX_MODE:-sharded}"
EVAL_INDEX_ROOT="${EVAL_INDEX_ROOT:-${OUTPUT_ROOT}/eval_index}"
EVAL_INDEX_SHARD_SIZE="${EVAL_INDEX_SHARD_SIZE:-50000}"
BUILD_EVAL_INDEX="${BUILD_EVAL_INDEX:-1}"
EVAL_RETRIEVER_DEVICE="${EVAL_RETRIEVER_DEVICE:-cuda}"
# Evaluation uses the validated 4K window. The packer keeps the cursor tail
# and only includes complete snippets. Training remains explicitly configured
# at 4096 below to keep its memory footprint predictable.
EVAL_MAX_CONTEXT_TOKENS="${EVAL_MAX_CONTEXT_TOKENS:-4096}"
EVAL_SKIP_NLL="${EVAL_SKIP_NLL:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_DOWNLOAD="${RUN_DOWNLOAD:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
INCLUDE_POLICY_VARIANTS="${INCLUDE_POLICY_VARIANTS:-0}"
RESAMPLE_TRAIN_EACH_EPOCH="${RESAMPLE_TRAIN_EACH_EPOCH:-0}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

EVAL_NLL_ARGS=()
if [[ "${EVAL_SKIP_NLL}" == "1" ]]; then
  EVAL_NLL_ARGS+=(--skip-eval-nll)
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${OUTPUT_ROOT}"

if [[ "${RUN_DOWNLOAD}" == "1" ]]; then
  "${SCRIPT_DIR}/download_aligncoder_data.sh"
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  if [[ "${RESAMPLE_TRAIN_EACH_EPOCH}" == "1" && "${EXPERIMENT_MODE}" != "intent_main" ]]; then
    echo "RESAMPLE_TRAIN_EACH_EPOCH=1 requires EXPERIMENT_MODE=intent_main." >&2
    exit 1
  fi
  if [[ "${EXPERIMENT_MODE}" == "intent_main" && "${EPOCH_BUDGET_MODE}" == "1" && "${TRAIN_EPOCHS}" -gt 1 ]]; then
    echo "WARNING: intent_main repeats generator-heavy Phase 2 ${TRAIN_EPOCHS} times."
    echo "Use EXPERIMENT_MODE=sequential_retriever_first ADAPTER_TYPE=none for the 24-32 hour run." >&2
  fi
  TRAIN_BUDGET_ARGS=()
  if [[ "${EPOCH_BUDGET_MODE}" == "1" ]]; then
    TRAIN_BUDGET_ARGS+=(--epoch-budget-mode)
  fi
  if [[ "${BUILD_TRAIN_INDEX}" == "1" ]]; then
    TRAIN_BUDGET_ARGS+=(--build-train-index)
  fi
  if [[ "${REFRESH_TRAIN_INDEX}" == "1" ]]; then
    TRAIN_BUDGET_ARGS+=(--refresh-train-index)
  fi
  TRAIN_RESAMPLE_ARGS=()
  if [[ "${RESAMPLE_TRAIN_EACH_EPOCH}" == "1" ]]; then
    TRAIN_RESAMPLE_ARGS+=(--resample-train-each-epoch)
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
  echo "Encoder: ${ENCODER_NAME}"
  echo "Generator: ${GENERATOR_NAME}"
  effective_gate_mode="${GATE_MODE}"
  if [[ "${EXPERIMENT_MODE}" == "retriever_only" ]]; then
    effective_gate_mode="always_retrieve"
  fi
  echo "Generator backbone: frozen; adapter: ${ADAPTER_TYPE}; gate: ${effective_gate_mode}"
  echo "Completion level: ${COMPLETION_LEVEL}"
  echo "Preference pool top-k: ${PREFERENCE_POOL_TOP_K}; max pairs/sample: ${MAX_PAIRS_PER_SAMPLE}"
  echo "Batch sizes: data_loader=${TRAIN_BATCH_SIZE}, encoder_microbatch=${BATCH_ENCODE_SIZE}, utility_score_microbatch=${UTILITY_SCORE_MICROBATCH_SIZE}, eval_loader=${EVAL_BATCH_SIZE}"
  echo "Train global index: build=${BUILD_TRAIN_INDEX}, refresh=${REFRESH_TRAIN_INDEX}"
  echo "Resample train each epoch: ${RESAMPLE_TRAIN_EACH_EPOCH}"
  echo "Train datasets: ${TRAIN_DATASETS}"

  "${PYTHON_BIN}" -m co_retrieval.cli.co_retrieval_cli train \
    --use-neural \
    --skip-train-eval \
    --dataset-path "${TRAIN_DATASETS}" \
    --output-dir "${OUTPUT_ROOT}/train" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --log-dir "${LOG_DIR}/train" \
    --encoder-name "${ENCODER_NAME}" \
    --generator-name "${GENERATOR_NAME}" \
    --experiment-mode "${EXPERIMENT_MODE}" \
    --intent-mode static \
    --gate-mode "${GATE_MODE}" \
    --adapter-type "${ADAPTER_TYPE}" \
    --retriever-loss lipo \
    --lipo-tau 1.0 \
    --num-epochs "${TRAIN_EPOCHS}" \
    --train-epochs "${TRAIN_EPOCHS}" \
    "${TRAIN_BUDGET_ARGS[@]}" \
    "${TRAIN_RESAMPLE_ARGS[@]}" \
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
    --preference-pool-top-k "${PREFERENCE_POOL_TOP_K}" \
    --max-pairs-per-sample "${MAX_PAIRS_PER_SAMPLE}" \
    --utility-score-microbatch-size "${UTILITY_SCORE_MICROBATCH_SIZE}" \
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
    --batch-encode-size "${BATCH_ENCODE_SIZE}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
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
  if [[ "${INCLUDE_ANALYSIS}" != "1" ]]; then
    EVAL_EXTRA_ARGS+=(--no-analysis)
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
    eval_index_dir="${EVAL_INDEX_ROOT}/${dataset_name}"
    mkdir -p "${out_dir}" "${LOG_DIR}/eval"

    if [[ "${EVAL_INDEX_MODE}" == "sharded" && "${BUILD_EVAL_INDEX}" == "1" ]]; then
      if [[ -f "${eval_index_dir}/manifest.json" ]]; then
        echo "Reusing existing eval index for ${dataset_name}: ${eval_index_dir}"
      else
        mkdir -p "${eval_index_dir}"
        echo "Building CPU mmap eval index for ${dataset_name}: ${eval_index_dir}"
        "${PYTHON_BIN}" -m co_retrieval.cli.co_retrieval_cli build-eval-index \
          --dataset-path "${dataset_path}" \
          --checkpoint-dir "${CHECKPOINT_DIR}" \
          --eval-index-dir "${eval_index_dir}" \
          --max-samples "${EVAL_MAX_SAMPLES}" \
          --batch-encode-size "${BATCH_ENCODE_SIZE}" \
          --eval-index-shard-size "${EVAL_INDEX_SHARD_SIZE}" \
          --max-chunk-lines 120 \
          --fallback-lines 40 \
          --device cuda \
          2>&1 | tee "${LOG_DIR}/eval/${dataset_name}.index.log"
      fi
    fi

    "${PYTHON_BIN}" -m co_retrieval.cli.co_retrieval_cli evaluate \
      --dataset-path "${dataset_path}" \
      --checkpoint-dir "${CHECKPOINT_DIR}" \
      --output-dir "${out_dir}" \
      --log-dir "${LOG_DIR}/eval/${dataset_name}" \
      --max-samples "${EVAL_MAX_SAMPLES}" \
      --top-k 3 \
      --batch-size "${EVAL_BATCH_SIZE}" \
      --batch-encode-size "${BATCH_ENCODE_SIZE}" \
      --eval-index-mode "${EVAL_INDEX_MODE}" \
      --eval-index-dir "${eval_index_dir}" \
      --eval-index-shard-size "${EVAL_INDEX_SHARD_SIZE}" \
      --eval-retriever-device "${EVAL_RETRIEVER_DEVICE}" \
      --max-context-tokens "${EVAL_MAX_CONTEXT_TOKENS}" \
      --max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
      --leave-one-out-analysis-samples "${LEAVE_ONE_OUT_ANALYSIS_SAMPLES}" \
      "${EVAL_NLL_ARGS[@]}" \
      "${EVAL_EXTRA_ARGS[@]}" \
      --generator-dtype bfloat16 \
      --device cuda \
      2>&1 | tee "${LOG_DIR}/eval/${dataset_name}.log"
  done
fi
