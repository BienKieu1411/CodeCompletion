#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT="${1:-checkpoints/rl/best_model.pt}"
DATASET="${2:-repoeval}"
LANGUAGE="${3:-python}"
OUTPUT_DIR="${4:-results/research/ablation}"
MAX_SAMPLES="${5:-200}"
TOP_K="${6:-3}"
MAX_GEN_LEN="${7:-64}"

python3 graphcoder_cli.py ablation-study \
  --checkpoint "$CHECKPOINT" \
  --dataset "$DATASET" \
  --language "$LANGUAGE" \
  --output-dir "$OUTPUT_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --top-k "$TOP_K" \
  --max-gen-length "$MAX_GEN_LEN"
