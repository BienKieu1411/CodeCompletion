# ICAR LiPO A100 80GB Run Notes

Copy only the `src` directory to the server, then run from the directory that
contains `src`.

```bash
python3 -m pip install -r src/scripts/requirements_a100.txt
```

Keep the pinned `tree-sitter==0.20.1` / `tree-sitter-languages==1.10.2`
versions unless you also validate parser loading. Newer mismatched parser
wheels can silently force the sampler into weaker random-cut fallback.

Download AlignCoder data:

```bash
RUN_DOWNLOAD=1 RUN_TRAIN=0 RUN_EVAL=0 \
  bash src/scripts/run_icar_lipo_a100_80gb.sh
```

Train only, with no train-time eval:

```bash
RUN_TRAIN=1 RUN_EVAL=0 \
  bash src/scripts/run_icar_lipo_a100_80gb.sh
```

By default this uses DeepSeek-Coder 6.7B base (`GENERATOR_NAME=deepseek-ai/deepseek-coder-6.7b-base`), full Python+Java AlignCoder train data (`TRAIN_DATASETS=data/github_repos/python/train.parquet,data/github_repos/java/train.parquet`, `MAX_TRAIN_SAMPLES=0`), and 10 neural full-pass training epochs (`TRAIN_EPOCHS=10`, `EPOCH_BUDGET_MODE=1`). In epoch-budget mode, each co-training epoch uses `len(train samples)` prompt steps, one pass over LiPO preference groups for retriever training, and one pass over gate labels. The command still uses a small bootstrap warmup (`WARMUP_STEPS=200` by default). The training command intentionally passes `--skip-train-eval`, so evaluation is only run by the separate evaluate phase.

`NUM_EPOCHS` remains accepted as a backward-compatible alias, but prefer
`TRAIN_EPOCHS` in new server runs.

Evaluate existing checkpoint only:

```bash
RUN_TRAIN=0 RUN_EVAL=1 \
  bash src/scripts/run_icar_lipo_a100_80gb.sh
```

Evaluation defaults to `INCLUDE_POLICY_VARIANTS=1`, so each eval dataset also
runs learned/always-retrieve/always-skip policy variants for gate ablation. Set
`INCLUDE_POLICY_VARIANTS=0` for a faster smoke run.

The default eval set labels follow AlignCoder's reporting groups:
`cceval_python`, `cceval_java`, `repoeval_line`, and `repoeval_api`.
For RepoEval, `test_0.parquet` and `test_1.parquet` are passed together and
evaluated as one benchmark, matching AlignCoder's concat behavior.

Train then evaluate:

```bash
bash src/scripts/run_icar_lipo_a100_80gb.sh
```

Run reviewer-blocker schedule ablations with the same train/eval config:

```bash
bash src/scripts/run_icar_reviewer_ablations_a100_80gb.sh
```

This trains/evaluates `intent_main`, `sequential_adapter_first`, and
`sequential_retriever_first` into separate checkpoint/result directories. Use
`ABLATION_MODES="intent_main sequential_adapter_first"` to run a subset.

Useful overrides:

```bash
MAX_TRAIN_SAMPLES=5000 TRAIN_EPOCHS=3 EVAL_MAX_SAMPLES=1000 CUDA_VISIBLE_DEVICES=0 \
  bash src/scripts/run_icar_lipo_a100_80gb.sh
```

Train on a single language for a quick ablation:

```bash
TRAIN_DATASETS=data/github_repos/python/train.parquet MAX_TRAIN_SAMPLES=5000 \
  bash src/scripts/run_icar_lipo_a100_80gb.sh
```

Run one schedule manually:

```bash
EXPERIMENT_MODE=sequential_adapter_first \
CHECKPOINT_DIR=checkpoints/icar_seq_adapter_first \
OUTPUT_ROOT=results/icar_seq_adapter_first \
LOG_DIR=logs/icar_seq_adapter_first \
  bash src/scripts/run_icar_lipo_a100_80gb.sh
```

Evaluation writes AlignCoder-style files under:

```text
results/icar_lipo_deepseek_6p7b_a100_80gb/eval/<dataset_label>/prediction.jsonl
results/icar_lipo_deepseek_6p7b_a100_80gb/eval/<dataset_label>/prediction_with_candidates.jsonl
results/icar_lipo_deepseek_6p7b_a100_80gb/eval/<dataset_label>/prediction_truncated.jsonl
results/icar_lipo_deepseek_6p7b_a100_80gb/eval/<dataset_label>/exact_match_idx.jsonl
results/icar_lipo_deepseek_6p7b_a100_80gb/eval/<dataset_label>/detailed_results.json
results/icar_lipo_deepseek_6p7b_a100_80gb/eval/<dataset_label>/results.json
results/icar_lipo_deepseek_6p7b_a100_80gb/eval/<dataset_label>/metrics.json
```

`results.json` uses AlignCoder-style percentage metrics (`em`, `es`,
`id_em`, `id_precision`, `id_recall`, `id_f1`). `metrics.json` keeps the richer
ICAR audit fields: gate calibration, policy variants, NLL/output correlation,
leave-one-out analysis, and oracle-safety flags.

Summarize ICAR against an AlignCoder result directory:

```bash
python3 src/scripts/summarize_aligncoder_comparison.py \
  --icar-root results/icar_lipo_deepseek_6p7b_a100_80gb \
  --aligncoder-root /path/to/AlignCoder/results
```

Positive `delta_*` values mean ICAR is ahead on that AlignCoder-style metric.
