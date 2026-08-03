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

By default this uses DeepSeek-Coder 6.7B base (`GENERATOR_NAME=deepseek-ai/deepseek-coder-6.7b-base`) frozen as the generator/teacher, UniXcoder-base as the trainable retriever encoder (`ENCODER_NAME=microsoft/unixcoder-base`), full Python+Java AlignCoder train data (`TRAIN_DATASETS=data/github_repos/python/train.parquet,data/github_repos/java/train.parquet`, `MAX_TRAIN_SAMPLES=0`), mixed completion sampling (`COMPLETION_LEVEL=mixed`), and 10 neural full-pass training epochs (`TRAIN_EPOCHS=10`, `EPOCH_BUDGET_MODE=1`). The launcher defaults to `sequential_retriever_first` with `ADAPTER_TYPE=none`, so it skips the optional soft-prompt warm-up and builds the expensive generator preference pool once. DeepSeekCoder remains frozen; the generator is used for utility supervision. Set `ADAPTER_TYPE=soft_prompt` to include the optional adapter phase, or set `EXPERIMENT_MODE=intent_main` explicitly for alternating co-training; that mode rebuilds Phase 2 once per epoch and is substantially slower. In epoch-budget mode, each full pass uses `len(train samples)` prompt steps, one pass over LiPO preference groups for retriever training, and one pass over gate labels. The training command intentionally passes `--skip-train-eval`, so evaluation is only run by the separate evaluate phase.

Training does not pre-encode a full global dense index by default
(`BUILD_TRAIN_INDEX=0`, `REFRESH_TRAIN_INDEX=0`). This avoids repeatedly
encoding hundreds of thousands of chunks with a retriever whose weights are
being updated. Retrieval during train is sample-local and uses the current
retriever weights; the separate eval phase still builds an index for benchmark
prediction.

`NUM_EPOCHS` remains accepted as a backward-compatible alias, but prefer
`TRAIN_EPOCHS` in new server runs.

Evaluate existing checkpoint only:

```bash
RUN_TRAIN=0 RUN_EVAL=1 \
  bash src/scripts/run_icar_lipo_a100_80gb.sh
```

Evaluation defaults to `INCLUDE_POLICY_VARIANTS=0` and `INCLUDE_ANALYSIS=0` for
the 8-12 hour target. Set `INCLUDE_POLICY_VARIANTS=1` for the gate ablation and
`INCLUDE_ANALYSIS=1` plus `LEAVE_ONE_OUT_ANALYSIS_SAMPLES=25` for the full
reviewer analysis pass.

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

Faster full-data debugging run:

```bash
PREFERENCE_POOL_TOP_K=5 TRAIN_BATCH_SIZE=6 BATCH_ENCODE_SIZE=96 \
RUN_TRAIN=1 RUN_EVAL=0 CUDA_VISIBLE_DEVICES=0 \
  bash src/scripts/run_icar_lipo_a100_80gb.sh
```

`PREFERENCE_POOL_TOP_K` bounds the dense ranking pool used for current/hard-
negative supervision. The expensive generator utility scores are now batched
per sample, while the sequential schedule is the main wall-clock saving
because it builds that pool once instead of once per epoch. `TRAIN_BATCH_SIZE`
is a parquet/data-loader batch, not a GPU optimization batch; Phase 3 LiPO
intentionally updates one preference group per step. `BATCH_ENCODE_SIZE` is
the real retriever GPU microbatch and controls dense encoding/index throughput;
128 is appropriate for UniXcoder-base on an A100 80GB, with 96/64 as OOM
fallbacks.

Recommended 72-hour run:

```bash
ENCODER_NAME=microsoft/unixcoder-base \
EXPERIMENT_MODE=sequential_retriever_first \
ADAPTER_TYPE=none \
COMPLETION_LEVEL=mixed \
PREFERENCE_POOL_TOP_K=5 \
TRAIN_BATCH_SIZE=8 \
BATCH_ENCODE_SIZE=128 \
MAX_TRAIN_SAMPLES=0 \
TRAIN_EPOCHS=10 \
EVAL_MAX_SAMPLES=0 \
RUN_TRAIN=1 \
RUN_EVAL=1 \
INCLUDE_POLICY_VARIANTS=0 \
INCLUDE_ANALYSIS=0 \
CUDA_VISIBLE_DEVICES=0 \
  bash src/scripts/run_icar_lipo_a100_80gb.sh
```

This schedule builds Phase 2 preference data once, so it is much more likely to
finish under a hard wall-clock budget. For a cheap encoder smoke test, set
`ENCODER_NAME=sentence-transformers/all-MiniLM-L6-v2`, but do not use MiniLM as
the main paper run unless code-specific encoders are still too slow.

Train on a single language for a quick ablation:

```bash
TRAIN_DATASETS=data/github_repos/python/train.parquet MAX_TRAIN_SAMPLES=5000 \
  bash src/scripts/run_icar_lipo_a100_80gb.sh
```

Run one schedule manually:

```bash
EXPERIMENT_MODE=sequential_retriever_first \
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
