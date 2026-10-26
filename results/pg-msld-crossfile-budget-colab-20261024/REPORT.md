# Paired cross-file budget evaluation

## Result

The 2,344-token cross-file cap gives a small increase in macro exact match, but does **not** improve macro edit similarity on this 400-task paired evaluation. The confidence intervals include zero, so this run is suggestive rather than evidence of a reliable overall gain.

| Cross-file cap | CCEval Java EM / ES | CCEval Python EM / ES | RepoEval API EM / ES | RepoEval Line EM / ES | 4-benchmark macro EM / ES |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.110 / 0.6173 | 0.170 / 0.6512 | 0.3864 / 0.6464 | 0.4398 / 0.6665 | 0.2766 / 0.6453 |
| 2,344 | 0.140 / 0.6306 | 0.210 / 0.6506 | 0.3786 / 0.6406 | 0.4460 / 0.6493 | 0.2936 / 0.6428 |
| Paired delta | +0.030 / +0.0133 | +0.040 / −0.0006 | −0.0078 / −0.0058 | +0.0062 / −0.0172 | **+0.0171 / −0.0026** |

Macro delta 95% bootstrap intervals:

- EM: +0.0171, 95% CI [−0.0002, +0.0356]
- ES: −0.0026, 95% CI [−0.0149, +0.0096]

RepoEval intervals resample repositories (8 per benchmark); CCEval intervals resample tasks. This respects the RepoEval repo-level aggregation. Per-benchmark paired intervals and ES win/tie/loss counts are in `paired_analysis.json`.

## What was held fixed

- The same 400 task IDs and benchmark composition: 100 each for CCEval Java, CCEval Python, RepoEval API, and RepoEval Line.
- The same trained PG-MSLD UniXcoder checkpoint (`2480` updates; SHA-256 `30a6807a…1686f7`) and the same BM25 candidate pools (top 100 where available).
- The exact same selected snippet identities **and selected snippet count for each task** in both arms. These were frozen from the previous evaluation; only their permitted cross-file token allocation changed.
- The same 3,072-token maximum prompt context, left-context handling, DeepSeek-Coder 1.3B checkpoint/revision, greedy decoding, and 64-token generation cap.
- RLCoder natural chunking: blank-line mini-blocks, merged/split to a maximum of 15 non-empty lines. Per-task candidate availability was respected (39–100; some pools do not contain 100 chunks).

Run: Colab A100-SXM4-40GB; paired inference completed in about 569 seconds. Both arms were run in the same process with Hugging Face Transformers and BF16, so the *within-run* comparison is controlled. The earlier reported evaluation used vLLM; absolute scores should not be compared across those runs as if backend/runtime were identical.

## Interpretation and limits

The larger cap raised mean cross-file tokens actually placed in prompts from 223.6 to 670.5 and reduced mean left-context tokens from 1,427.8 to 1,282.6. The cap is not a target: most prompts did not fill 2,344 tokens because the frozen slate had limited content or selected no snippets. Thus this tests **more room for the same retrieved slate**, not whether the policy would select more snippets under the larger budget.

The direction is mixed: CCEval benefits numerically, while RepoEval ES declines, especially line completion. The macro changes are small and uncertain. Do not claim that increasing cross-file budget improves completion overall from this run alone. A next test should let the selection policy adapt its snippet count to the larger budget while retaining the same task IDs and generator settings; report both fixed-slate and adaptive-slate arms.

## Artifacts

- `metrics.json`: run configuration and aggregate metrics.
- `paired_analysis.json`: paired deltas, task/repository bootstrap intervals, and selection-overlap checks.
- `predictions_crossfile_512.jsonl`, `predictions_crossfile_2344.jsonl`: per-task prompts, predictions, scores, and selected snippets.
- Evaluation implementation: `experiments/pg_msld_budget_eval/colab_eval.py`.
