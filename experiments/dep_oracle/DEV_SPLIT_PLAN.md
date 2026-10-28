# Dev split plan (E10) — draft 2026-10-28

## Why
Every knob in E9c / CADRE v2.9 was picked by looking at test:
budget set {384,1536,2304} (designed after E9 FAILed), min-left 256, copy gate 0.95,
slack rule, grounding beta 0.15, and any future early-exit threshold.
A reviewer will ask. We need numbers where knobs are chosen on data the reported score never saw.

## Facts that shape the design (checked 2026-10-28)
| benchmark | tasks | repos | tasks/repo |
|---|---|---|---|
| CCEval-py | 2665 | 471 | median 4, max 69 |
| CCEval-java | 2139 | 239 | median 5, max 136 |
| RepoEval-line | 1600 | 8 | 200 each |
| RepoEval-API | 1600 | 8 (same repos as line) | 200 each |

Repo ids: CCEval from `metadata.repository` in
`docs/research/audits/aligncoder_dataset/data/cceval/{python,java}/test.jsonl` (join on task_id);
RepoEval from the task_id prefix. Unit of splitting = repo (tasks in one repo share code, so a
task-level split leaks).

## Track A — repo-disjoint cross-fitting on the benchmarks (0 GPU, main track)
1. `dev_split.py`: assign each repo to fold 0/1 by `sha256(seed + repo) % 2`, seed fixed in the
   docstring. CCEval: 2 folds over ~470 / ~240 repos, balanced on task count (greedy fill if the
   hash split is off by >10%). RepoEval: 4 repos per fold; line and API use the same repo->fold map.
   Write `dev_folds.json` once, hash it into HANDOFF, never regenerate.
2. Tuning protocol for any knob K: choose K on fold 0, score on fold 1; choose on fold 1, score
   on fold 0; report the union (all 8004 tasks scored by a K that never saw them).
   Compare with the test-tuned K; the gap is the "tuning optimism" we disclose.
3. Knobs tunable offline from stored generations (E9d Modal: b384/b1536/b2304 raw+main+conf):
   copy gate tau, slack threshold, grounding beta, early-exit conf threshold, min-left only if
   both values were generated.
   Budget-set choice needs E9 (256/1024/1536) + E9c generations together: pull the raw
   prediction files from Drive (`MyDrive/dep_oracle/runs/{e9_w3072,e9c}`) first.
4. Caveat for RepoEval: 4 repos per fold is small; report per-fold numbers too, not only the union.

Prereg (to put in the script docstring before running):
- Hypothesis: cross-fitted conf3 (E9c) keeps >= 80% of its test-tuned gain over CADRE v2.8 on
  pooled primary benchmarks, and McNemar vs v2.8 stays p < 0.01.
- Control: same generations, knobs tuned on test (current numbers).
- FAIL = gain shrinks below 80% or p >= 0.01 -> stop and report.

## Track B — external dev set from training repos (needs GPU, optional)
For knobs that need new generations (new budget values, fill-the-window budget):
- Source: `github_repos/{python,java}/train.parquet` (RLCoder/AlignCoder training corpus, disjoint
  from test repos — verify no repo overlap with CCEval/RepoEval before use).
- Build ~1000 tasks per language with the CCEval recipe (cross-file-dependent line), RLCoder
  retrieval for them (RLCoder only; AlignCoder retriever is never rerun).
- Cost: retrieval + 3-5 budgets x 2000 tasks on Modal L4, est. ~1h.
- Only start if Track A shows a knob whose cross-fitted value differs from the test-tuned one.

## Order
1. Pull Drive E9/E9c raw predictions; write `dev_split.py` + `dev_folds.json` (smoke on 50 tasks).
2. Track A on knobs already generated -> HANDOFF entry.
3. Decide Track B from Track A's result.
