# dep_oracle: E- and D-series experiments

Repo-level code completion with a frozen generator (DeepSeek-Coder-1.3B, greedy, 64 tokens, 2048 window) on top of the
saved AlignCoder top-10 candidates. Benchmarks: RepoEval API (1600) and CCEval Python (2665); dev = 1000 per dataset.

Copied from the working folder `NewCodeCompletion/experiments/dep_oracle` (not a git repo). Data, candidates and outputs
(`aligncoder_data/`, `*_prediction_with_candidates.jsonl`, `modal_out/`, sweeps) are not included.

- Core modules: `dep_oracle.py` (data, prompt building, EM, McNemar), `budget_sweep.py`, `likelihood_probe.py`,
  `chunk_pilot.py`, `joint_retrieval.py`, `modal_probe.py`.
- E0–E8 (`e*.py`, Modal wrappers `modal_e*.py`): chunk order, chunk scorers, per-sample budget policy (E2, the main
  positive result), in-file compaction, hidden-state probe, budget x Caret-style selection, cascade, error decomposition
  (E7/E7b), contrastive decoding.
- D1/D1.2/D2 (`d*.py`): call-site retrieval at decode time, compact call sites + confidence gating, cross-budget likelihood.
- Each script's docstring states the hypothesis, control and decision threshold. Results and script hashes:
  `HANDOFF.md`; summary: `REPORT_2026-10-26.md`; decision tree: `NEXT_SESSION_BRIEF.md`; Colab setup: `HANDOFF_COLAB.md`.
