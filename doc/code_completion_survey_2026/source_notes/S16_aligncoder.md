# S16 — AlignCoder

- **Publication:** ASE 2025 Research Papers. Its arXiv v1 was posted on 2026-01-27; that upload date does not make it a 2026 venue paper.
- **Task:** Repository-level completion on CrossCodeEval and RepoEval.
- **Method:** Generates candidate completions to build an enhanced retrieval query, then trains AlignRetriever with reinforcement learning; authors report +18.1% EM over baselines on CrossCodeEval.
- **Use here:** Aligns retrieval to the intended continuation, but candidate-completion query enhancement is not an explicit repository dependency representation and may require extra generation/inference stages.
- **Limitations:** Reported improvement is paper-protocol specific; the “18.1%” is not an expected gain on the project's current 400-sample evaluation.
- **Short exact quote:** “candidate completions to construct an enhanced query”
- **Sources:** [ASE 2025 paper record](https://conf.researchr.org/details/ase-2025/ase-2025-papers/34/AlignCoder-Aligning-Retrieval-with-Target-Intent-for-Repository-Level-Code-Comple), [arXiv version](https://arxiv.org/abs/2601.19697).
