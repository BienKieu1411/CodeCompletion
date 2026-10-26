# S15 — RLCoder

- **Publication:** ICSE 2025 Research Track.
- **Task:** Repository-level completion on CrossCodeEval and RepoEval.
- **Method:** Uses completion-model feedback to train a retriever with reinforcement learning and controls retrieval/stop behavior; it directly targets context selection for code completion.
- **Use here:** Historical comparison point for target-aware retrieval optimization. It does not provide an explicit typed repository relation graph as its main contribution.
- **Limitations:** Results are protocol/backbone-specific; do not assume a retriever trained by RL has learned file dependency semantics unless that behavior is separately measured.
- **Short exact quote:** “model-based retrieval methods face challenges due to the lack of labeled data”
- **Sources:** [ICSE 2025 record](https://conf.researchr.org/details/icse-2025/icse-2025-research-track/24/RLCoder-Reinforcement-Learning-for-Repository-Level-Code-Completion), [arXiv manuscript](https://arxiv.org/abs/2407.19487).
