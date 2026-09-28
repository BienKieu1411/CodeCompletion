# Refresh targets

Revisit these only when updating this investigation or before making a paper-level claim:

- **Notebook status:** whether the engine-path, scorer CLI, unsupported flags, head checkpoint, and test-selection issues in S01 have been corrected; inspect a new revision without overwriting the current local artifact.
- **Quantum-inspired retrieval evidence:** peer-reviewed follow-ups to QIEPSM (arXiv:2501.04591), Q-Interference (arXiv:2608.17288), and direct code-search/repository-completion tests using phase-aware fusion.
- **Negative evidence:** replications or independent evaluations of standalone quantum-inspired document/chunk retrieval; revisit S06 cautiously due to low confidence.
- **Repository-completion baselines:** new versions/results for AlignCoder, CodeRAG (EMNLP 2025), Repoformer, and benchmark revisions.
- **User AST-GR/chunker:** label semantics, repo split disjointness, parser/language coverage, source-span IDs, candidate recall, target leakage, and reproducible manifests.
- **Decision numbers to refresh after experiments:** EM/ES, AST-GR Recall@K/nDCG, top-K candidate recall, seeds and paired confidence intervals, latency, peak memory, and inference context-token budget.

Do not carry forward unverified notebook claims about CrossCodeEval TEST selection or training completion; verify against actual executed run artifacts.
