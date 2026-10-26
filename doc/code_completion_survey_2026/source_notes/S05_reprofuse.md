# S05 — RepoFuse

- **Publication:** FSE 2026 Industry Papers; conference record and ACM DOI.
- **Task:** Repository-level completion deployed at Ant Group.
- **Method:** Split context into structural “semantic context” obtained by lightweight static analysis and “similar context” from similarity retrieval; select evidence by relevance under latency/token constraints.
- **Reported result:** Authors report a 156 ms average production latency, 18% acceptance-rate improvement in a one-week A/B test, and CrossCodeEval EM improvements for Python/Java using 25% of baseline token requirements.
- **Limitations:** Industry track, one company and short A/B period; these results need not generalize to other IDEs or generators. It is valuable evidence for separating dependency evidence from analogy, not a controlled academic retriever-only benchmark.
- **Short exact quote:** “semantic context” and “similar context”
- **Sources:** [FSE 2026 record](https://conf.researchr.org/details/fse-2026/fse-2026-industry-papers/28/RepoFuse-A-Dual-Context-Approach-to-Repository-Level-Code-Completion-at-Industrial-S), [ACM DOI](https://doi.org/10.1145/3803437.3805230).
