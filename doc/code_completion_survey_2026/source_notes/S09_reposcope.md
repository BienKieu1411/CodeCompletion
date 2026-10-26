# S09 — RepoScope

- **Publication:** ICSE 2026 Research Track.
- **Task:** Repository-level full-function code generation (CoderEval/DevEval), not prefix-only FIM completion.
- **Method:** Builds a Repository Structural Semantic Graph and retrieves four context views combining structural and similarity evidence; predicts call chains and serializes context while preserving structure. The paper states it uses static analysis without extra training or multiple LLM queries.
- **Reported result:** Up to 36.35% relative pass@1 improvement on its full-function-generation benchmarks.
- **Limitations:** Task mismatch is substantial: full function generation and available target signature/context differ from left-only continuation. Transferable point: a useful graph must represent and expose typed relations in the prompt, not just rank graph-near chunks.
- **Short exact quote:** “four-view context”
- **Sources:** [ICSE record](https://conf.researchr.org/details/icse-2026/icse-2026-research-track/169/RepoScope-Leveraging-Call-Chain-Aware-Multi-View-Context-for-Repository-Level-Code-G), [DOI](https://doi.org/10.1145/3744916.3773211), [arXiv manuscript](https://arxiv.org/html/2507.14791).
