# S14 — CARET

- **Status at cutoff:** arXiv preprint, submitted 2026-10-04; no venue publication verified in this bounded search.
- **Task:** Test-time scaling for repository-level FIM completion.
- **Method:** Routes among retrieval contexts using sample agreement, then generates multiple candidates over a cached prompt and selects with reverse-context likelihood.
- **Reported result:** Preprint reports improvements over greedy decoding across its 18 model/benchmark combinations at increased generation cost.
- **Limitations:** Reverse-context scoring reads code after the cursor; it is incompatible with a strictly left-context-only task and changes generator-time compute, not just retrieval.
- **Short exact quote:** “reverse-context likelihood”
- **Source:** [arXiv preprint](https://arxiv.org/abs/2610.04837).
