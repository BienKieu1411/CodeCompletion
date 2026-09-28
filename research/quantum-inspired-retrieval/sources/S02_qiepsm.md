# S02 — Quantum-inspired embeddings projection and similarity metrics

- **Authors / year:** Kankeu et al., 2025.
- **Type:** arXiv research preprint, arXiv:2501.04591.
- **Source:** [arXiv abstract](https://arxiv.org/abs/2501.04591) · [HTML paper](https://arxiv.org/html/2501.04591).
- **Credibility / recency / bias risk:** 3 / 4 / 3. Primary research with controlled IR benchmarks, but not a repository-completion study and not a peer-reviewed venue according to the record consulted.

## Findings relevant to this project

The paper maps BERT embeddings to separable quantum-inspired states and evaluates a fidelity similarity/projection head on TREC DL 2019/2020. It is the closest methodological precedent for the notebook's per-coordinate phase/fidelity scorer. Results are competitive, not evidence of a general retrieval breakthrough: the paper reports small fidelity-only gaps on the stronger sentence encoder. In its frozen-backbone ablation, the quantum-inspired head trails the classical compression head; joint tuning of the encoder's last layers matters.

**Short quote:** “the quantum-inspired approach by -0.48±0.31% on TREC19” (paper's frozen-head comparison).

## Transfer boundary

This supports testing an end-to-end learned scorer, but argues against assuming a fidelity head alone adds value. TREC passage retrieval does not establish gains for code chunks, AST relevance, or completion EM/ES. The notebook's 768-dimensional uncompressed phase map is not the same as the paper's compressed circuit head.
