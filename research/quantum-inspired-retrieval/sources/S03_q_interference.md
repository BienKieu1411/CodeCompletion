# S03 — Q-Interference: phase-aware interaction with exact factorization

- **Authors / year:** Nahid et al., 2026-08-18, v1.
- **Type:** arXiv preprint, arXiv:2608.17288.
- **Source:** [arXiv abstract](https://arxiv.org/abs/2608.17288) · [HTML paper](https://arxiv.org/html/2608.17288).
- **Credibility / recency / bias risk:** 2 / 5 / 3. Very recent primary preprint; evaluate as a design lead, not settled evidence.

## Findings relevant to this project

The paper uses learned amplitudes and phases to score token interactions as `sum_r a_q a_k cos(phi_q - phi_k)`. Its key systems result is an exact rewrite into cosine-feature and sine-feature matrix products, avoiding an extra token-pair-feature tensor while retaining the same score. It stays entirely classical and retains the ordinary next-token objective.

**Short quote:** “aligned phases contribute constructively while conflicting phases contribute destructively.”

On the paper's own language-modeling tables, quality is mixed across datasets: the controlled model improves the internal GPT on WikiText-103 but is weaker on other corpora, and pretrained GPT-Neo/OPT remain stronger. This is not a code retriever evaluation. The reported evidence supports the mechanism and implementation pattern, not a claim that phase-aware retrieval will beat AlignCoder.

## Transfer boundary

Use the factorization idea if a channel-interference scorer is tested. Keep the candidate-level scorer small, since repo retrieval has only a few evidence channels and no need to construct an O(T²d) interaction tensor. Compare against real-valued fusion controls.
