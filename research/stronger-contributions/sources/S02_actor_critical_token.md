# S02 — ACToR: generation-time critical-token retrieval

- **Type / status:** Primary arXiv preprint, submitted 2026-09-01; listed as under review on arXiv.
- **URL:** https://arxiv.org/abs/2609.01601
- **Credibility / recency / bias risk:** 3 / 5 / 3.
- **Primary locations:** Abstract; §III–V; arXiv HTML.

## Evidence extracted

- ACToR identifies critical generation positions from token mismatch, entropy, and subsequent attention; it trains a lightweight token classifier with ordinary cross-entropy.
- At inference, the generator decodes token-by-token, performs an additional targeted retrieval when the classifier fires, and re-decodes the current position.
- It also introduces endpoint-weighted pooling for the dense retriever.
- Evaluations are on RepoExec and the Python subset of CoderEval (355 and 230 tasks respectively), which emphasize repository-level function generation rather than the exact cursor-completion protocol in CrossCodeEval/RepoEval.
- The abstract reports relative gains on those benchmarks. As a recent under-review preprint, this is a prior-art warning, not settled evidence of general superiority.

## Verbatim excerpt

> “triggers targeted retrieval on demand”

## Relevance and limits

Generation-time/token-level retrieval is already claimed. A new project should not claim “adaptive retrieval at important code positions” broadly. A static, pre-generation evidence-need predictor trained from AST-resolved target relations is a different candidate mechanism, but needs a direct comparison and a careful task-boundary claim.

## Citation

Duan, K. et al. (2026). *Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation*. arXiv:2609.01601. Under review at the time of this search.
