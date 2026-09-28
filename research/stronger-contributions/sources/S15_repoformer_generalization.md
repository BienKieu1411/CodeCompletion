# S15 — Repoformer: selective retrieval across backbones

- **Type / status:** Peer-reviewed primary paper, ICML 2024.
- **URL:** https://arxiv.org/abs/2403.10059
- **Credibility / recency / bias risk:** 5 / 3 / 2.
- **Primary locations:** Abstract and experiments.

## Evidence extracted

- Repoformer studies selective retrieval for repository-level code completion, including when retrieval is unnecessary or harmful.
- The paper reports that its approach can accommodate different generation models, retrievers, and programming languages.
- This is evidence that broad “works across multiple code LMs” or “model-agnostic retrieval” claims are already occupied.

## Verbatim excerpt

> “accommodate different generation models”

## Relevance and limits

Repoformer does not, from the checked abstract, establish that its retriever is trained with a multi-generator average target-likelihood reward. Thus an ensemble-reward training objective remains a narrow candidate to investigate, but must be tested against Repoformer and cannot claim cross-backbone generalization as a new concept.

## Citation

Wu, D. et al. (2024). *Repoformer: Selective Retrieval for Repository-Level Code Completion*. ICML 2024. https://arxiv.org/abs/2403.10059
