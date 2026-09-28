# S07 — Repoformer: Selective Retrieval for Repository-Level Code Completion

- **Authors / year:** Wu et al., ICML 2024.
- **Type:** peer-reviewed ML paper and official project page.
- **Source:** [official project page](https://repoformer.github.io/).
- **Credibility / recency / bias risk:** 5 / 3 / 2.

## Findings relevant to this project

Repoformer learns whether retrieval is needed and selectively retrieves context. The project page reports that Repoformer-3B performs on par with a larger StarCoder-16B RAG system while reducing retrieval overhead. This is a repository-completion reason to preserve a no-retrieval action or gate in evaluation; relevance ranking alone is not the whole objective.

**Short quote:** “selectively triggering retrieval”.

## Transfer boundary

This is not quantum-inspired evidence. It is a task-specific control: test the proposed scorer with the same no-context option and report both retrieval and completion outcomes.
