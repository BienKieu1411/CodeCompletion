# S01 — AIRCoder: adaptive multi-dimensional retrieval

- **Type / status:** Peer-reviewed primary paper, ACL 2026.
- **URL:** https://aclanthology.org/2026.acl-long.1166/
- **Credibility / recency / bias risk:** 5 / 5 / 2.
- **Primary locations:** Abstract; §3.1–3.3; §4; ACL Anthology PDF, pp. 1–8.

## Evidence extracted

- AIRCoder already combines AST-preserving chunking with eight retrieval metrics over textual similarity, dependency existence, and structural hierarchy.
- It offers both training-free Reciprocal Rank Fusion and a query-conditioned MLP that weights the metrics.
- The trainable fusion labels each query-candidate example by generator Edit Similarity and optimizes a pairwise MSE ranking objective.
- The paper reports an average 4.63% exact-match improvement over its strongest baseline and 10.2× efficiency; these are the authors' reported benchmark results, not independent replication.
- The paper itself notes a target-context boundary: its static dependency extraction can help when a target line exposes identifiers, while incomplete/empty target lines can misalign retrieval. This is relevant to any target-derived structural supervision proposal.

## Verbatim excerpt

> “we utilize a Pairwise Mean Squared Error Loss”

## Relevance and limits

AST chunking, hybrid structural metrics, and query-conditioned fusion are not defensible standalone novelty claims anymore. Its pairwise loss is outside the user's constraint. The remaining opening must be a different source of supervision or a more specific, falsifiable mechanism; merely replacing pairwise MSE with BCE would be a weak contribution without that difference.

## Citation

Shi, C., Gao, M., & Gao, Z. (2026). *AIRCoder: Adaptive Integration of Multi-dimensional Retrieval for Repository-level Code Completion*. ACL 2026, 25458–25470. https://doi.org/10.18653/v1/2026.acl-long.1166
