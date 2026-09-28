# S08 — RepoShapley: coalition-aware context filtering

- **Type / status:** Peer-reviewed primary paper, Findings of ACL 2026.
- **URL:** https://aclanthology.org/2026.findings-acl.505/
- **Credibility / recency / bias risk:** 5 / 5 / 2.
- **Primary locations:** Abstract; §§1–4; ACL Anthology full text/PDF.

## Evidence extracted

- RepoShapley explicitly models context utility as interaction-dependent and proposes coalition-aware filtering.
- Its pipeline probes individual chunks with teacher-forced likelihood, fits a surrogate game for saturation/interference, computes Shapley values for small sets, verifies a selected coalition with a frozen generator, then distills KEEP/DROP and retrieval triggers into one controller.
- It reports completion gains across benchmarks/backbones. Those are author-reported experimental results.

## Verbatim excerpt

> “chunk utility is often interaction-dependent”

## Relevance and limits

This occupies the broad claim “select context jointly because chunks interact.” A new method cannot use that phrase as its novelty. A possible distinction is a directly trained, sequential AST-relation action policy whose policy-gradient reward scores the actual assembled prompt, with no Shapley-label pipeline or KEEP/DROP distillation. This is still a narrow distinction and requires a direct RepoShapley comparison.

## Citation

Huo, Y. et al. (2026). *RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion*. Findings of ACL 2026, 10390–10412. https://doi.org/10.18653/v1/2026.findings-acl.505
