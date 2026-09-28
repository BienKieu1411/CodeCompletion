# S08 — CodeRAG: Finding Relevant and Necessary Knowledge

- **Authors / year:** Zhang et al., EMNLP 2025.
- **Type:** peer-reviewed EMNLP paper.
- **Source:** [ACL Anthology record and abstract](https://aclanthology.org/2025.emnlp-main.1187/).
- **Credibility / recency / bias risk:** 5 / 4 / 2.

## Findings relevant to this project

CodeRAG targets the gap between relevant and necessary repository context and combines log-probability-guided query construction, multi-path retrieval, and a BestFit reranker. It reports improvements on ReccEval and CCEval. The paper supports keeping multiple retrieval paths and evaluating context utility, rather than replacing every retrieval signal with one dense similarity.

**Short quote:** “multi-path code retrieval”.

## Transfer boundary

The source does not test quantum-inspired phase scoring. It is a current task-specific baseline/control for the evidence channels and downstream objective the proposed phase head would need to improve.
