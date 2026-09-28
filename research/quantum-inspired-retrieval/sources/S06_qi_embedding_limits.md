# S06 — On the Representational Limits of Quantum-Inspired Document Embeddings

- **Type / year:** single-author arXiv preprint, 2026, arXiv:2604.09430.
- **Source:** [arXiv abstract](https://arxiv.org/abs/2604.09430) · [HTML paper](https://arxiv.org/html/2604.09430).
- **Credibility / recency / bias risk:** 2 / 5 / 5. Useful adversarial signal only: preprint, small/custom corpora and limited query counts; not independent replication.

## Findings relevant to this project

The paper reports that its tested quantum-inspired document embeddings are weak as standalone retrievers and that hybrid configurations can sometimes recover performance, with lexical BM25 often carrying most of the result. It also warns that matching a teacher embedding geometry does not necessarily preserve local retrieval neighborhoods.

**Short quote:** “better suited to auxiliary or hybrid components”.

## Transfer boundary

Do not generalize the paper's numbers to code completion. Its value here is to make the standalone-QI hypothesis earn a direct comparison against BM25 and classical hybrid fusion. Source credibility is deliberately down-weighted.
