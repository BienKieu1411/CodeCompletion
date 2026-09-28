# S04 — A Quantum Interference Inspired Neural Matching Model for Ad-hoc Retrieval

- **Authors / year:** Jiang, Zhang, Gao, and Song, SIGIR 2020.
- **Type:** peer-reviewed information-retrieval conference paper.
- **Source:** [DOI / ACM record](https://doi.org/10.1145/3397271.3401070) · [author-hosted paper PDF](https://cic.tju.edu.cn/faculty/zhangpeng/paper/Ad-hocRetrieval.pdf).
- **Credibility / recency / bias risk:** 5 / 2 / 3.

## Findings relevant to this project

QINM motivates interference as a way to model dependencies among query-document matching units, rather than treating each matching signal as independent and simply accumulating relevance evidence. This is the closest established IR precedent for using interference to represent agreement/conflict between retrieval evidence. It evaluates ad-hoc text retrieval, not code search or repository-level completion.

**Short quote:** “ignores the dependencies between terms”.

## Transfer boundary

The transferable idea is cross-evidence interaction, not the paper's exact quantum-probability formalism. For this project, candidate-wise phase fusion over lexical, dense, AST, and dependency scores is a testable analogue; it remains a hypothesis until code-completion metrics validate it.
