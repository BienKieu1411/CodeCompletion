# S06 — Better Call Grep / GrepRAG

- **Publication:** ISSTA 2026 / Proceedings of the ACM on Software Engineering; DOI 10.1145/3832136. The arXiv page identifies the ISSTA 2026 publication.
- **Task:** Repository-level completion on CrossCodeEval and RepoEval-Updated.
- **Method:** Investigates LLM-generated ripgrep commands as index-free retrieval; the improved GrepRAG post-processes results with identifier-weighted reranking and structure-aware deduplication.
- **Reported result:** Authors report 7.04–15.58% relative code-EM improvement over best baselines on CrossCodeEval. The naive grep stage is fast, but query generation and the full pipeline must be measured separately.
- **Limitations:** Lexical ambiguity, context redundancy and fragmented chunks remain issues; the main generators are much larger than DeepSeek-Coder-1.3B. Tool-generated grep is a useful complementary retrieval source, not semantic dependency proof.
- **Short exact quote:** “identifier-weighted re-ranking and structure-aware deduplication”
- **Sources:** [ISSTA DOI](https://doi.org/10.1145/3832136), [arXiv HTML with method/results](https://arxiv.org/html/2601.23254).
