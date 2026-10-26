# S04 — ACAR

- **Publication:** EMNLP 2026 Main; official ACL Anthology record.
- **Task:** Direct fill-in-the-middle repository-level completion on RepoEval, ReccEval and CrossCodeEval, across Python, Java, TypeScript and C#.
- **Method:** Fuse lexical retrieval with AST/LSP structural retrieval at line level using reciprocal-rank fusion, then syntax-aware post-processing of generated code.
- **Reported result:** Qwen2.5-Coder-3B reaches 42.98 EM on ReccEval and 32.86 on CrossCodeEval, compared with 36.25 and 28.40 for the strongest retrieval baseline reported. Non-decoding overhead is reported as 87–95 ms under warm-cache conditions.
- **Limitations:** These numbers use a stronger/different backbone and a distinct protocol; the timing is warm-cache and excludes decoding. It is a strong mechanism-level baseline, not an apples-to-apples score target for DeepSeek-Coder-1.3B.
- **Short exact quote:** “lexical and AST/LSP-based structural retrieval”
- **Sources:** [EMNLP paper](https://aclanthology.org/2026.emnlp-main.81/), [paper PDF](https://aclanthology.org/2026.emnlp-main.81.pdf).
