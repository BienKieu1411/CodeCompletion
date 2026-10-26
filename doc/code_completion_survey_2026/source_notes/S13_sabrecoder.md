# S13 — SabreCoder

- **Publication:** ACL 2026, Long Papers.
- **Task:** Long-code completion efficiency, via structured block-sparse attention rather than repository candidate retrieval.
- **Method:** Parses semantic chunks, creates chunk-level sparse attention patterns using dependency analysis and similarity, then maps them to GPU-efficient blocks.
- **Reported result:** 45–55% lower time-to-first-token with accuracy within 3% in its evaluated LCC/CrossCodeEval settings.
- **Use here:** Adjacent axis if generator prefill dominates; it does not train UniXcoder or replace context retrieval.
- **Short exact quote:** “Structure-aware block-sparse attention”
- **Sources:** [ACL paper](https://aclanthology.org/2026.acl-long.1189/), [full PDF](https://aclanthology.org/2026.acl-long.1189.pdf).
