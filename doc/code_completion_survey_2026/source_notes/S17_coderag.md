# S17 — CodeRAG

- **Publication:** EMNLP 2025 Main.
- **Task:** Repository-level code completion; ReccEval and CrossCodeEval.
- **Method:** Log-probability-guided query construction, multi-path retrieval, and preference-aligned BestFit reranking.
- **Use here:** Strong prior art for generator-informed retrieval and for combining complementary retrieval channels. It also means a new method should identify a gap beyond simply adding another retrieval path.
- **Limitations:** Its BestFit/query pipeline is not the same as one-shot UniXcoder retrieval; exact resource and model costs need to be controlled.
- **Short exact quote:** “log probability guided query construction”
- **Sources:** [EMNLP 2025 paper](https://aclanthology.org/2025.emnlp-main.1187/), [full PDF](https://aclanthology.org/2025.emnlp-main.1187.pdf).
