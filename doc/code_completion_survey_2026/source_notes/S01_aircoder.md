# S01 — AIRCoder

- **Publication:** ACL 2026, Long Papers; official ACL Anthology record.
- **Task:** Repository-level completion, CrossCodeEval and RepoEval.
- **Method:** Structure-preserving chunks; eight query–candidate signals: sparse/dense textual similarity, dependency existence, path proximity, file/class/function lexical similarity, and dense structural similarity. It offers training-free RRF and an adaptive MLP. The MLP is trained on generator-derived Edit Similarity using pairwise MSE of positive/negative score differences.
- **Useful result:** With DeepSeek-Coder-1.3B, the paper reports CCEval-Python EM 31.26 for its RRF variant, versus 30.92 for CodeRAG and 30.28 for DraCo in its protocol.
- **Limitations for this project:** The learned variant requires pairwise records; dependency extraction uses Qwen3-8B; candidate chunking/context budget and evaluation need exact reproduction before numerical comparison. RRF is a simpler first baseline.
- **Short exact quote:** “eight complementary metrics across three dimensions”
- **Sources:** [ACL paper](https://aclanthology.org/2026.acl-long.1166/), [full PDF](https://aclanthology.org/2026.acl-long.1166.pdf).
