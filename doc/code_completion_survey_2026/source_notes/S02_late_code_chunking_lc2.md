# S02 — Late Code Chunking (LC²)

- **Publication:** ACL 2026, Short Papers.
- **Task:** Direct repository-level code completion; RepoEval and CrossCodeEval.
- **Method:** Separate the representation used to retrieve from the context later shown to the generator. It retrieves 512-token chunks, expands context by up to another 512 tokens, and can append signatures/docstrings for up to three functions. The retrieval query is the last 64 tokens of the unfinished code. The main table uses DeepSeek-Coder-1.3B with UniXcoder and retrieves top 5 under a 4,096-token cross-file context cap.
- **Useful result:** On CCEval-Python, Fixed-Token is 22.01 EM and LC² is 26.35 (4.34 points, approximately 19.7% relative). Table 2 also reports Java 23.38 to 24.22.
- **Limitations:** The paper itself notes that irrelevant function augmentation can hurt; exact top-K, prompt and generation settings differ from this project. It modifies how retrieved evidence is expanded, not the retriever objective alone.
- **Short exact quote:** “Code Retrieval Context”
- **Sources:** [ACL paper](https://aclanthology.org/2026.acl-short.64/), [full PDF](https://aclanthology.org/2026.acl-short.64.pdf).
