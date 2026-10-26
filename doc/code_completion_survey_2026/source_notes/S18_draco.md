# S18 — DraCo

- **Publication:** ACL 2024 Long Papers.
- **Task:** Repository-level code completion.
- **Method:** Dataflow-guided retrieval from a repository-specific context graph; uses static analysis to identify relevant dataflow/type relations and constructs prompts with retrieved context.
- **Use here:** Establishes that graph/dataflow-aware repository retrieval predates the current work. A graph contribution cannot be claimed as novel by itself.
- **Limitations:** Graph construction and static analysis are language/tooling dependent; compare on the same FIM task and latency budget.
- **Short exact quote:** “Dataflow-Guided Retrieval Augmentation”
- **Sources:** [ACL 2024 paper](https://aclanthology.org/2024.acl-long.431/), [arXiv manuscript](https://arxiv.org/abs/2405.19782).
