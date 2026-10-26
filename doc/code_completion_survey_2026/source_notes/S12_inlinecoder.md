# S12 — InlineCoder

- **Publication:** FSE 2026 Research Papers.
- **Task:** Repository-level full-function generation on DevEval and RepoExec.
- **Method:** Drafts an anchor completion, inlines it into callers (upstream) and retrieves callees (downstream) along the call graph.
- **Use here:** Demonstrates caller and callee relations as distinct evidence roles, but the method is multi-stage and relies on a draft generation; this is not the user's desired one-pass left-context retriever.
- **Short exact quote:** “Upstream Inlining” and “Downstream Retrieval”
- **Sources:** [FSE 2026 record](https://conf.researchr.org/details/fse-2026/fse-2026-research-papers/161/In-Line-with-Context-Repository-Level-Code-Generation-via-Context-Inlining), [arXiv manuscript](https://arxiv.org/abs/2601.00376).
