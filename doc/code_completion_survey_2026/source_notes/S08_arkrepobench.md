# S08 — ArkRepoBench

- **Publication:** Findings of ACL 2026.
- **Task:** Repository-level ArkTS/HarmonyOS completion; benchmark contribution.
- **Method/contribution:** 7,519 samples from 20 repositories; explicitly separates Single-File, Cross-File Independent, and Cross-File Dependent cases using dependency analysis.
- **Useful result:** The paper finds cross-file context can be double-edged and sparse Jaccard can beat dense retrieval on ArkTS. More important for this project is the distinction between context being present and a dependency actually being necessary.
- **Limitations:** ArkTS/HarmonyOS findings may not transfer to Python/Java; use the dependency taxonomy as an evaluation design principle, not as evidence that Jaccard wins on CCEval.
- **Short exact quote:** “Cross-File Independent” and “Cross-File Dependent”
- **Sources:** [ACL paper](https://aclanthology.org/2026.findings-acl.969/), [full PDF](https://aclanthology.org/2026.findings-acl.969.pdf).
