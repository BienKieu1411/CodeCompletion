# S07 — SpecAgent

- **Publication:** ACL 2026, Long Papers.
- **Task:** Code completion with indexing-time speculative repository exploration.
- **Method:** Proactively explores repository files while indexing and precomputes speculative context anticipating likely future edits, shifting work away from interactive retrieval time.
- **Reported result:** The paper reports absolute gains of 9–11% (48–58% relative) over its strongest baselines on a synthetic benchmark designed to avoid future-context leakage.
- **Limitations:** It relies on repository indexing and speculative future edits; this differs from a single query using only left context plus repository code. The benchmark caution is highly relevant: accidental use of future target-file content can inflate completion scores.
- **Short exact quote:** “proactively exploring repository files during indexing”
- **Sources:** [ACL paper](https://aclanthology.org/2026.acl-long.786/), [full PDF](https://aclanthology.org/2026.acl-long.786.pdf).
