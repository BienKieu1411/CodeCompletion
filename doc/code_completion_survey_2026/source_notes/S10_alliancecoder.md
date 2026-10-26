# S10 — AllianceCoder / “What to Retrieve…”

- **Publication:** ICSE 2026 Research Track.
- **Task:** Repository-level function generation on CoderEval and RepoExec, not direct FIM completion.
- **Finding:** The study reports that in-context code and potential APIs help, while retrieved similar code can introduce noise; AllianceCoder uses chain-of-thought decomposition and API retrieval by semantic description.
- **Use here:** Supports separating “dependency/API evidence” from “similar-looking examples” and treating similarity snippets as potentially harmful.
- **Limitations:** Different target granularity, prompt and generation protocol. Do not compare its Pass@1 to CCEval EM/ES.
- **Short exact quote:** “retrieved similar code often introduces noise”
- **Source:** [ICSE 2026 paper record](https://conf.researchr.org/details/icse-2026/icse-2026-research-track/182/What-to-Retrieve-for-Effective-Retrieval-Augmented-Code-Generation-An-Empirical-Stud).
