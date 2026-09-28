# S09 — AlignCoder project extraction

- **Type:** local deep-read of the paper.
- **Path:** [`doc/papers/ALIGNCODER_DEEPREAD.md`](../../../doc/papers/ALIGNCODER_DEEPREAD.md).
- **Credibility / recency / bias risk:** 4 / 3 / 2. The source notes identify paper tables and pages; consult the PDF before making a publication-level claim.

## Project-relevant facts from the extraction

AlignCoder combines query enhancement, an AlignRetriever initialized from UniXcoder, dependency context, and a DeepSeekCoder-1B target-PPL signal. The extraction reports that only the lowest-PPL candidate receives positive winner feedback; query enhancement uses multiple sampled completions; dependency context contributes in ablations. The user's AST chunking addresses a separate boundary issue but does not by itself establish that AST-GR relevance optimizes final EM/ES.

**Short quote:** “only candidate with the lowest target PPL receives” positive feedback.

## Transfer boundary

Use the local report to identify controls and baseline behavior. Recheck the original AlignCoder PDF for final numerical claims in any paper draft.
