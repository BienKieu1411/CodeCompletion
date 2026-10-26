# Research plan — repository-level code completion (cutoff: 2026-10-24)

## Question

What did the 2026 code-completion literature actually add to repository context selection, and what is a credible next experiment for a single-pass, left-context-only FIM system using UniXcoder and DeepSeek-Coder-1.3B?

## Scope

- Primary scope: repository-level fill-in-the-middle / code completion that retrieves or selects cross-file context.
- Adjacent scope, kept separate: full-function repository code generation, agentic repository exploration, long-context attention, and code-search-only work.
- Historical comparison: RepoCoder/Repoformer, graph/dataflow retrieval, RLCoder and AlignCoder, and CodeRAG.
- Publication cutoff: 2026-10-24. Venue publication is distinguished from arXiv posting and from industry-track deployment evidence.

## Research steps

1. Search official ACL Anthology, conference programs/proceedings, publisher records, and author-hosted paper/code only when needed for technical details.
2. Read abstracts and method/evaluation sections for the core papers; record a short exact quote, scope, method, comparison limits, and source URL in one note per work.
3. Triangulate the main claims across independent methods: hybrid structural retrieval (AIRCoder, ACAR, RepoFuse), interaction-aware context selection (RepoShapley), graph/static-analysis retrieval (DraCo, GraphCoder, RepoHyper, RepoScope), lexical retrieval (GrepRAG), and chunk/context handling (LC²).
4. Run an adversarial pass: inspect task mismatch, model/backbone mismatch, metric and budget mismatch, reported latency conditions, and future-context leakage risks.
5. Translate only the evidence that transfers to the project's constraints into a bounded proposal; do not make a novelty or “will beat AlignCoder” claim without experiments and a broader novelty search.

## Status

Completed as a bounded literature survey, not an exhaustive systematic review. The source index and limitations are in `sources.csv`, `findings.md`, `adversarial_review.md`, `refresh_targets.md`, and `report.md`.
