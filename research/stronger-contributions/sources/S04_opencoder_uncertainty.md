# S04 — OpenCoder: uncertainty-aware repository code generation

- **Type / status:** Primary arXiv preprint, July 2026; not a peer-reviewed venue in the source record at search time.
- **URL:** https://arxiv.org/abs/2607.24884
- **Credibility / recency / bias risk:** 2 / 5 / 4.
- **Primary locations:** Abstract; §1–2; evaluation and limitations.

## Evidence extracted

- OpenCoder models uncertainty over heterogeneous evidence sources (API knowledge, repository context, similar code), filters/fuses evidence, and includes generation verification and repair.
- Its factorial analysis reports source interactions that vary by evidence and LLM backend rather than one universal additive ranking.
- The abstract reports results on a 32-task RepoExec-inline evaluation; the authors also report that its GPT gain matches a verification-and-repair control and that the Gemini effect is not statistically supported.

## Verbatim excerpt

> “retrieval utility is interaction-dependent”

## Relevance and limits

Broad uncertainty-aware evidence fusion and interaction analysis are already being explored. This is adjacent repository-level code generation, not a direct retriever-only completion comparison; the small evaluation and preprint status limit strength of its performance claims. It weakens generic “uncertainty-aware retrieval” novelty, not the specific use of AST-derived target-to-evidence labels.
