# Adversarial review

## Strongest objections to the proposed direction

1. **“This is just another hybrid retriever.”** AIRCoder already combines text, dependency existence, and structural hierarchy with query-adaptive fusion; ACAR combines lexical and AST/LSP retrieval; RepoFuse combines static semantic context and similar code. A proposal that merely adds a graph feature or fuses BM25 and UniXcoder is not novel enough.
   - **Response:** constrain the claim to whether *typed, query-anchored repository relations* improve retrieval decisions on left-context-only FIM examples after controlling candidate pool, generator, chunking, and budget. This is a research hypothesis, not a novelty claim already cleared.

2. **“The current benchmarks may not require repository reasoning.”** A test sample may contain cross-file context even if the target can be completed from local context; apparent retrieval gains may come from language priors, lexical overlap, or data leakage.
   - **Response:** annotate or approximate dependency-required / independent / single-file strata; report effect separately. ArkRepoBench motivates this distinction, but its ArkTS labels cannot simply be assumed for CCEval/RepoEval.

3. **“Static dependency graphs are incomplete or stale.”** Dynamic dispatch, reflection, generated code, aliases, unresolved imports, and language-specific tooling can break static edges. GrepRAG notes graph-index construction/maintenance latency; exact lexical matching can outperform semantic or graph retrieval on explicit symbols.
   - **Response:** preserve sparse lexical and UniXcoder retrieval as parallel evidence; use typed edges only with confidence/source metadata; test parser failures and unresolved-edge coverage. Do not make graph retrieval a hard filter.

4. **“A one-pass candidate ranker cannot model set interactions.”** RepoShapley demonstrates complementarity and interference. A candidate that looks weak alone might only help alongside another; pointwise marginal scores or a simple listwise target may still choose a bad slate.
   - **Response:** begin with a one-pass relation-aware reranker and evaluate full-slate utility; if interactions dominate, add a small slate-level utility experiment. Do not claim that a listwise objective alone solves coalition effects. RepoShapley is a warning and a costly reference, not a free solution.

5. **“The best 2026 methods may be inaccessible under the local model/runtime.”** ACAR reports its headline numbers on Qwen2.5-Coder 3B and larger backbones; RepoFuse has production infrastructure; RepoShapley fine-tunes a generator/controller; SpecAgent amortizes work at indexing. These are not equivalent to DeepSeek-Coder-1.3B with a separately trained UniXcoder.
   - **Response:** use them as system-design baselines, reproduce only the mechanism that fits, and maintain an exact same-model/same-prompt local comparison.

6. **“Maybe the real bottleneck is chunking or prompt serialization, not retrieval.”** LC² gains materially by separating retrieval chunks from comprehension context; it also reports that irrelevant signature augmentation can reduce quality. GrepRAG shows chunk fragmentation and duplicates waste context.
   - **Response:** freeze the already selected fixed chunking for the first structural-retrieval ablation; separately test context expansion/AST hints later. Otherwise improvements cannot be attributed.

7. **“Current reported SOTA may be contaminated or not comparable.”** SpecAgent explicitly discusses future-context leakage. GrepRAG uses very large generators; LC² and AIRCoder have different budgets and candidate counts; ACAR reports warm-cache overhead. FSE industry metrics are single-company reports.
   - **Response:** state the cutoff, source status, model size, prompt, decoding, retrieval budget, and timing condition in every comparison. Keep official venue year separate from arXiv upload year.

## What would falsify the proposed contribution

- Typed relation features do not improve EM/ES or dependency recall over a same-pool lexical+dense reranker.
- Gains disappear on dependency-required examples or are equally present on single-file examples.
- Relation extraction is too incomplete/slow, or the method requires multiple retrieval rounds/LLM calls to work.
- Improvement is attributable to changed prompt, larger context, chunk expansion, or generator decoding rather than relations.
- A broad novelty search finds a directly equivalent left-context, typed-edge, one-pass FIM ranker with the same training signal.
