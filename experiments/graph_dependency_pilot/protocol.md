# Preregistered pilot protocol: left-prefix dependency graph

Date: 2026-10-24  
Status: frozen before remote generation/evaluation

## Question and hypothesis

Does adding high-confidence, prefix-visible cross-file import-to-definition
edges (a) improve the snippets ranked by the frozen AlignCoder retriever and
(b) improve completion when the same dependency links are exposed to the
generator?

H1 (retrieval): graph reranking improves dependency-definition coverage@10
over the unmodified AlignCoder retriever on the selected high-dependency set.
H2 (completion): graph reranking and/or compact dependency notes improve
paired completion ES over the AlignCoder-code-only arm. These are hypotheses,
not assumed outcomes.

## Sample and leakage boundary

Use 64 deterministic, uniformly sampled eligible tasks from each of
CCEval-Python, CCEval-Java, RepoEval-Line, and RepoEval-API (256 tasks total).
Eligibility requires at least one explicit import in the left context that
resolves to a symbol/module in a different source file in that task's
repository pool. Select only from the first 400 rows per benchmark. This is a
targeted diagnostic subset, not a representative benchmark estimate.

Eligibility and graph extraction may inspect only `left_context`, target file
path, and repository source files. `right_context` and `groundtruth` are held
out until prediction scoring. They are never passed to chunking, graph
construction, query enhancement, candidate scoring, or generation.

## Fixed pipeline and interventions

Freeze model revisions and AlignCoder settings from the prior repository run
manifest: AlignRetriever `a179c0fdbc5a56d83ae142064ef1ea773a841753`,
DeepSeek-Coder-1.3B-Base `c919139c3a9b4070729c8b2cca4847ab29ca8d94`, default
AlignCoder mini-blocks, BM25 top-100 for seed and enhanced query, eight
query-enhancement samples at temperature 0.8/top-p 0.95, API blocks, retriever
top-10, stop block, 1,536 cross-file tokens, 2,048 total input tokens, and 64
max generated tokens. Query-enhancement outputs are generated once and reused
across all four arms; final decoding is greedy. All models remain frozen.

1. `aligncoder_code`: unmodified AlignCoder retrieval; generator sees retrieved
   code blocks only.
2. `graph_rerank_code`: same BM25/API candidate pool and frozen cosine scores;
   rerank with `cosine + 0.12 * resolved-edge coverage`; generator sees code
   only.
3. `aligncoder_dep_prompt`: unmodified AlignCoder top-10; generator sees code
   and up to two compact, resolved dependency-edge notes per matching chunk.
4. `graph_rerank_dep_prompt`: graph-reranked top-10 plus those notes.

The graph uses explicit imports parsed from the left prefix and exact
cross-file symbol/module resolution. The coefficient 0.12 is fixed in advance
and will not be tuned on these 256 examples.

## Outcomes and decision rule

Primary retrieval outcome: mean per-task resolved dependency coverage@10.
Primary completion outcome: paired ES delta versus `aligncoder_code`, reported
separately for each benchmark (AlignCoder fuzzy character ES for CCEval; both
AlignCoder's fuzzy micro ES and official repo-macro edit ES for RepoEval).
Also report EM and identifier precision/recall/F1 where applicable. Save all
prompts, query-enhancement outputs, BM25 top-100, complete rerank candidate
scores, selected top-10s, raw/postprocessed predictions, and per-example
metrics.

Call the pilot directionally promising only if the graph raises dependency
coverage@10 and yields positive paired completion ES in at least three of four
benchmark groups, with no benchmark losing more than 1.0 absolute ES point.
Even then, do not claim it beats AlignCoder on the full task: the selected
subset is high-dependency and small. The next step would be a larger,
repository-disjoint evaluation with a representative test sample and a
clustered confidence interval.

## Runtime and reproducibility

Run once on one Google Colab A100 40-GiB session using vLLM offline batched
inference. Record the exact runner, selection
manifest, model revisions, runtime device, outputs, and session status. Stop
the Colab session only after the result archive is downloaded and verified.
