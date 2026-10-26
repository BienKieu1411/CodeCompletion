# Prefix-dependency graph pilot

This is a deliberately narrow, paired pilot for the question: can explicit
left-prefix-to-repository-definition edges improve (1) the snippets selected
for completion and (2) the generator's use of those snippets?

## Sample selection and leakage boundary

`prepare_pilot.py` selects 64 tasks from each of CCEval-Python, CCEval-Java,
RepoEval-Line, and RepoEval-API (256 total). It keeps tasks only when an import
visible in `left_context` resolves to a declaration/module in a different file
in that task's repository candidate pool. Selection does not inspect
`groundtruth` or `right_context`; those fields are included only for scoring.
The selection is intentionally high-dependency and is not a representative
benchmark sample.

The bundle deduplicates source files by content hash and ships the cloned
AlignCoder API extractor plus its language parsers.

## Four paired arms

| Arm | Retrieval | Generator input |
|---|---|---|
| `aligncoder_code` | AlignCoder BM25 top-100 → frozen AlignRetriever top-10 | Code snippets only |
| `graph_rerank_code` | Same candidate pool, rank by cosine + `0.12 × prefix-edge coverage` | Code snippets only |
| `aligncoder_dep_prompt` | AlignCoder BM25 → AlignRetriever top-10 | Code plus compact resolved-edge notes |
| `graph_rerank_dep_prompt` | Graph reranking | Code plus compact resolved-edge notes |

The run follows the repository's recorded AlignCoder evaluation configuration:
blank-line mini-block chunking (up to 15 nonempty lines), BM25 seed and
enhanced-query top-100, 8 query-enhancement samples (temperature 0.8/top-p
0.95), import-derived API blocks, AlignRetriever top-10, stop block, 1,536
cross-file tokens, 2,048 total context tokens, and 64 generated tokens. The
same query-enhancement outputs are shared across arms. The final generator is
greedy and frozen. Dependency notes replace code tokens under the same
1,536-token cross-file ceiling; total generator input remains capped at 2,048
tokens.

Graph edges are typed import-to-definition/module relations extracted only
from the visible prefix and other repository files. The graph score rewards a
candidate chunk only when it contains a resolved imported symbol (or is an
exact imported module). This pilot does not claim a full dataflow graph.

## Outputs and interpretation

The runner saves prompt text, query-enhancement generations, BM25 top-100,
both top-10 lists, per-task predictions, retrieval dependency coverage, and
paired-bootstrap confidence intervals. Completion metrics are EM, AlignCoder
ES, and identifier F1; RepoEval also reports repo-macro EM/ES.

This is a small hypothesis test, not evidence that the method beats AlignCoder
on the full benchmarks. A positive result should trigger a larger,
repository-disjoint evaluation; a retrieval-coverage gain without a paired
completion gain is not a success.

## Run

Use `/Users/kieugiangbien/bienkieu_env/bin/python` to prepare the bundle, then
Google Colab CLI (`colab new --gpu A100`, `colab upload`, `colab exec`). The
runner uses vLLM offline batched inference and public Hugging Face model
downloads. It validates that Colab supplied a 40-GiB A100 before loading the
models.
