# Prefix-dependency graph pilot — results

## Takeaway

On this deliberately dependency-heavy 256-example subset, adding prefix-derived
dependency edges to rerank AlignCoder candidates substantially increased
retrieval coverage of resolvable imports. It did **not** improve completion
scores in a statistically reliable way. The experiment supports “the graph can
steer retrieval toward imported dependencies,” but does not yet support “those
retrieved snippets make DeepSeek complete code better.”

Treat this as a diagnostic pilot, not a benchmark claim: the sample is
dependency-filtered and contains only 64 examples from each benchmark.

## Experimental setup

- 64 examples each from CCEval-Python, CCEval-Java, RepoEval-Line, and
  RepoEval-API; 256 total. Of the first 400 rows scanned per benchmark, 336,
  330, 372, and 400 respectively met the dependency eligibility rule.
- Eligibility required an import visible in the left context to resolve to a
  declaration/module in another source file in that repository pool. Selection
  did not inspect right context or ground truth; those fields were used only
  for completion scoring.
- Frozen AlignCoder AlignRetriever (revision a179c0f…) and
  deepseek-ai/deepseek-coder-1.3b-base (revision c919139…); AlignCoder default
  blank-line mini-blocking (up to 15 nonempty lines), BM25 top-100,
  AlignRetriever top-10, 1,536 cross-file tokens, 2,048 total prompt tokens.
- The same AlignCoder-style query-enhancement generations (8 samples,
  temperature 0.8, top-p 0.95) were shared by all four arms; final completion
  generation was greedy.
- vLLM 0.32.0 ran the generator on one Colab A100 40GB. It generated 2,048
  query-enhancement completions in 40.7s and 1,024 final completions in 31.7s
  after engine initialization. The Colab session has been stopped.

## Completion scores

Edit similarity (ES) and exact match (EM) are percentages. The three
intervention arms are paired against the same tasks and shared query
enhancements.

| Benchmark | AlignCoder code ES / EM | Graph rerank, code only ES / EM | AlignCoder + dependency notes ES / EM | Graph rerank + dependency notes ES / EM |
|---|---:|---:|---:|---:|
| CCEval-Java | 53.52 / 6.25 | 51.20 / 6.25 | 54.25 / 6.25 | 51.38 / 6.25 |
| CCEval-Python | 39.34 / 7.81 | 41.02 / 6.25 | 39.22 / 7.81 | 38.84 / 7.81 |
| RepoEval-API | 41.38 / 1.56 | 40.03 / 1.56 | 41.39 / 1.56 | 39.73 / 1.56 |
| RepoEval-Line | 24.09 / 1.56 | 22.95 / 1.56 | 24.53 / 1.56 | 24.73 / 1.56 |

RepoEval repo-macro ES:

| Benchmark | AlignCoder code | Graph rerank, code only | AlignCoder + dependency notes | Graph rerank + dependency notes |
|---|---:|---:|---:|---:|
| RepoEval-API | 32.75 | 31.49 | 33.19 | 32.15 |
| RepoEval-Line | 17.12 | 16.38 | 17.22 | 17.58 |

Paired bootstrap deltas against AlignCoder-code, pooled over the 256 tasks:

| Intervention | ES delta (pp), 95% CI | Identifier-F1 delta (pp), 95% CI |
|---|---:|---:|
| Graph rerank, code only | -0.82 [-2.02, 0.34] | -0.53 [-1.72, 0.60] |
| AlignCoder retrieval + dependency notes | +0.26 [-0.34, 0.89] | +0.60 [-0.11, 1.38] |
| Graph rerank + dependency notes | -0.89 [-2.35, 0.47] | -0.95 [-2.53, 0.48] |

These are 2,000-draw, paired example-level bootstrap intervals. After inference
finished, a key collision was found because RepoEval line/API can reuse
task_id strings for different completion positions. The pairing was corrected
to use (benchmark, task_id); prediction scores themselves were unchanged.
RepoEval tasks also come from only two repositories here, so these intervals
must not be read as repository-level confidence intervals.

All intervals include zero. The dependency-note-only arm trends positive on
identifier F1, but this pilot does not establish a reliable completion gain.
The combined graph-rerank-plus-notes arm is not additive here.

## Retrieval dependency coverage

Coverage is the mean fraction of each task's resolved imported symbols
appearing in its selected top-10. Hit rate is the fraction of tasks with at
least one such dependency represented.

| Benchmark | AlignRetriever coverage | Graph-reranked coverage | Coverage change | Hit rate: Align → graph |
|---|---:|---:|---:|---:|
| CCEval-Java | 64.53% | 88.98% | +24.45 pp | 84.38% → 96.88% |
| CCEval-Python | 52.04% | 84.48% | +32.44 pp | 78.13% → 100.00% |
| RepoEval-API | 23.48% | 33.82% | +10.35 pp | 51.56% → 60.94% |
| RepoEval-Line | 12.75% | 45.31% | +32.56 pp | 21.88% → 68.75% |

## Interpretation and next experiment

The graph score used here is a simple import-to-definition/module edge
coverage bonus (cosine + 0.12 × coverage) over the exact same BM25 candidate
pool. It is a transparent reranker, not a learned graph encoder. It retrieves
more snippets that contain the statically resolved imported symbols, but
symbol coverage is only a proxy for whether a snippet helps predict the next
code. The ES results suggest that pushing dependency coverage alone can displace
other useful context or provide dependencies that the generator does not use.

Next, inspect paired per-example traces—especially where dependency coverage
rose but ES fell—then run a small held-out tuning set for the graph weight.
Separate three questions in the follow-up:

1. Does the reranker improve retrieval relevance (dependency resolution and
   target/API relevance), not just symbol presence?
2. Do dependency annotations help when the retrieved code is held fixed?
3. Does graph reranking help when the generator receives the same token budget
   and identical code snippets?

Only after that should a repository-disjoint, full-benchmark run be used to
claim a completion improvement. Do not tune the graph weight on these test
examples and then report the same examples as final evaluation.

## Artifacts

- Recommended full bundle with corrected paired-bootstrap intervals:
  graph_dependency_pilot_results_reanalyzed.tar.gz
- Original run archive: graph_dependency_pilot_results.tar.gz. It preserves
  the raw run output; its bootstrap interval section predates the
  (benchmark, task_id) collision fix described above.
- Readable metrics: outputs/metrics.json
- Raw predictions: outputs/predictions.jsonl.gz
- Full retrieval evidence: outputs/retrieval_trace.jsonl.gz
- Actual prompts: outputs/prompts.jsonl.gz
- Resolved run config: outputs/run_config.json
- Preparation/selection bundle and manifest: pilot_bundle.tar.gz,
  selection.json, tasks.jsonl.gz, pools.jsonl.gz, files.jsonl.gz
- Pilot code and preregistered protocol:
  experiments/graph_dependency_pilot/
