# RLCoder retrieval with a dependency-graph left context

## Question

On the same 256 dependency-relevant tasks, does replacing the source prefix
with a structured dependency graph in **both** retrieval and generation change
retrieval quality and completion quality under RLCoder's released retriever?

## Paired arms

| Arm | Query sent to BM25 and RLRetriever | Left-context segment sent to DeepSeek |
| --- | --- | --- |
| `raw_left_context` | RLCoder's last 20 non-empty prefix lines | Original left context |
| `dependency_graph` | Compact, prefix-visible import-to-definition graph | The same serialized graph |

Both arms use RLCoder's released `nov3630/RLRetriever` checkpoint, BM25 top-100,
RLRetriever top-10 including its stop block, default blank-line mini-blocking
(maximum 15 non-empty lines), DeepSeek-Coder-1.3B-Base, and RLCoder's 1,536 /
2,048 / 64 cross-file, total-context, and generation budgets. Generation is
greedy. No query enhancement, score bonus, or per-snippet graph notes are used.

The graph contains only resolved dependencies derived from the visible left
prefix and repository source files. It is token-fitted to the 252 content-token
query window. The complete graph and the exact graph presented to either model
are both saved. The original prefix is used for Python statement-boundary
post-processing during scoring only; it is never passed to the graph-arm
generator.

## Data and interpretation

The task IDs are the same fixed 64 examples per CCEval-Python, CCEval-Java,
RepoEval-Line, and RepoEval-API used in the earlier dependency-filtered pilot.
This is a targeted diagnostic set, not a full-benchmark estimate. The selected
RepoEval examples come from only two repositories, so RepoEval cluster-level
inference is limited.

## Saved output

- `retrieval_trace.jsonl.gz`: both queries, full dependency graph, BM25 top-100,
  every RLRetriever-ranked candidate (including ranks outside top-10), exact
  selected snippets, and dependency-recall metrics.
- `prompts.jsonl.gz`: exact prompt text and token IDs for every generated task.
- `predictions.jsonl.gz`: raw and RLCoder-postprocessed generations with EM, ES,
  identifier metrics, and RepoEval's repo-macro metrics.
- `retrieval_metrics.jsonl.gz`, `metrics_by_benchmark.csv`, and `summary.json`.

## Run

Using the local `bienkieu_env`:

```bash
python experiments/rlcoder_dependency_graph/package_bundle.py
colab new --gpu A100 --session rlgraph-20261026
colab upload experiments/rlcoder_dependency_graph/rlcoder_dependency_graph_bundle.tar.gz /content/rlcoder_dependency_graph_bundle.tar.gz --session rlgraph-20261026
colab exec --session rlgraph-20261026 --file experiments/rlcoder_dependency_graph/remote_setup.py
```

The remote setup starts a detached process and writes its log under
`/content/rlcoder_dependency_graph/runner.log`; polling the Colab client is not
required for the subprocess to continue while the session remains allocated.
