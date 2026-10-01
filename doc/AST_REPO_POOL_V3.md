# AST Repository Pool v3

This is the new data contract for repo-level code completion retrieval. It is
independent of the removed PPO/RRPO code.

## What is stored

`train.parquet` contains one compressed row per repository. Each row stores:

- all parseable Python/Java source files;
- AST chunks measured with the DeepSeek-Coder tokenizer;
- eligible target spans and dependency evidence;
- quality statistics and a deterministic train/valid split.

The pool does not store a fixed five examples per repository. The epoch sampler
chooses at most 2,000 unique training repositories, samples a target file first,
then samples a target kind/span from that file, and mines the cross-file BM25
pool from the visible left context. There is no right context and no gold chunk
in the retrieval query.

## Quality contract

The current default filter is deliberately strict:

- 4–200 unique `.py`/`.java` source files;
- at least 3 substantive files, each substantive file having at least 12 code
  lines;
- at least 2 substantive production files (test/fixture files are retained but
  cannot be the whole repository);
- 120–200,000 substantive code lines per repository;
- file size at most 500,000 bytes;
- explicit generated files and generated-source headers excluded;
- benchmark exact-file overlaps excluded;
- at least two other chunked files for every retained target;
- files with parser errors are excluded; a repository is kept only if enough
  parseable files and targets remain.

For chunks, the DeepSeek tokenizer is the authority:

- hard maximum: 384 generator tokens;
- soft packing target: 96 generator tokens;
- retriever cap: 508 UniXcoder tokens;
- cross-file prompt budget: 2,344 generator tokens, at most 10 complete snippets;
- short AST siblings are packed when the combined span fits;
- oversized leaves use a tokenizer-bounded binary-search window splitter;
- residual short fragments are retained rather than dropped, padded, or
  duplicated. Comments are AST children and remain in source-backed chunks.

Targets are AST-aware `member_suffix`, physical `line`, and multi-line
`api_statement` cuts. A target requires at least 8 previous lines, 64 DeepSeek
tokens of prefix context, at least 6 DeepSeek target tokens by default, and a
bounded target length. The first repository file
is kept as the anchor and is not sampled as a target, following the established
AlignCoder/RLCoder target-file convention.

## Build and sample

```bash
PY=/Users/kieugiangbien/bienkieu_env/bin/python

$PY -m src.data.build_ast_repo_pool \
  --root datasets/data4aligncoder \
  --exclusions local_server_results/data_audit/benchmark_file_hashes.json \
  --output local_server_results/ast_repo_pool_v3/train.parquet \
  --work-dir local_server_results/ast_repo_pool_v3/work \
  --workers 2

$PY -m src.train.epoch_data_loader \
  --pool local_server_results/ast_repo_pool_v3/train.parquet \
  --output local_server_results/ast_repo_pool_v3/epoch_001.parquet \
  --epoch 1 --seed 123
```

The build is resumable at repository-shard level. Epoch sampling is
deterministic for a fixed `(seed, epoch)` and changes the repository subset and
target choice across epochs. The resulting epoch artifact has one task per
selected repository whenever that repository has a valid cross-file candidate
pool.

## Benchmark input preparation

The same AST chunker, input-only cleanup, prefix-only AST hints and BM25
proposal pool are used for the four frozen test outputs. Labels are copied
unchanged; no `groundtruth` or `right_context` is read while building the
`ast_payload` input view. RepoEval `test_0` and `test_1` are merged in source
order per benchmark.

```bash
$PY -m src.data.prepare_test_data \
  --root datasets/data4aligncoder \
  --output-dir local_server_results/ast_test_inputs_v1 \
  --cache local_server_results/ast_test_inputs_v1/chunks.sqlite
```

The test output keeps the original six columns and adds one compressed
`ast_payload` column containing `model_left_context`, the complete ranked
candidate pool, and the budgeted cross-file view.
