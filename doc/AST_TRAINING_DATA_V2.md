# AST data v2: train labels and shared train/test chunking

Implementation date: 2026-10-01. This is a **new artifact schema**, not a drop-in replacement for the old PPO notebooks. Original datasets/checkpoints are not overwritten. Preparation runs on CPU; it does not train or call a generator.

## What changed

- Reject training repositories with fewer than **4 distinct, nonempty, parseable Python/Java files** after source filtering. Duplicate content does not count as an extra file. Initial eligibility additionally requires at least two files with 8 nonblank/non-comment-prefix lines; this is a heuristic, not a semantic quality score.
- Pack adjacent AST nodes in the same source scope. Group imports, fields, assignments and small methods instead of emitting each as a one-line candidate. Soft packing target: 96 generator tokens; hard core caps: 384 generator / 508 retriever tokens. Substantial definitions can stay separate. The floor is not enforced by padding or duplicating code, and a residual short chunk is retained.
- Large scopes partition into body groups, retaining source-backed enclosing headers. Oversized expressions, strings or control blocks use explicitly tagged bounded source windows, not isolated terminal AST nodes. Syntax-error fallback uses bounded windows, not one chunk per line. Fallbacks are not claimed to be semantic AST units.
- An AST core fit probe is bounded to 16,384 source bytes. Larger units are split, not passed wholesale to the tokenizer. This conservative resource guard can split a highly compressible unit even if its token count would fit; it does not discard benchmark source tails. Training targets above that byte bound are excluded as oversized proposals.
- All chunks retain original UTF-8 byte offsets, raw text, source hash and header spans. Comments are kept as source-backed children. The model-facing representation is separate. A short standalone file or an indivisible single-line construct can still produce a short/one-line chunk; an oversized unit is split into bounded windows rather than dropped.
- New train cuts cover **member suffixes after `.`**, statement lines and multiline API statements. Gold is an exact source slice; no code is invented. Targets have 6–64 generator tokens, or up to 160 for multiline API statements. Prefix minimum: 8 lines and 64 bounded-prefix tokens.

## Input cleanup, not label rewriting

`src/code_input_cleanup.py` is shared by train and test:

- Non-English comments and identifiers are **not** discarded merely for their language.
- Strip recognized license/copyright banners, separator-only comments, empty comments and consecutive identical comments. Preserve pragmas such as `noqa`, `type: ignore`, formatting directives and SPDX identifiers.
- Collapse runs to at most **two blank lines outside literals**; remove trailing spaces on completed non-literal lines; normalize CRLF outside literals.
- Preserve strings, docstrings, regex/URL contents, indentation and final cursor whitespace. Do not replace the literal characters `\\n` as though they were line breaks.
- For an incomplete prefix, clean only a parseable preceding prefix and preserve the uncertain suffix. For a chunk crossing a string/comment boundary, preserve raw text rather than guess. Not every malformed language construct can be safely normalized.
- Inline comment removal uses whitespace so Java tokens cannot accidentally concatenate. Recount actual tokenizer lengths and fall back to raw text if cleanup exceeds a cap.

Blank-line cleanup does **not** rewrite original source offsets. Completion labels and all original test fields remain unchanged.

## Training labels and splits

Up to five tasks per eligible repo are selected deterministically. Selection cycles through member/line/API target kinds with import-supported cross-file symbol evidence, retaining up to one general-control task per repo. Evidence uses AST call/member/type references, not arbitrary argument/local identifiers: `missing(name)` must not count as a dependency merely because another class has a field `name`. “General control” means **no detected evidence**, not a proved zero-retrieval-utility example. Static matching is conservative in some cases and approximate in others; it is not full name/type resolution.

There are two different labels:

1. `target_code`: generated now from the original training source, for completion scoring.
2. Helpful/harmful/context-utility labels: **not generated now**. `label_metadata.utility_labels` remains null until the frozen generator is measured on rendered context sets. A symbol match is never silently used as a positive utility target.

Gold can determine whether a supervised training task has dependency evidence, but it never enters the query or BM25 candidate mining. No matching definition is injected into the pool. `evidence_in_pool` is only a source-span-overlap diagnostic, not oracle recall or measured usefulness. The current file, including its hidden suffix, is excluded from all related-source chunks.

Existing repo train/valid membership is mapped back to the pinned raw source and verified against old prefix+gold bytes. Exact shared substantive files are grouped to avoid crossing splits; conflicting old split anchors cause exclusion rather than silent reassignment. Benchmark-file hashes exclude exact nontrivial overlaps. This does not detect every fork/near-duplicate.

Source: `AlignCoder/Data4AlignCoder`, revision `ce72fe3ef4987e15a9219fea4174bb80700e742c`. Both language source-file SHA256s and tokenizer revisions are checked/saved. Old train source is downloaded separately from `BienKieu/CodeCompletion` and used to preserve split anchors, not to reuse its old targets.

## Benchmark inspection and scope

All 2,665 CCEval Python stored prefixes end after a dot, whereas all 3,922 old train/valid tasks used line/block-boundary cuts. This is an observed structural mismatch, **not proof of why PPO failed**. CCEval Python target median is 12 tokens; RepoEval API target p99 is 133 tokens, with 393/1,600 multiline targets. Accordingly the new task maker supports suffix cuts and longer API labels.

RepoEval stores line-joined segments whose boundary newlines may be omitted. `cursor_contract=line_segments` distinguishes this from CCEval's exact-prefix fragments. Do not infer a midline cursor simply because the stored prefix lacks a final newline. The preparation stage does not silently invent that boundary; the generator prompt adapter must respect the benchmark contract.

Test labels, row count and original columns are frozen. No repo-size filter, target selection, difficulty filter or model-based test selection is applied. Empty-pool test tasks remain in the output. RepoEval shards are merged in source order, separately for line and API.

When two related files share a relative path but contain different source bytes, preserve both with content-qualified internal IDs and keep the original name in `source_path`. Do not silently overwrite either file. All variants matching the current file path are conservatively excluded from retrieval. Original benchmark fields remain unmodified.

## Files and execution

Use `/Users/kieugiangbien/bienkieu_env/bin/python` on this machine.

```bash
python -m src.build_ast_training_dataset \
  --root datasets/data4aligncoder \
  --old-train local_server_results/old_prebuilt_train/train.parquet \
  --exclusions local_server_results/data_audit/benchmark_file_hashes.json \
  --output local_server_results/ast_training_v2/train.parquet \
  --work-dir local_server_results/ast_training_v2/work_reference_roles --workers 4

python -m src.verify_ast_training_dataset \
  --dataset local_server_results/ast_training_v2/train.parquet \
  --root local_server_results/data4aligncoder_raw/data \
  --old-train local_server_results/old_prebuilt_train/train.parquet \
  --exclusions local_server_results/data_audit/benchmark_file_hashes.json

$PY -m src.data.prepare_test_data \
  --root datasets/data4aligncoder \
  --output-dir local_server_results/ast_test_inputs_v1 \
  --cache local_server_results/ast_test_inputs_v1/chunks.sqlite
```

The source builder saves atomic per-repository progress. Re-running with the same work directory resumes only if its source/config/code contract matches. Native AST work is isolated in worker processes and fails explicitly on timeout; it is not silently converted into missing tasks. Existing output Parquets are never overwritten.

Final outputs are `train.parquet`, `cceval_python.parquet`, `cceval_java.parquet`, `repoeval_line.parquet`, `repoeval_api.parquet`. Test outputs live under the requested output directory after preparation finishes. A `.parquet.incomplete` is **not** a finished artifact; existing outputs are never overwritten.

Every prepared test file preserves its original columns and adds a zlib-compressed JSON `ast_payload` column:

- `left_context_model`: conservative cleaned visible prefix, before prompt budgeting.
- `candidate_pool`: query-only BM25 top-64 shortlist from the chunked cross-file corpus, with no gold-derived features. This is a proposal pool, not a claim the model must choose 64 or cannot search beyond it.
- `cross_file_context`: up to 10 complete snippets under a 2,344-token DeepSeek budget; no chunk is truncated for admission.
- Config/version/cursor-contract fields, with no gold or right context in this model-input view.

For training, the compressed repository-pool payload holds the target cuts and
diagnostic dependency evidence. Use `ast_payload` for frozen-test model inputs;
do not use `groundtruth`/`right_context` while constructing retrieval inputs.
Old PPO loaders do not consume this schema: loader/renderer integration is a
separate step.

## Verification

Run:

```bash
python -m unittest src.test_ast_training_data src.test_code_input_cleanup \
  src.test_prepare_ast_model_inputs src.test_late_code_context
```

Tests cover UTF-8/CRLF spans, literal preservation, incomplete cursor input, multilingual comments, small/duplicate repos, sibling packing, oversized leaves, cross-file evidence, no current-file leakage, split conflicts, identical train/test input views, gold/suffix invariance, and exact original-field preservation through test-shard merging.

`src/measure_ast_fragmentation.py` reproducibly compares the legacy `ast_chunks` function against the new chunker on the same seeded 100 Python + 100 Java source files. The first paired run found one-line chunks fall from 67.5% to 5.7% for Python and from 71.5% to 2.3% for Java; median core tokens rose from 15 to 212 and from 14 to 196, respectively. No core exceeded 384 tokens. This is a new sample, not a reproduction of the screenshot's unknown file selection, and does not measure completion quality or superiority over AlignCoder.

Machine-readable logs and reports are under `local_server_results/data_audit/` and beside each completed Parquet. A final `.report.json` verifies all original input fields; `.verification.json` verifies all generated training rows against the original repository sources.
