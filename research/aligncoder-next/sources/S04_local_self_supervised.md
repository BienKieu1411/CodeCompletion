# S04 — Local self-supervised retrieval report

- **Type:** local primary-paper extraction
- **Local source:** [`REVELA_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/REVELA_DEEPREAD.md)
- **Underlying paper:** [REVELA](https://arxiv.org/abs/2506.16552)

## Evidence extracted

- REVELA trains dense retrievers without annotated or synthetic query-document pairs.
- Its signal is next-token prediction conditioned on local and cross-document context, where retriever similarity weights in-batch cross-document attention.
- This is a plausible pretraining stage for repository-specific representations, but the local report has no direct evidence that generic REVELA training beats target-conditioned completion supervision on RepoEval/CrossCodeEval.

## Verbatim excerpt from the local extraction

> “REVELA cung cấp một route self-supervised non-pairwise dựa trên next-token prediction.”

## Research use

Test REVELA-style NTP as initialization or auxiliary pretraining. Do not use it as the main superiority claim until it survives a target-utility KD comparison.

## Limitations

The transfer from general/domain retrieval benchmarks to repository-level completion is an open hypothesis.
