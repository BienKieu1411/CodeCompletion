# S01 — Local AlignCoder deep-read

- **Type:** local primary-paper extraction
- **Local source:** [`doc/papers/ALIGNCODER_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/ALIGNCODER_DEEPREAD.md)
- **Underlying paper:** [AlignCoder](https://arxiv.org/abs/2601.19697)

## Evidence extracted

- AlignCoder addresses query–target misalignment with query enhancement, AlignRetriever training, and dependency chunks.
- The online path samples multiple completions; `k=4` is the reported sweet spot, while additional candidates can add noise.
- The reward keeps only the candidate with the lowest target perplexity. The local report identifies this as a hard winner signal implemented with cross-entropy over retriever logits.
- Base chunks use Split-Aggregate over blank-line blocks. Dependency chunks already use tree-sitter parsing for imports, signatures, classes, and methods; therefore the precise AST contribution is to improve base evidence units, not to replace all structure in AlignCoder.
- The local report records substantial ablation drops when dependency context, query enhancement, or retriever training is removed.

## Verbatim excerpt from the local extraction

> “AlignCoder giải quyết query–target semantic misalignment bằng hai thành phần: query enhancement và AlignRetriever.”

## Research use

This is the baseline failure map: preserve target alignment and dependency evidence, but test soft utility supervision, AST-bounded units, null/no-retrieval handling, and lower-latency inference.

## Limitations

This is a report of the local PDF/code snapshot, not a fresh reproduction. Native paper and local implementation must be compared before publishing numbers.
