# S03 — Local target-supervision and distillation reports

- **Type:** local synthesis of primary-paper extractions
- **Local sources:** [`CODERAG_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/CODERAG_DEEPREAD.md), [`GRACE_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/GRACE_DEEPREAD.md), [`REACC_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/REACC_DEEPREAD.md), [`REPLUG` discussion in the canonical framework](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/TCD_KD_FRAMEWORK.md#23-replug-lsr-là-prior-art-gần-nhất-phải-có)

## Evidence extracted

- CodeRAG builds a multi-path AST/dataflow/sparse/dense retrieval pipeline and distills an expensive BESTFIT reranker into a smaller student with token-level cross-entropy over offline teacher examples.
- GRACE/related reports emphasize that context usefulness is conditional on the current completion and that feature interaction matters.
- ReACC and RepoCoder show that target-aware retrieval can help but that prompt construction and retrieval cost are coupled.
- The canonical framework correctly identifies REPLUG-LSR as the closest prior art for frozen-generator likelihood supervision; using KL on a likelihood-derived candidate distribution is not automatically novel.

## Verbatim excerpt from the local extraction

> “Đây là ví dụ trực tiếp của offline teacher-to-student distillation cho reranker, không cần pairwise training student.”

## Research use

Distill the final generator's candidate utility distribution rather than imitate only a text reranker's discrete decision. Treat this as a controlled adaptation of prior art, not an unqualified novelty claim.

## Limitations

Most local evidence measures retrieval or reranking utility, not calibrated end-to-end exact match or execution success.
