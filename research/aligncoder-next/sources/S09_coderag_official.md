# S09 — CodeRAG official paper

- **Type:** primary paper
- **URL:** https://arxiv.org/abs/2509.16112
- **Paper:** CodeRAG: Finding Relevant and Necessary Knowledge for Retrieval-Augmented Repository-Level Code Completion

## Evidence extracted

CodeRAG combines log-probability-guided query construction, multi-path retrieval, and BESTFIT reranking. It also reports offline distillation of a large reranker into a smaller model with token-level cross-entropy and consensus filtering. The paper explicitly identifies query construction, single-path retrieval, and retriever–generator misalignment as problems.

## Verbatim excerpt

> “inappropriate query construction, single-path code retrieval, and misalignment between code retriever and code LLM”

## Research use

Use multi-path candidate union and consensus/quality filters, but replace discrete reranker imitation with a soft candidate-utility distribution from the final generator.

## Limitations

Its BESTFIT teacher is a reranker and its probe uses line-based current-file chunks; neither is direct evidence for final-generator KD over AST units.
