# S12 — GrepRAG official paper

- **Type:** primary paper
- **URL:** https://arxiv.org/abs/2601.23254
- **Paper:** GrepRAG: An Empirical Study and Optimization of Grep-Like Retrieval for Code Completion

## Evidence extracted

GrepRAG studies index-free lexical retrieval, then adds identifier-weighted reranking and structure-aware deduplication. It reports that exact lexical fragments can be strong, while high-frequency ambiguous terms and rigid truncation create noise and fragmentation.

## Verbatim excerpt

> “identifier-weighted re-ranking and structure-aware deduplication”

## Research use

Keep a lexical path in the fixed candidate pool and deduplicate by AST/source span. Do not assume a large embedding teacher subsumes identifiers.

## Limitations

Lexical retrieval can miss semantic aliases, dynamic dispatch, and dependencies not expressed by shared identifiers.
