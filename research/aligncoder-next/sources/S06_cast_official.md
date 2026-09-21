# S06 — CAST official paper

- **Type:** primary paper
- **URL:** https://arxiv.org/abs/2506.15655
- **Paper:** CAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree

## Evidence extracted

CAST recursively splits oversized AST nodes and merges adjacent siblings within a budget. The paper evaluates retrieval and generation across RepoEval, CrossCodeEval, and SWE-bench, supporting AST-aware chunking as a strong structural baseline.

## Verbatim excerpt

> “AST recursive split/merge”

## Research use

Adopt the user's AST chunking as a candidate-pool representation, while preserving parent chains and source spans for later expansion and debugging.

## Limitations

AST coherence is not semantic relevance; a syntactically complete chunk can still be irrelevant or harmful to the completion.
