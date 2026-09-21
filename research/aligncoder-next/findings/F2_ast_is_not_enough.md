# F2 — AST chunking is necessary evidence hygiene, not a complete method

## Claim

The user's AST chunking is the correct structural starting point, but AST alone is not a sufficient superiority argument. The framework must combine AST units with lexical, dense, and program-relation evidence and control cross-file context length.

## Evidence chain

1. **S02/S06:** CAST supports recursive split-then-merge and reports gains over fixed chunks; the local report also preserves the user's AST-GR direction.
2. **S07:** the controlled 2026 study finds that Function chunks underperform, but that Sliding Window and cAST are both on the Pareto front and that cross-file context length is dominant.
3. **S12:** GrepRAG finds identifier-weighted lexical retrieval and structure-aware deduplication useful, exposing cases where semantic dense retrieval alone can miss exact code symbols.
4. **S13:** DraCo provides complementary evidence that dataflow/type relations can add information not recoverable from text similarity alone.

## Inference

The candidate pool should be a union of AST-bounded units from four paths:

- exact import/symbol/dependency lookup;
- BM25 or identifier-aware lexical search;
- pretrained/fine-tuned dense retrieval;
- optional dataflow/type relation expansion.

The student should select among the union, not be forced to choose between “AST” and “embedding.”

## Falsification test

Run AST-only, sliding-window-only, and hybrid pools with identical candidate count, context budget, generator, and decoding. If the hybrid does not improve candidate coverage or completion, remove the extra path rather than retaining it by intuition.

## Practical implication

AST boundaries fix information loss at chunk construction; a separate late-expansion or parent-signature rule fixes comprehension context. These must be evaluated independently because expansion can consume the budget and reintroduce noise.
