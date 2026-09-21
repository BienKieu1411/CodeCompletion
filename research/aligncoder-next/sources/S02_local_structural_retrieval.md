# S02 — Local structural retrieval reports

- **Type:** local synthesis of primary-paper extractions
- **Local sources:** [`CAST_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/CAST_DEEPREAD.md), [`LATE_CODE_CHUNKING_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/LATE_CODE_CHUNKING_DEEPREAD.md), [`GREPRAG_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/GREPRAG_DEEPREAD.md), [`REPOHYPER_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/REPOHYPER_DEEPREAD.md)

## Evidence extracted

- CAST uses AST recursive split-then-merge to preserve syntactic units while keeping chunks dense and reconstructable.
- Late Code Chunking separates short retrieval units from expanded comprehension context, suggesting a two-view representation.
- GrepRAG shows that identifier-precise lexical retrieval and structure-aware deduplication complement dense and graph retrieval.
- RepoHyper and related local reports warn that larger retrieval context is not automatically better; the generator can be harmed by inaccurate context.

## Verbatim excerpt from the local extraction

> “AST split-then-merge là prior art cho structural chunking.”

## Research use

Use AST-bounded units for the candidate pool, retain lexical/dataflow paths, and test late expansion only after fixed-pool retrieval is understood.

## Limitations

The local reports do not establish that AST boundaries dominate all sliding-window/context-budget configurations.
