# S13 — DraCo official paper

- **Type:** primary paper
- **URL:** https://arxiv.org/abs/2405.19782
- **Paper:** DraCo: Enhancing Code Completion with Dataflow-Guided Retrieval

## Evidence extracted

DraCo uses a type-sensitive dependency graph and dataflow-guided retrieval for repository completion. It is evidence that program relations can add information beyond text similarity, but graph precision and language support are implementation risks.

## Verbatim excerpt

> “type-sensitive dependency graph”

## Research use

Treat dataflow/import/symbol lookup as a complementary candidate source or late expansion rule, not as the only retriever.

## Limitations

Static graphs can be incomplete for dynamic languages, reflection, generated code, and unresolved imports.
