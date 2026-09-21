# S17 — Jina code embedding model card

- **Type:** official model card / model documentation
- **URL:** https://huggingface.co/jinaai/jina-code-embeddings-1.5b
- **Model:** jina-code-embeddings-1.5b

## Evidence extracted

The model card describes a code embedding model derived from a code-generation backbone, with code retrieval and code-completion-oriented instructions, broad language support, and a large context window. It is a reasonable auxiliary semantic teacher or candidate miner.

## Verbatim excerpt

> “code2completion”

## Research use

Use Jina to expand candidate coverage or provide an auxiliary semantic score in a fixed-pool ablation. Do not make Jina's cosine similarity the main label unless it correlates with final completion utility.

## Limitations

The model card does not show that cosine similarity predicts exact-match completion utility or repository dependency necessity.
