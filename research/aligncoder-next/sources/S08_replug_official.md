# S08 — REPLUG-LSR official paper

- **Type:** primary prior-art paper
- **URL:** https://arxiv.org/abs/2301.12652
- **Paper:** REPLUG: Retrieval-Augmented Black-Box Language Model

## Evidence extracted

REPLUG uses a frozen black-box language model to score retrieved documents through likelihood and trains a retriever from the resulting distribution. This is the closest general precedent for using a generator's probability distribution as retrieval supervision without updating the generator.

## Verbatim excerpt

> “the LM can be used to supervise the retrieval model”

## Research use

TCD-KD must be framed as a repository-code adaptation with AST evidence units, null policy, and completion-specific controls—not as a new generic KL principle.

## Limitations

REPLUG is not itself a repository-level code-completion method, so its transfer requires controlled experiments.
