# S14 — SelfRACG official paper

- **Type:** primary paper
- **URL:** https://arxiv.org/abs/2507.19033
- **Paper:** SelfRACG: Self-Expressing Information Needs for Retrieval-Augmented Code Generation

## Evidence extracted

SelfRACG argues that matching an external query to repository content can miss the information needed for the next code fragment. It trains a model to express information needs and use them for retrieval, offering a one-pass alternative to repeated completion sampling.

## Verbatim excerpt

> “enables LLMs to self-express their information needs”

## Research use

Test a one-pass distilled need/query head only after direct utility KD is established; it is a latency/representation alternative, not the first main objective.

## Limitations

The information-need representation can be hard to evaluate independently and may introduce another teacher-student mismatch.
