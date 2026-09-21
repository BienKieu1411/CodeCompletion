# S15 — Repoformer official paper

- **Type:** primary paper
- **URL:** https://arxiv.org/abs/2403.10059
- **Paper:** Repoformer: Selective Retrieval for Repository-Level Code Completion

## Evidence extracted

Repoformer trains a code model to decide when retrieval is useful, reducing unnecessary retrieval while retaining completion quality. The paper reports substantial speedups, making selective retrieval a relevant comparison for AlignCoder's always-on multi-sample path.

## Verbatim excerpt

> “self-supervised learning approach to enable a code LM to accurately self-evaluate”

## Research use

Include a null candidate and measure the no-retrieval decision explicitly. A student that retrieves a slightly different context but always adds it may lose to a calibrated selective policy.

## Limitations

Repoformer’s selective objective is not the same as a candidate-distribution KD objective; its gains do not validate the proposed formula.
