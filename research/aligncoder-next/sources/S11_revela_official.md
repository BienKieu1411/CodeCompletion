# S11 — REVELA official paper

- **Type:** primary paper
- **URL:** https://arxiv.org/abs/2506.16552
- **Paper:** REVELA: Dense Retriever Learning via Language Modeling

## Evidence extracted

REVELA treats retrieval as learning token dependencies. Its in-batch attention is weighted by retriever similarity, so next-token prediction can update the retriever without annotated or synthetic query-document pairs. The paper reports gains on BEIR, CoIR, and BRIGHT, but not repository-level code completion.

## Verbatim excerpt

> “Without annotated or synthetic query-document pairs”

## Research use

Use as an optional Stage 0 representation pretraining experiment. The main task loss should remain target-conditioned candidate-distribution KD.

## Limitations

Domain retrieval gains do not establish completion gains; NTP can learn topical/code co-occurrence instead of target utility.
