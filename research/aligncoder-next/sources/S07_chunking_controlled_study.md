# S07 — Controlled chunking study, 2026

- **Type:** recent primary empirical study; adversarial evidence
- **URL:** https://arxiv.org/abs/2605.04763
- **Paper:** How Does Chunking Affect Retrieval-Augmented Code Completion? A Controlled Empirical Study

## Evidence extracted

The study crosses four chunking strategies, four retrievers, five generators, and nine configurations on RepoEval and CrossCodeEval, totaling 864 settings. It reports that Function chunking underperforms other strategies on RepoEval, while cross-file context length is a dominant parameter and chunk size has a weaker, non-monotonic effect. Sliding Window and cAST occupy the reported cost-quality Pareto front.

## Verbatim excerpt

> “cross-file context length is the dominant parameter”

## Research use

This is the main opposition to an AST-only narrative. Every AST experiment must control context length, include a sliding-window comparator, and report Pareto cost/quality rather than only EM.

## Limitations

The study is recent and may use generator/retriever versions different from the local AlignCoder reproduction. Its result constrains claims; it does not prove a particular hybrid pool for this project.
