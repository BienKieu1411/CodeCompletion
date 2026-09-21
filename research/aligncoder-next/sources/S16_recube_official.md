# S16 — ReCUBE official benchmark

- **Type:** primary benchmark paper
- **URL:** https://arxiv.org/abs/2603.25770
- **Paper:** ReCUBE: Evaluating Repository-Level Context Utilization in Code Generation

## Evidence extracted

ReCUBE masks a file and evaluates reconstruction using the remaining source files, documentation, and dependency specifications. It adds usage-aware tests for internal and cross-file integration and reports that even strong models struggle to use repository context fully; caller-centric exploration improves strict pass rate.

## Verbatim excerpt

> “repository-level context utilization remains highly challenging”

## Research use

Use ReCUBE or a similar usage-aware stress set to check whether retrieval gains reflect real dependency utilization rather than only benchmark-local token overlap.

## Limitations

ReCUBE is a different task from next-fragment completion, so it is a stress test and not a replacement for RepoEval/CrossCodeEval.
