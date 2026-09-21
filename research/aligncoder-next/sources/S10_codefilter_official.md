# S10 — CODEFILTER official paper

- **Type:** primary paper
- **URL:** https://arxiv.org/abs/2508.05970
- **Paper:** Impact-driven Context Filtering For Cross-file Code Completion

## Evidence extracted

CODEFILTER assigns retrieved chunks positive, neutral, or negative impact using likelihood-based measurements, then trains an adaptive filtering framework. It reports accuracy improvements and shorter prompts on RepoEval and CrossCodeLongEval.

## Verbatim excerpt

> “only a small subset positively contributes to the completion”

## Research use

Include harmful-context rate and pointwise keep/drop/null filtering in the ablation ladder. This supports a no-pairwise filter because each candidate can be supervised independently by a utility signal.

## Limitations

Polarity labels still use the target completion during offline dataset construction; they cannot be computed from target access at serving time.
