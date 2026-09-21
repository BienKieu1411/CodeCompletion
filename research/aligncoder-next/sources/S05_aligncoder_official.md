# S05 — AlignCoder official paper

- **Type:** primary paper
- **URL:** https://arxiv.org/abs/2601.19697
- **Paper:** AlignCoder: Aligning Retrieval with Target Intent for Repository-Level Code Completion

## Evidence extracted

The paper's central design is to generate multiple candidate completions, append them to the unfinished query, and train a retriever with target-aware feedback. The abstract reports evaluation on CrossCodeEval and RepoEval across five backbone code LLMs and an 18.1% EM improvement over baselines on CrossCodeEval.

## Verbatim excerpt

> “Our approach generates multiple candidate completions to construct an enhanced query.”

## Research use

AlignCoder is the direct end-to-end target. Any proposed method must match its generator, data access, candidate budget, and formatter before interpreting a gain.

## Limitations

The abstract does not expose the hard-winner implementation details or the exact contribution of AST versus query enhancement; those come from the local deep-read and paper body.
