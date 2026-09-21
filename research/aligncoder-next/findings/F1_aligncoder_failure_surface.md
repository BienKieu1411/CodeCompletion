# F1 — What must be beaten in AlignCoder

## Claim

The highest-value replacement is not “AST instead of AlignCoder.” It is a target-aligned retrieval system that keeps AlignCoder's useful signals while removing its hard winner, repeated online sampling, and weak base chunk boundaries.

## Evidence chain

1. **S01/S05:** AlignCoder's gain comes from query enhancement, dependency context, and target-aware retriever supervision; the paper reports an 18.1% CrossCodeEval EM improvement over baselines.
2. **S01:** the local implementation reduces each sampled candidate set to an argmin target-PPL winner and trains a cosine classifier with a hard label. This discards the relative utility of the other candidates.
3. **S01/S09:** CodeRAG independently identifies query construction and retriever–generator misalignment, while showing an offline teacher-to-student route is implementable.
4. **S07:** a recent controlled study warns that context length and cost-quality trade-offs can dominate a simple chunking story.

## Inference

The proposed main comparison is:

```text
same repository snapshot + same final generator + same candidate budget
AlignCoder hard-winner retriever
versus
AST-hybrid candidate pool + frozen-generator soft utility KD + null gate
```

This isolates the supervision and evidence-unit changes instead of comparing unrelated generators.

## Confidence and uncertainty

- High confidence that repeated sampling and hard winner labels are real AlignCoder design costs.
- Medium confidence that soft labels improve EM; REPLUG makes the mechanism plausible, but repository completion transfer remains unverified.
- Low confidence that a larger semantic encoder by itself beats the aligned hard-winner retriever.

## Falsification test

If hard target-NLL labels and soft target-NLL distributions are equal under the same pool, then the claimed benefit is not the soft KD objective. If the AST-hybrid pool has lower candidate coverage, retriever supervision cannot be judged fairly.
