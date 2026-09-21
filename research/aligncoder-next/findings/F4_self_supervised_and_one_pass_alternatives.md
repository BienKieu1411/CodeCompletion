# F4 — Self-supervised and one-pass alternatives

## Claim

There are two credible alternatives to direct utility KD, but neither should replace D1 as the first experiment: repository-specific NTP pretraining and one-pass information-need/query distillation.

## D2: REVELA-style NTP pretraining

**S04/S11** show that a retriever can be trained from next-token prediction with cross-document attention and no annotated or synthetic query-document pairs. This can teach repository co-occurrence and code-unit compatibility before target-conditioned fine-tuning.

Risk: generic co-occurrence may retrieve popular definitions rather than the evidence that reduces target uncertainty. Therefore D2 is a representation prior, not the final completion objective.

## D3: one-pass need/query head

**S14** argues that external content matching can miss the information needed for the next code fragment, while **S09** and **S12** demonstrate query construction and lexical intent signals. A distilled query/need head could replace AlignCoder's repeated completion sampling at serving time.

Risk: an intermediate query representation can lose the generator's direct utility signal and add another mismatch.

## Decision rule

- If D1 is accurate but too slow, test D3.
- If D1 is data/initialization limited, test D2.
- If D1 is not accurate, do not hide the failure with either alternative; inspect candidate recall, teacher calibration, and formatter fairness first.

## Falsification test

For D2 and D3, hold the candidate pool and final generator fixed. They must beat D1 or materially reduce serving cost at statistically unchanged EM/ES to remain in the main framework.
