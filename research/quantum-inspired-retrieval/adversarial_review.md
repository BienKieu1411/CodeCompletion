# Adversarial review — 2026-09-27

## 1. What if the phase effect is only a nonlinear classical kernel?

That is plausible. Product-state fidelity is classically tractable, and phase-interference fusion can be factorized into ordinary real-valued operations. Compare against additive fusion and a parameter-matched MLP on identical features. If they tie, the quantum-inspired framing has no demonstrated methodological value.

## 2. What if AST-GR labels optimize structural inclusion, not completion utility?

Then pointwise AST-GR BCE may improve structural retrieval while adding irrelevant tokens or failing to help the generator. Report retrieval-label metrics and end-to-end EM/ES under a fixed context budget. No completion gain means the retriever direction did not reach the target.

## 3. What if the apparent gain comes from the pool, not the scorer?

The multi-path candidate union may improve recall independently of phase. First compare scorers on exactly the same frozen pool; then compare full candidate-generation systems and report pre-reranking recall.

## 4. What if phases suppress useful lexical evidence or learn benchmark artifacts?

Run phase-off/amplitude-only ablations; slice identifier-heavy vs semantic/dependency-heavy queries; use repository-disjoint untouched data; inspect calibration and false positives. Do not select on CrossCodeEval TEST.

## 5. What if the marginal completion objective is mislabeled as KD—or mismatches inference?

The optional `L_completion` uses the frozen downstream generator's probability of the gold continuation, not its token-distribution targets. That is not KL knowledge distillation, but it does use generator-derived candidate likelihoods and could violate a stricter “no generator signal” interpretation. It also marginalizes single-candidate contexts while inference may pack top-K candidates together. Keep it out of the first teacher-free AST-GR/BCE pilot; if tested later, state the supervision precisely and reproduce deployed context assembly.

## Strongest counterargument

The best evidence for QI similarity heads is modest and from non-code domains; the supplied notebook is unexecuted and selects checkpoints on a CrossCodeEval TEST subset. A classical hybrid AST retriever may be the higher-probability path to beating AlignCoder. The notebook does contain its active online engine; the actual concerns are test-set selection, lack of a matched scorer control, non-AST training data, and reward/prompt confounders—not a missing engine. AST relation-factorized phase therefore remains a low-cost falsifiable branch, not the main framework until empirical gates pass.
