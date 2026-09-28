# Research plan: a stronger contribution beyond quantum-inspired scoring

**Date:** 2026-09-27  
**Question:** What technically distinct, testable contribution could plausibly improve repository-level code completion over AlignCoder?  
**Constraints:** No knowledge distillation; no pairwise example construction or pairwise ranking loss; do not modify the notebook or implementation.

## Decision under test

The most direct candidate is an **AST-GR-constrained, slate-level retrieval policy**. It chooses a complete, ordered set of AST evidence under a fixed prompt budget, and learns from the change in the frozen completion model's gold-continuation log-likelihood for that assembled prompt. The primary policy loss is REINFORCE with a scalar reward and baseline. There are no teacher-token targets, KL loss, positive/negative pairs, or pairwise ranking examples.

This is a candidate for a controlled pilot, not a claim that it will beat AlignCoder. AlignCoder/RLCoder already use generator target likelihood to train retrieval; RepoShapley already covers context-set interaction and coalition utility. The narrow potential contribution is the AST-relation-constrained action space and direct optimization of the deployed, budgeted context slate without Shapley labels or a distilled KEEP/DROP controller. Novelty risk remains medium-to-high.

## Falsifiable hypotheses

**H1 — Structural slate hypothesis.** At identical candidate-pool recall, generator, and cross-file token budget, selecting AST-GR-related evidence as a joint slate yields higher downstream completion quality than independent top-K retrieval and AlignCoder's candidate-level selection.

**H2 — Objective hypothesis.** The measured gain is due to joint slate utility and AST-relation constraints, not extra candidate coverage, a larger prompt, or more generator calls.

**H3 — Scope alternative.** Version/freshness-aware context selection may be more novel, but is relevant only if historical snapshots or stale indexes are in the target use case; it is not the main proposal for static CrossCodeEval/RepoEval.

**H4 — Quantum scorer.** Keep phase/fidelity scoring as a controlled ablation. Do not make it the paper's central novelty without a gain against parameter-matched real-valued controls and the slate-policy baseline.

## Literature checks

- Reuse local DeepRead reports for AlignCoder, RLCoder, REVELA, RepoShapley, CAST, GRACE, CodeRAG, RepoHyper, and related papers.
- Verify 2025–2026 primary work on AST/hybrid retrieval, candidate utility, context interactions, adaptive/token-aware retrieval, and stale context.
- Separate direct repository completion from repository-level generation, editing, general retrieval, and QA.
- State source status and novelty uncertainty; targeted search is not proof of global novelty.

## Planned report

1. Direct answer and ranked alternative contributions.
2. Prior-art overlap and narrowest defensible contribution.
3. Exact retriever, frozen evaluator, final generator, reward, and policy loss.
4. Training/inference flow with no KD and no pairwise construction.
5. Matched experiment, ablations, cost, and stop/go criteria.
6. Alternative freshness direction and quantum's role.
7. Sources, evidence status, and refresh targets.

## Risks and stop criteria

- **Novelty:** The method can look like an engineering combination of AlignCoder/RLCoder rewards, RepoShapley set utility, and AST chunking. Demonstrate that AST relation constraints add value beyond generic slate policy.
- **Reward cost:** Generator-scored slate rollouts are expensive and can overfit the evaluator. Start with a fixed proposal pool, one frozen evaluator, a small rollout budget, and a separate generator-backend evaluation.
- **Reward validity:** Token likelihood is not correctness. Report exact match/edit similarity and test-backed metrics where the benchmark supports them.
- **Data leakage:** The gold continuation may score training slates only. It must not affect retrieval candidates, features, or inference.
- **Unknown artifact:** The workspace contains the quantum notebook but no separately auditable AST-GR schema/implementation. Inspect its label and edge definitions before treating them as available.
- **Stop:** Do not claim “beats AlignCoder” unless repository-disjoint, token-matched end-to-end results show a repeatable positive gain on the untouched test set.

## Changelog

- 2026-09-27: replaced the weaker standalone AST-label hypothesis with a slate-level policy candidate after checking AlignCoder, RLCoder, RepoShapley, AIRCoder, and REVELA.
