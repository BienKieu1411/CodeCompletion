# F02 — Recommended candidate: AST-GR-constrained slate policy

> **Superseded 2026-09-28:** Do not implement this earlier target-likelihood slate-RL recipe as written. The current candidate uses online free-running completion outcome reward and a fully updated UniXcoder AST-GR policy; see [current report](../2026-09-28_online_ast_gr_outcome_rl.md) and [plan](../online_ast_gr_rl_plan.md). This file is retained only as historical analysis.

## Research question

At a fixed candidate pool and context budget, does choosing a complete AST-related evidence slate by its effect on the downstream completion model outperform independent chunk ranking and AlignCoder?

## Mechanism

- Build a fixed proposal pool from BM25/identifier matches, UniXcoder dense retrieval, and AST-GR neighborhood expansion. Deduplicate by stable source-span/node identity.
- The actor is initialized from **microsoft/unixcoder-base**, with a policy head that selects AST units or relation-derived support units in order, subject to a token budget; **STOP** is always available.
- The gold continuation is used only during training to score a sampled assembled prompt with a **frozen** code generator. It is never an inference feature or part of candidate generation.
- For a sampled slate S, let q be the visible prefix and y the gold continuation:

  ~~~text
  R(S) = (log P_G(y | q,S) - log P_G(y | q,∅)) / |y| - λ · overflow(S)
  ~~~

  The primary run uses a hard token cap so overflow(S)=0; a cost term is an ablation. A value baseline b(q) reduces variance.
- Optimize the actor with:

  ~~~text
  L_policy = - E_S[(R(S)-b(q)) · Σ_t log πθ(a_t | q,a_<t)] - β H(πθ)
  ~~~

  A baseline/critic may have a secondary regression loss. The frozen generator is not updated.

## Why this is different—and where overlap remains

- **Different from AlignCoder/RLCoder:** reward applies to the entire ordered context slate under the actual prompt budget, rather than assigning hard winner reward to a candidate in isolation. AlignCoder's generation-time sampled query is not required.
- **Different from RepoShapley:** no offline Shapley surrogate, coalition oracle, or distilled KEEP/DROP controller; the current policy samples the slate and receives one scalar outcome reward.
- **Potentially distinctive:** actions are constrained and informed by typed AST-GR relations, so the selector can add complementary evidence as a structural support group rather than treat every snippet as independent.
- **Overlap/risk:** RepoShapley already owns the broad set-interaction claim and AlignCoder/RLCoder own likelihood-reward retriever training. “Combining them without KD/pairs” is not enough by itself. The AST-relation action-space ablation must carry the contribution.

## Explicit model roles

| Component | Initial choice | Train? | Role |
|---|---|---:|---|
| Retriever/policy | UniXcoder-base plus a compact slate-policy head | Train policy head first; freeze encoder in fixed-pool pilot | Propose/rank AST evidence actions and stop under the budget |
| Reward evaluator | deepseek-ai/deepseek-coder-1.3b-base | No | Teacher-force the gold continuation and return a scalar log-likelihood reward during training only |
| Final generator | The exact same frozen model for the primary paired comparison; start with deepseek-ai/deepseek-coder-1.3b-base, then repeat on a larger available backbone | No | Generate the final completion at inference and provide a matched reward in the first pilot |

The generator supplies **one scalar reward per sampled slate**. No output-distribution KL, hidden-state imitation, teacher action labels, positive/rejected examples, or pairwise ranking loss is created. This is on-policy policy-gradient training, not online/offline distillation. The quantum notebook currently uses this exact checkpoint as its reward evaluator; compare it to AlignCoder's reported DeepSeek-Coder-1B setup only after verifying the checkpoint revision and prompt protocol.

## Necessary first audit

The current workspace has no separately auditable AST-GR implementation/schema; the visible **src/** artifact is the quantum notebook. Before training, inspect the actual AST-GR labels, edge types, parser/language support, source spans, and leakage rules. If the graph lacks typed relations, derive only relations that can be verified from the parser and narrow the claim accordingly.

## Failure modes

- Slate reward is expensive: each rollout needs a generator forward over the gold continuation. Bound rollouts and use a fixed candidate pool for the first pilot; cache exact prompt/reward hashes when safe.
- NLL may reward stylistic likelihood rather than correctness. Treat it as a training signal, not the main success metric.
- A static candidate pool caps recall. Report pool recall before the policy and do not attribute a recall change to selection.
- REINFORCE variance may swamp the gain. A learned value baseline and entropy regularization are allowed; no pairwise objective is needed.
- If the AST relation edges fail to outperform an unconstrained slate policy, the AST-specific novelty claim fails.
