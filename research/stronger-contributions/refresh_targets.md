# Refresh targets — online AST-GR outcome RL

Research snapshot: 2026-09-28. Recheck before expensive training and before submission.

1. **Recheck reward prior art:** inspect full methods and ablations for RLCoder, AlignCoder, RepoShapley, IRCoCo, RepoGenReflex and new repository-completion RL papers; establish exactly whether any already performs on-policy retriever-gradient updates from free-running ES/EM.
2. **Validate reward metric:** reuse the exact completion normalizer/scorer used by CrossCodeEval/RepoEval; measure correlation of EditSim/EM reward with held-out final completion outcomes, separately for Python and Java.
3. **Audit leakage and pool recall:** mask target spans and near-duplicates; split by repository; report proposal-pool recall before interpreting actor policy results.
4. **Measure live-encoder feasibility:** on one T4, verify policy-action log-prob gradients reach UniXcoder parameters and profile candidate-pool encoding memory/time. Cached frozen embeddings cannot support a full-retriever training claim.
5. **Smoke-test online serving:** vLLM on physical T4 #0, trainer on T4 #1, one batched text-generation request, metric reward, and checkpoint/resume. No prompt-logprob endpoint is required by this objective.
6. **Run the 100–200-episode pilot:** log reward/EM/ES, entropy, STOP-rate, gradient norm, GPU memory and episodes/hour. Do not scale unless reward has useful variance and validation trends improve.
7. **Matched comparison:** same generator revision, data/repository split, AST candidate pool, prompt, output decoding and cross-file budget for PPL-RL, outcome-RL, frozen RLRetriever and AlignCoder; use paired repo-level uncertainty intervals.
8. **Revisit the novelty claim:** if free-running outcome reward does not beat matched PPL reward, or AST-GR does not beat flat chunks, stop claiming a new framework. Do not claim “beats AlignCoder” without held-out matched end-to-end EM evidence.
9. **Read RRPO against the AST-GR proposal:** compare its sequential action distribution, per-prefix output reward, fixed-reference ratio/KL, deterministic baseline, candidate pool, and fixed-k protocol with terminal-only token-budgeted code retrieval.
10. **Measure credit at equal compute:** compare one terminal completion with per-prefix completion scoring; report generator calls, output tokens, EM/ES, and quality per GPU-hour.
11. **Audit the critic/state definition:** verify `V(q, selected AST set, remaining token budget)` predicts held-out returns; compare against the RRPO-style fixed reference baseline and log approximate KL/clip fraction.
12. **Keep the novelty boundary current:** PPO, sequential context selection, metric reward, AST actions, and no-pairwise claims each have separate prior art; only a matched, independently validated mechanism can support a contribution claim.
13. **Audit actual PPO mechanics:** verify saved old action log-probs and masks, re-encoded current log-probs, nontrivial ratios after update, both clipping branches, at least two reuse passes in the pilot, and encoder gradients from actor loss.
14. **Measure within-query reward spread:** on 128 repository-stratified train queries compare two distinct AST slates with the same greedy generator/metric; report ties, absolute ES spread, target lengths and pool misses before committing 12 hours.
15. **Measure the hardware contract:** smoke-test the user's pinned vLLM wheel in FP16 on T4 #0 and one full-backward UniXcoder process on T4 #1; derive rollout/hour and export guard time from actual wall clock, not prior head-only training logs.
16. **Test the decision, not the acronym:** compare PPO-Clip to an unclipped full-encoder policy-gradient control and a frozen-encoder PPO control at matched calls; only run GRPO group-of-four if its extra completions fit and zero-std groups are uncommon.
