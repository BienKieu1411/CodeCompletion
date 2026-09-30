# Research plan — Posterior AST-Intent Support Retrieval

> **SUPERSEDED 2026-09-28:** PAISR/support-BCE cần tạo nhãn retrieval và đã bị thay theo ràng buộc mới. Nguồn hiện hành là [true PPO-Clip method report](2026-09-28_true_ppo_unixcoder_ast_retrieval.md) và [PPO/GRPO research plan](ppo_grpo_training_plan.md); phần dưới đây chỉ là lịch sử, không phải hướng triển khai.

**Date:** 2026-09-28
**Status:** One candidate selected for falsification; novelty and benchmark gain are not established.
**Constraints:** no KD or preference pairs/pairwise loss; UniXcoder retriever; DeepSeek-Coder-1.3B generator; two Kaggle T4 GPUs, vLLM on one and retrieval training on the other.

## Research question

Does posterior-weighted support retrieval over generated completion intents improve repository completion under fixed query/context budgets, and does it outperform AlignCoder's raw hypothesis query and CodeRAG's log-probability-guided query construction?

Implementation evidence is pinned in [S16](sources/S16_aligncoder_query_budget_impl.md): AlignCoder appends generated answers after local context, defaults to eight samples and a 256-token UniXcoder query, then retains the tokenized query tail. Overflow prevalence and impact are unmeasured.

## Method, one sentence

Parse K generated completions into an empirical sample distribution over typed AST intents (not a calibrated Bayesian posterior); fine-tune full UniXcoder with masked pointwise labels for target-resolved evidence support; select evidence to maximize intent coverage under a fixed budget while reserving tokens for cursor-local query.

## Models and training

- **UniXcoder-base:** train all retriever parameters.
- **DeepSeek-Coder-1.3B-base:** frozen; generate/cache K raw train hypotheses and their multiplicities once, then produce final benchmark completions. No teacher logits, KL, online KD or generator reward during updates.
- **Tree-sitter + AST-GR/static resolver:** create support labels from train targets only.
- **Primary loss:** intent-frequency-weighted pointwise BCE over query/intent/chunk support labels. No chosen/rejected pairs and no pairwise/listwise ranking loss.
- **Hardware:** vLLM DeepSeek on GPU 0; one UniXcoder trainer on GPU 1; CPU for parsing/resolution. DDP/torchrun is unnecessary for one trainer GPU.

## Decision-oriented experiment sequence

### Gate 0 — verify that a real failure exists

Replay exact AlignCoder query creation and official UniXcoder tokenization on train/dev. Measure overflow rate, retained cursor-neighborhood (last 32/64 local tokens), retained hypothesis spans, hypothesis-to-target intent match, resolver coverage, and target-support Recall@50. Break out Python/Java and report repository-bootstrap intervals.

**Stop:** overflow is rare, or overflow does not remove cursor tokens / associate with support-recall loss.

### Gate 1 — isolate representation at fixed cost

Use the same frozen UniXcoder, candidate pool, K, generator calls, 256-token query cap, and final context cap:

1. AlignCoder raw concatenation.
2. Local-only query.
3. Local plus raw hypotheses with role-aware truncation.
4. CodeRAG-style log-probability-probed local chunks, if the module can be reproduced; report probe-token cost separately.
5. Local plus weighted typed AST-intent sketch.

Compare candidate recall, target-intent/support recall, cursor retention, EM and ES. Keep an overflow stratum.

**Stop:** AST-intent sketch does not improve on the simple truncation-safe control.

### Gate 1b — isolate the budgeted selector

Fix PAISR query representation, support scores and candidate pool. Compare independent top chunks, greedy posterior-intent coverage per token, and dense top-budget under the same cross-file token cap. Drop the selector claim if it does not beat independent ranking.

### Gate 2 — isolate the training loss

With query representation fixed, compare frozen UniXcoder, simple target-symbol BCE without hypothesis posterior, matched AlignCoder retriever if reproducible, and full UniXcoder fine-tuned with PAISR pointwise support BCE. Freeze candidate-pool construction and negative sampling; mask unresolved labels and report hypothesis-target match, resolver coverage and pool misses.

**Stop:** fine-tuning does not improve evidence support or completion, or gains rely on unverified negative labels.

### Gate 3 — end-to-end claim

Run AlignCoder, CodeRAG where reproducible, RLCoder, BM25, dense UniXcoder, and PAISR with identical repo-disjoint split, exact model revisions, prompts, sampling K/seeds, candidate pool, final generator, local context, and cross-file token cap. Primary outcome is EM/ES with paired repository-level uncertainty intervals. Retrieval/NLL gains alone do not support “beats AlignCoder.”

## Falsifiable hypotheses

- **H1:** a meaningful fraction of augmented queries overflows and right-tail truncation drops cursor-local tokens.
- **H2:** this displacement lowers target-support recall and/or completion quality within the overflow stratum.
- **H3:** posterior AST-intent support coverage beats raw hypotheses, truncation-only controls, and CodeRAG-style query construction at equal query budget.
- **H4:** pointwise support training adds gains beyond the representation change.
- **H5:** the combined method improves matched end-to-end completion over AlignCoder.

H1/H2 failure means the motivating weakness is too small; H3 failure reduces PAISR to query-truncation engineering; H4 failure removes the training contribution; H5 failure means no “surpasses AlignCoder” claim.

## Prior-art boundaries

- **AlignCoder/RLCoder:** already use sampled predictions and generator-likelihood retriever feedback. Do not claim those ideas.
- **RepoShapley:** already studies coalition-aware context contribution. Do not claim generic joint slate utility.
- **AIRCoder:** already includes AST chunks and structural/hybrid retrieval. Do not claim AST chunking.
- **CodeRAG:** direct prior for query information loss, log-probability-guided query construction, AST knowledge units, multi-path retrieval and preference-aligned BESTFIT. Compare directly; do not claim query construction or AST retrieval generally.
- **RepoHyper:** nearby gold-target graph retrieval supervision. Compare directly; do not claim graph retrieval or target supervision generally.
- **OpenCoder/DyRetriever/CECoder:** adjacent uncertainty, dynamic graph, and code-element retrieval. Bound the claim to posterior-weighted support coverage over multiple generated hypotheses plus per-intent supervision.

See [F05](findings/F05_query_hypothesis_prior_art.md). This is a targeted prior-art map, not proof of global novelty.

## Practical constraints

Cache K hypotheses and AST labels to avoid rerunning vLLM per epoch. Measure CPU parsing throughput and both T4 VRAMs in a pilot. Do not increase generator calls or context/query budget in the primary comparison. Test data remains untouched until dev choices are frozen.

## Deliverables

1. Overflow/support-retention audit and summary table.
2. Frozen-encoder representation ablation.
3. Pointwise-BCE retriever checkpoint and exact config/seed.
4. Matched end-to-end benchmark and per-repo bootstrap intervals.
5. Updated novelty/source audit before drafting a “first” claim.
6. Independent adversarial review and resolved prior-art gaps.

## Current decision

PAISR is the sole direction selected for the next falsification step. It formulates retrieval as support coverage over a posterior of completion intents and has a concrete AlignCoder query-budget failure to test. CodeRAG raises the novelty bar: **the method is not established as strong or successful.** First run Gate 0; do not launch full Kaggle training before Gate 0 and a small Gate 1 pilot pass.
