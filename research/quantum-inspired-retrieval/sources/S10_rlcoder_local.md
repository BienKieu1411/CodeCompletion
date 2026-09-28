# S10 — RLCoder project extraction

- **Type:** local deep-read of the paper.
- **Path:** [`doc/papers/RLCODER_DEEPREAD.md`](../../../doc/papers/RLCODER_DEEPREAD.md).
- **Credibility / recency / bias risk:** 4 / 3 / 2. Local extraction is a navigation aid; original PDF remains the primary source.

## Project-relevant facts from the extraction

RLCoder uses a code generator's target perplexity to reward repository candidates, with a hard winner signal and an empty/no-retrieval candidate. Its retriever is UniXcoder; the generator is used as a reward/evaluator, not trained as part of the method. This is relevant because the notebook also turns generator losses into a hard winner, but does so in an offline frozen-cache CE pipeline.

**Short quote:** “only candidate with the lowest PPL receives the positive reward.”

## Transfer boundary

This source explains the notebook's inherited target-PPL signal; it does not validate the notebook's phase head or imply that target-PPL winner labels are required for the user's AST-GR plan.
