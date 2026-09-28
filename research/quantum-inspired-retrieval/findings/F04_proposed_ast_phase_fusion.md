# F04 — Proposed pilot: AST relation-factorized fidelity (working name)

**Objective:** Test if quantum-inspired phase helps when attached to meaningful code relations, not arbitrary hidden dimensions. Preserve the user's AST chunker and AST-GR data; no KD, no pairwise samples/loss, no policy-gradient RL.

## Model and evidence factors

- **Retriever:** `microsoft/unixcoder-base`, initialized from the same checkpoint as the AlignCoder retriever control.
- **Generator:** for end-to-end testing, freeze and use the same generator/prompt/decoding as the matched AlignCoder run. The notebook's utility scorer defaults to `deepseek-ai/deepseek-coder-1.3b-base`; that is suitable for a low-cost pilot only if it is also the downstream evaluation generator.
- Candidate pool: union lexical/identifier, dense, AST-symbol, and dependency/graph paths; deduplicate by stable AST node/span ID. Keep candidate recall as a separate metric.
- Build relation-factor features for each query-candidate: visible identifier/signature, AST role/scope/type, AST-GR relation/edge/path, and dependency/call/import evidence. All query features must be computable without hidden target tokens at inference.

## Scoring hierarchy

First encode each factor into a normalized two-level complex state and compute a local fidelity `F_r(q,c)`. The simple relation-factorized score is

```text
s_fact(q,c) = sum_r w_r * log(epsilon + F_r(q,c))
```

This is a weighted product kernel: all factors must match reasonably well; one near-zero factor can dominate, so also test a normalized/soft-clipped log variant. Next, if the factorized head is promising, add cross-factor phase interference:

```text
a_r = softplus(g_r(features_r, q, c))
phi_r = pi * tanh(h_r(features_r, q, c))
z(q,c) = sum_r a_r * exp(i * phi_r)
s_int(q,c) = alpha * log(epsilon + |z(q,c)|^2) + b
```

The interference intensity includes `2*a_r*a_t*cos(phi_r-phi_t)`: AST relation evidence and lexical/API evidence can reinforce or conflict. This is an ordinary classical computation; use sine/cosine real features, not a quantum simulator. Treat the second scorer as a distinct ablation rather than claiming it follows automatically from the product-fidelity result.

## Objective: no KD and no pairwise training

**First pilot (teacher-free):** if the existing AST-GR artifact exposes independent binary query-candidate labels, use class-balanced pointwise `BCEWithLogitsLoss`. First verify its actual label schema, positive definition, negative sampling, and leakage rules; the AST-GR implementation was not inspected in this audit. Preserve an existing graded/multilabel semantics rather than silently thresholding it. There is no KL to a teacher, generator-PPL winner, RL reward, or candidate-pair construction.

**Optional task-utility objective, only if desired later:** use the actual gold continuation `y` and the frozen downstream generator `G`, not a teacher token distribution. For fixed candidate set `C`, let `pi_i = softmax(s(q,c_i)/T)` and `ell_i = -log P_G(y | visible_prefix, c_i)`. A latent-context objective is

```text
L_completion = -log sum_i pi_i * exp(-ell_i)
```

This is a marginal likelihood of the ground-truth continuation, not KD and not pairwise preference training. It does use generator-derived candidate likelihoods, so it is **not** teacher-free; precompute `ell_i` once for a fixed pool to avoid re-mining/re-scoring after every retriever update. Because inference may pack multiple candidates together, validate the single-context objective against the actual top-K prompt before treating it as the main loss. For the first quantum-mechanism pilot, keep pointwise AST-GR BCE as the only training loss and use benchmark EM/ES for the task-level decision.

## Required controls

1. UniXcoder cosine.
2. Calibrated additive fusion on identical lexical/dense/AST/dependency features.
3. Parameter-matched real MLP over those features.
4. Relation-factorized fidelity with fixed phases; then learned phases.
5. Cross-factor interference scorer; phase-off and amplitude-only ablations.
6. Notebook-style per-hidden-dimension learned-phase log fidelity as a separate reproduction target.

Use identical fixed candidate pools, AST chunk IDs, prompt prefix, labels, optimizer/update budget, and seeds. First isolate score quality; later evaluate full candidate generation and completion.

## Decision criteria

- Retrieval: AST-GR Recall@K, positive coverage, nDCG/MRR, and candidate-pool recall before reranking.
- Completion: matched AlignCoder generator and inference configuration, EM/ES, no-context/harmful-context behavior, latency, and context-token budget.
- Data: repository-disjoint train/dev/test; development data selects checkpoints; test is run once; at least 3 seeds with paired uncertainty.
- Advance only if the phase method beats cosine **and** the parameter-matched classical MLP on fixed-pool ranking, then improves end-to-end completion over the matched AlignCoder control across seeds. If MLP matches phase, keep the simpler classical scorer.

**Status:** Research proposal only; not implemented or run.
