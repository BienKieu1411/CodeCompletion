# F3 — Distributional target-utility KD is the clean non-pairwise core

## Claim

The main training loss should be a candidate-distribution knowledge-distillation loss derived from a frozen code generator's target-token likelihood. It is a complete objective by itself; it does not need RL, LiPO, chosen/rejected pairs, or a separate pairwise ranking loss.

## Construction

For an example with visible prefix `x`, target continuation `y`, candidate units `c_1...c_K`, and a null candidate `c_0`, let the frozen generator produce masked target NLL:

```text
ell_i = mean target-token NLL(y | x, c_i)
g_i = ell_0 - ell_i
p_T(i) = softmax(g_i / tau_G)
```

The student produces one logit per candidate plus one null logit:

```text
p_S = softmax(student_logits)
L_main = KL(stopgrad(p_T) || p_S)
```

The teacher is run offline and cached. Only the student is updated. The null candidate is part of the same distribution, so the objective learns retrieval usefulness and no-retrieval behavior together.

## Evidence chain

1. **S08:** REPLUG establishes the general pattern of frozen-LM likelihood supervision for a retriever.
2. **S03/S09:** CodeRAG establishes practical offline teacher-to-student training without pairwise student construction, although its teacher is a reranker.
3. **S10/S15:** context filtering and selective retrieval show why a null/keep decision is needed, not just a top-K ranking.
4. **S01/S18:** AlignCoder/RLCoder's local hard winner is a natural controlled baseline for the same target-NLL signal.

## What is novel only as a hypothesis

The novelty candidate is the combination of completion-utility distribution KD, AST-bounded repository evidence, hybrid fixed pools, and calibrated null/packing policy. The KL operation alone is prior art and must not be presented as new.

## Optional losses, clearly separated

- **Stage 0 only:** REVELA-style NTP pretraining, if tested.
- **Optional auxiliary:** pointwise keep/drop or harmful-context regression from the same teacher utility.
- **Not in the main line:** RL, policy gradient, LiPO, chosen/rejected pairs, or online teacher calls.

## Falsification test

Compare soft KD, hard target-NLL CE, and an untrained/pretrained retrieval baseline on the identical pool. If soft KD does not improve calibration or EM, keep the simpler hard-label system.
