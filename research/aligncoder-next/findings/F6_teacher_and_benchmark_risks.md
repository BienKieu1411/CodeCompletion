# F6 — Risks that can make a false AlignCoder win

## Risk 1: teacher mismatch

Target NLL from a frozen generator is a proxy for EM, execution, and human usefulness. **S01, S08, and S17** support the mechanism but do not prove calibration. Use teacher-ranked completion diagnostics and, where possible, held-out generator or test-based checks.

## Risk 2: context-length confounding

**S07** reports that context length can dominate chunk strategy. Every comparison needs the same local-prefix budget, cross-file budget, formatter, and number of generator forwards. A longer prompt is not a fair retrieval win.

## Risk 3: candidate-pool confounding

A student cannot recover candidates absent from its pool. Report pool coverage before training and keep the pool identical for task-only KD, task-plus-Jina, hard labels, and soft labels.

## Risk 4: benchmark overfitting

**S16** shows that repository context utilization remains difficult even for strong models and adds usage-aware tests. Use it as a stress check, while keeping RepoEval/CrossCodeEval as the direct completion comparison.

## Risk 5: structural false positives

AST and dataflow units can be syntactically valid but semantically unnecessary. **S02, S10, and S13** jointly motivate overlap removal, null gating, and harmful-context diagnostics.

## Minimum adversarial review

Before any superiority claim, answer:

- Did the proposed model see any target-derived feature while building the pool or query?
- Did it use more generator calls or a larger cross-file context than AlignCoder?
- Was Jina used as a teacher, a miner, or both, and are those effects separated?
- Does soft KD beat hard target-NLL labels, or only an underpowered baseline?
- Does the result survive a sliding-window and lexical-only comparator?
