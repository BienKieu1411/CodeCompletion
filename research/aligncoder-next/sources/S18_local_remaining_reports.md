# S18 — Remaining local paper reports

- **Type:** local synthesis of the remaining primary-paper extractions
- **Local sources:** [`PAPERS_INDEX.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/PAPERS_INDEX.md), [`C2LLM_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/C2LLM_DEEPREAD.md), [`REPOSHAPLEY_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/REPOSHAPLEY_DEEPREAD.md), [`LIPO_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/LIPO_DEEPREAD.md), [`RLCODER_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/RLCODER_DEEPREAD.md), [`REACC_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/REACC_DEEPREAD.md), [`REPOCODER_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/REPOCODER_DEEPREAD.md), [`REPOHYPER_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/REPOHYPER_DEEPREAD.md), [`STEPCODER_DEEPREAD.md`](/Users/kieugiangbien/Downloads/Project/CodeCompletion/doc/papers/STEPCODER_DEEPREAD.md)

## Evidence extracted

- C2LLM contributes PMA-style code embedding/pooling as a representation ablation, not a target-utility teacher.
- REPOSHAPLEY exposes context interaction: a chunk can be useful alone but harmful or redundant in a coalition. This motivates overlap control and filtering diagnostics.
- RLCoder reinforces the target-PPL supervision precedent and the weakness of hard/weighted winner-style updates.
- ReACC supports lexical+dense hybrid retrieval; RepoCoder demonstrates iterative retrieval–generation but also its latency and error-propagation cost.
- RepoHyper and GRACE support structural/graph expansion and fusion, while exposing parser and graph-completeness risks.
- StepCoder shows AST-aware curriculum and execution-aware supervision are possible, but its RL component is explicitly out of scope for the present framework.
- LiPO is a pairwise/listwise preference direction and is retained only as a negative constraint for this project.

## Verbatim excerpt from the local extraction

> “Khoảng trống nghiên cứu còn lại”

## Research use

This group supplies the ablation menu and constraints: PMA as an encoder variant, Shapley/interaction as a diagnostic, hybrid lexical+dense retrieval, structural expansion, and execution-aware evaluation without importing RL or pairwise training.

## Limitations

These are synthesized local reports; every retained paper-specific numerical claim must be checked against its PDF and native implementation before publication.
