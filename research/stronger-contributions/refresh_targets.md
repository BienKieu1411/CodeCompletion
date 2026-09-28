# Refresh targets

Research snapshot: 2026-09-27. Refresh before submission or expensive training because repository-retrieval papers are appearing quickly.

1. Search ACL/EMNLP/ICLR/NeurIPS/ICSE/ASE and arXiv for repository-level completion, context-set selection, slate-policy RL, generator likelihood rewards, AST-GR actions, and code retrieval.
2. Verify whether a new method directly combines AST relation-constrained evidence actions with full-prompt completion reward. Current claim is only a targeted-search gap, not global novelty.
3. Read the full RepoShapley implementation and compare its context candidate construction, ordering, reward, and deployment controller. It is the closest direct context-interaction baseline.
4. Revisit AIRCoder and ACToR status/results; distinguish pairwise fusion and generation-time retrieval from the proposed training.
5. Recheck REVELA's implementation and whether its in-batch document interactions violate the user's intended meaning of “no pairwise.”
6. Audit the actual AST-GR schema and code: typed edges, labels, source spans, language/parser support, split leakage, and whether target-derived signals are used at inference.
7. Check whether CrossCodeEval/RepoEval candidate-pool and prompt protocols have changed; reproduce the exact AlignCoder baseline before claiming gains.
