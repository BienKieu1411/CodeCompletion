# F01 — What is already occupied

## Conclusion

Do not claim AST chunking, generic hybrid retrieval, target-PPL supervision, context-set interaction, adaptive retrieval, or abstention as the central novelty.

## Evidence map

1. **AST chunks and adaptive hybrid retrieval:** AIRCoder uses AST-preserving chunks, eight textual/dependency/structural metrics, and query-conditioned fusion. Its learned fusion uses pairwise MSE, which is excluded by the project constraint. See [S01](../sources/S01_aircoder_acl2026.md).
2. **Generator-likelihood reward for retrieval:** RLCoder and AlignCoder both use a frozen code LM's target continuation likelihood/PPL to update a retriever, with hard winner-style candidate rewards. See [RLCoder DeepRead](../../../doc/papers/RLCODER_DEEPREAD.md) and [AlignCoder DeepRead](../../../doc/papers/ALIGNCODER_DEEPREAD.md).
3. **Coalition interaction:** RepoShapley directly argues that context value is interaction-dependent, estimates Shapley-style marginal utility, verifies selected sets with a frozen generator, and distills selection decisions. See [S08](../sources/S08_reposhapley_acl2026.md).
4. **Generic RL set selection:** Context-Picker uses multi-stage RL over evidence subsets for QA. This is adjacent evidence that slate selection alone is not new. See [S10](../sources/S10_context_picker_adjacent.md).
5. **Adaptive retrieval:** ACToR retrieves at critical generation tokens; CodeRAG probes queries and uses multiple retrieval paths; Repoformer learns when to abstain. Reuse their existing local extractions; see [refresh targets](../refresh_targets.md).
6. **LM loss for retriever training:** REVELA uses NTP as the main objective in general/domain retrieval, including code retrieval but not repository completion. See [S09](../sources/S09_revela_ntp_retriever.md).
7. **Cross-model applicability:** AlignCoder evaluates across five backbones and Repoformer reports support for different generation models/retrievers. Generalization by itself is not new; only a specifically multi-generator training reward remains a candidate. See [S15](../sources/S15_repoformer_generalization.md).

## Consequence for the claim

The potential gap is not “context utility,” “RL,” or “AST” separately. The narrow claim to test is whether an **AST-GR-constrained action space for selecting an ordered, complete context slate**, directly optimized by the final prompt's scalar completion reward, improves code completion under the exact same candidate coverage and token budget—without pairwise data or distilled KEEP/DROP labels.

This targeted search has not proven global novelty. The distinction could still be judged incremental unless the AST-relation ablation is decisive.
