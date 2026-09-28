# Research plan: quantum-inspired scoring for repository retrieval

**Created:** 2026-09-27  
**Question:** Can a classical phase-interference scorer improve repository-level code-completion retrieval beyond AlignCoder when the repository evidence is AST-structured?  
**Decision sought:** whether to test a small quantum-inspired scoring head, and what experiment can distinguish a useful mechanism from quantum terminology without introducing KD or pairwise training.

## Scope and constraints

- Task: repository-level code completion; retrieval should improve final completion, not merely similarity metrics.
- Target baseline: AlignCoder. Its current retrieval, query-enhancement, and candidate-generation controls must be evaluated under matched conditions.
- Existing project direction: AST-based chunks and AST-GR labels. Treat those as the structural substrate; audit the label-generation contract before implementation.
- User constraints: no distillation and no pairwise sample generation / pairwise ranking objective. Use independent candidate labels and a pointwise loss.
- The supplied Kaggle notebook is a local artifact to audit, not a result to trust. Do not modify or execute it: it has unset execution counts and external Kaggle/GPU side effects.
- Meaning of “quantum-inspired”: a classical complex-valued scoring parameterization. No quantum hardware or quantum-advantage claim.

## Falsifiable hypotheses

**H1 — phase head over a single dense embedding.** At a fixed candidate pool and equal encoder/training setup, learned-phase log fidelity improves AST-GR candidate ranking and end-to-end EM/ES over cosine. Refute if it fails to beat cosine on held-out repositories across seeds.

**H2 — phase interference across evidence channels.** A phase-interference fusion of lexical, dense, AST, and dependency evidence improves over additive fusion and a parameter-matched real-valued MLP, especially when these evidence sources are complementary. Refute if the real-valued controls match it or if gains do not reach completion metrics.

**H3 — channel complementarity is the useful mechanism, not the “quantum” label.** Gains should concentrate on examples where evidence views disagree but the correct AST-GR candidate has support in a complementary view. Refute if improvement is uniform/random or phase ablation has no measurable effect.

## Report genre and structure

Decision/validation report: (1) notebook audit, (2) evidence and limitations, (3) proposed no-KD/no-pairwise experiment, (4) go/no-go gates, (5) adversarial review.

## Sourcing strategy

1. Primary quantum-inspired retrieval and representation papers: QINM/SIGIR; QIEPSM/arXiv; Q-Interference/arXiv; ACL complex embeddings.
2. Primary repository-completion controls: AlignCoder local extraction; Repoformer/ICML; CodeRAG/EMNLP.
3. Adversarial source: recent QI embedding retrieval evaluation, explicitly down-weighted because it is a single preprint with small/custom evaluations.
4. Local implementation evidence: static parse of `src/rlcoder_learned_phase_kaggle.ipynb`; local source files are treated as ground truth about the notebook, not about model effectiveness.

## Triangulation rule and source scoring

For a broad thesis about the method, require at least three independent source types (e.g., peer-reviewed IR paper, newer preprint, direct local code/benchmark report). If this cannot be met for repository-level completion, label that claim **insufficient evidence**. Per-source scores in `sources.csv` use 1–5: credibility (venue/provenance), recency (relative to 2026-09-27), and bias risk (5 = high risk).

## Adversarial search / risks

- Search for negative or mixed results, not only phase-interference proposals.
- Test whether product fidelity simply creates a sharp/ill-calibrated kernel in 768 dimensions.
- Compare against classical MLP and linear fusion on exactly the same evidence features.
- Check candidate-generation recall separately: a reranker cannot recover an AST unit absent from its pool.
- Prevent benchmark-test model selection, repository overlap, target leakage, and candidate-dependent truncation of the current-file prefix.
- Distinguish label-ranking gains from end-to-end completion gains.

## Stop criteria

Do not recommend the phase method as the main framework unless it (a) beats cosine and classical fusion on fixed-pool retrieval, (b) improves end-to-end EM/ES against the matched AlignCoder baseline on untouched evaluation repositories, and (c) has no material inference-cost or stability regression. Otherwise keep AST-GR plus the strongest classical fusion and record a negative result.

## Changelog

- 2026-09-27: initial dossier; includes the local notebook audit and targeted research on phase-aware interaction and retrieval.
