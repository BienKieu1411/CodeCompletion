# Research plan: directions to beat AlignCoder

**Date:** 2026-09-20  
**Decision question:** Given the 15 local paper reports and the existing AST chunking/GR work, which technically distinct directions are worth testing to exceed AlignCoder on repository-level code completion?

## 1. Scope and hard constraints

- Target: beat AlignCoder on the same generator, split, prompt protocol, and evaluation metrics before claiming superiority.
- Keep the user's AST-based chunking and AST-based GR training as the structural starting point.
- Do not use LiPO, chosen/rejected construction, pairwise ranking loss, or policy-gradient/RL training.
- Do not use a teacher at serving time unless a direction is explicitly marked as a latency ablation.
- Separate evidence-backed facts, inferences, and hypotheses. No direction is called “best” until controlled experiments support it.
- Preserve the existing `doc/TCD_KD_FRAMEWORK.md` as the current canonical implementation specification; this dossier explores alternatives and sharpens the decision.

## 2. Local evidence to reuse

The local reports cover AlignCoder, RLCoder, CAST/AST chunking, CodeRAG, GrepRAG, GRACE, Late Code Chunking, LiPO, ReACC, RepoCoder, RepoHyper, REVELA, and StepCoder. The most relevant signals are:

1. AlignCoder obtains its gain from target-aware query enhancement, dependency evidence, and retriever training, but uses multiple sampled completions and a hard winner based on target perplexity.
2. CAST supports AST-bounded split-then-merge chunks, but structural coherence alone is not relevance.
3. CodeRAG shows that an expensive teacher can be distilled offline into a smaller reranker with token cross-entropy and consensus filtering; it does not distill the final generator's utility distribution.
4. GrepRAG and DraCo show that lexical identifiers, structure-aware deduplication, and dataflow evidence remain complementary to dense retrieval.
5. REVELA provides a non-pairwise self-supervised route based on next-token prediction and in-batch cross-document attention, but it is not yet evidence for repository completion.
6. Repoformer and GRACE show that retrieval can be harmful and that selective/no-retrieval decisions matter.

## 3. External validation and opposition search

The external search must test, rather than merely support, the proposed story:

- **Chunking opposition:** the 2026 controlled chunking study reports that cross-file context length can dominate chunk strategy and that Function chunking is never Pareto-optimal. This prevents the claim “AST automatically wins.”
- **Teacher opposition:** target NLL may be misaligned with exact match, compilation, or tests; semantic embedding similarity may reward topical similarity rather than completion utility.
- **KD opposition:** a student can imitate teacher scores while losing complementary evidence and calibration; teacher quality and label coverage are bottlenecks.
- **Graph opposition:** static dependency edges can be incomplete in dynamic languages; lexical retrieval can be necessary for implicit or string-based APIs.
- **Benchmark opposition:** RepoEval/CrossCodeEval measure completion behavior, while ReCUBE adds usage-aware repository context utilization. Improvements must not be benchmark-specific.

## 4. Falsifiable hypotheses

### H1 — Main hypothesis: target-utility distribution KD is a better replacement for AlignCoder's hard winner

With the same candidate pool and final generator, an AST-bounded retriever trained on a soft distribution induced by frozen generator target NLL, including an explicit null candidate, will improve EM/ES over AlignCoder's hard winner. The student is trained with distributional KD; there is no pairwise sample generation.

**Falsifier:** no improvement over a hard-label target-NLL baseline on at least two repository-level slices, or candidate recall is lower enough to explain all gains/losses.

### H2 — AST is necessary but not sufficient

AST-bounded units plus parent/signature metadata will beat line chunks at fixed context budget, but AST-only will not beat a hybrid AST + lexical + dense + dependency pool. Context budget and retrieval coverage will explain more variance than AST boundaries alone.

**Falsifier:** AST-only and hybrid are indistinguishable after controlling for candidate recall and cross-file token budget, or a sliding-window baseline wins consistently.

### H3 — Non-pairwise self-supervised pretraining is a useful prior, not the main objective

REVELA-style next-token/self-supervised repository pretraining on AST units will improve initialization and cold-start retrieval, but target-utility KD is required for the final AlignCoder-level gain.

**Falsifier:** NTP-only matches or beats task-utility KD under equal training budget, or the extra pretraining gain disappears after controlling for corpus and compute.

### H4 — Selection/filtering matters more after retrieval recall is fixed

A pointwise utility/null gate or harmful-context filter will improve completion more than replacing one dense encoder with another when the candidate pool is held fixed, especially for noisy, duplicated, or implicit-dependency cases.

**Falsifier:** all retrieved contexts are non-harmful, or filtering gains vanish when generator context length is matched.

## 5. Directions to implement as competing experiments

### D1 — AST-Hybrid Target-Utility KD (recommended main line)

Fixed AST evidence pool from exact symbol/import, BM25, pretrained UniXcoder, Jina, and optional dataflow lookup. A frozen code generator scores every candidate and the null candidate with target-token NLL. Convert relative utility to a soft candidate distribution. Train UniXcoder to reproduce that distribution; serve only the student and a frozen completion generator.

Why it is plausible: it retains AlignCoder's target alignment, removes winner-take-all supervision and online multi-sampling, and uses the user's AST improvement. Why it may fail: REPLUG-style likelihood supervision is close prior art, and NLL may not equal EM.

### D2 — AST-REVELA pretraining + D1 task adaptation

Pretrain the retriever on repository-local next-token prediction with cross-document attention over AST units. Then fine-tune with D1's target-utility distribution. This tests whether generic repository continuation knowledge helps the target-conditioned objective.

Why it is plausible: no labels and no pairwise construction. Why it may fail: generic co-occurrence can prefer popular but irrelevant code and consume substantial compute.

### D3 — One-pass need/query distillation + AST retrieval

Use an offline teacher to label an information-need or query representation once, then distill it into a small student. At serving, generate one query/need vector and retrieve from the hybrid AST index. This removes AlignCoder's four-sample online draft loop.

Why it is plausible: SelfRACG, CodeRAG, and GrepRAG show that explicit information needs and multi-path retrieval are useful. Why it may fail: the query teacher can be less faithful than direct generator-utility supervision.

### D4 — Utility-aware context filtering and late expansion

Keep the hybrid pool and student fixed, but add a pointwise keep/drop/null head and deterministic AST parent/signature expansion after retrieval. Use a teacher utility or harmful-context signal; do not create chosen/rejected pairs.

Why it is plausible: CODEFILTER, Repoformer, GRACE, and Late Code Chunking all indicate that retrieved context can be redundant or harmful. Why it may fail: expansion may spend the context budget on plausible but unnecessary parents.

## 6. Priority and experiment order

1. Reproduce AlignCoder and a hard target-NLL retriever under a controlled protocol.
2. Build and freeze one AST-hybrid candidate pool; measure candidate coverage before training.
3. Run D1 task-only KD with the null candidate.
4. Add Jina as an auxiliary semantic signal only if D1's candidate coverage is inadequate; do not assume a larger embedding teacher is better.
5. Run D4 filtering/late expansion with the same pool and checkpoint.
6. Run D2 as a separate pretraining ablation, not mixed into the first claim.
7. Run D3 only if D1 is accurate but too slow or too expensive at serving.

## 7. Required evaluation and stop criteria

Report CrossCodeEval Python/Java/C#/TypeScript where available, RepoEval line/API, and a usage-oriented stress set such as ReCUBE if the data/protocol can be reproduced. For every run log:

- EM primary; ES and identifier metrics secondary; execution/pass metrics only when the benchmark supports them.
- Candidate recall/coverage, target-NLL calibration, harmful-context rate, null-gate rate, and prompt token count.
- Retrieval latency, generator latency, total latency, and number of generator forwards.
- Three seeds or bootstrap confidence intervals on the same examples.
- Per-language and per-difficulty slices: explicit imports, implicit calls, nested scopes, long files, and duplicate-heavy repositories.

Stop a direction when:

- its candidate coverage is below the controlled baseline;
- teacher-ranked completion does not improve a small pilot, indicating a scoring or formatter problem;
- it improves NLL but not EM/ES after calibration and context-budget checks;
- it needs online teacher/draft sampling to match AlignCoder's quality without a measured latency budget;
- it fails to beat the hard target-NLL baseline and AlignCoder on the prespecified primary slices after three seeds.

## 8. Deliverables

- `sources.csv` and one source note per important paper/report.
- Finding notes that distinguish facts, inferences, and open risks.
- A decision report with a ranked comparison and a single recommended main line.
- A dated delta note so future research can refresh unstable claims without rereading the entire dossier.
