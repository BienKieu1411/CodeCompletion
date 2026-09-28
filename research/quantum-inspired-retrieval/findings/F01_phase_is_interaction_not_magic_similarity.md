# F01 — The most plausible use of phase is evidence interaction

**Thesis:** A phase representation is most justifiable when the model must represent whether distinct evidence signals reinforce or suppress one another. A single learned-phase fidelity over pooled UniXcoder embeddings is a weaker and less interpretable hypothesis.

**Evidence:** QINM motivates interference to capture dependencies among matching units (S04); complex word embeddings use relative phase in composition (S05); Q-Interference defines amplitude-weighted cosine phase differences and gives an exact real-valued factorization (S03). QIEPSM tests fidelity similarity/projection but finds that the quantum-inspired head alone is not consistently superior to a classical head when the backbone is frozen (S02).

**Transfer status:** Mechanistically motivated across IR/NLP, but no source here tests phase fusion on repository-level code completion. Therefore the thesis supports an ablation, not an expected benchmark win.

**Test:** On the same candidate pools and same AST-GR pointwise labels, compare (a) cosine, (b) additive calibrated lexical+dense+AST+dependency fusion, (c) parameter-matched real MLP, (d) phase-interference fusion; run phase-off ablation. Measure fixed-pool ranking and end-to-end EM/ES.

**Sources:** S02–S05, S07–S08.
