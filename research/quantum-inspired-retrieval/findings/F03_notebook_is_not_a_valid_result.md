# F03 — The Kaggle notebook is an unexecuted online hard-reward prototype, not a result

**Thesis:** `src/rlcoder_learned_phase_kaggle.ipynb` contains a functioning-in-principle learned-phase training path, but it does not isolate whether phase helps and its checkpoint selection protocol contaminates the test estimate.

**Evidence:** Code cells have no saved outputs/execution counts (S01). Cell 6 correctly launches the embedded `online_learned_phase.py`; the active script accepts the learned-phase scorer, mines candidates/rewards each epoch, and saves the full model state. It uses frozen-generator target NLL to make hard-winner labels and CE—not KD or pairwise loss (S01, S10). However, the scheduled validation is loaded from CrossCodeEval TEST and used to select the best macro-EM/ES checkpoint; the active run has no same-pool cosine/phase control; mixed-language reward weighting and prompt-prefix length are confounded; candidate chunks/labels are not the user's AST-GR/AST implementation (S01).

**Correction:** An earlier audit incorrectly inferred a missing active engine and omitted head checkpoint by looking only at the separately generated `controlled_comparison.py`. Those claims do not apply to the active online engine.

**Consequence:** Do not use the notebook's future selected score as an untouched test result. Reuse the phase math only after moving to repository-disjoint development data and a controlled same-pool comparison with the AST-GR pipeline.
