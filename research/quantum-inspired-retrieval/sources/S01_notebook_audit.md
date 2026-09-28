# S01 — Static audit of `rlcoder_learned_phase_kaggle.ipynb`

- **Type / provenance:** local Jupyter notebook supplied in this repository.
- **Path:** [`src/rlcoder_learned_phase_kaggle.ipynb`](../../../src/rlcoder_learned_phase_kaggle.ipynb)
- **Inspection:** parsed notebook JSON and all generated Python string literals into AST with the project's required `bienkieu_env`; no cells were executed. Notebook has 7 cells; code cells have null execution counts and no saved outputs.
- **Generated artifacts:** `rlcoder_compat.py`, `online_learned_phase.py`, a top-level `controlled_comparison.py`, and `generation_eval/controlled_comparison.py`. Cell 6 correctly launches `online_learned_phase.py`. The controlled-comparison script is not the active launcher.
- **Credibility / recency / bias risk:** 5 / 5 / 3 for what the artifact contains; no credibility for unexecuted performance claims.

## Active path and model/loss

- Retriever: `microsoft/unixcoder-base` by default, fine-tuned with a learned-phase head.
- Generator/reward model: `deepseek-ai/deepseek-coder-1.3b-base` by default; frozen during reward scoring.
- Scorer: one separable complex state per encoder coordinate. `theta = tanh(e) * pi/2 + pi/2`; learned phase is `pi + pi*tanh(Linear(e))`; pairwise log fidelity sums per-coordinate log overlaps. With 768 hidden dimensions the dense phase projection has 590,592 weights and biases. It is a classical scorer; no quantum circuit/hardware.
- Training is **online reward mining**, not KD: each epoch mines candidates with the current retriever, scores target continuation NLL using the frozen generator, assigns the minimum-loss candidate as hard winner, then updates the retriever with cross entropy. It re-mines/re-scores after an epoch. For the QIEPSM scorers `main()` rejects soft/hybrid mode; default hard mode is the only supported one. No pairwise examples/loss are built.
- Candidate construction remains legacy line/blank-line-block retrieval rather than the user's AST chunker/AST-GR. Training prefixes/targets are synthetically split by whitespace, not benchmark completion boundaries.

## Confirmed validity and implementation concerns

1. **Test-set selection:** `build_validation` loads `cceval/{python,java}/test.parquet`; the same fixed subset is evaluated every scheduled epoch and macro EM/ES selects `best_crosscodeeval`. This is test-set model selection if run as configured. The notebook explicitly describes 200 samples per language from TEST.
2. **No matched active control:** cell 6 launches only the online scorer run. It does not run a paired cosine/fixed-phase/learned-phase comparison on the same frozen candidates and rewards. The separately generated controlled script is not invoked and its parser only exposes cosine/`quantum_fidelity`, not `qiepsm_learned_phase`.
3. **Mixed-language weighted reward bug:** the online scorer passes the first pool's language to `reward_weights` for every batch. In a mixed Python/Java run, identifier weighting is therefore wrong for one language.
4. **First-token weighting quirk:** `reward_weights` flattens labels and checks `i < 1`, so the special first-token weight applies only at flattened position 0 (which may be a masked prompt token), not to each example's first target token.
5. **Prompt-context confound:** the compatibility prompt builder subtracts each candidate's token length from the local-prefix budget. Candidate utility is measured with varying amounts of current-file context.
6. **Benchmark-limit mismatch:** the active script accepts `--benchmark-limit`, but its benchmark subprocess passes `--limit 0`; selection uses the saved validation JSONL. Changing the wrapper's benchmark limit does not constrain that evaluation.
7. **Smoke semantics:** with `SMOKE=True` the launcher requests one epoch, while scheduled benchmark evaluation starts at epoch 2. This agrees with the markdown's claim that smoke has no validation; it cannot yield a benchmark-selected best checkpoint.

## Important correction to the first audit pass

My first extraction examined the separately generated `controlled_comparison.py` but missed the active `online_learned_phase.py` literal. Therefore the earlier claims that the active engine path/scorer flags were missing, that it was an offline-only trainer, and that its learned-phase checkpoint head was omitted were incorrect. The active engine exists, accepts the learned-phase scorer, runs online reward mining, and saves `model.state_dict()` in `latest.pt`. The checkpoint omission applies only to the unused comparison script, not the active training path.

## Minimal quote (local artifact)

> “CrossCodeEval TEST”

## Conclusion

Treat the notebook as a real but unexecuted online hard-reward prototype, not as a result. Its main scientific limitation for this project is that it does not test the user's AST-GR/AST chunking setup or isolate phase against a matched control; its main evaluation flaw is selecting checkpoints on the test subset. Do not run unchanged for a paper result, and do not overwrite the supplied notebook.
