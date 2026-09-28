# S10 — Context-Picker: adjacent set-selection RL

- **Type / status:** Primary arXiv preprint, December 2025; long-context/multi-hop QA rather than code completion.
- **URL:** https://arxiv.org/abs/2512.14465
- **Credibility / recency / bias risk:** 2 / 4 / 3.
- **Primary locations:** Abstract and method.

## Evidence extracted

- Context-Picker frames evidence-subset selection as a multi-stage reinforcement-learning problem.
- It uses recall- then precision-oriented policy stages and mines minimal sufficient sets via leave-one-out analysis for denser supervision.
- Its evaluations are QA benchmarks, not repository-level code completion.

## Verbatim excerpt

> “minimal sufficient sets”

## Relevance and limits

Set-selection RL is not novel in general. The remaining claim must be code-specific and narrow (for example, AST-GR-constrained evidence actions and completion-likelihood reward), and direct code-completion prior art remains the decisive comparison. This adjacent paper is a warning against claiming generic slate selection as the contribution.

## Citation

Zhu, S. et al. (2025). *Context-Picker: Dynamic context selection using multi-stage reinforcement learning*. arXiv:2512.14465.
