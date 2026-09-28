# S07 — CECoder: fine-grained code-element usage retrieval

- **Type / status:** Primary paper in IEEE ISSREW 2025.
- **URL:** https://ieeexplore.ieee.org/document/11262312/
- **Credibility / recency / bias risk:** 4 / 4 / 3.
- **Primary locations:** IEEE abstract and bibliographic record.

## Evidence extracted

- CECoder targets retrieval of concrete code-element usages, not only function-level semantic similarity.
- It retrieves an initial context, generates draft code, splits the draft into blocks of code-element invocations, retrieves usage snippets for those blocks, then reranks and selectively integrates the snippets.
- It evaluates on DevEval (1,825 repository-level code-generation tasks) and reports improvements over RepoCoder.

## Verbatim excerpt

> “retrieval of concrete code element usage”

## Relevance and limits

This overlaps the goal of finding evidence about how APIs are used. A narrower distinction would be to predict evidence needs before generation from visible-prefix features, train with target-AST-derived static labels, and avoid draft-generation/retrieval loops. CECoder is repository-level generation rather than the exact next-fragment completion setting; it remains close prior art.
