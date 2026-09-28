# S05 — Stale repository context diagnostic study

- **Type / status:** Primary arXiv diagnostic preprint, May 2026.
- **URL:** https://arxiv.org/abs/2605.14478
- **Credibility / recency / bias risk:** 2 / 5 / 4.
- **Primary locations:** Abstract; §2.5; §3–4; §6.

## Evidence extracted

- The study varies current-only, stale-only, no-retrieval, and mixed current/stale context while keeping tasks controlled.
- On 17 curated signature-change examples from five Python repositories, stale-only retrieval induced stale helper references in 15/17 Qwen2.5-Coder-7B-Instruct outputs and 13/17 GPT-4.1-mini outputs; current-only context had zero stale references in the retained sample.
- The authors explicitly limit the claim: this is a narrow diagnostic, not a freshness mitigation; the sample is small, signature-drift-only, Python-only, and uses static call-pattern oracles.
- The paper identifies commit/index freshness metadata and current-evidence recall as future mitigation avenues, not tested methods.

## Verbatim excerpt

> “The current contribution is a diagnostic protocol and a set of empirical observations.”

## Relevance and limits

Version/freshness-aware retrieval is a possible separate contribution if the project indexes repository history or stale caches. It is not a strong primary direction for static single-snapshot CrossCodeEval/RepoEval, and the existing study does not show that freshness metadata improves standard completion accuracy.
