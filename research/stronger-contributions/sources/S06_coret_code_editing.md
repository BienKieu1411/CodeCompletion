# S06 — CoRet: repository-level retriever for code editing

- **Type / status:** Peer-reviewed primary paper, ACL 2025 short paper.
- **URL:** https://aclanthology.org/2025.acl-short.62/
- **Credibility / recency / bias risk:** 5 / 4 / 2.
- **Primary locations:** Abstract; §1–3; ACL Anthology PDF.

## Evidence extracted

- CoRet trains a dense retriever for code editing using code semantics, repository hierarchy, and call-graph context.
- It formulates training around likelihood of retrieving the ground-truth edited chunks among chunks from the same repository, and reports improved retrieval recall on SWE-bench and Long Code Arena bug-localization.
- This is important adjacent prior art for repository-conditioned retriever objectives, but the task query is an issue/change request and the primary outcome is localization/recall—not cursor-based code completion.

## Verbatim excerpt

> “a loss function explicitly designed for repository-level retrieval”

## Relevance and limits

Do not claim that repository-specific retriever training or call-graph-enriched retrieval is new. The potential distinction for this project is completion-specific, multi-label evidence support derived from gold continuation AST relations and evaluated end-to-end; novelty still requires exact literature review.
