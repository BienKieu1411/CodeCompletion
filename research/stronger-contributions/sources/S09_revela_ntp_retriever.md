# S09 — REVELA: next-token prediction as retriever supervision

- **Type / status:** Peer-reviewed primary paper, ICLR 2026; local full-paper DeepRead at doc/papers/REVELA_DEEPREAD.md.
- **URL:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/0343104ddbfc48f35f06aaae88980e48-Abstract-Conference.html
- **Credibility / recency / bias risk:** 5 / 5 / 3.
- **Primary locations:** Abstract and full paper, method and CoIR experiments; see local DeepRead for detailed extraction.

## Evidence extracted

- REVELA uses a language-model next-token-prediction objective to train a dense retriever through cross-document attention.
- It is not knowledge distillation: the language-model NTP loss is the main training loss and gradients reach the retriever through the attention weights.
- Experiments cover general/domain retrieval and CoIR, not repository-level code completion against AlignCoder.
- Its in-batch cross-document similarity matrix is still a set of document-to-document interactions. Therefore it avoids explicit labeled preference pairs, but may not satisfy a strict interpretation of “no pairwise construction.”

## Verbatim excerpt

> “Dense Retriever Learning via Language Modeling”

## Relevance and limits

REVELA weakens a broad novelty claim around using LM loss to learn a retriever. An AST-conditioned completion adaptation is possible, but it introduces joint-attention implementation cost and an inference discretization gap, and its in-batch interaction design should be checked against the user's no-pairwise constraint. It is not the primary recommendation.

## Citation

*REVELA: Dense Retriever Learning via Language Modeling*. ICLR 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/0343104ddbfc48f35f06aaae88980e48-Abstract-Conference.html
