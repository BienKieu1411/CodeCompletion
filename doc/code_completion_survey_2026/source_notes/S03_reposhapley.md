# S03 — RepoShapley

- **Publication:** Findings of ACL 2026.
- **Task:** Repository-level completion, with context filtering and retrieval triggering.
- **Method:** Teacher-forced generator probes estimate signed chunk effects; a surrogate models saturation/interference; Shapley values and bounded verification select a coalition. Keep/drop decisions are distilled into discrete control tokens.
- **Useful lesson:** Context items interact: an individually weak chunk may be useful when paired with a complementary one, and some items harm decoding. Default retrieval size is K=10.
- **Limitations:** Exact enumeration scales as 2^K; single-chunk probes may miss higher-order interactions; offline teacher labeling needs many forward/decode passes; the method fine-tunes a control/generator model, so it is not a drop-in UniXcoder-only training recipe.
- **Short exact quote:** “some snippets help only when paired with complementary context”
- **Sources:** [ACL paper](https://aclanthology.org/2026.findings-acl.505/), [full PDF](https://aclanthology.org/2026.findings-acl.505.pdf).
