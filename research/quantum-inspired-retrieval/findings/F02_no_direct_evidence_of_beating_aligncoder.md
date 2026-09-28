# F02 — Current evidence cannot support “quantum-inspired will beat AlignCoder”

**Thesis:** The reviewed quantum-inspired papers provide a modeling analogy and early controlled experiments, but no direct repository-completion comparison. Any claim of superiority is currently underdetermined.

**Evidence:** QIEPSM evaluates TREC DL, not code (S02); Q-Interference evaluates autoregressive language modeling and reports mixed quality relative to its controls (S03); QINM is ad-hoc text retrieval (S04); the ACL complex embedding paper is sentence classification (S05). The recent embedding-limits preprint reports weak standalone retrieval in its own small/custom evaluation, but is low-confidence and only an adversarial warning (S06). Repository-specific work shows multiple paths and selective/no retrieval matter (S07–S08).

**Transfer status:** Insufficient evidence to state an expected gain over AlignCoder.

**Decision:** Run a cheap, fixed-pool scorer ablation first. Promote it only if it beats both cosine and strong classical fusion on untouched repository completion data.
