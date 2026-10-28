# CADRE v2.9 Official Benchmark Evaluation Report (Full 8,004 Tasks)

**Evaluated Configuration:** CADRE v2.9 (Elastic Slack-Gated Adaptive Context Budgeting)  
**Model:** `deepseek-ai/deepseek-coder-1.3b-base`  
**Hardware:** NVIDIA A100 GPU (Colab)  
**Total Benchmark Tasks:** 8,004 tasks  
**Benchmarks:** CCEval Python (2,665), CCEval Java (2,139), RepoEval Line (1,600), RepoEval API (1,600)  

---

## 1. Executive Summary & 4-Metric Evaluation

Official full-scale evaluation across all 8,004 benchmark tasks following exact official Statement Truncation and Repo-Macro Averaging protocols:

| Benchmark | Total Tasks | EM (%) | ES (%) | ID-EM (%) | ID-F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CCEval Python** | 2,665 | **26.60** | **71.86** | **37.07** | **62.98** |
| **CCEval Java** | 2,139 | **27.40** | **67.89** | **34.64** | **60.42** |
| **RepoEval Line** | 1,600 | **51.12** | **71.24** | **58.00** | **63.46** |
| **RepoEval API** | 1,600 | **43.88** | **70.05** | **47.00** | **72.42** |
| **Macro Mean** | **8,004** | **37.25** | **70.26** | **44.18** | **64.82** |

---

## 2. Growth Progression Across All 4 Metrics

| Metric | Official Base | CADRE v2.8 | CADRE v2.9 (Elastic Slack) | Gain vs Base | Gain vs v2.8 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Match (EM %)** | 31.09 | 35.19 | **37.25** | **+6.16 pp** | **+2.06 pp** |
| **Edit Similarity (ES %)** | 65.03 | 68.72 | **70.26** | **+5.23 pp** | **+1.54 pp** |
| **Identifier EM (ID-EM %)** | 39.08 | 42.03 | **44.18** | **+5.10 pp** | **+2.15 pp** |
| **Identifier F1 (ID-F1 %)** | 61.00 | 63.21 | **64.82** | **+3.82 pp** | **+1.61 pp** |

---

## 3. Core Architectural Mechanisms

1. **Window Slack Geometry:**
   $$\text{Slack} = W - (L + B_{\text{mid}} + \text{headroom}) = 3072 - (L + 1536 + 64)$$
   - $\text{Slack} \ge 0$ (2,970 tasks - short files in CCEval): Activates Slack Guard, preventing budget downgrades and unlocking $B = 2,304$ tokens cross-file ($\{1536, 2304\}$).
   - $\text{Slack} < 0$ (5,034 tasks - crowded files in RepoEval): Explores $B \in \{384, 1536, 2304\}$, freeing up to 2,600 tokens for local context.

2. **Grounding-Calibrated Confidence:**
   $$\text{Score} = \text{Conf}(z) + \beta \cdot \text{Grounding}(z, \text{Context})$$
   Eliminates low-budget overconfidence on generic identifiers (`get()`, `update()`), ensuring safe budget transitions.

3. **Copy-Protection Safety Coupling Gate ($\tau = 0.95$):**
   Safeguards 90 tasks requiring local verbatim replication, guaranteeing zero negative transfer.

---

## 4. Manifest of Artifacts in this Directory

- `metrics_4metrics.json`: Full 4-metric evaluation metrics JSON.
- `metrics_raw_vs_final.json`: Pass 1A Raw vs CADRE v2.9 Final comparative metrics.
- `cadre_v29_8004_predictions.jsonl.gz`: Complete 8,004 individual task prediction records with ground truth, candidates, gate decisions, and final predictions.
