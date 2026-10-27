#!/usr/bin/env python3
"""E9-pre (CPU, 0 CU): how much do CADRE v2.8's three prompt variants complement each other?

Input: partner's cadre_v28_8004_predictions.jsonl.gz (BienKieu ba5983c; window 2560, RLCoder top-10 chunks,
cross-file cap 1536, min in-file 1024). Variants per task: pass1_raw (chunks only), pass1_tree_cssi (+tree stub),
cadre_tree_spec_raw (+tree +target spec); cadre_v28_final = copy-gate over raw / tree_spec.

Descriptive only (no decision rides on it): it sizes the "variant" axis of a variant x budget grid before any GPU.
    per dataset: simple_score EM of each variant, oracle over the 3 variants, gap to the best variant,
    pairwise disagreement. Compare with the budget-axis oracle gap on AlignCoder (E2: 5.5 / 4.7 at 2k).
usage: python e9_cadre_variant_oracle.py <predictions.jsonl.gz>
"""
import gzip
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dep_oracle as D

V = ["pass1_raw", "pass1_tree_cssi", "cadre_tree_spec_raw", "cadre_v28_final"]
rows = defaultdict(list)
for r in map(json.loads, gzip.open(sys.argv[1], "rt", encoding="utf-8")):
    rows[r["benchmark"]].append({v: D.simple_score(r[v], r["groundtruth"])[0] for v in V})
res = {}
for bm, rs in rows.items():
    n = len(rs)
    pct = lambda x: round(100.0 * x / n, 2)
    em = {v: pct(sum(x[v] for x in rs)) for v in V}
    best = max(V[:3], key=lambda v: em[v])
    orc = pct(sum(max(x[v] for v in V[:3]) for x in rs))
    res[bm] = {"n": n, "em_simple": em, "best_variant": best, "oracle3": orc,
               "oracle_gap": round(orc - em[best], 2),
               "raw_vs_tree_spec_disagree": pct(sum(x["pass1_raw"] != x["cadre_tree_spec_raw"] for x in rs))}
print(json.dumps(res, indent=1))
