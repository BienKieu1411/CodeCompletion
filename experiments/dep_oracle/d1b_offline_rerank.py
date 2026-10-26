#!/usr/bin/env python3
r"""D1b-CPU (0 CU): Can Repo Call-Site Agreement & Scope-Partitioned Anti-Copycat scoring
improve candidate selection offline on E5 (8-sample pools) and E2 (multi-budget greedy)?

Theory (from C1 & C3):
  1. Call-Site Pattern Agreement: If candidate line `p` invokes `fn(...)` or `obj.attr`,
     how well do the identifiers/kwargs inside `p` overlap with peer call-sites of `fn` in the repo?
  2. Scope-Partitioned Anti-Copycat (C3):
     - If `p` is copied verbatim from a cross-file chunk (`cross_text`) and NOT present in `in-file left`,
       it is likely a copycat error (37.5% of RepoEval usage errors).
     - If `p` uses identifiers that only appear in `cross_text` (not in `ex["left"]` and not defined as
       a top-level class/def in the repo), those are foreign local variables (`S_distractor`).

Usage:
  .\.venv-revela\Scripts\python.exe experiments/dep_oracle/d1b_offline_rerank.py
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dep_oracle as D  # noqa: E402
import d1_callsite_cpu as D1  # noqa: E402
import e2_confirm as E2  # noqa: E402
import e5_budget_tts as E5  # noqa: E402
import e7_errors as E7  # noqa: E402
import likelihood_probe as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def candidate_structural_features(p_line, line_pfx, ex, cs_idx, defs_idx, left_words, left_recent, left_lines_set):
    """Compute structural call-site agreement and scope-partition features for a candidate first line."""
    p1 = p_line.strip()
    if not p1:
        return {"empty": 1.0, "cs_overlap": 0.0, "kw_match": 0.0, "local_ratio": 0.0, "foreign_local": 1.0}
    full_line = (line_pfx + p1).strip()
    callees, dots = D1.extract_triggers(full_line)
    p_ids = set(D.identifiers(p1))
    p_kws = set(D1.KW_RE.findall(p1))

    # 1. Call-site overlap
    cs_blocks = []
    for _, short in callees[:2]:
        cs_blocks.extend(cs_idx.find_callsites(short, max_sites=3))
    for rec, _ in dots[:2]:
        cs_blocks.extend(cs_idx.find_member_usages(rec, max_sites=2))
    cs_text = "\n".join(b["code_content"] for b in cs_blocks)
    cs_ids = set(D.identifiers(cs_text))
    cs_kws = set(D1.KW_RE.findall(cs_text))

    callee_names = {s for _, s in callees} | {r for r, _ in dots}
    arg_ids = p_ids - callee_names
    cs_overlap = len(arg_ids & cs_ids) / max(len(arg_ids), 1) if arg_ids else 0.0
    kw_match = len(p_kws & cs_kws) / max(len(p_kws), 1) if p_kws else 0.0

    # 2. Scope provenance of identifiers in p1
    # Local identifiers (in last 30 lines of left context or anywhere in left context)
    in_recent = len(arg_ids & left_recent) / max(len(arg_ids), 1) if arg_ids else 1.0
    # Foreign local identifier: not in left_words, and not a known repo def/class
    n_foreign = sum(1 for x in p_ids if x not in left_words and defs_idx.where(x) is None)

    return {
        "empty": 0.0,
        "cs_overlap": cs_overlap,
        "kw_match": kw_match,
        "in_recent": in_recent,
        "foreign_local": float(n_foreign),
    }


def eval_on_e5():
    print("=== Evaluating Call-Site + Scope Tie-Breaking on E5 (1000 dev/dataset) ===")
    for ds in ("repoeval", "cceval"):
        subset, cpath = E7.CFG[ds]
        D.DATASET = ds
        gts = {t: r["gt"] for t, r in P.load_jsonl(os.path.join(HERE, "probe_inputs", ds, "gt.jsonl")).items()}
        d = os.path.join(HERE, "modal_out", f"e5_{ds}")
        rows = [r for fn in sorted(os.listdir(d)) if fn.startswith("items_")
                for r in map(json.loads, open(os.path.join(d, fn), encoding="utf-8"))]
        ids = {r["task_id"] for r in rows}
        exs = D.load_examples(os.path.join(HERE, "aligncoder_data"), subset, ids)

        for pool_name in ("mixed", "fixed1536"):
            em_vote, em_cs_vote = [], []
            for r in rows:
                t = r["task_id"]
                ex = exs[t]
                pool = sum((r["samples"][str(b)][:k] for b, k in E5.POOLS[pool_name].items()), [])
                # Baseline vote
                pred_vote = E5.select(pool, r["lines"], "vote")
                em_vote.append(D.simple_score(pred_vote, gts[t])[0])

                # Call-site + Scope tie-broken vote
                line_pfx = "" if ex["left"].endswith("\n") else ex["left"].split("\n")[-1]
                cs_idx = D1.RepoCallSiteIndex(ex)
                defs_idx = D.ExampleDefs(ex)
                left_words = E7.words(ex["left"])
                left_recent = cs_idx.local_recent_ids
                left_lines_set = {" ".join(l.strip().split()) for l in ex["left"].splitlines() if l.strip()}

                # Group by normalized first line (as in E5.select)
                groups = {}
                for cand in pool:
                    l1 = E5.first_line(cand).strip()
                    if not l1:
                        continue
                    groups.setdefault(l1, []).append(cand)
                if not groups:
                    em_cs_vote.append(em_vote[-1])
                    continue

                def score_line(l1):
                    cnt = len(groups[l1])
                    f = candidate_structural_features(l1, line_pfx, ex, cs_idx, defs_idx, left_words, left_recent, left_lines_set)
                    # Vote count + structural bonus (breaks ties and rescues strong call-site matches within 1 vote of top)
                    bonus = 0.6 * f["cs_overlap"] + 0.5 * f["kw_match"] + 0.4 * f["in_recent"] - 0.8 * f["foreign_local"]
                    return cnt + bonus

                best_l1 = max(groups, key=score_line)
                pred_cs = groups[best_l1][0]
                em_cs_vote.append(D.simple_score(pred_cs, gts[t])[0])

            n = len(rows)
            w = sum(a < b for a, b in zip(em_vote, em_cs_vote))
            l_ = sum(a > b for a, b in zip(em_vote, em_cs_vote))
            print(f"  {ds:10s} {pool_name:10s}: vote={100*sum(em_vote)/n:5.2f} -> cs_scope_vote={100*sum(em_cs_vote)/n:5.2f} "
                  f"[+{w}/-{l_}, p={D.mcnemar(w, l_):.4f}]")


if __name__ == "__main__":
    eval_on_e5()
