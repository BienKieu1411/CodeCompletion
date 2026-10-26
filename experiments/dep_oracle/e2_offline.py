#!/usr/bin/env python3
"""E2 offline: per-sample budget policies on the cached budget sweep (CPU only, no new generation).

kaggle_sweep_<ds>_out/sweep/b<B>/prediction.jsonl holds, for 1000 samples per dataset and B in 0/256/512/1024/1536,
the greedy completion and per-token log-probs. A policy that only needs (a) static features and (b) the generator's
confidence on a SUBSET S of budgets can be scored exactly from these files; its cost is |S| generations.

Policies:
    fixed_B                one budget for every sample
    conf_S                 argmax over S of first-line mean log-prob                    (|S| generations)
    conf_cal_S             same + a per-budget offset fitted by grouped CV               (|S| generations)
    vote_S                 most frequent first line over S, ties -> confidence           (|S| generations)
    learned_S              per-budget logistic regression on static + conf_S, grouped CV (|S| generations)
Pooled = both datasets, one policy (dataset-agnostic features), the setting that matters: F3 showed fixed
strategies trade RepoEval against CCEval.

usage: python e2_offline.py --out e2_offline_summary.json
"""
import argparse
import json
import os
import sys
import zlib
from collections import Counter, defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep as BS  # noqa: E402
import dep_oracle as D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGETS = [0, 256, 512, 1024, 1536]


def first_line(pred):
    for l in pred.split("\n"):
        if l.strip():
            return l.strip()
    return ""


def load(ds):
    root = os.path.join(HERE, f"kaggle_sweep_{ds}_out", "sweep")
    feats = {r["task_id"]: r for r in map(json.loads, open(os.path.join(root, "features.jsonl"), encoding="utf-8"))}
    em, conf, fl = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for b in BUDGETS:
        for r in map(json.loads, open(os.path.join(root, f"b{b}", "prediction.jsonl"), encoding="utf-8")):
            t = r["task_id"]
            em[b][t] = D.simple_score(r["pred"], feats[t]["gt"])[0]
            conf[b][t] = BS.first_line_conf(r["pieces"], r["lp"])
            fl[b][t] = first_line(r["pred"])
    ids = sorted(set.intersection(*[set(em[b]) for b in BUDGETS]))
    fn = sorted(next(iter(feats.values()))["features"])
    static = {t: [feats[t]["features"][k] for k in fn] for t in ids}
    group = {t: (t.split("/")[0] if not t.startswith("project_cc") else f"cc{zlib.crc32(t.encode()) % 50}")
             for t in ids}
    return {"ids": ids, "em": em, "conf": conf, "fl": fl, "static": static, "group": group}


def folds_of(groups, k=5):
    uniq = sorted(set(groups.values()))
    return {g: zlib.crc32(g.encode()) % k for g in uniq}


def cv_offsets(ids, groups, em, conf, S, k=5):
    """Per-budget additive offsets on confidence, grid-searched on the training folds."""
    fo = folds_of(groups, k)
    grid = [x / 10 for x in range(-10, 11)]
    choice = {}
    for f in range(k):
        tr = [t for t in ids if fo[groups[t]] != f]
        te = [t for t in ids if fo[groups[t]] == f]
        off = {b: 0.0 for b in S}
        for _ in range(2):  # coordinate ascent
            for b in S[1:]:
                best = max(grid, key=lambda x: sum(em[max(S, key=lambda c: conf[c][t] + (x if c == b else off[c]))][t]
                                                   for t in tr))
                off[b] = best
        for t in te:
            choice[t] = max(S, key=lambda c: conf[c][t] + off[c])
    return choice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "e2_offline_summary.json"))
    a = ap.parse_args()
    data = {ds: load(ds) for ds in ("repoeval", "cceval")}
    # pooled view
    P = {"ids": [], "em": defaultdict(dict), "conf": defaultdict(dict), "fl": defaultdict(dict), "static": {},
         "group": {}, "ds": {}}
    for ds, d in data.items():
        for t in d["ids"]:
            P["ids"].append(t)
            P["ds"][t] = ds
            P["static"][t] = d["static"][t]
            P["group"][t] = d["group"][t]
            for b in BUDGETS:
                P["em"][b][t], P["conf"][b][t], P["fl"][b][t] = d["em"][b][t], d["conf"][b][t], d["fl"][b][t]
    ids, em, conf, fl = P["ids"], P["em"], P["conf"], P["fl"]

    def score(choice):
        out = {}
        for ds in data:
            sub = [t for t in ids if P["ds"][t] == ds]
            out[ds] = round(100 * sum(em[choice[t]][t] for t in sub) / len(sub), 2)
        out["macro"] = round((out["repoeval"] + out["cceval"]) / 2, 2)
        return out

    res = {"n": {ds: len(d["ids"]) for ds, d in data.items()}, "policies": {}}
    pol = res["policies"]
    for b in BUDGETS:
        pol[f"fixed_{b}"] = {**score({t: b for t in ids}), "gens": 1}
    pol["oracle_all"] = {**score({t: max(BUDGETS, key=lambda b: em[b][t]) for t in ids}), "gens": "GT"}
    swing = {ds: sum(len({em[b][t] for b in BUDGETS}) > 1 for t in data[ds]["ids"]) for ds in data}
    res["n_swing_samples"] = swing

    for r in (2, 3, 5):
        for S in combinations(BUDGETS, r):
            S = list(S)
            name = "".join(f"_{b}" for b in S)
            pol[f"oracle{name}"] = {**score({t: max(S, key=lambda b: em[b][t]) for t in ids}), "gens": "GT"}
            pol[f"conf{name}"] = {**score({t: max(S, key=lambda b: conf[b][t]) for t in ids}), "gens": r}
            pol[f"conf_cal{name}"] = {**score(cv_offsets(ids, P["group"], em, conf, S)), "gens": r}
            vote = {}
            for t in ids:
                c = Counter(fl[b][t] for b in S)
                top = max(c.values())
                vote[t] = max((b for b in S if c[fl[b][t]] == top), key=lambda b: conf[b][t])
            pol[f"vote{name}"] = {**score(vote), "gens": r}
            if r <= 3:
                feats = {t: P["static"][t] + [conf[b][t] for b in S] for t in ids}
                pol[f"learned{name}"] = {**score(BS._cv_policy(ids, P["group"], feats, em, S)), "gens": r}
    feats = {t: P["static"][t] for t in ids}
    pol["learned_static_only"] = {**score(BS._cv_policy(ids, P["group"], feats, em, BUDGETS)), "gens": 1}

    best_fixed = max((k for k in pol if k.startswith("fixed_")), key=lambda k: pol[k]["macro"])
    res["best_fixed_macro"] = best_fixed
    for g in (2, 3, 5):
        cands = {k: v for k, v in pol.items() if v["gens"] == g}
        res[f"top5_with_{g}_gens"] = sorted(cands, key=lambda k: -cands[k]["macro"])[:5]
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    show = [best_fixed, "fixed_512", "fixed_1024", "fixed_1536", "oracle_all", "learned_static_only"]
    for g in (2, 3, 5):
        show += res[f"top5_with_{g}_gens"][:3]
    show += ["oracle_512_1536", "conf_512_1536", "conf_cal_512_1536", "oracle_0_512_1536", "conf_0_512_1536"]
    print("swing samples:", swing)
    for k in dict.fromkeys(show):
        print(f"{k:28s} {pol[k]}")


if __name__ == "__main__":
    main()
