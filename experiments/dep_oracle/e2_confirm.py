#!/usr/bin/env python3
"""E2 confirmation: policies chosen on the 1000-sample dev sweeps, applied unchanged to the held-out test samples.

Fixed before the hold-out run (2026-10-25), from e2_offline.py on dev:
    PRIMARY   learned_256_1024_1536 : pooled per-budget logistic regression on static features + first-line confidence
                                      at budgets 256/1024/1536 (3 generations), fitted on ALL dev samples of both datasets
    secondary learned_512_1536 (2 gens), conf_256_1024_1536 (3 gens, no learning), learned_static (1 gen)
    references fixed_B; per-dataset best fixed on dev (RepoEval 512, CCEval 1536); best single fixed on dev (1024)
Success: PRIMARY >= per-dataset best fixed on BOTH datasets, and pooled McNemar vs it p < 0.05.

usage: python e2_confirm.py   (reads kaggle_sweep_*_out/sweep and modal_out/e2hold_*)
"""
import json
import os
import sys
import zlib
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep as BS  # noqa: E402
import dep_oracle as D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGETS = [256, 512, 1024, 1536]
DEV_BEST = {"repoeval": 512, "cceval": 1536}


def load(root):
    feats = {r["task_id"]: r for r in map(json.loads, open(os.path.join(root, "features.jsonl"), encoding="utf-8"))}
    em, conf = defaultdict(dict), defaultdict(dict)
    for b in BUDGETS:
        for r in map(json.loads, open(os.path.join(root, f"b{b}", "prediction.jsonl"), encoding="utf-8")):
            t = r["task_id"]
            em[b][t] = D.simple_score(r["pred"], feats[t]["gt"])[0]
            conf[b][t] = BS.first_line_conf(r["pieces"], r["lp"])
    ids = sorted(set.intersection(*[set(em[b]) for b in BUDGETS]))
    fn = sorted(next(iter(feats.values()))["features"])
    return ids, em, conf, {t: [feats[t]["features"][k] for k in fn] for t in ids}


def pooled(roots):
    ids, em, conf, st, ds = [], defaultdict(dict), defaultdict(dict), {}, {}
    for name, root in roots.items():
        i, e, c, s = load(root)
        for t in i:
            ids.append(t)
            ds[t] = name
            st[t] = s[t]
            for b in BUDGETS:
                em[b][t], conf[b][t] = e[b][t], c[b][t]
    return ids, em, conf, st, ds


def fit_predict(Xtr, ytr, Xte, S):
    import numpy as np
    Xtr, Xte = np.array(Xtr, float), np.array(Xte, float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    probs = {}
    for b in S:
        w, c = BS._fit_logreg(Xtr, np.array(ytr[b], float))
        probs[b] = 1 / (1 + np.exp(-(Xte @ w + c)))
    return [max(S, key=lambda b: (round(float(probs[b][k]), 6), b == 1536)) for k in range(len(Xte))]


def main():
    dev = pooled({ds: os.path.join(HERE, f"kaggle_sweep_{ds}_out", "sweep") for ds in DEV_BEST})
    hold = pooled({ds: os.path.join(HERE, "modal_out", f"e2hold_{ds}") for ds in DEV_BEST})
    did, dem, dconf, dst, _ = dev
    hid, hem, hconf, hst, hds = hold

    def learned(S, use_conf=True):
        f = lambda conf, st, t: st[t] + ([conf[b][t] for b in S] if use_conf else [])
        ch = fit_predict([f(dconf, dst, t) for t in did], {b: [dem[b][t] for t in did] for b in S},
                         [f(hconf, hst, t) for t in hid], S)
        return dict(zip(hid, ch))

    pols = {f"fixed_{b}": {t: b for t in hid} for b in BUDGETS}
    pols["per_dataset_best_fixed(dev)"] = {t: DEV_BEST[hds[t]] for t in hid}
    pols["PRIMARY learned_256_1024_1536"] = learned([256, 1024, 1536])
    pols["learned_512_1536"] = learned([512, 1536])
    pols["learned_static"] = learned(BUDGETS, use_conf=False)
    pols["conf_256_1024_1536"] = {t: max([256, 1024, 1536], key=lambda b: hconf[b][t]) for t in hid}
    pols["oracle_4budgets"] = {t: max(BUDGETS, key=lambda b: hem[b][t]) for t in hid}

    ref = pols["per_dataset_best_fixed(dev)"]
    res = {"n_holdout": {ds: sum(hds[t] == ds for t in hid) for ds in DEV_BEST}, "policies": {}}
    for name, ch in pols.items():
        row = {}
        for ds in DEV_BEST:
            sub = [t for t in hid if hds[t] == ds]
            w = sum(hem[ch[t]][t] > hem[ref[t]][t] for t in sub)
            l_ = sum(hem[ch[t]][t] < hem[ref[t]][t] for t in sub)
            row[ds] = {"em": round(100 * sum(hem[ch[t]][t] for t in sub) / len(sub), 2),
                       "vs_ref": [w, l_, round(D.mcnemar(w, l_), 4)]}
        w = sum(hem[ch[t]][t] > hem[ref[t]][t] for t in hid)
        l_ = sum(hem[ch[t]][t] < hem[ref[t]][t] for t in hid)
        row["macro"] = round((row["repoeval"]["em"] + row["cceval"]["em"]) / 2, 2)
        row["pooled_vs_ref"] = [w, l_, round(D.mcnemar(w, l_), 4)]
        res["policies"][name] = row
    p = res["policies"]["PRIMARY learned_256_1024_1536"]
    r = res["policies"]["per_dataset_best_fixed(dev)"]
    ok = all(p[ds]["em"] >= r[ds]["em"] for ds in DEV_BEST) and p["pooled_vs_ref"][2] < 0.05 \
        and p["pooled_vs_ref"][0] > p["pooled_vs_ref"][1]
    res["decision"] = "PASS" if ok else "FAIL"
    out = os.path.join(HERE, "modal_out", "e2_confirm_summary.json")
    json.dump(res, open(out, "w"), indent=1)
    print("hold-out n:", res["n_holdout"])
    for k, v in res["policies"].items():
        print(f"{k:32s} RepoEval {v['repoeval']['em']:5.2f} {v['repoeval']['vs_ref']}  "
              f"CCEval {v['cceval']['em']:5.2f} {v['cceval']['vs_ref']}  macro {v['macro']:5.2f} pooled {v['pooled_vs_ref']}")
    print("DECISION:", res["decision"])


if __name__ == "__main__":
    main()
