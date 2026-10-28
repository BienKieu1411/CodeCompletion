#!/usr/bin/env python3
"""E10b (CPU): E2's learned budget policy (AlignCoder candidates, window 2k) re-cross-fitted with REPO folds.

E2 (HANDOFF 11) cross-fitted dev (1000 random tasks) <-> hold-out: a task-level split, so the same repos sit on
both sides (all 8 RepoEval repos; most CCEval repos). Its 5-fold CV on dev also used pseudo-random groups for
CCEval. Here: union of dev + hold-out sweeps (RepoEval-API 1600, CCEval-py 2665, budgets 256/512/1024/1536),
folds from dev_folds.json (repo-disjoint), same features and logistic fit as e2_confirm.py, fit on fold f, apply
to fold 1-f. "Per-dataset best fixed" is also chosen on fold f only. Scorer = simple_score, as in E2.

Pre-registered: E2 claim survives a repo split iff learned_256_1024_1536 (repo cross-fit) >= per-dataset best
fixed (repo cross-fit) on BOTH datasets AND pooled McNemar p < 0.05. Descriptive: learned vs conf_256_1024_1536
(no learning, needs no fit). FAIL -> report; the E2 numbers in the brief get a caveat.
usage: python e10b_e2_repo_crossfit.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dep_oracle as D  # noqa: E402
import e2_confirm as E  # noqa: E402

BUDGETS = E.BUDGETS
S3 = [256, 1024, 1536]
BM = {"repoeval": "repoeval_api", "cceval": "cceval_python"}


def main():
    folds = json.load(open(os.path.join(HERE, "dev_folds.json"), encoding="utf-8"))["tasks"]
    parts = [E.pooled({ds: os.path.join(HERE, f"kaggle_sweep_{ds}_out", "sweep") for ds in BM}),
             E.pooled({ds: os.path.join(HERE, "modal_out", f"e2hold_{ds}") for ds in BM})]
    ids, em, conf, st, ds = [], {b: {} for b in BUDGETS}, {b: {} for b in BUDGETS}, {}, {}
    for pid, pem, pconf, pst, pds in parts:
        for t in pid:
            ids.append(t)
            st[t], ds[t] = pst[t], pds[t]
            for b in BUDGETS:
                em[b][t], conf[b][t] = pem[b][t], pconf[b][t]
    fold = {t: folds[f"{BM[ds[t]]}|{t}"]["fold"] for t in ids}
    pols = {k: {} for k in ("best_fixed", "learned3", "conf3", "oracle4")}
    for f in (0, 1):
        tr = [t for t in ids if fold[t] == f]
        te = [t for t in ids if fold[t] == 1 - f]
        best = {d: max(BUDGETS, key=lambda b: sum(em[b][t] for t in tr if ds[t] == d)) for d in BM}
        feat = lambda t: st[t] + [conf[b][t] for b in S3]
        ch = E.fit_predict([feat(t) for t in tr], {b: [em[b][t] for t in tr] for b in S3}, [feat(t) for t in te], S3)
        for t, c in zip(te, ch):
            pols["learned3"][t] = c
            pols["best_fixed"][t] = best[ds[t]]
        pols.setdefault("_best_fixed_chosen", {})[str(f)] = best
    for t in ids:
        pols["conf3"][t] = max(S3, key=lambda b: conf[b][t])
        pols["oracle4"][t] = max(BUDGETS, key=lambda b: em[b][t])
    res = {"n": {d: sum(ds[t] == d for t in ids) for d in BM}, "best_fixed_chosen_on_fold": pols.pop("_best_fixed_chosen"),
           "em": {}, "per_fold": {}, "mcnemar": {}}

    def acc(ch, ks):
        return round(100.0 * sum(em[ch[t]][t] for t in ks) / len(ks), 2) if ks else None
    for name, ch in pols.items():
        res["em"][name] = {d: acc(ch, [t for t in ids if ds[t] == d]) for d in BM}
        res["per_fold"][name] = {f: {d: acc(ch, [t for t in ids if ds[t] == d and fold[t] == f]) for d in BM}
                                 for f in (0, 1)}

    def mc(x, y, sel):
        w = sum(em[x[t]][t] > em[y[t]][t] for t in ids if sel(t))
        l_ = sum(em[x[t]][t] < em[y[t]][t] for t in ids if sel(t))
        return [w, l_, round(D.mcnemar(w, l_), 4)]
    for a, b in (("learned3", "best_fixed"), ("conf3", "best_fixed"), ("learned3", "conf3")):
        res["mcnemar"][f"{a} vs {b}"] = {"pooled": mc(pols[a], pols[b], lambda t: True),
                                        **{d: mc(pols[a], pols[b], lambda t, d=d: ds[t] == d) for d in BM}}
    e = res["em"]
    ok = all(e["learned3"][d] >= e["best_fixed"][d] for d in BM) and \
        res["mcnemar"]["learned3 vs best_fixed"]["pooled"][2] < 0.05
    res["verdict"] = "PASS" if ok else "FAIL"
    res["reference_E2_task_level"] = {"learned3": {"repoeval": 43.12, "cceval": 29.83},
                                      "best_fixed": {"repoeval": 41.38, "cceval": 28.86}}
    out = os.path.join(HERE, "results_e10", "e10b_summary.json")
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
