#!/usr/bin/env python3
"""E6: confidence cascade -- keep E2's gain at a lower generation cost (offline, CPU, no new generation).

E2 always generates at 3 budgets (256/1024/1536) and picks one. A cascade generates at B=1024 first and only
escalates (generates at the other budgets) when a gate says the 1024 output is unreliable.

Gates (score on the 1024 output, escalate when score < tau):
    conf     first-line confidence at 1024 (training-free)
    learned  logistic P(EM@1024 = 1 | static features, conf@1024), fitted on the train fold
Escalation:
    par      generate 256 and 1536 together, pick by conf argmax (cost 3) or by E2's learned selector (cost 3)
    seq      1024 -> 1536 -> 256, stop at the first output whose conf >= tau_conf, else conf argmax (cost 2 or 3)
tau is set on the TRAIN fold as the quantile giving escalation rate q; cross-fit dev <-> hold-out, so every test
sample is scored by a model and threshold that never saw it (same folds as e2_fulltest).

Decision (fixed before running): the cascade WORKS if some q chosen on train for mean cost <= 1.6 generations keeps
>= 80% of E2's pooled EM gain over the per-dataset best fixed budget on the full test, on both datasets.

usage: python e6_cascade.py   (reads kaggle_sweep_*_out/sweep and modal_out/e2hold_*)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep as BS  # noqa: E402
import dep_oracle as D  # noqa: E402
import e2_confirm as E2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
S3 = [256, 1024, 1536]
QS = [round(x, 2) for x in np.arange(0, 1.01, 0.05)]
TARGET_COST = 1.6


def logreg(Xtr, ytr, Xte):
    Xtr, Xte = np.array(Xtr, float), np.array(Xte, float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    w, c = BS._fit_logreg((Xtr - mu) / sd, np.array(ytr, float))
    return 1 / (1 + np.exp(-(((Xte - mu) / sd) @ w + c)))


def fold(tr, te):
    """Return, for test ids, per-q choices and costs of every cascade variant."""
    tid, tem, tconf, tst, _ = tr
    eid, eem, econf, est, _ = te
    # E2 learned selector over 3 budgets (fallback option) and its full-cost choice
    e2 = dict(zip(eid, E2.fit_predict([tst[t] + [tconf[b][t] for b in S3] for t in tid],
                                      {b: [tem[b][t] for t in tid] for b in S3},
                                      [est[t] + [econf[b][t] for b in S3] for t in eid], S3)))
    argc = {t: max(S3, key=lambda b: econf[b][t]) for t in eid}
    # gate scores
    g_conf_tr = np.array([tconf[1024][t] for t in tid])
    g_conf_te = {t: econf[1024][t] for t in eid}
    f = lambda st, conf, t: st[t] + [conf[1024][t]]
    p_tr = logreg([f(tst, tconf, t) for t in tid], [tem[1024][t] for t in tid], [f(tst, tconf, t) for t in tid])
    p_te = dict(zip(eid, logreg([f(tst, tconf, t) for t in tid], [tem[1024][t] for t in tid],
                                [f(est, econf, t) for t in eid])))
    out = {}
    for q in QS:
        tau_c = np.quantile(g_conf_tr, q) if q > 0 else -np.inf
        tau_l = np.quantile(p_tr, q) if q > 0 else -np.inf
        if q >= 1:
            tau_c = tau_l = np.inf
        row = {}
        for gname, score, tau in (("conf", g_conf_te, tau_c), ("learned", p_te, tau_l)):
            esc = {t: score[t] < tau for t in eid}
            row[f"{gname}_par_conf"] = ({t: argc[t] if esc[t] else 1024 for t in eid},
                                        {t: 3 if esc[t] else 1 for t in eid})
            row[f"{gname}_par_e2"] = ({t: e2[t] if esc[t] else 1024 for t in eid},
                                      {t: 3 if esc[t] else 1 for t in eid})
        ch, cost = {}, {}
        for t in eid:
            seen = []
            for b in (1024, 1536, 256):
                seen.append(b)
                if econf[b][t] >= tau_c:
                    break
            ch[t] = seen[-1] if econf[seen[-1]][t] >= tau_c else max(seen, key=lambda b: econf[b][t])
            cost[t] = len(seen)
        row["conf_seq"] = (ch, cost)
        out[q] = row
    return out, e2, argc


def main():
    dev = E2.pooled({ds: os.path.join(HERE, f"kaggle_sweep_{ds}_out", "sweep") for ds in E2.DEV_BEST})
    hold = E2.pooled({ds: os.path.join(HERE, "modal_out", f"e2hold_{ds}") for ds in E2.DEV_BEST})
    em, ds_of = {b: {} for b in E2.BUDGETS}, {}
    for part in (dev, hold):
        for b in E2.BUDGETS:
            em[b].update(part[1][b])
        ds_of.update(part[4])
    ids = sorted(ds_of)
    ref = {t: E2.DEV_BEST[ds_of[t]] for t in ids}

    # cross-fit: test on hold with train dev, and vice versa; also keep train-fold curves for choosing q
    res_te, res_tr, e2c, argc = {}, {}, {}, {}
    for tr, te in ((dev, hold), (hold, dev)):
        o, e, a = fold(tr, te)
        o_tr, _, _ = fold(tr, tr)  # in-fold curve, used only to pick q
        e2c.update(e)
        argc.update(a)
        for q in QS:
            for k, (ch, cost) in o[q].items():
                res_te.setdefault(k, {}).setdefault(q, ({}, {}))
                res_te[k][q][0].update(ch)
                res_te[k][q][1].update(cost)
            for k, (ch, cost) in o_tr[q].items():
                res_tr.setdefault((k, id(te)), {})[q] = (ch, cost)

    def summ(ch, cost, sub=None):
        sub = sub or ids
        out = {}
        for d in ("repoeval", "cceval"):
            s = [t for t in sub if ds_of[t] == d]
            w = sum(em[ch[t]][t] > em[ref[t]][t] for t in s)
            l_ = sum(em[ch[t]][t] < em[ref[t]][t] for t in s)
            out[d] = {"em": round(100 * sum(em[ch[t]][t] for t in s) / len(s), 2),
                      "cost": round(sum(cost[t] for t in s) / len(s), 3), "vs_ref": [w, l_, round(D.mcnemar(w, l_), 4)]}
        w = sum(em[ch[t]][t] > em[ref[t]][t] for t in ids)
        l_ = sum(em[ch[t]][t] < em[ref[t]][t] for t in ids)
        out["cost"] = round(sum(cost[t] for t in ids) / len(ids), 3)
        out["pooled_vs_ref"] = [w, l_, round(D.mcnemar(w, l_), 4)]
        return out

    one = {t: 1 for t in ids}
    three = {t: 3 for t in ids}
    refs = {"per_dataset_best_fixed": summ(ref, one), "fixed_1024": summ({t: 1024 for t in ids}, one),
            "E2 learned_256_1024_1536": summ(e2c, three), "E2 conf_256_1024_1536": summ(argc, three)}
    curves = {k: {str(q): summ(*v[q]) for q in QS} for k, v in res_te.items()}

    # q chosen on the train fold: max train EM with train cost <= TARGET_COST; then apply that q on test fold
    chosen = {}
    for k in res_te:
        ch, cost = {}, {}
        for tr, te in ((dev, hold), (hold, dev)):
            tr_ids, te_ids = tr[0], te[0]
            best = None
            for q in QS:
                c_tr, k_tr = res_tr[(k, id(te))][q][1], res_tr[(k, id(te))][q][0]
                mc = sum(c_tr.values()) / len(c_tr)
                e = sum(em[k_tr[t]][t] for t in tr_ids) / len(tr_ids)
                if mc <= TARGET_COST and (best is None or e > best[1]):
                    best = (q, e)
            q = best[0]
            chosen.setdefault(k, []).append(q)
            ch.update({t: res_te[k][q][0][t] for t in te_ids})
            cost.update({t: res_te[k][q][1][t] for t in te_ids})
        chosen[k] = {"q_per_fold": chosen[k], **summ(ch, cost)}

    e2g = {d: refs["E2 learned_256_1024_1536"][d]["em"] - refs["per_dataset_best_fixed"][d]["em"]
           for d in ("repoeval", "cceval")}
    for k, v in chosen.items():
        v["gain_kept"] = {d: round((v[d]["em"] - refs["per_dataset_best_fixed"][d]["em"]) / e2g[d], 3)
                          for d in e2g}
    ok = {k: v["cost"] <= TARGET_COST + 1e-9 and all(g >= 0.8 for g in v["gain_kept"].values())
          for k, v in chosen.items()}
    res = {"refs": refs, "chosen_at_cost<=%.1f" % TARGET_COST: chosen, "decision": ok, "curves": curves}
    out = os.path.join(HERE, "modal_out", "e6_cascade_summary.json")
    json.dump(res, open(out, "w"), indent=1)

    for k, v in refs.items():
        print(f"{k:28s} cost {v['cost']:.2f}  RepoEval {v['repoeval']['em']:5.2f}  CCEval {v['cceval']['em']:5.2f}"
              f"  pooled {v['pooled_vs_ref']}")
    print(f"\ncascades, q chosen on train for cost <= {TARGET_COST}:")
    for k, v in chosen.items():
        print(f"{k:20s} q {v['q_per_fold']} cost {v['cost']:.2f} (RE {v['repoeval']['cost']:.2f} CC "
              f"{v['cceval']['cost']:.2f})  RepoEval {v['repoeval']['em']:5.2f} {v['repoeval']['vs_ref']}  "
              f"CCEval {v['cceval']['em']:5.2f} {v['cceval']['vs_ref']}  pooled {v['pooled_vs_ref']}  "
              f"kept {v['gain_kept']}  {'WORKS' if ok[k] else 'no'}")
    print("\ncurves (test, cross-fit): q -> cost / RepoEval / CCEval")
    for k, c in curves.items():
        print(k, "  ".join(f"{q}:{v['cost']:.2f}/{v['repoeval']['em']:.1f}/{v['cceval']['em']:.1f}"
                           for q, v in c.items() if float(q) in (0, .2, .3, .4, .5, .6, .8, 1)))


if __name__ == "__main__":
    main()
