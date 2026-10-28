#!/usr/bin/env python3
"""E10 (CPU, 0 GPU): repo-disjoint cross-fitting of every knob of E9c / CADRE v2.9 (DEV_SPLIT_PLAN.md, Track A).

Data: E9d Modal generations (modal_out/e9d/b{384,1536,2304}.jsonl.gz; `final` = CADRE v2.8 pipeline at that
cross-file budget, `conf` = first-line mean log-prob). b1536 = CADRE v2.8 (anchor reproduced, HANDOFF 35-37).
Folds: dev_folds.json (dev_split.py, repo-disjoint, written once). EM = CADRE official per-sample EM; RepoEval
repos all have 200 tasks, so the per-repo macro equals the per-sample mean, inside a fold too.

Cross-fit of a menu M of policies: choose m* in M maximising the mean official EM over the 4 benchmarks on fold
f, score m* on fold 1-f, union the two halves (every task scored by a choice that never saw it). "Test-tuned" =
m* chosen on all tasks (what E9c / v2.9 report today). One policy for all benchmarks, as in E2.
Menus:
    M_set   fixed 384 / 1536 / 2304 and conf-argmax over each budget subset of size >= 2 (7 policies; E9c = conf3)
    M_v29   conf + beta * grounding, budget 384 dropped when L <= t (L = left-context tokens);
            t in {never, 256..2048, always} x beta in {0..0.5}. v2.9 as published: t = 3072-1536-64 = 1472,
            beta = 0.15. grounding = share of identifiers in the candidate's first line that occur in left context
            or the RLCoder top-10 chunks (our re-implementation: v2.9 code is not pushed).
    learned per-budget logistic (budget_sweep._fit_logreg) on conf x3, grounding x3, log L, pairwise agreement
            of first lines; trained on fold f, applied to fold 1-f (E2's recipe, but with repo folds).
Transfer check: knobs chosen on CCEval (py+java) applied to RepoEval, and the reverse.

Pre-registered (written before the full run):
    H1 PRIMARY (E9c holds without test tuning): on the pooled primary sets (RepoEval-API + CCEval-py, mean of the
       two official EMs) the cross-fitted M_set gain over b1536 >= 80% of the test-tuned gain, AND pooled
       McNemar (per-sample official EM, cross-fit vs b1536, both sets) p < 0.01.  FAIL -> stop and report.
    H2 (v2.9 knobs): descriptive. Report chosen (t, beta) per fold, cross-fit EM, and cross-fit M_v29 vs
       cross-fit M_set (McNemar). No decision rides on it.
    H3 (learned policy under a repo split): learned > cross-fit conf3 on BOTH primary sets AND pooled McNemar
       p < 0.05. FAIL -> the paper's selector is plain conf argmax; the "learned" variant is not claimed.
usage: python e10_crossfit.py [--limit N]   (smoke: first N tasks, writes e10_smoke.json)
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "data_in", "cadre_pkg"))
import budget_sweep as BS  # noqa: E402
import cadre_v28_unified as C  # noqa: E402
import dep_oracle as D  # noqa: E402

B = [384, 1536, 2304]
BMS = ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]
PRIMARY = ["repoeval_api", "cceval_python"]
T_GRID = [-1, 256, 512, 768, 1024, 1280, 1472, 1792, 2048, 10 ** 9]  # -1 = never drop 384, 1e9 = always
BETA = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CACHE = os.path.join(HERE, "results_e10", "e10_cache.json.gz")


def first_line(s):
    return next((l.strip() for l in s.split("\n") if l.strip()), "")


def build_cache(keys, R):
    """official per-sample EM per budget, left-context tokens, grounding per budget (CPU, ~minutes)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", local_files_only=True)
    tasks = {(r["benchmark"], r["task_id"]): r for r in
             map(json.loads, gzip.open(os.path.join(HERE, "data_in", "e9data", "tasks.jsonl.gz"), "rt", encoding="utf-8"))}
    trace = {(r["benchmark"], r["task_id"]): r for r in
             map(json.loads, gzip.open(os.path.join(HERE, "data_in", "e9data",
                                                    "rlcoder_full_8004_retrieval_trace.jsonl.gz"), "rt", encoding="utf-8"))}
    cache = {}
    for i, k in enumerate(keys):
        t = tasks[k]
        ctx = set(ID.findall(t["left_context"]))
        for c in trace[k]["rlcoder_top10"]:
            ctx |= set(ID.findall(c["code_content"]))
        row = {"L": len(tok(t["left_context"], add_special_tokens=False)["input_ids"]), "em": {}, "g": {}, "fl": {}}
        for b in B:
            r = R[b][k]
            row["em"][b] = C.evaluate_records_official([dict(r, pred=r["final"])], "pred")[k[0]]["em"] > 0
            ids = ID.findall(first_line(r["final"]))
            row["g"][b] = sum(x in ctx for x in ids) / len(ids) if ids else 0.0
            row["fl"][b] = first_line(r["final"])
        cache["|".join(k)] = row
        if i % 1000 == 0:
            print("cache", i, len(keys), flush=True)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    R = {b: {(r["benchmark"], r["task_id"]): r for r in
             map(json.loads, gzip.open(os.path.join(HERE, "modal_out", "e9d", f"b{b}.jsonl.gz"), "rt", encoding="utf-8"))}
         for b in B}
    folds = json.load(open(os.path.join(HERE, "dev_folds.json"), encoding="utf-8"))["tasks"]
    keys = sorted(set.intersection(*[set(R[b]) for b in B]))
    if a.limit:
        keys = keys[:a.limit] + [k for k in keys if k[0] == "repoeval_api"][:a.limit]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if not a.limit and os.path.exists(CACHE):
        cache = json.load(gzip.open(CACHE, "rt", encoding="utf-8"))
    else:
        cache = build_cache(keys, R)
        if not a.limit:
            json.dump(cache, gzip.open(CACHE, "wt", encoding="utf-8"))
    X = {k: cache["|".join(k)] for k in keys}
    for k in keys:  # json turns int keys into str
        for f in ("em", "g", "fl"):
            X[k][f] = {int(b): v for b, v in X[k][f].items()}
    conf = {k: {b: float(R[b][k]["conf"]) for b in B} for k in keys}
    fold = {k: folds["|".join(k)]["fold"] for k in keys}

    def em_of(choice, ks):  # mean official EM per benchmark over ks
        out = {}
        for bm in BMS:
            kk = [k for k in ks if k[0] == bm]
            if kk:
                out[bm] = 100.0 * sum(X[k]["em"][choice(k)] for k in kk) / len(kk)
        return out

    def obj(choice, ks):
        e = em_of(choice, ks)
        return sum(e.values()) / len(e)

    # ---- policy menus: name -> choice(k) -> budget
    M_set = {f"fixed{b}": (lambda b: lambda k: b)(b) for b in B}
    for S in [s for n in (2, 3) for s in combinations(B, n)]:
        M_set["conf" + "_".join(map(str, S))] = (lambda S: lambda k: max(S, key=lambda b: conf[k][b]))(S)

    def v29(t, beta):
        def ch(k):
            allowed = [b for b in B if not (b == 384 and X[k]["L"] <= t)]
            return max(allowed, key=lambda b: conf[k][b] + beta * X[k]["g"][b])
        return ch
    M_v29 = {f"t{t}_beta{beta}": v29(t, beta) for t in T_GRID for beta in BETA}

    def crossfit(M, fit_sets=None):
        """returns (choice over all keys, chosen name per fold, test-tuned name)."""
        chosen, choice = {}, {}
        for f in (0, 1):
            fit = [k for k in keys if fold[k] == f]
            best = max(M, key=lambda m: (round(obj(M[m], fit), 6), m == "conf384_1536_2304"))
            chosen[f] = best
            for k in keys:
                if fold[k] == 1 - f:
                    choice[k] = M[best](k)
        tt = max(M, key=lambda m: (round(obj(M[m], keys), 6), m == "conf384_1536_2304"))
        return (lambda k: choice[k]), chosen, tt

    def transfer(M):
        out = {}
        for src, dst in (("cceval", "repoeval"), ("repoeval", "cceval")):
            fit = [k for k in keys if k[0].startswith(src)]
            best = max(M, key=lambda m: round(obj(M[m], fit), 6))
            out[f"{src}->{dst}"] = {"chosen": best, "em": {bm: round(v, 2) for bm, v in em_of(
                M[best], [k for k in keys if k[0].startswith(dst)]).items()}}
        return out

    # ---- learned: per-budget logistic, fit on fold f, apply on 1-f
    import numpy as np

    def feat(k):
        fl = X[k]["fl"]
        return [conf[k][b] for b in B] + [X[k]["g"][b] for b in B] + [math.log1p(X[k]["L"])] + \
            [float(fl[x] == fl[y]) for x, y in combinations(B, 2)]
    F = {k: feat(k) for k in keys}
    lchoice = {}
    for f in (0, 1):
        tr = [k for k in keys if fold[k] == f]
        te = [k for k in keys if fold[k] == 1 - f]
        Xtr, Xte = np.array([F[k] for k in tr]), np.array([F[k] for k in te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        P = {}
        for b in B:
            w, c = BS._fit_logreg(Xtr, np.array([float(X[k]["em"][b]) for k in tr]))
            P[b] = 1 / (1 + np.exp(-(Xte @ w + c)))
        for i, k in enumerate(te):
            lchoice[k] = max(B, key=lambda b: (round(float(P[b][i]), 6), b == 1536))

    # ---- assemble
    base = M_set["fixed1536"]
    cf_set, ch_set, tt_set = crossfit(M_set)
    cf_v29, ch_v29, tt_v29 = crossfit(M_v29)
    systems = {"b1536 (CADRE v2.8)": base, "test-tuned M_set": M_set[tt_set], "crossfit M_set": cf_set,
               "conf3 (E9c)": M_set["conf384_1536_2304"], "v2.9 published (t1472, b0.15)": v29(1472, 0.15),
               "v2.9 alt (t704, b0.15)": v29(704, 0.15), "test-tuned M_v29": M_v29[tt_v29],
               "crossfit M_v29": cf_v29, "learned (repo cross-fit)": (lambda k: lchoice[k]),
               "oracle3": (lambda k: max(B, key=lambda b: (X[k]["em"][b], b == 1536)))}
    res = {"n": len(keys), "folds_sha": hashlib.sha256(open(os.path.join(HERE, "dev_folds.json"), "rb").read()).hexdigest()[:8],
           "em": {}, "per_fold": {}, "chosen": {"M_set": ch_set, "M_set_test_tuned": tt_set,
                                                "M_v29": ch_v29, "M_v29_test_tuned": tt_v29},
           "transfer": {"M_set": transfer(M_set), "M_v29": transfer(M_v29)}, "mcnemar": {}}
    for s, ch in systems.items():
        e = em_of(ch, keys)
        res["em"][s] = {bm: round(v, 2) for bm, v in e.items()} | {
            "primary_mean": round(sum(e[b] for b in PRIMARY if b in e) / len([b for b in PRIMARY if b in e]), 2)}
        res["per_fold"][s] = {f: {bm: round(v, 2) for bm, v in em_of(ch, [k for k in keys if fold[k] == f]).items()}
                              for f in (0, 1)}

    def mc(x, y, sets):
        w = sum(X[k]["em"][x(k)] and not X[k]["em"][y(k)] for k in keys if k[0] in sets)
        l_ = sum(X[k]["em"][y(k)] and not X[k]["em"][x(k)] for k in keys if k[0] in sets)
        return [w, l_, round(D.mcnemar(w, l_), 4)]
    res["mcnemar"]["crossfit M_set vs b1536, primary"] = mc(cf_set, base, PRIMARY)
    res["mcnemar"]["crossfit M_v29 vs crossfit M_set, primary"] = mc(cf_v29, cf_set, PRIMARY)
    res["mcnemar"]["learned vs conf3, primary"] = mc(systems["learned (repo cross-fit)"], M_set["conf384_1536_2304"], PRIMARY)
    for bm in PRIMARY:
        res["mcnemar"][f"learned vs conf3, {bm}"] = mc(systems["learned (repo cross-fit)"], M_set["conf384_1536_2304"], [bm])
    E = res["em"]
    gain_tt = E["test-tuned M_set"]["primary_mean"] - E["b1536 (CADRE v2.8)"]["primary_mean"]
    gain_cf = E["crossfit M_set"]["primary_mean"] - E["b1536 (CADRE v2.8)"]["primary_mean"]
    res["H1"] = {"gain_test_tuned": round(gain_tt, 2), "gain_crossfit": round(gain_cf, 2),
                 "kept": round(gain_cf / gain_tt, 3) if gain_tt > 0 else None,
                 "verdict": "PASS" if gain_tt > 0 and gain_cf >= 0.8 * gain_tt and
                 res["mcnemar"]["crossfit M_set vs b1536, primary"][2] < 0.01 else "FAIL"}
    lc = all(E["learned (repo cross-fit)"][bm] > E["conf3 (E9c)"][bm] for bm in PRIMARY)
    res["H3"] = "PASS" if lc and res["mcnemar"]["learned vs conf3, primary"][2] < 0.05 else "FAIL"
    path = os.path.join(HERE, "results_e10", "e10_smoke.json" if a.limit else "e10_summary.json")
    json.dump(res, open(path, "w", encoding="utf-8"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
