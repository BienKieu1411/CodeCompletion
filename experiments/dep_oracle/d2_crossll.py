#!/usr/bin/env python3
"""D2: cross-context likelihood as a prefix-only selector over the per-budget greedy completions (frozen generator).

Question: E2 picks the budget whose completion is most confident under ITS OWN prompt. Is a completion that the
other prompts also find likely (cross-context agreement) more often correct? If yes, selection generalises from
"budget choice" to "agreement across context variants", and E2 becomes a special case.

Data: full test = dev sweep (Kaggle T4) + hold-out sweep (Modal L4), budgets 256/512/1024/1536, no new generation.
    score    GPU. For every sample, the first-line segment of each budget's saved greedy completion is scored
             (teacher-forced log-probs) under each of the 4 budget prompts (KV cache of the prompt reused).
    analyze  CPU. m(a, b) = mean log-prob over the first non-empty line of candidate a under prompt b
             (same definition as budget_sweep.first_line_conf). Policies on S3 = 256/1024/1536 (E2's set):
               E2 learned3   static + own-prompt conf at S3, logistic per budget          (reference)
               conf3         argmax own-prompt conf                                        (E2 without learning)
               xmean3/xmin3  argmax mean / min over the 3 prompts of m(a, .)
               agree3        argmax (#budgets in S3 giving the same first line, conf)
               PRIMARY learned3x: E2 learned3 features + per slot a: xmean, xmin, agree
               learned4x     same on all 4 budgets
             Learned policies cross-fitted dev <-> hold-out (as b1b_abstention.py): every score is out-of-sample.

Decision (fixed before running): PASS if PRIMARY >= E2 learned3 + 1 EM on BOTH datasets and pooled McNemar vs
E2 learned3 has more wins with p < 0.05. Otherwise selection stays at E2 (H2 closed).
Sanity: m(a, a) must reproduce the saved first-line conf (re-tokenisation of the decoded text).

Cost note: in deployment the 3 prompt prefills already exist (one per generation); cross-scoring only adds
short teacher-forced continuations on the cached prompts.

usage:
    python d2_crossll.py score --dataset repoeval --cands prediction_with_candidates.jsonl --out modal_out/d2_repoeval
    python d2_crossll.py score --dataset cceval --cands cceval_python_prediction_with_candidates.jsonl --out modal_out/d2_cceval
    python d2_crossll.py analyze
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep as BS  # noqa: E402
import dep_oracle as D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGETS = [256, 512, 1024, 1536]
S3 = [256, 1024, 1536]
DEV_BEST = {"repoeval": 512, "cceval": 1536}
MAX_SEG = 48


def first_seg(pred):
    """Prefix of the completion up to and including the newline that ends its first non-empty line ('' if none).
    The newline is kept because budget_sweep.first_line_conf (E2's signal) counts it."""
    pos = 0
    for line in pred.split("\n"):
        end = pos + len(line)
        if line.strip():
            return pred[:end + 1]
        pos = end + 1
    return ""


def sweep_roots(ds):
    return [os.path.join(HERE, f"kaggle_sweep_{ds}_out", "sweep"), os.path.join(HERE, "modal_out", f"e2hold_{ds}")]


def load_preds(ds):
    preds = {b: {} for b in BUDGETS}
    for root in sweep_roots(ds):
        for b in BUDGETS:
            for r in map(json.loads, open(os.path.join(root, f"b{b}", "prediction.jsonl"), encoding="utf-8")):
                preds[b][r["task_id"]] = r["pred"]
    return preds

# ----------------------------------------------------------------------------------
# score (GPU)
# ----------------------------------------------------------------------------------

def cmd_score(a):
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    preds = load_preds(a.dataset)
    ids = sorted(set.intersection(*[set(preds[b]) for b in BUDGETS]))
    if a.limit:
        ids = ids[:a.limit]
    cands = D.load_cands(a.cands)
    exs = D.load_examples(a.data_dir, a.subset, set(ids))
    ids = [t for t in ids if t in exs]
    tok = D.get_tokenizer(a.generator)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "scores.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["task_id"] for l in open(path, encoding="utf-8")}
    todo = [t for t in ids if t not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fout:
        for n, t in enumerate(todo):
            segs, src = [], {}
            for b in BUDGETS:
                s = first_seg(preds[b][t])
                if not s.strip():
                    src[b] = None
                    continue
                if s not in segs:
                    segs.append(s)
                src[b] = segs.index(s)
            seg_ids = [tok.encode(s, add_special_tokens=False)[:MAX_SEG] for s in segs]
            row = {"task_id": t, "segs": segs, "src": {str(b): src[b] for b in BUDGETS},
                   "pieces": [[tok.decode([i]) for i in x] for x in seg_ids], "lp": {}}
            if segs:
                k, Lc = len(segs), max(map(len, seg_ids))
                cand = torch.tensor([x + [tok.eos_token_id] * (Lc - len(x)) for x in seg_ids]).cuda()
                cmask = torch.tensor([[1] * len(x) + [0] * (Lc - len(x)) for x in seg_ids]).cuda()
                for b in BUDGETS:
                    prompt, _ = D.build_prompt(tok, exs[t], cands[t]["candidates"], max_cross=b)
                    p_ids = tok.encode(prompt)[-D.MAX_CTX:]
                    P = len(p_ids)
                    with torch.no_grad():
                        o = model(input_ids=torch.tensor([p_ids]).cuda(), use_cache=True)
                        leg = o.past_key_values
                        leg = leg.to_legacy_cache() if hasattr(leg, "to_legacy_cache") else leg
                        past = DynamicCache.from_legacy_cache(
                            tuple((kk.expand(k, -1, -1, -1), vv.expand(k, -1, -1, -1)) for kk, vv in leg))
                        att = torch.cat([torch.ones(k, P, dtype=cmask.dtype, device=cmask.device), cmask], 1)
                        pos = torch.arange(P, P + Lc, device=cand.device).unsqueeze(0).expand(k, -1)
                        o2 = model(input_ids=cand, attention_mask=att, position_ids=pos, past_key_values=past)
                        logits = torch.cat([o.logits[:, -1:].expand(k, -1, -1), o2.logits[:, :-1]], 1).float()
                        lps = torch.log_softmax(logits, -1).gather(-1, cand.unsqueeze(-1)).squeeze(-1)
                    row["lp"][str(b)] = [[round(v, 4) for v in lps[j, :len(seg_ids[j])].tolist()] for j in range(k)]
            fout.write(json.dumps(row) + "\n")
            if n % 50 == 0:
                fout.flush()
                print(f"{n + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------------
# analyze (CPU)
# ----------------------------------------------------------------------------------

def load_scores(ds):
    rows = {}
    path = os.path.join(HERE, "modal_out", f"d2_{ds}", "scores.jsonl")
    if not os.path.exists(path):
        print(f"[warn] no scores for {ds}", flush=True)
        return rows
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # partially written last line while scoring is still running
            continue
        if len(r["lp"]) == len(BUDGETS) or not r["segs"]:
            rows[r["task_id"]] = r
    return rows


def cross_matrix(r, S):
    """m[a][b]: first-line mean log-prob of budget a's candidate under budget b's prompt (-10 if empty)."""
    m = {}
    for a in S:
        j = r["src"][str(a)]
        m[a] = {b: (BS.first_line_conf(r["pieces"][j], r["lp"][str(b)][j]) if j is not None else -10.0) for b in S}
    return m


def agree(r, S, a):
    j = r["src"][str(a)]
    if j is None:
        return 0
    key = r["segs"][j].strip()
    return sum(1 for b in S if r["src"][str(b)] is not None and r["segs"][r["src"][str(b)]].strip() == key)


def xfeats(r, S):
    m = cross_matrix(r, S)
    out = []
    for a in S:
        vals = [m[a][b] for b in S]
        out += [sum(vals) / len(vals), min(vals), float(agree(r, S, a))]
    return out


def argmax_pick(S, score):
    return max(S, key=lambda b: (round(score(b), 6), b == 1536))


def crossfit(tr, te, S, feat):
    """Logistic per budget on feat(t); fitted on tr ids, applied to te ids. tr/te: (ids, em)."""
    import numpy as np
    (tid, tem), (eid, _) = tr, te
    Xtr = np.array([feat(t) for t in tid], float)
    Xte = np.array([feat(t) for t in eid], float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    probs = {}
    for b in S:
        w, c = BS._fit_logreg(Xtr, np.array([tem[b][t] for t in tid], float))
        probs[b] = 1 / (1 + np.exp(-(Xte @ w + c)))
    return {t: argmax_pick(S, lambda b: float(probs[b][k])) for k, t in enumerate(eid)}


def cmd_analyze(a):
    import e2_confirm as E2
    dev = E2.pooled({ds: sweep_roots(ds)[0] for ds in DEV_BEST})
    hold = E2.pooled({ds: sweep_roots(ds)[1] for ds in DEV_BEST})
    ids = dev[0] + hold[0]
    em = {b: {**dev[1][b], **hold[1][b]} for b in BUDGETS}
    conf = {b: {**dev[2][b], **hold[2][b]} for b in BUDGETS}
    st = {**dev[3], **hold[3]}
    dsof = {**dev[4], **hold[4]}
    sc = {}
    for ds in DEV_BEST:
        sc.update(load_scores(ds))
    ids = [t for t in ids if t in sc]
    if a.limit:
        ids = ids[:a.limit]
    dev_ids = [t for t in dev[0] if t in set(ids)]
    hold_ids = [t for t in hold[0] if t in set(ids)]

    # sanity: own-prompt score re-computed from re-tokenised text vs saved first-line conf
    diffs = []
    for t in ids:
        m = cross_matrix(sc[t], BUDGETS)
        for b in BUDGETS:
            if sc[t]["src"][str(b)] is not None:
                diffs.append(abs(m[b][b] - conf[b][t]))
    diffs.sort()
    sanity = {"n": len(diffs), "median_abs_diff": round(diffs[len(diffs) // 2], 4),
              "pct_lt_0.05": round(100 * sum(d < 0.05 for d in diffs) / len(diffs), 1)}

    def learned(S, use_x):
        def feat(t):
            f = st[t] + [conf[b][t] for b in S]
            return f + (xfeats(sc[t], S) if use_x else [])
        return {**crossfit((dev_ids, em), (hold_ids, em), S, feat),
                **crossfit((hold_ids, em), (dev_ids, em), S, feat)}

    pols = {"fixed_best(dev)": {t: DEV_BEST[dsof[t]] for t in ids},
            "E2 learned3 (ref)": learned(S3, False),
            "conf3": {t: argmax_pick(S3, lambda b: conf[b][t]) for t in ids}}
    for S, tag in ((S3, "3"), (BUDGETS, "4")):
        mats = {t: cross_matrix(sc[t], S) for t in ids}
        pols[f"xmean{tag}"] = {t: argmax_pick(S, lambda b: sum(mats[t][b].values()) / len(S)) for t in ids}
        pols[f"xmin{tag}"] = {t: argmax_pick(S, lambda b: min(mats[t][b].values())) for t in ids}
        pols[f"agree{tag}"] = {t: max(S, key=lambda b: (agree(sc[t], S, b), round(conf[b][t], 6))) for t in ids}
    pols["PRIMARY learned3x"] = learned(S3, True)
    pols["learned4x"] = learned(BUDGETS, True)
    pols["oracle3"] = {t: max(S3, key=lambda b: em[b][t]) for t in ids}
    pols["oracle4"] = {t: max(BUDGETS, key=lambda b: em[b][t]) for t in ids}

    ref = pols["E2 learned3 (ref)"]
    res = {"sanity_own_prompt": sanity, "n": {ds: sum(dsof[t] == ds for t in ids) for ds in DEV_BEST}, "policies": {}}
    for name, ch in pols.items():
        row = {}
        for ds in DEV_BEST:
            sub = [t for t in ids if dsof[t] == ds]
            if not sub:
                row[ds] = {"em": float("nan"), "vs_E2": [0, 0, 1.0]}
                continue
            w = sum(em[ch[t]][t] > em[ref[t]][t] for t in sub)
            l_ = sum(em[ch[t]][t] < em[ref[t]][t] for t in sub)
            row[ds] = {"em": round(100 * sum(em[ch[t]][t] for t in sub) / len(sub), 2),
                       "vs_E2": [w, l_, round(D.mcnemar(w, l_), 4)]}
        w = sum(em[ch[t]][t] > em[ref[t]][t] for t in ids)
        l_ = sum(em[ch[t]][t] < em[ref[t]][t] for t in ids)
        row["pooled_vs_E2"] = [w, l_, round(D.mcnemar(w, l_), 4)]
        res["policies"][name] = row
    p, r = res["policies"]["PRIMARY learned3x"], res["policies"]["E2 learned3 (ref)"]
    ok = all(p[ds]["em"] >= r[ds]["em"] + 1 for ds in DEV_BEST) and p["pooled_vs_E2"][2] < 0.05 \
        and p["pooled_vs_E2"][0] > p["pooled_vs_E2"][1]
    res["decision"] = "PASS" if ok else "FAIL"
    print("sanity (own prompt vs saved conf):", sanity)
    print("n:", res["n"])
    for k, v in res["policies"].items():
        print(f"{k:20s} RepoEval {v['repoeval']['em']:5.2f} {v['repoeval']['vs_E2']}  "
              f"CCEval {v['cceval']['em']:5.2f} {v['cceval']['vs_E2']}  pooled {v['pooled_vs_E2']}")
    print("DECISION:", res["decision"])
    if not a.limit:
        json.dump(res, open(os.path.join(HERE, "modal_out", "d2_crossll_summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--data_dir", default=os.path.join(HERE, "aligncoder_data"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    D.DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    if a.out is None:
        a.out = os.path.join(HERE, "modal_out", f"d2_{a.dataset}")
    {"score": cmd_score, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
