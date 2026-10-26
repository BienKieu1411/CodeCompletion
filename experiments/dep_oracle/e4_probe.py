#!/usr/bin/env python3
"""E4: predict the per-sample budget from ONE prefill of the frozen generator (no extra generation).

E2 picks the budget from the generator's confidence after generating at 3 budgets (3x cost). E4 asks whether the
generator's hidden state, read once while it encodes the B=1024 prompt, already says which budget will work.
Same AlignCoder candidates, same prompts as budget_sweep.py; the outcome of each budget comes from the cached sweeps
(dev: kaggle_sweep_*_out, hold-out: modal_out/e2hold_*), so only the prefill is new GPU work.

Features per sample (all GT-free):
    hidden    last-token hidden state at layers 8 / 16 / 24 of the B=1024 prompt (3 x 2048), PCA-reduced
    nexttok   entropy and top-1 probability of the next-token distribution at that position
    static    the 9 budget_sweep.static_features
Probe: per-budget L2 logistic regression (as E2), budgets 256/512/1024/1536, argmax P(correct). Cross-fit dev <-> hold-out
as e2_confirm / e2 full test. PCA size and L2 chosen by grouped inner CV on the training side only.

Decision (fixed before running), full test, one pooled policy:
    PASS if probe policy (1 prefill + 1 generation) beats the per-dataset best fixed budget on BOTH datasets with pooled
    McNemar p < 0.05 AND beats learned_static (static features only).
    Also reported: share of the E2 3-generation gain it keeps.

usage:
    python e4_probe.py extract --dataset repoeval --cands prediction_with_candidates.jsonl --out e4   (GPU)
    python e4_probe.py analyze                                                                      (CPU)
"""
import argparse
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep as BS  # noqa: E402
import dep_oracle as D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = [8, 16, 24]
PROBE_BUDGET = 1024
BUDGETS = [256, 512, 1024, 1536]
DEV_BEST = {"repoeval": 512, "cceval": 1536}


def cmd_extract(a):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    cands = D.load_cands(a.cands)
    ids = sorted(cands)
    exs = D.load_examples(a.data_dir, a.subset, set(ids))
    ids = [t for t in ids if t in exs]
    if a.limit:
        ids = ids[:a.limit]
    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    seqs = []
    for t in ids:
        prompt, _ = D.build_prompt(tok, exs[t], cands[t]["candidates"], max_cross=PROBE_BUDGET)
        seqs.append(tok.encode(prompt)[-D.MAX_CTX:])
    order = sorted(range(len(ids)), key=lambda i: len(seqs[i]))
    H = np.zeros((len(ids), len(LAYERS), model.config.hidden_size), dtype=np.float16)
    NT = np.zeros((len(ids), 2), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(order), a.batch_size):
            idx = order[s:s + a.batch_size]
            L = max(len(seqs[i]) for i in idx)
            inp = torch.tensor([seqs[i] + [tok.pad_token_id] * (L - len(seqs[i])) for i in idx]).cuda()
            att = torch.tensor([[1] * len(seqs[i]) + [0] * (L - len(seqs[i])) for i in idx]).cuda()
            out = model(input_ids=inp, attention_mask=att, output_hidden_states=True)
            last = att.sum(1) - 1
            ar = torch.arange(len(idx), device=inp.device)
            for j, li in enumerate(LAYERS):
                H[idx, j] = out.hidden_states[li][ar, last].float().cpu().numpy().astype(np.float16)
            p = torch.softmax(out.logits[ar, last].float(), -1)
            ent = -(p * torch.log(p + 1e-12)).sum(-1)
            NT[idx, 0], NT[idx, 1] = ent.cpu().numpy(), p.max(-1).values.cpu().numpy()
            if (s // a.batch_size) % 50 == 0:
                print(f"{s + len(idx)}/{len(ids)}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "hidden.npz"), task_ids=np.array(ids), hidden=H, nexttok=NT)
    print("saved", H.shape, flush=True)


# ----------------------------------------------------------------------------------
# analyze (CPU, numpy)
# ----------------------------------------------------------------------------------

def load_sweep(root):
    feats = {r["task_id"]: r for r in map(json.loads, open(os.path.join(root, "features.jsonl"), encoding="utf-8"))}
    em = {}
    for b in BUDGETS:
        em[b] = {}
        for r in map(json.loads, open(os.path.join(root, f"b{b}", "prediction.jsonl"), encoding="utf-8")):
            em[b][r["task_id"]] = D.simple_score(r["pred"], feats[r["task_id"]]["gt"])[0]
    fn = sorted(next(iter(feats.values()))["features"])
    ids = sorted(set.intersection(*[set(em[b]) for b in BUDGETS]))
    return ids, em, {t: [feats[t]["features"][k] for k in fn] for t in ids}


def group_of(t):
    return t.split("/")[0] if not t.startswith("project_cc") else f"cc{zlib.crc32(t.encode()) % 50}"


def fit_policy(Xtr, Ytr, Xte, pca_k, l2, np):
    """Standardize, PCA on the hidden block (first `pca_k` comps), per-budget logistic, argmax."""
    mu, sd = Xtr["h"].mean(0), Xtr["h"].std(0) + 1e-4
    Htr, Hte = (Xtr["h"] - mu) / sd, (Xte["h"] - mu) / sd
    blocks_tr, blocks_te = [Xtr["s"]], [Xte["s"]]
    if pca_k:
        _, _, Vt = np.linalg.svd(Htr - Htr.mean(0), full_matrices=False)
        P = Vt[:pca_k].T
        blocks_tr.append(Htr @ P)
        blocks_te.append(Hte @ P)
    A, B = np.hstack(blocks_tr), np.hstack(blocks_te)
    m, s = A.mean(0), A.std(0) + 1e-6
    A, B = (A - m) / s, (B - m) / s
    probs = {}
    for b in BUDGETS:
        w, c = BS._fit_logreg(A, Ytr[b].astype(float), l2=l2, iters=400)
        probs[b] = 1 / (1 + np.exp(-(B @ w + c)))
    return [max(BUDGETS, key=lambda b: (round(float(probs[b][k]), 6), b == 1536)) for k in range(len(B))]


def cmd_analyze(a):
    import numpy as np
    parts = {}
    for name, sub in (("dev", "kaggle_sweep_{}_out/sweep"), ("hold", "modal_out/e2hold_{}")):
        ids, em, st, ds = [], {b: {} for b in BUDGETS}, {}, {}
        for d in DEV_BEST:
            i, e, s = load_sweep(os.path.join(HERE, sub.format(d)))
            ids += i
            st.update(s)
            ds.update({t: d for t in i})
            for b in BUDGETS:
                em[b].update(e[b])
        parts[name] = (ids, em, st, ds)
    hid, nt = {}, {}
    for d in DEV_BEST:
        z = np.load(os.path.join(HERE, "modal_out", f"e4_{d}", "hidden.npz"))
        for t, h, n in zip(z["task_ids"], z["hidden"], z["nexttok"]):
            hid[str(t)] = h.astype(np.float32).reshape(-1)
            nt[str(t)] = n
    for name in parts:  # keep samples with hidden states
        ids, em, st, ds = parts[name]
        parts[name] = ([t for t in ids if t in hid], em, st, ds)

    def mats(ids, st, use_hidden=True, use_nt=True):
        s = np.array([st[t] + (list(nt[t]) if use_nt else []) for t in ids], float)
        h = np.array([hid[t] for t in ids], float) if use_hidden else np.zeros((len(ids), 1))
        return {"s": s, "h": h}

    def cross_fit(variant):
        out = {}
        for tr, te in (("dev", "hold"), ("hold", "dev")):
            ti, tem, tst, _ = parts[tr]
            ei, _, est, _ = parts[te]
            uh, un = variant != "static", variant in ("probe", "nexttok")
            Xtr, Xte = mats(ti, tst, uh and variant == "probe", un), mats(ei, est, uh and variant == "probe", un)
            Ytr = {b: np.array([tem[b][t] for t in ti]) for b in BUDGETS}
            # inner grouped CV on the training side picks (pca_k, l2)
            grid = [(k, l2) for k in ((16, 64, 128) if variant == "probe" else (0,)) for l2 in (1.0, 10.0, 100.0)]
            groups = sorted({group_of(t) for t in ti})
            fold = {g: zlib.crc32(g.encode()) % 4 for g in groups}
            best, best_em = None, -1
            for k, l2 in grid:
                tot = 0
                for f in range(4):
                    a_ = [i for i, t in enumerate(ti) if fold[group_of(t)] != f]
                    b_ = [i for i, t in enumerate(ti) if fold[group_of(t)] == f]
                    sub = lambda X, ix: {"s": X["s"][ix], "h": X["h"][ix]}
                    ch = fit_policy(sub(Xtr, a_), {b: Ytr[b][a_] for b in BUDGETS}, sub(Xtr, b_), k, l2, np)
                    tot += sum(Ytr[c][i] for c, i in zip(ch, b_))
                if tot > best_em:
                    best, best_em = (k, l2), tot
            ch = fit_policy(Xtr, Ytr, Xte, best[0], best[1], np)
            out.update(dict(zip(ei, ch)))
            print(f"  {variant} fit on {tr}: picked pca={best[0]} l2={best[1]}", flush=True)
        return out

    ids = parts["dev"][0] + parts["hold"][0]
    em = {b: {**parts["dev"][1][b], **parts["hold"][1][b]} for b in BUDGETS}
    ds = {**parts["dev"][3], **parts["hold"][3]}
    pols = {"per_dataset_best_fixed": {t: DEV_BEST[ds[t]] for t in ids},
            "fixed_1024": {t: 1024 for t in ids},
            "learned_static (1 gen)": cross_fit("static"),
            "static+nexttok (1 prefill + 1 gen)": cross_fit("nexttok"),
            "PROBE hidden+static+nexttok (1 prefill + 1 gen)": cross_fit("probe"),
            "oracle_4": {t: max(BUDGETS, key=lambda b: em[b][t]) for t in ids}}
    ref = pols["per_dataset_best_fixed"]
    res = {"n": {d: sum(ds[t] == d for t in ids) for d in DEV_BEST}, "policies": {}}
    for n, ch in pols.items():
        row = {}
        for d in DEV_BEST:
            sub = [t for t in ids if ds[t] == d]
            w = sum(em[ch[t]][t] > em[ref[t]][t] for t in sub)
            l_ = sum(em[ch[t]][t] < em[ref[t]][t] for t in sub)
            row[d] = {"em": round(100 * sum(em[ch[t]][t] for t in sub) / len(sub), 2),
                      "vs_best_fixed": [w, l_, round(D.mcnemar(w, l_), 4)],
                      "budget_mix": {b: sum(ch[t] == b for t in sub) for b in BUDGETS}}
        w = sum(em[ch[t]][t] > em[ref[t]][t] for t in ids)
        l_ = sum(em[ch[t]][t] < em[ref[t]][t] for t in ids)
        row["pooled_vs_best_fixed"] = [w, l_, round(D.mcnemar(w, l_), 4)]
        res["policies"][n] = row
        print(f"{n:48s} RepoEval {row['repoeval']['em']:5.2f} {row['repoeval']['vs_best_fixed']}  "
              f"CCEval {row['cceval']['em']:5.2f} {row['cceval']['vs_best_fixed']}  pooled {row['pooled_vs_best_fixed']}")
    p = res["policies"]["PROBE hidden+static+nexttok (1 prefill + 1 gen)"]
    st_ = res["policies"]["learned_static (1 gen)"]
    r = res["policies"]["per_dataset_best_fixed"]
    beats_fixed = all(p[d]["em"] >= r[d]["em"] for d in DEV_BEST) and p["pooled_vs_best_fixed"][2] < 0.05 \
        and p["pooled_vs_best_fixed"][0] > p["pooled_vs_best_fixed"][1]
    beats_static = sum(p[d]["em"] for d in DEV_BEST) > sum(st_[d]["em"] for d in DEV_BEST)
    e2 = {"repoeval": 43.12, "cceval": 29.83}  # E2 learned_256_1024_1536, full test (e2_fulltest_summary.json)
    res["share_of_e2_gain"] = {d: round((p[d]["em"] - r[d]["em"]) / (e2[d] - r[d]["em"]), 2) for d in DEV_BEST}
    res["decision"] = "PASS" if beats_fixed and beats_static else "FAIL"
    json.dump(res, open(os.path.join(HERE, "modal_out", "e4_summary.json"), "w"), indent=1)
    print("share of E2 (3-gen) gain kept:", res["share_of_e2_gain"])
    print("DECISION:", res["decision"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="e4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=8)
    a = ap.parse_args()
    D.DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    {"extract": cmd_extract, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
