#!/usr/bin/env python3
"""Cross-file budget sweep + "can a cheap signal pick the budget?" (frozen generator, no training).

Same prompt as AlignCoder (saved candidates, construct_prompts), only the cross-file token budget
changes; the in-file left context gets the rest of the 2048-token window.

    sweep    GPU. For each budget in --budgets: greedy completion + per-token log-probs.
             Also writes static (pre-generation) features per sample.
    analyze  CPU. EM/ES per budget, per-sample oracle upper bound, and budget-selection policies:
               fixed_<b>          : always budget b
               conf_firstline     : budget whose completion has the highest mean log-prob on its first
                                    line (needs one generation per budget: k x cost)
               static_learned     : per-budget logistic regression on static features, argmax predicted
                                    P(EM=1); 5-fold CV grouped by repository (1 generation: cheap)
               static+conf_learned: same + the confidence features (k x cost)
             Every policy is compared with the AlignCoder default (1536) by paired McNemar.

usage (Kaggle):
  python budget_sweep.py sweep   --dataset repoeval --cands prediction_with_candidates.jsonl --n 1000 --out out
  python budget_sweep.py analyze --out out
"""
import argparse
import json
import math
import os
import random
import sys
import time
import zlib
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dep_oracle as D  # noqa: E402

BUDGETS_DEFAULT = "0,256,512,1024,1536"
DEFAULT_BUDGET = 1536

# ----------------------------------------------------------------------------------
# static features (computable before generation, no ground truth)
# ----------------------------------------------------------------------------------

def _ids(text):
    return set(D.identifiers(text))


def _jacc(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def _max_window_sim(q, text, win=10, stride=5):
    lines = [l for l in text.split("\n") if l.strip()]
    best = 0.0
    for i in range(0, max(1, len(lines) - win + 1), stride):
        best = max(best, _jacc(q, _ids("\n".join(lines[i:i + win]))))
    return best


def static_features(tok, ex, saved):
    lines = [l for l in ex["left"].split("\n") if l.strip()]
    q = _ids("\n".join(lines[-10:]))
    sel = []
    for b in saved:
        if b["file_path"] == "":
            break
        sel.append(b)
    sims = sorted((_jacc(q, _ids(b["code_content"])) for b in sel), reverse=True)
    _, info = D.build_prompt(tok, ex, saved)  # default budget
    left_tokens = len(tok.encode(ex["left"], add_special_tokens=False))
    visible = info["infile_text"]
    hidden = ex["left"][: max(0, len(ex["left"]) - len(visible))]
    visible_wo_query = "\n".join([l for l in visible.split("\n") if l.strip()][:-10])
    return {
        "log_left_tokens": math.log1p(left_tokens),
        "log_hidden_tokens": math.log1p(max(0, left_tokens - info["infile_tokens"])),
        "n_selected": len(sel),
        "n_api_selected": sum(1 for b in sel if b.get("file_path") == "file_path"),
        "cross_sim_max": sims[0] if sims else 0.0,
        "cross_sim_top3": sum(sims[:3]) / 3.0,
        "infile_hidden_sim_max": _max_window_sim(q, hidden) if hidden else 0.0,
        "infile_visible_sim_max": _max_window_sim(q, visible_wo_query) if visible_wo_query else 0.0,
        "query_n_ids": len(q),
    }

# ----------------------------------------------------------------------------------
# sweep (GPU)
# ----------------------------------------------------------------------------------

def cmd_sweep(a):
    import torch
    from transformers import AutoModelForCausalLM
    budgets = [int(x) for x in a.budgets.split(",")]
    cands = D.load_cands(a.cands)
    ids = D.pick_subset(list(cands), a.n, a.seed)
    if a.holdout:  # E2 confirmation: every test sample NOT in the n-sample development subset
        ids = sorted(set(cands) - set(ids))
    exs = D.load_examples(a.data_dir, a.subset, set(ids))
    ids = [t for t in ids if t in exs]
    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    os.makedirs(a.out, exist_ok=True)

    fpath = os.path.join(a.out, "features.jsonl")
    if not os.path.exists(fpath):
        with open(fpath, "w", encoding="utf-8") as f:
            for t in ids:
                feats = static_features(tok, exs[t], cands[t]["candidates"])
                f.write(json.dumps({"task_id": t, "gt": exs[t]["gt"], "features": feats}) + "\n")
        print(f"features written for {len(ids)} samples", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    eos = tok.eos_token_id
    for b in budgets:
        bdir = os.path.join(a.out, f"b{b}")
        os.makedirs(bdir, exist_ok=True)
        pred_path = os.path.join(bdir, "prediction.jsonl")
        done = set()
        if os.path.exists(pred_path):
            done = {json.loads(l)["task_id"] for l in open(pred_path, encoding="utf-8")}
        todo = [t for t in ids if t not in done]
        print(f"[b{b}] {len(done)} done, {len(todo)} to go", flush=True)
        prompts = []
        for t in todo:
            blocks = cands[t]["candidates"] if b > 0 else []
            prompt, info = D.build_prompt(tok, exs[t], blocks, max_cross=max(b, 0))
            prompts.append((t, tok.encode(prompt)[-D.MAX_CTX:], info["cross_tokens"], info["infile_tokens"]))
        prompts.sort(key=lambda x: len(x[1]))
        t0 = time.time()
        with open(pred_path, "a", encoding="utf-8") as fout:
            for i in range(0, len(prompts), a.batch_size):
                batch = prompts[i:i + a.batch_size]
                L = max(len(x[1]) for x in batch)
                inp = torch.tensor([[tok.pad_token_id] * (L - len(x[1])) + x[1] for x in batch]).cuda()
                att = torch.tensor([[0] * (L - len(x[1])) + [1] * len(x[1]) for x in batch]).cuda()
                with torch.no_grad():
                    out = model.generate(inp, attention_mask=att, max_new_tokens=D.MAX_NEW, do_sample=False,
                                         pad_token_id=tok.pad_token_id, output_scores=True,
                                         return_dict_in_generate=True)
                seqs = out.sequences[:, L:]
                lps = torch.stack([torch.log_softmax(s.float(), dim=-1) for s in out.scores], dim=1)  # B,T,V
                chosen = lps.gather(-1, seqs[:, :lps.size(1)].unsqueeze(-1)).squeeze(-1)  # B,T
                for j, (t, _, ct, it) in enumerate(batch):
                    toks = seqs[j].tolist()
                    lp = chosen[j].tolist()
                    keep = []
                    for k, tid in enumerate(toks[:len(lp)]):
                        if tid == eos:
                            break
                        keep.append(k)
                    fout.write(json.dumps({
                        "task_id": t, "pred": tok.decode(seqs[j], skip_special_tokens=True),
                        "lp": [round(lp[k], 4) for k in keep],
                        "pieces": [tok.decode([toks[k]]) for k in keep],
                        "cross_tokens": ct, "infile_tokens": it}) + "\n")
                fout.flush()
                if (i // a.batch_size) % 20 == 0:
                    print(f"[b{b}] {i + len(batch)}/{len(prompts)} {time.time() - t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------------
# analyze (CPU, numpy only)
# ----------------------------------------------------------------------------------

def first_line_conf(pieces, lps):
    """Mean log-prob over the first non-empty line of the completion."""
    started, vals, text = False, [], ""
    for p, lp in zip(pieces, lps):
        if not started and p.strip() == "":
            continue
        started = True
        before = text
        text += p
        vals.append(lp)
        if "\n" in p and before.strip():
            break
    return sum(vals) / len(vals) if vals else -10.0


def _fit_logreg(X, y, l2=1.0, iters=300, lr=0.5):
    import numpy as np
    w = np.zeros(X.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(iters):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return w, b


def _cv_policy(ids, groups, feats, em, budgets, folds=5, seed=0):
    """Per-budget logistic regression on features -> argmax P(EM). Grouped K-fold. Returns choice per id."""
    import numpy as np
    uniq = sorted(set(groups.values()))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    fold_of_group = {g: i % folds for i, g in enumerate(uniq)}
    X_all = np.array([feats[t] for t in ids], dtype=float)
    choice = {}
    for f in range(folds):
        tr = [i for i, t in enumerate(ids) if fold_of_group[groups[t]] != f]
        te = [i for i, t in enumerate(ids) if fold_of_group[groups[t]] == f]
        if not tr or not te:
            continue
        mu, sd = X_all[tr].mean(0), X_all[tr].std(0) + 1e-6
        Xtr, Xte = (X_all[tr] - mu) / sd, (X_all[te] - mu) / sd
        probs = {}
        for b in budgets:
            y = np.array([em[b][ids[i]] for i in tr], dtype=float)
            w, c = _fit_logreg(Xtr, y)
            probs[b] = 1 / (1 + np.exp(-(Xte @ w + c)))
        for k, i in enumerate(te):
            best = max(budgets, key=lambda b: (round(float(probs[b][k]), 6), b == DEFAULT_BUDGET))
            choice[ids[i]] = best
    return choice


def cmd_analyze(a):
    feats_rows = {}
    for l in open(os.path.join(a.out, "features.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        feats_rows[r["task_id"]] = r
    budgets = sorted(int(d[1:]) for d in os.listdir(a.out) if d.startswith("b") and d[1:].isdigit())
    em, es, conf, conf_all = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for b in budgets:
        for l in open(os.path.join(a.out, f"b{b}", "prediction.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            t = r["task_id"]
            e, s = D.simple_score(r["pred"], feats_rows[t]["gt"])
            em[b][t], es[b][t] = e, s
            conf[b][t] = first_line_conf(r["pieces"], r["lp"])
            conf_all[b][t] = sum(r["lp"]) / len(r["lp"]) if r["lp"] else -10.0
    ids = sorted(set.intersection(*[set(em[b]) for b in budgets]))
    n = len(ids)
    pct = lambda x: round(100.0 * x / n, 2)
    groups = {t: t.split("/")[0] if not t.startswith("project_cc") else str(zlib.crc32(t.encode()) % 50) for t in ids}

    def evaluate(choice):
        e = sum(em[choice[t]][t] for t in ids if t in choice)
        s = sum(es[choice[t]][t] for t in ids if t in choice)
        win = sum(em[choice[t]][t] == 1 and em[DEFAULT_BUDGET][t] == 0 for t in ids)
        loss = sum(em[choice[t]][t] == 0 and em[DEFAULT_BUDGET][t] == 1 for t in ids)
        return {"em": pct(e), "es": round(s / n, 2), "em_gained_vs_1536": win, "em_lost_vs_1536": loss,
                "mcnemar_p": D.mcnemar(win, loss), "budget_mix": dict(Counter(choice[t] for t in ids))}

    res = {"n": n, "budgets": budgets}
    res["fixed"] = {f"b{b}": {"em": pct(sum(em[b].get(t, 0) for t in ids)),
                              "es": round(sum(es[b].get(t, 0) for t in ids) / n, 2)} for b in budgets}
    oracle_choice = {t: max(budgets, key=lambda b: (em[b][t], es[b][t])) for t in ids}
    res["oracle_upper_bound"] = evaluate(oracle_choice)
    res["n_samples_where_some_budget_beats_1536"] = sum(
        any(em[b][t] > em[DEFAULT_BUDGET][t] for b in budgets) for t in ids)
    res["policies"] = {}
    res["policies"]["conf_firstline"] = evaluate({t: max(budgets, key=lambda b: conf[b][t]) for t in ids})
    res["policies"]["conf_allTokens"] = evaluate({t: max(budgets, key=lambda b: conf_all[b][t]) for t in ids})
    fnames = sorted(next(iter(feats_rows.values()))["features"])
    static = {t: [feats_rows[t]["features"][k] for k in fnames] for t in ids}
    res["policies"]["static_learned"] = evaluate(_cv_policy(ids, groups, static, em, budgets))
    both = {t: static[t] + [conf[b][t] for b in budgets] for t in ids}
    res["policies"]["static+conf_learned"] = evaluate(_cv_policy(ids, groups, both, em, budgets))
    # which static features separate "small budget wins" from "1536 wins"?
    small = [b for b in budgets if b < DEFAULT_BUDGET]
    sep = {}
    for k_i, k in enumerate(fnames):
        a_ = [static[t][k_i] for t in ids if any(em[b][t] > em[DEFAULT_BUDGET][t] for b in small)]
        b_ = [static[t][k_i] for t in ids if em[DEFAULT_BUDGET][t] > max(em[b][t] for b in small)]
        if a_ and b_:
            sep[k] = {"mean_when_smaller_budget_wins": round(sum(a_) / len(a_), 4),
                      "mean_when_1536_wins": round(sum(b_) / len(b_), 4), "n": [len(a_), len(b_)]}
    res["feature_separation"] = sep
    res["notes"] = ("simple EM/ES (first k lines, k = #GT lines). Learned policies use grouped 5-fold CV "
                    "(group = repository for RepoEval, pseudo-random groups for CCEval). conf_* and "
                    "static+conf need one generation per budget; static_learned needs one.")
    with open(os.path.join(a.out, "sweep_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="sweep_out")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", default=BUDGETS_DEFAULT)
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--holdout", action="store_true", help="sweep the test samples outside the dev subset")
    a = ap.parse_args()
    D.DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    {"sweep": cmd_sweep, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
