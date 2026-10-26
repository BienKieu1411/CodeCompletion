#!/usr/bin/env python3
"""E8: context-aware (contrastive) decoding against the cross-file context, frozen generator, prefix only.

Motivation (E7b): ~40% of CCEval "usage" failures have the definitions of every GT symbol in the prompt, yet the
generator picks the wrong member / argument. CAD (Shi et al. 2023) amplifies what the context contributes:
    score_t = (1 + a) * logits(with cross-file) - a * logits(without cross-file)
"without" = the SAME prompt minus the cross-file part (same in-file tail, same file-path line), so the contrast
isolates the cross-file context. Greedy, 64 new tokens, DeepSeek-Coder-1.3B, AlignCoder candidates.

Conditions: budget B in {512, 1536} x alpha in {0, 0.25, 0.5, 1, ada}; ada = AdaCAD-style per-token
alpha_t = JSD(p_with, p_without). alpha 0 = plain greedy (anchor: must match the budget sweep for that B).
Pilot: the usual 200 samples per dataset (chunk_pilot.pilot_ids: 200 of the 1000-sample dev subset).

Decision (fixed before running): PRIMARY = alpha 0.5 at the per-dataset best fixed budget (RepoEval 512,
CCEval 1536). CAD HELPS if PRIMARY > alpha 0 at the same budget on both datasets and pooled McNemar p < 0.05.
Other alphas are exploratory. Also reported: fixes/breaks per E7/E7b error class (usage x error type, copy group).

usage:
    python e8_cad.py run --dataset repoeval --cands prediction_with_candidates.jsonl --out e8 --shard 0 --nshards 2
    python e8_cad.py analyze --root modal_out
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk_pilot as C  # noqa: E402
import dep_oracle as D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUDGETS = [512, 1536]
ALPHAS = ["0", "0.25", "0.5", "1", "ada"]
BEST = {"repoeval": 512, "cceval": 1536}
PRIMARY = "0.5"


def prompts_for(tok, ex, saved, b):
    """Token ids of the prompt with cross-file context and of the same prompt without it."""
    prompt, info = D.build_prompt(tok, ex, saved, max_cross=b)
    with_ids = tok.encode(prompt)[-D.MAX_CTX:]
    path_ids = tok.encode(f"\n\n# file path: {ex['path']}\n\n", add_special_tokens=False)
    infile_ids = tok.encode(ex["left"], add_special_tokens=False)[-info["infile_tokens"]:] if info["infile_tokens"] else []
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    return with_ids, bos + path_ids + infile_ids


def cad_generate(model, tok, pairs, alpha, max_new):
    """Batched greedy CAD. pairs: list of (with_ids, without_ids). Returns list of (text, token log-probs)."""
    import torch
    import torch.nn.functional as F
    k = len(pairs)
    rows = [p[0] for p in pairs] + ([p[1] for p in pairs] if alpha != "0" else [])
    L = max(map(len, rows))
    pad = tok.pad_token_id
    inp = torch.tensor([[pad] * (L - len(r)) + r for r in rows]).cuda()
    att = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in rows]).cuda()
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    out_tok = [[] for _ in range(k)]
    out_lp = [[] for _ in range(k)]
    done = torch.zeros(k, dtype=torch.bool, device="cuda")
    past = None
    cur, cur_pos = inp, pos
    with torch.no_grad():
        for _ in range(max_new):
            o = model(input_ids=cur, attention_mask=att, position_ids=cur_pos, past_key_values=past, use_cache=True)
            past = o.past_key_values
            lg = o.logits[:, -1, :].float()
            lw = F.log_softmax(lg[:k], -1)
            if alpha == "0":
                score = lw
            else:
                lo = F.log_softmax(lg[k:], -1)
                if alpha == "ada":
                    pw, po = lw.exp(), lo.exp()
                    m = 0.5 * (pw + po)
                    jsd = 0.5 * ((pw * (lw - m.clamp_min(1e-12).log())).sum(-1)
                                 + (po * (lo - m.clamp_min(1e-12).log())).sum(-1))
                    a = jsd.clamp(min=0).unsqueeze(-1)
                else:
                    a = float(alpha)
                score = (1 + a) * lw - a * lo
            nxt = score.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
            lp = lw.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
            for j in range(k):
                if not done[j]:
                    out_tok[j].append(nxt[j].item())
                    out_lp[j].append(round(lp[j].item(), 4))
            done = done | (nxt == tok.eos_token_id)
            if done.all():
                break
            step = nxt if alpha == "0" else torch.cat([nxt, nxt])
            cur = step.unsqueeze(-1)
            att = torch.cat([att, torch.ones_like(cur)], -1)
            cur_pos = cur_pos[:, -1:] + 1
    res = []
    for j in range(k):
        toks = [t for t in out_tok[j] if t != tok.eos_token_id]
        res.append((tok.decode(toks, skip_special_tokens=True), out_lp[j][:len(toks)]))
    return res


def cmd_run(a):
    import torch
    from transformers import AutoModelForCausalLM
    cands = D.load_cands(a.cands)
    ids = C.pilot_ids(cands, a.n, a.seed, a.n_pilot)
    exs = D.load_examples(a.data_dir, a.subset, set(ids))
    ids = [t for t in ids if t in exs][a.shard::a.nshards]
    if a.limit:
        ids = ids[:a.limit]
    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"items_{a.shard:02d}.jsonl")
    done = {json.loads(l)["task_id"] for l in open(path)} if os.path.exists(path) else set()
    todo = [t for t in ids if t not in done]
    with open(path, "a") as f:
        for s in range(0, len(todo), a.batch_size):
            chunk = todo[s:s + a.batch_size]
            rec = {t: {"task_id": t, "gt": exs[t]["gt"], "pred": {}} for t in chunk}
            for b in BUDGETS:
                pairs = [prompts_for(tok, exs[t], cands[t]["candidates"], b) for t in chunk]
                for al in ALPHAS:
                    for t, (txt, lp) in zip(chunk, cad_generate(model, tok, pairs, al, D.MAX_NEW)):
                        rec[t]["pred"][f"{b}_{al}"] = {"text": txt, "lp": lp}
            for t in chunk:
                f.write(json.dumps(rec[t]) + "\n")
            f.flush()
            print(f"shard {a.shard}: {s + len(chunk)}/{len(todo)}", flush=True)


def cmd_analyze(a):
    res = {}
    pooled = defaultdict(lambda: [0, 0])
    for ds in BEST:
        rows = []
        d = os.path.join(a.root, f"{a.tag}_{ds}")
        for fn in sorted(os.listdir(d)):
            if fn.startswith("items_"):
                rows += [json.loads(l) for l in open(os.path.join(d, fn), encoding="utf-8")]
        rows = list({r["task_id"]: r for r in rows}.values())
        em = {r["task_id"]: {k: D.simple_score(v["text"], r["gt"])[0] for k, v in r["pred"].items()} for r in rows}
        ids = sorted(em)
        # anchor: alpha 0 vs the budget sweep greedy output at the same budget
        sweep = {}
        for b in BUDGETS:
            p = os.path.join(HERE, f"kaggle_sweep_{ds}_out", "sweep", f"b{b}", "prediction.jsonl")
            sweep[b] = {r["task_id"]: r["pred"] for r in map(json.loads, open(p, encoding="utf-8"))}
        anchor = {b: round(100 * sum(D.simple_score(sweep[b][t], next(r["gt"] for r in rows if r["task_id"] == t))[0]
                                     for t in ids if t in sweep[b]) / len(ids), 2) for b in BUDGETS}
        same = {b: round(100 * sum(next(r for r in rows if r["task_id"] == t)["pred"][f"{b}_0"]["text"].strip()
                                   == sweep[b][t].strip() for t in ids if t in sweep[b]) / len(ids), 1)
                for b in BUDGETS}
        cats = {}
        for fn, key in ((f"e7_rows_{ds}.jsonl", "cls"), (f"e7b_rows_{ds}.jsonl", "err")):
            p = os.path.join(HERE, "modal_out", fn)
            if os.path.exists(p):
                for r in map(json.loads, open(p, encoding="utf-8")):
                    cats.setdefault(r["task_id"], {})[key] = r[key]
                    if key == "err":
                        cats[r["task_id"]]["copied"] = r["pred_copied"]
                        cats[r["task_id"]]["defs"] = r["defs"]
        ref = f"{BEST[ds]}_0"
        conds = {}
        for c in sorted(em[ids[0]]):
            w = sum(em[t][c] > em[t][ref] for t in ids)
            l_ = sum(em[t][c] < em[t][ref] for t in ids)
            b = c.split("_")[0]
            w0 = sum(em[t][c] > em[t][f"{b}_0"] for t in ids)
            l0 = sum(em[t][c] < em[t][f"{b}_0"] for t in ids)
            by = Counter()
            for t in ids:
                if em[t][c] != em[t][f"{b}_0"]:
                    ct = cats.get(t, {})
                    g = ct.get("err", ct.get("cls", "solved_somewhere")) if ct.get("cls") == "usage" else ct.get("cls", "?")
                    if ct.get("copied"):
                        g += "+copied"
                    by[f"{g}:{'fix' if em[t][c] > em[t][f'{b}_0'] else 'break'}"] += 1
            conds[c] = {"em": round(100 * sum(em[t][c] for t in ids) / len(ids), 2),
                        "vs_best_fixed": [w, l_, round(D.mcnemar(w, l_), 4)],
                        "vs_alpha0_same_B": [w0, l0, round(D.mcnemar(w0, l0), 4)],
                        "changes_by_class": dict(by.most_common())}
            if c.endswith("_" + PRIMARY) and int(b) == BEST[ds]:
                pooled["primary"][0] += w0
                pooled["primary"][1] += l0
        res[ds] = {"n": len(ids), "anchor_sweep_em": anchor, "alpha0_identical_to_sweep_pct": same, "conditions": conds}
    w, l_ = pooled["primary"]
    prim = {ds: res[ds]["conditions"][f"{BEST[ds]}_{PRIMARY}"] for ds in BEST}
    ok = all(prim[ds]["vs_alpha0_same_B"][0] > prim[ds]["vs_alpha0_same_B"][1] for ds in BEST) and D.mcnemar(w, l_) < 0.05
    res["primary_pooled_vs_alpha0"] = [w, l_, round(D.mcnemar(w, l_), 4)]
    res["decision"] = "PASS" if ok else "FAIL"
    json.dump(res, open(os.path.join(a.root, "e8_cad_summary.json"), "w"), indent=1)
    for ds in BEST:
        r = res[ds]
        print(f"{ds} n={r['n']} sweep anchor {r['anchor_sweep_em']} alpha0==sweep {r['alpha0_identical_to_sweep_pct']}")
        for c, v in r["conditions"].items():
            print(f"  {c:9s} EM {v['em']:5.2f}  vs best fixed {v['vs_best_fixed']}  vs a0 {v['vs_alpha0_same_B']}")
    print("PRIMARY pooled vs alpha0:", res["primary_pooled_vs_alpha0"], "DECISION:", res["decision"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="e8")
    ap.add_argument("--root", default=os.path.join(HERE, "modal_out"))
    ap.add_argument("--tag", default="e8")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_pilot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    a = ap.parse_args()
    D.DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    cmd_run(a) if a.cmd == "run" else cmd_analyze(a)


if __name__ == "__main__":
    main()
