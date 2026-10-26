#!/usr/bin/env python3
"""E5: is the context BUDGET a useful extra axis for Caret-style test-time selection?

Caret (arXiv 2610.04837) samples several completions per retrieval context and picks one by agreement (vote) plus
reverse-context verification (RCV: how likely the code AFTER the cursor is once the candidate line is inserted).
Its candidate diversity comes from sampling and from switching retrievers; the context budget stays fixed.
E5 keeps AlignCoder's retrieval and asks whether spreading the same number of samples over several budgets
(i.e. several in-file / cross-file splits) beats sampling at one budget.

Generation (1000-sample dev subset per dataset, AlignCoder candidates and prompts, DeepSeek-Coder-1.3B):
    samples  temperature 0.8, top-p 0.95, 64 tokens: 8 at B=1536, 4 at B=1024, 4 at B=512
    greedy   one per budget 512 / 1024 / 1536 (with first-line confidence, as E2)
Candidate key = first non-empty line. RCV(c) = mean log p(first RCV_TOKENS tokens of the right context |
in-file left tail + c + newline); no cross-file context, so scores are comparable across budgets.

Pools of EQUAL size (8 samples):
    fixed1536   8 samples at 1536                         (Caret-like: diversity from sampling only)
    mixed       3 at 512 + 3 at 1024 + 2 at 1536           (diversity from sampling + budget)
Selectors: vote (ties -> mean candidate log-prob), rcv, caret = 0.75 * minmax(rcv) + 0.25 * vote share.
Settings: "right context allowed" (rcv, caret) and "prefix only" (vote).

Decision (fixed before running): the budget axis HELPS if mixed > fixed1536 on both datasets with the caret selector,
pooled McNemar p < 0.05; reported separately for the prefix-only vote selector.

usage:
    python e5_budget_tts.py run --dataset repoeval --cands prediction_with_candidates.jsonl --out e5 --shard 0 --nshards 2
    python e5_budget_tts.py analyze --prev_dir_fmt probe_inputs/{}
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep as BS  # noqa: E402
import chunk_pilot as C  # noqa: E402
import dep_oracle as D  # noqa: E402
import likelihood_probe as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_SAMPLES = {1536: 8, 1024: 4, 512: 4}
POOLS = {"fixed1536": {1536: 8}, "mixed": {512: 3, 1024: 3, 1536: 2}}
RCV_TOKENS, LEFT_TAIL = 128, 1024
MAX_SEQS = 8  # sampled sequences per generate() call: 24 x 2048-token prefill OOMs on L4


def first_line(s):
    for l in s.split("\n"):
        if l.strip():
            return l
    return ""


def cmd_run(a):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoModelForCausalLM

    cands = D.load_cands(a.cands)
    ids = D.pick_subset(list(cands), a.n, a.seed)
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
    print(f"right context non-empty: {sum(bool(exs[t]['right'].strip()) for t in ids)}/{len(ids)}", flush=True)
    torch.manual_seed(a.seed)

    def gen(prompts, n, sample):
        """prompts: list of id lists -> list (per prompt) of lists of (text, first_line_conf)."""
        out = []
        per = max(1, MAX_SEQS // n)
        for i in range(0, len(prompts), per):
            batch = prompts[i:i + per]
            L = max(map(len, batch))
            inp = torch.tensor([[tok.pad_token_id] * (L - len(x)) + x for x in batch]).cuda()
            att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in batch]).cuda()
            with torch.no_grad():
                g = model.generate(inp, attention_mask=att, max_new_tokens=D.MAX_NEW, do_sample=sample,
                                   temperature=0.8 if sample else None, top_p=0.95 if sample else None,
                                   num_return_sequences=n, pad_token_id=tok.pad_token_id,
                                   output_scores=not sample, return_dict_in_generate=True)
            seqs = g.sequences[:, L:]
            confs = [None] * len(seqs)
            if not sample:
                lps = torch.stack([torch.log_softmax(s.float(), -1) for s in g.scores], 1)
                ch = lps.gather(-1, seqs[:, :lps.size(1)].unsqueeze(-1)).squeeze(-1)
                for j in range(len(seqs)):
                    toks = seqs[j].tolist()
                    keep = [k for k in range(len(ch[j])) if toks[k] != tok.eos_token_id]
                    confs[j] = BS.first_line_conf([tok.decode([toks[k]]) for k in keep], [ch[j][k].item() for k in keep])
            texts = [tok.decode(s, skip_special_tokens=True) for s in seqs]
            for j in range(len(batch)):
                out.append([(texts[j * n + r], confs[j * n + r]) for r in range(n)])
        return out

    todo = [t for t in ids if t not in done]
    with open(path, "a") as f:
        for s in range(0, len(todo), a.chunk):
            chunk = todo[s:s + a.chunk]
            rec = {t: {"task_id": t, "samples": {}, "greedy": {}} for t in chunk}
            for b, n in N_SAMPLES.items():
                prompts = [tok.encode(D.build_prompt(tok, exs[t], cands[t]["candidates"], max_cross=b)[0])[-D.MAX_CTX:]
                           for t in chunk]
                for t, outs in zip(chunk, gen(prompts, n, True)):
                    rec[t]["samples"][str(b)] = [x for x, _ in outs]
                for t, outs in zip(chunk, gen(prompts, 1, False)):
                    rec[t]["greedy"][str(b)] = {"pred": outs[0][0], "conf": outs[0][1]}
            # reverse-context verification + candidate log-prob, per distinct first line
            for t in chunk:
                ex = rec[t]
                lines = []
                for b in N_SAMPLES:
                    lines += [first_line(x) for x in ex["samples"][str(b)]] + [first_line(ex["greedy"][str(b)]["pred"])]
                uniq = sorted(set(lines))
                right_ids = tok.encode(exs[t]["right"], add_special_tokens=False)[:RCV_TOKENS]
                left_ids = tok.encode(exs[t]["left"], add_special_tokens=False)[-LEFT_TAIL:]
                left_txt = tok.decode(left_ids)
                seqs_r, seqs_c = [], []
                for c in uniq:
                    c_ids = tok.encode(c, add_special_tokens=False)
                    seqs_c.append((tok.encode(left_txt, add_special_tokens=False), c_ids))
                    seqs_r.append((tok.encode(left_txt + c + "\n", add_special_tokens=False), right_ids))
                rr = C.lp_sums(model, tok, seqs_r, model.device, a.batch_size) if right_ids else [0.0] * len(uniq)
                cc = C.lp_sums(model, tok, seqs_c, model.device, a.batch_size)
                ex["lines"] = {c: {"rcv": round(r / max(len(right_ids), 1), 5),
                                   "cand_lp": round(l / max(len(sc[1]), 1), 5)}
                               for c, r, l, sc in zip(uniq, rr, cc, seqs_c)}
                f.write(json.dumps(ex) + "\n")
            f.flush()
            print(f"shard {a.shard}: {s + len(chunk)}/{len(todo)}", flush=True)


# ----------------------------------------------------------------------------------
# analyze (CPU)
# ----------------------------------------------------------------------------------

def select(cands, lines, method):
    """cands: list of full preds. Returns the chosen full pred."""
    keys = [first_line(x) for x in cands]
    cnt = Counter(keys)
    if method == "vote":
        best = max(cnt, key=lambda k: (cnt[k], lines[k]["cand_lp"]))
    elif method == "rcv":
        best = max(cnt, key=lambda k: lines[k]["rcv"])
    else:  # caret blend
        r = [lines[k]["rcv"] for k in cnt]
        lo, hi = min(r), max(r)
        best = max(cnt, key=lambda k: 0.75 * ((lines[k]["rcv"] - lo) / (hi - lo) if hi > lo else 1.0)
                   + 0.25 * cnt[k] / len(keys))
    return next(x for x, k in zip(cands, keys) if k == best)


def cmd_analyze(a):
    res = {"policies": {}}
    rows, gts = {}, {}
    for ds in ("repoeval", "cceval"):
        gts[ds] = {t: r["gt"] for t, r in P.load_jsonl(a.prev_dir_fmt.format(ds) + "/gt.jsonl").items()}
        d = os.path.join(HERE, "modal_out", f"e5_{ds}")
        rows[ds] = {}
        for fn in sorted(os.listdir(d)):
            if fn.startswith("items_"):
                for r in map(json.loads, open(os.path.join(d, fn))):
                    rows[ds][r["task_id"]] = r
    choices = {}
    for ds, rs in rows.items():
        for t, r in rs.items():
            em = lambda p: D.simple_score(p, gts[ds][t])[0]
            pools = {name: sum((r["samples"][str(b)][:k] for b, k in spec.items()), [])
                     for name, spec in POOLS.items()}
            c = {}
            for name, pool in pools.items():
                for m in ("vote", "rcv", "caret"):
                    c[f"{name}/{m}"] = em(select(pool, r["lines"], m))
                c[f"{name}/oracle"] = max(em(x) for x in pool)
            g = r["greedy"]
            for b in ("512", "1024", "1536"):
                c[f"greedy_{b}"] = em(g[b]["pred"])
            c["greedy3/conf (E2-style)"] = em(g[max(g, key=lambda b: g[b]["conf"])]["pred"])
            c["greedy3/caret"] = em(select([g[b]["pred"] for b in g], r["lines"], "caret"))
            c["greedy3+fixed1536/caret"] = em(select([g[b]["pred"] for b in g] + pools["fixed1536"], r["lines"], "caret"))
            choices[(ds, t)] = c
    names = list(next(iter(choices.values())))
    ref = "fixed1536/caret"
    for n in names:
        row = {}
        for ds in rows:
            ks = [k for k in choices if k[0] == ds]
            w = sum(choices[k][n] > choices[k][ref] for k in ks)
            l_ = sum(choices[k][n] < choices[k][ref] for k in ks)
            row[ds] = {"em": round(100 * sum(choices[k][n] for k in ks) / len(ks), 2), "vs_fixed1536_caret": [w, l_, round(D.mcnemar(w, l_), 4)]}
        w = sum(choices[k][n] > choices[k][ref] for k in choices)
        l_ = sum(choices[k][n] < choices[k][ref] for k in choices)
        row["pooled_vs_ref"] = [w, l_, round(D.mcnemar(w, l_), 4)]
        res["policies"][n] = row
        print(f"{n:28s} RepoEval {row['repoeval']['em']:5.2f} {row['repoeval']['vs_fixed1536_caret']}  "
              f"CCEval {row['cceval']['em']:5.2f} {row['cceval']['vs_fixed1536_caret']}  pooled {row['pooled_vs_ref']}")
    # prefix-only: mixed/vote vs fixed1536/vote
    pv = {}
    for ds in rows:
        ks = [k for k in choices if k[0] == ds]
        w = sum(choices[k]["mixed/vote"] > choices[k]["fixed1536/vote"] for k in ks)
        l_ = sum(choices[k]["mixed/vote"] < choices[k]["fixed1536/vote"] for k in ks)
        pv[ds] = [w, l_, round(D.mcnemar(w, l_), 4)]
    w = sum(choices[k]["mixed/vote"] > choices[k]["fixed1536/vote"] for k in choices)
    l_ = sum(choices[k]["mixed/vote"] < choices[k]["fixed1536/vote"] for k in choices)
    pv["pooled"] = [w, l_, round(D.mcnemar(w, l_), 4)]
    res["prefix_only_mixed_vs_fixed_vote"] = pv
    m, f_ = res["policies"]["mixed/caret"], res["policies"]["fixed1536/caret"]
    ok = all(m[ds]["em"] > f_[ds]["em"] for ds in rows) and m["pooled_vs_ref"][2] < 0.05
    res["decision"] = "HELPS" if ok else "no clear gain"
    res["n"] = {ds: len(rs) for ds, rs in rows.items()}
    print("prefix-only mixed vs fixed (vote):", pv)
    print("n:", res["n"], "DECISION:", res["decision"])
    json.dump(res, open(os.path.join(HERE, "modal_out", "e5_summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="e5")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--prev_dir_fmt", default=os.path.join(HERE, "probe_inputs", "{}"))
    a = ap.parse_args()
    D.DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
