#!/usr/bin/env python3
"""GT-free generator proxy for chunk utility (LongCodeZip-style AMI), checked against the Phase-2 dLL labels.

E1 showed a text-only scorer cannot recover dLL (Spearman <= 0.20). The information seems to sit in the generator.
This asks the generator directly, without the ground truth: does chunk c make the LAST k non-empty lines of the
visible left context (q, right before the cursor) more likely?
    AMI_k(c) = log p(q | c + path + left_without_q) - log p(q | path + left_without_q)
Same pool, same in-file window W(B) and same prompt layout as chunk_pilot.label_prompts; only the target changes
(q instead of the GT). B = 1024 (as the E1 gate), k in 1, 3, 6.

Generation (as e1_eval.py, same packing: whole chunks that fit B, best chunk nearest the cursor):
    ami{k}_fill_1024   all chunks ranked by AMI_k
    ami{k}_pos_1024    only chunks with AMI_k > 0 (GT-free stop rule)

Decision (fixed before running), per dataset:
    G1: best k's within-sample Spearman(AMI_k, dLL) at B=1024 >= 0.30   (dense 0.07-0.14, best E1b scorer 0.14-0.20)
    G2: best ami EM >= E1b dense_fill_1024 + 3 (RepoEval) / + 2 (CCEval), same packing and subset
All 3 k values are reported; picking the best k on eval data is a mild selection, read it as such.

usage:
    python e1_ami.py run     --dataset repoeval --cands ... --labels pilot/chunks.jsonl --out e1ami
    python e1_ami.py analyze --dataset repoeval --out modal_out/e1ami_repoeval --prev_dir probe_inputs/repoeval \
                             --pilot_dir modal_out/pilot_repoeval --e1_dir modal_out/e1heval_repoeval
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk_pilot as C  # noqa: E402
import dep_oracle as D  # noqa: E402
import likelihood_probe as P  # noqa: E402

BUDGET = 1024
KS = [1, 3, 6]
MAX_Q = 128  # target tokens
SLACK = 32


def split_tail(left, k):
    """(prefix, q): q = the last k non-empty lines of `left` plus any partial line after them."""
    pos, seen = len(left), 0
    while pos > 0 and seen < k:
        j = left.rfind("\n", 0, pos - 1)
        line = left[j + 1:pos]
        if line.strip():
            seen += 1
        pos = j + 1 if j >= 0 else 0
    return left[:pos], left[pos:]


def ami_prompts(tok, ex, pool, budget, k):
    path_ids = tok.encode(f"\n\n# file path: {ex['path']}\n\n", add_special_tokens=False)
    window = D.MAX_CTX - (budget + len(path_ids) + 10)
    prefix, q = split_tail(ex["left"], k)
    q_ids = tok.encode(q, add_special_tokens=False)[-MAX_Q:]
    pre_ids = tok.encode(prefix, add_special_tokens=False)[-max(window - len(q_ids), 1):] if prefix else []

    def ids(block):
        cross = tok.encode(D.block_str(block)[: budget * 10], add_special_tokens=False)[:budget] if block else []
        return tok.encode(tok.decode(cross + path_ids + pre_ids))[-(D.MAX_CTX - len(q_ids)):]
    return q_ids, ids(None), [ids(b) for b in pool]


def pack(tok, ranked, budget):
    used, keep = 0, []
    for b in ranked:
        n = len(tok.encode(D.block_str(b), add_special_tokens=False)) + 2
        if used + n > budget:
            break
        used += n
        keep.append(b)
    return keep[::-1]


def cmd_run(a):
    import torch
    from transformers import AutoModelForCausalLM

    cands = D.load_cands(a.cands)
    ids = C.pilot_ids(cands, a.n, a.seed, a.n_pilot)
    exs = D.load_examples(a.data_dir, a.subset, set(ids))
    ids = [t for t in ids if t in exs]
    if a.limit:
        ids = ids[:a.limit]
    labels = {(r["task_id"], r["budget"]): r for r in map(json.loads, open(a.labels, encoding="utf-8"))}
    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    os.makedirs(a.out, exist_ok=True)

    path = os.path.join(a.out, "ami.jsonl")
    ami = {}
    if os.path.exists(path):
        for r in map(json.loads, open(path)):
            ami[(r["task_id"], r["k"])] = r["ami"]
    pools = {t: C.pool_at(tok, exs[t], cands[t]["candidates"], BUDGET) for t in ids}
    t0, n_seq = time.time(), 0
    with open(path, "a") as f:
        for i, t in enumerate(ids):
            assert len(pools[t]) == len(labels[(t, BUDGET)]["cands"]), t
            for k in KS:
                if (t, k) in ami:
                    continue
                q, base, per = ami_prompts(tok, exs[t], pools[t], BUDGET, k)
                lps = C.lp_sums(model, tok, [(base, q)] + [(x, q) for x in per], model.device, a.batch_size)
                n_seq += len(per) + 1
                ami[(t, k)] = [round(x - lps[0], 4) for x in lps[1:]]
                f.write(json.dumps({"task_id": t, "k": k, "n_q_tokens": len(q), "ami": ami[(t, k)]}) + "\n")
            f.flush()
            if i % 20 == 0:
                el = time.time() - t0
                print(f"ami {i}/{len(ids)} {el:.0f}s {n_seq / max(el, 1):.1f} seq/s", flush=True)

    for k in KS:
        for mode in ("fill", "pos"):
            cond = f"ami{k}_{mode}_{BUDGET}"
            cdir = os.path.join(a.out, cond)
            os.makedirs(cdir, exist_ok=True)
            pred_path = os.path.join(cdir, "prediction.jsonl")
            done = {json.loads(l)["task_id"] for l in open(pred_path)} if os.path.exists(pred_path) else set()
            prompts = []
            for t in ids:
                if t in done:
                    continue
                s, pool = ami[(t, k)], pools[t]
                order = sorted(range(len(pool)), key=lambda i: -s[i])
                if mode == "pos":
                    order = [i for i in order if s[i] > 0]
                bl = pack(tok, [pool[i] for i in order], BUDGET)
                prompt, info = D.build_prompt(tok, exs[t], bl, max_cross=BUDGET + SLACK)
                prompts.append((t, tok.encode(prompt)[-D.MAX_CTX:], info["cross_tokens"], [x["_src"] for x in bl]))
            prompts.sort(key=lambda x: len(x[1]))
            with open(pred_path, "a") as fout:
                for i in range(0, len(prompts), a.gen_batch_size):
                    batch = prompts[i:i + a.gen_batch_size]
                    L = max(len(x[1]) for x in batch)
                    inp = torch.tensor([[tok.pad_token_id] * (L - len(x[1])) + x[1] for x in batch]).cuda()
                    att = torch.tensor([[0] * (L - len(x[1])) + [1] * len(x[1]) for x in batch]).cuda()
                    with torch.no_grad():
                        gen = model.generate(inp, attention_mask=att, max_new_tokens=D.MAX_NEW, do_sample=False,
                                             pad_token_id=tok.pad_token_id)
                    for (t, _, ct, srcs), g in zip(batch, gen[:, L:]):
                        fout.write(json.dumps({"task_id": t, "pred": tok.decode(g, skip_special_tokens=True),
                                               "cross_tokens": ct, "srcs": srcs}) + "\n")
                    fout.flush()
            print(f"[{cond}] done", flush=True)


def cmd_analyze(a):
    gt = {t: r["gt"] for t, r in P.load_jsonl(os.path.join(a.prev_dir, "gt.jsonl")).items()}
    labels = {(r["task_id"], r["budget"]): r for r in map(json.loads, open(os.path.join(a.pilot_dir, "chunks.jsonl")))}
    ami = {}
    for r in map(json.loads, open(os.path.join(a.out, "ami.jsonl"))):
        ami[(r["task_id"], r["k"])] = r["ami"]
    res = {"dataset": a.dataset, "spearman": {}, "spearman_by_src": {}, "sign_acc": {}, "em": {}}
    for k in KS:
        rho, acc, by = [], [], {"cross": [], "infile": []}
        for (t, kk), s in ami.items():
            if kk != k:
                continue
            cs = labels[(t, BUDGET)]["cands"]
            d = [c["dll"] for c in cs]
            x = C.spearman(s, d)
            if x is not None:
                rho.append(x)
            acc += [(si > 0) == (di > 0) for si, di in zip(s, d)]
            for src in by:
                idx = [i for i, c in enumerate(cs) if c["src"] == src]
                y = C.spearman([s[i] for i in idx], [d[i] for i in idx])
                if y is not None:
                    by[src].append(y)
        res["spearman"][f"ami{k}"] = round(sum(rho) / len(rho), 4)
        res["spearman_by_src"][f"ami{k}"] = {s: round(sum(v) / len(v), 4) if v else None for s, v in by.items()}
        res["sign_acc"][f"ami{k}"] = round(sum(acc) / len(acc), 4)
    dense = [C.spearman([c["dense"] for c in r["cands"]], [c["dll"] for c in r["cands"]])
             for (t, b), r in labels.items() if b == BUDGET and (t, KS[0]) in ami]
    dense = [x for x in dense if x is not None]
    res["spearman"]["dense"] = round(sum(dense) / len(dense), 4)

    def em_of(p):
        return {t: D.simple_score(r["pred"], gt[t])[0] for t, r in P.load_jsonl(p).items()}
    preds = {c: em_of(os.path.join(a.out, c, "prediction.jsonl")) for c in sorted(os.listdir(a.out))
             if os.path.exists(os.path.join(a.out, c, "prediction.jsonl"))}
    for c in ("dense_fill_1024", "cross_soft_fill_1024", "bi_onehot_fill_1536", "cross_soft_fill_1536",
              "dense_fill_1536"):
        p = os.path.join(a.e1_dir, c, "prediction.jsonl")
        if os.path.exists(p):
            preds["ref_e1b_" + c] = em_of(p)
    p0 = json.load(open(os.path.join(a.pilot_dir, "pilot_summary.json")))["P0"]
    preds["ref_regen_P0"] = em_of(os.path.join(a.pilot_dir, "regen_" + p0, "prediction.jsonl"))
    e0 = os.path.join(os.path.dirname(a.e1_dir), f"e0_{a.dataset}", f"set_asc_{BUDGET}", "prediction.jsonl")
    if os.path.exists(e0):
        preds["ref_dll_upper_1024"] = em_of(e0)
    ids = sorted(set.intersection(*[set(v) for v in preds.values()]))
    ref = preds["ref_e1b_dense_fill_1024"]
    for c, em in preds.items():
        w = sum(em[t] > ref[t] for t in ids)
        l_ = sum(em[t] < ref[t] for t in ids)
        res["em"][c] = {"em": round(100 * sum(em[t] for t in ids) / len(ids), 2),
                        "vs_dense_fill_1024": [w, l_, round(D.mcnemar(w, l_), 4)]}
    res["n"] = len(ids)
    best_k = max(KS, key=lambda k: res["spearman"][f"ami{k}"])
    need = 3.0 if a.dataset == "repoeval" else 2.0
    best_em = max(v["em"] for c, v in res["em"].items() if c.startswith("ami"))
    dense_em = res["em"]["ref_e1b_dense_fill_1024"]["em"]
    res["decision"] = {"best_k": best_k, "G1_spearman": res["spearman"][f"ami{best_k}"],
                       "G1": "PASS" if res["spearman"][f"ami{best_k}"] >= 0.30 else "FAIL",
                       "G2_best_ami_em": best_em, "dense_fill_1024": dense_em,
                       "G2": "PASS" if best_em >= dense_em + need else "FAIL"}
    with open(os.path.join(a.out, "ami_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--labels")
    ap.add_argument("--prev_dir")
    ap.add_argument("--pilot_dir")
    ap.add_argument("--e1_dir")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="e1ami")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_pilot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    a = ap.parse_args()
    D.DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
