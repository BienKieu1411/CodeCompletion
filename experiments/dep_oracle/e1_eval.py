#!/usr/bin/env python3
"""E1 step 3: does the learned scorer recover dLL on the EVAL pilot (200 samples/dataset) and does it pay off?

(1) Spearman: within-sample Spearman(score, dLL) on the Phase-2 pilot labels (chunks.jsonl, GT-based), budgets
    512/1024/1536. Baseline = AlignRetriever dense (stored in the labels: 0.03-0.14).
(2) Generation, 1 pass, greedy, as the pilot. Packing (as E0): rank by score, keep the chunks that fit WHOLE into B,
    place the best chunk nearest the cursor (E0 "asc"), max_cross = B + SLACK.
        <m>_fill_B   all chunks ranked by the scorer
        <m>_stop_B   only chunks scored above the NULL candidate (learned stop; not for dense)
    m in dense, bi_soft, bi_onehot, cross_soft; B in 512, 1024. References with the same packing and subset:
    E0 set_asc_B (dLL > 0, GT-based upper bound) and pilot regen_P0.

Decision (fixed before running), per dataset:
    G1 (learnable): best variant's Spearman at B=1024 >= 0.30 (dense ~0.03-0.14)        -> else stop this line
    G2 (useful, directional at n=200): best variant's stop/fill EM >= regen_P0 + 3 (RepoEval) / + 2 (CCEval)
Checkpoint per variant = epoch with the best DEV Spearman (training-side split), never chosen on eval.

usage:
    python e1_eval.py run     --dataset repoeval --cands ... --labels pilot/chunks.jsonl --models e1_models --out e1_eval
    python e1_eval.py analyze --dataset repoeval --out modal_out/e1eval_repoeval --prev_dir probe_inputs/repoeval \
                              --pilot_dir modal_out/pilot_repoeval --e0_dir modal_out/e0_repoeval
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk_pilot as C  # noqa: E402
import dep_oracle as D  # noqa: E402
import likelihood_probe as P  # noqa: E402
from e1_train import NULL_TEXT, VARIANTS, Scorer  # noqa: E402

BUDGETS_SP = [512, 1024, 1536]
BUDGETS_GEN = [512, 1024]
SLACK = 32
MODELS = ["dense"] + VARIANTS


def best_epoch(models_dir, v):
    rows = [json.loads(l) for l in open(os.path.join(models_dir, v, "metrics.jsonl"))]
    rows = [r for r in rows if r["epoch"] > 0]
    return os.path.join(models_dir, v, f"epoch{max(rows, key=lambda r: r['spearman'])['epoch']}")


def pack(tok, ranked, budget):
    used, keep = 0, []
    for b in ranked:
        n = len(tok.encode(D.block_str(b), add_special_tokens=False)) + 2
        if used + n > budget:
            break
        used += n
        keep.append(b)
    return keep[::-1]  # best chunk last = nearest the cursor


def cmd_run(a):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    import joint_dense as JD

    cands = D.load_cands(a.cands)
    ids = C.pilot_ids(cands, a.n, a.seed, a.n_pilot)
    labels = {(r["task_id"], r["budget"]): r for r in map(json.loads, open(a.labels, encoding="utf-8"))}
    exs = D.load_examples(a.data_dir, a.subset, set(ids))
    ids = [t for t in ids if t in exs]
    if a.limit:
        ids = ids[:a.limit]
    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    os.makedirs(a.out, exist_ok=True)

    # ---- scores for every model, every pool --------------------------------------------
    sc_path = os.path.join(a.out, "scores.jsonl")
    scores = {}
    if os.path.exists(sc_path):
        for r in map(json.loads, open(sc_path)):
            scores[(r["model"], r["task_id"], r["budget"])] = r["s"]
    pools = {(t, b): C.pool_at(tok, exs[t], cands[t]["candidates"], b) for t in ids for b in BUDGETS_SP}
    for m in MODELS:
        if all((m, t, b) in scores for t in ids for b in BUDGETS_SP):
            continue
        sc = (Scorer(snapshot_download(JD.RETRIEVER_REPO, revision=JD.RETRIEVER_REV), "bi") if m == "dense" else
              Scorer(best_epoch(a.models, m), "cross" if m.startswith("cross") else "bi"))
        sc.enc.eval()
        with open(sc_path, "a") as f:
            for t in ids:
                q = "\n".join([l for l in exs[t]["left"].split("\n") if l.strip()][-JD.QUERY_LINES:])
                for b in BUDGETS_SP:
                    if (m, t, b) in scores:
                        continue
                    pool = pools[(t, b)]
                    s = sc.score(q, [D.block_str(x) for x in pool] + [NULL_TEXT])
                    scores[(m, t, b)] = s
                    f.write(json.dumps({"model": m, "task_id": t, "budget": b, "s": s}) + "\n")
        del sc
        torch.cuda.empty_cache()
        print(f"scored {m}", flush=True)

    # ---- generation -----------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    gen_budgets = [int(x) for x in a.gen_budgets.split(",")]
    conds = [f"{m}_{k}_{b}" for m in MODELS for k in ("fill", "stop") for b in gen_budgets
             if not (m == "dense" and k == "stop")]
    for cond in conds:
        m, k, b = cond.rsplit("_", 2)
        b = int(b)
        cdir = os.path.join(a.out, cond)
        os.makedirs(cdir, exist_ok=True)
        pred_path = os.path.join(cdir, "prediction.jsonl")
        done = {json.loads(l)["task_id"] for l in open(pred_path)} if os.path.exists(pred_path) else set()
        prompts = []
        for t in ids:
            if t in done:
                continue
            pool, s = pools[(t, b)], scores[(m, t, b)]
            order = sorted(range(len(pool)), key=lambda i: -s[i])
            if k == "stop":
                order = [i for i in order if s[i] > s[-1]]
            bl = pack(tok, [pool[i] for i in order], b)
            prompt, info = D.build_prompt(tok, exs[t], bl, max_cross=b + SLACK)
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
    scores = {}
    for r in map(json.loads, open(os.path.join(a.out, "scores.jsonl"))):
        scores[(r["model"], r["task_id"], r["budget"])] = r["s"]
    res = {"dataset": a.dataset, "spearman": {}, "null_rule_acc": {}, "em": {}}
    for m in MODELS:
        for b in BUDGETS_SP:
            rho, acc = [], []
            for (mm, t, bb), s in scores.items():
                if mm != m or bb != b:
                    continue
                d = [c["dll"] for c in labels[(t, b)]["cands"]]
                assert len(d) == len(s) - 1, (m, t, b)
                x = C.spearman(s[:-1], d)
                if x is not None:
                    rho.append(x)
                acc += [(si > s[-1]) == (di > 0) for si, di in zip(s[:-1], d)]
            res["spearman"][f"{m}_B{b}"] = round(sum(rho) / len(rho), 4) if rho else None
            if m != "dense":
                res["null_rule_acc"][f"{m}_B{b}"] = round(sum(acc) / len(acc), 4) if acc else None

    def em_of(path):
        rows = P.load_jsonl(path)
        return {t: D.simple_score(r["pred"], gt[t])[0] for t, r in rows.items()}, rows

    preds = {}
    for c in sorted(os.listdir(a.out)):
        p = os.path.join(a.out, c, "prediction.jsonl")
        if os.path.exists(p):
            preds[c] = em_of(p)
    p0 = json.load(open(os.path.join(a.pilot_dir, "pilot_summary.json")))["P0"]
    preds["ref_regen_P0"] = em_of(os.path.join(a.pilot_dir, "regen_" + p0, "prediction.jsonl"))
    for b in BUDGETS_SP:  # E0 has set_asc for all three budgets
        p = os.path.join(a.e0_dir, f"set_asc_{b}", "prediction.jsonl")
        if os.path.exists(p):
            preds[f"ref_dll_upper_{b}"] = em_of(p)
    ids = sorted(set.intersection(*[set(v[0]) for v in preds.values()]))
    ref = preds["ref_regen_P0"][0]
    for c, (em, rows) in preds.items():
        w = sum(em[t] > ref[t] for t in ids)
        l_ = sum(em[t] < ref[t] for t in ids)
        row = {"em": round(100 * sum(em[t] for t in ids) / len(ids), 2), "vs_regen_P0": [w, l_, D.mcnemar(w, l_)]}
        if "srcs" in next(iter(rows.values())):
            row["mean_chunks"] = round(sum(len(rows[t]["srcs"]) for t in ids) / len(ids), 2)
            row["mean_infile_chunks"] = round(sum(rows[t]["srcs"].count("infile") for t in ids) / len(ids), 2)
        res["em"][c] = row
    res["n"] = len(ids)
    sp = {m: res["spearman"][f"{m}_B1024"] for m in VARIANTS if res["spearman"].get(f"{m}_B1024") is not None}
    best = max(sp, key=sp.get) if sp else None
    need = 3.0 if a.dataset == "repoeval" else 2.0
    best_em = max((v["em"] for c, v in res["em"].items() if c.split("_")[0] + "_" + c.split("_")[1] in VARIANTS),
                  default=None)
    p0_em = res["em"]["ref_regen_P0"]["em"]
    res["decision"] = {"G1_best_variant": best, "G1_spearman": sp.get(best),
                       "G1": None if best is None else ("PASS" if sp[best] >= 0.30 else "FAIL"),
                       "G2_best_learned_em": best_em, "regen_P0": p0_em,
                       "G2": None if best_em is None else ("PASS" if best_em >= p0_em + need else "FAIL")}
    with open(os.path.join(a.out, "e1_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--labels")
    ap.add_argument("--models", default="e1_models")
    ap.add_argument("--prev_dir")
    ap.add_argument("--pilot_dir")
    ap.add_argument("--e0_dir")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="e1_eval")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_pilot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--gen_budgets", default=",".join(map(str, BUDGETS_GEN)))
    a = ap.parse_args()
    D.DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
