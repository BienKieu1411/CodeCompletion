#!/usr/bin/env python3
"""Phase 2 pilot: do per-chunk likelihood labels (dLL) compose into a good prompt?

See PLAN_likelihood_probe.md, "Thiết kế chi tiết Pha 2". For each sample and cross-file budget B:
    dLL(c) = log p(GT | c + path + in-file W(B)) - log p(GT | path + in-file W(B))
with W(B) the in-file window the final prompt has at budget B. Candidates = AlignCoder's saved top-10 cross-file
chunks + hidden in-file chunks (same pool as dense_joint). Then chunks are packed by dLL and the frozen generator
completes the prompt.

Conditions (greedy, batch 8, left padding, as the earlier runs):
    dll_fill_B      whole pool sorted by dLL, budget B                  (B in 512, 1024, 1536)
    dll_pos_B       only chunks with dLL > 0, sorted, budget B           (empty -> in-file only, i.e. STOP)
    rand_joint1536  same pool, random order, budget 1536                 (control: order vs pool)
    regen_<cond>    Phase-1 constructions re-generated on this GPU       (same-hardware reference)
Every condition also stores log p(GT | final prompt) -> `dll_bestB` (analyze) picks the dll_pos_B budget with the
highest GT likelihood. Primary metric (fixed before running): dll_pos_1536 vs regen_P0.

usage:
    python chunk_pilot.py run     --dataset repoeval --cands prediction_with_candidates.jsonl --out pilot
    python chunk_pilot.py analyze --dataset repoeval --out pilot --prev_dir probe_inputs/repoeval \
                                  --probe_out modal_out/full_repoeval
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dep_oracle as D  # noqa: E402
import joint_retrieval as J  # noqa: E402
import likelihood_probe as P  # noqa: E402

BUDGETS = [512, 1024, 1536]
MAX_INFILE_CANDS = 128  # as joint_dense.py
PRIMARY = "dll_pos_1536"
REGEN = ["regen_" + c for c in P.CONDITIONS]
CONDITIONS = ([f"dll_fill_{b}" for b in BUDGETS] + [f"dll_pos_{b}" for b in BUDGETS] + ["rand_joint1536"]
              + REGEN)
QUERY_LINES = 20  # as joint_dense.py


def pilot_ids(cands, n, seed, n_pilot):
    ids = D.pick_subset(list(cands), n, seed)  # the Phase-1 subset
    return sorted(random.Random(seed).sample(ids, min(n_pilot, len(ids))))


def pool_at(tok, ex, saved, budget):
    """Candidate pool in a fixed order: saved cross-file top-10 (STOP removed), then hidden in-file chunks."""
    return J.saved_blocks(saved) + J.hidden_infile_blocks(tok, ex, budget)[-MAX_INFILE_CANDS:]


def label_prompts(tok, ex, pool, budget):
    """Token ids of the base prompt (no chunk) and of one prompt per chunk, in-file window fixed to W(budget)."""
    path_ids = tok.encode(f"\n\n# file path: {ex['path']}\n\n", add_special_tokens=False)
    window = D.MAX_CTX - (budget + len(path_ids) + 10)
    left_ids = tok.encode(ex["left"], add_special_tokens=False)[-window:]

    def ids(block):
        cross = tok.encode(D.block_str(block)[: budget * 10], add_special_tokens=False)[:budget] if block else []
        return tok.encode(tok.decode(cross + path_ids + left_ids))[-D.MAX_CTX:]
    return ids(None), [ids(b) for b in pool]


def lp_sums(model, tok, seqs, dev, bs):
    out = []
    for i in range(0, len(seqs), bs):
        out += [sum(x) for x in P.target_logprobs(model, tok, seqs[i:i + bs], dev)]
    return out


def cmd_run(a):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    import joint_dense as JD

    cands = D.load_cands(a.cands)
    ids = pilot_ids(cands, a.n, a.seed, a.n_pilot)
    exs = D.load_examples(a.data_dir, a.subset, set(ids))
    ids = [t for t in ids if t in exs]
    if a.limit:
        ids = ids[:a.limit]
    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    os.makedirs(a.out, exist_ok=True)
    scorer = JD.DenseScorer(snapshot_download(JD.RETRIEVER_REPO, revision=JD.RETRIEVER_REV))
    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    dev = model.device

    # ---- stage A: per-chunk dLL labels ------------------------------------------------
    lab_path = os.path.join(a.out, "chunks.jsonl")
    labels = {}
    if os.path.exists(lab_path):
        for r in map(json.loads, open(lab_path, encoding="utf-8")):
            labels[(r["task_id"], r["budget"])] = r
    t0 = time.time()
    with open(lab_path, "a", encoding="utf-8") as f:
        for k, t in enumerate(ids):
            ex, saved = exs[t], cands[t]["candidates"]
            gt = tok.encode(ex["gt"], add_special_tokens=False)[:P.MAX_TARGET]
            query = "\n".join([l for l in ex["left"].split("\n") if l.strip()][-QUERY_LINES:])
            for b in BUDGETS:
                if (t, b) in labels:
                    continue
                pool = pool_at(tok, ex, saved, b)
                base, per = label_prompts(tok, ex, pool, b)
                lps = lp_sums(model, tok, [(base, gt)] + [(x, gt) for x in per], dev, a.batch_size)
                dense = scorer.scores(query, pool)
                r = {"task_id": t, "budget": b, "base_lp": round(lps[0], 4), "n_gt_tokens": len(gt),
                     "cands": [{"src": c["_src"], "n_tok": len(tok.encode(D.block_str(c), add_special_tokens=False)),
                                "dll": round(lp - lps[0], 4), "dense": round(s, 5)}
                               for c, lp, s in zip(pool, lps[1:], dense)]}
                labels[(t, b)] = r
                f.write(json.dumps(r) + "\n")
                f.flush()
            if k % 10 == 0:
                print(f"labels {k}/{len(ids)} {time.time() - t0:.0f}s", flush=True)

    # ---- stage B: pack and generate -----------------------------------------------------
    def blocks(cond, t, cache):
        ex, saved = exs[t], cands[t]["candidates"]
        if cond.startswith("regen_"):
            c = cond[len("regen_"):]
            return P.blocks_for(c, tok, ex, saved, scorer, cache), P.BUDGET[c]
        b = 1536 if cond == "rand_joint1536" else int(cond.rsplit("_", 1)[1])
        pool, lab = pool_at(tok, ex, saved, b), labels[(t, b)]["cands"]
        assert len(pool) == len(lab), (t, b, len(pool), len(lab))
        if cond == "rand_joint1536":
            order = list(range(len(pool)))
            random.Random(f"{a.seed}-{t}").shuffle(order)
            return [pool[i] for i in order], b
        order = sorted(range(len(pool)), key=lambda i: -lab[i]["dll"])  # stable: ties keep pool order
        if cond.startswith("dll_pos_"):
            order = [i for i in order if lab[i]["dll"] > 0]
        return [pool[i] for i in order], b

    for cond in a.conditions.split(","):
        cdir = os.path.join(a.out, cond)
        os.makedirs(cdir, exist_ok=True)
        pred_path = os.path.join(cdir, "prediction.jsonl")
        done = set()
        if os.path.exists(pred_path):
            done = {json.loads(l)["task_id"] for l in open(pred_path, encoding="utf-8")}
        todo = [t for t in ids if t not in done]
        print(f"[{cond}] {len(done)} done, {len(todo)} to go", flush=True)
        prompts = []
        for t in todo:
            bl, b = blocks(cond, t, {})
            prompt, info = D.build_prompt(tok, exs[t], bl, max_cross=b)
            prompts.append((t, tok.encode(prompt)[-D.MAX_CTX:], info["cross_tokens"], info["infile_tokens"],
                            J.fitted_sources(tok, bl, b)))
        prompts.sort(key=lambda x: len(x[1]))
        with open(pred_path, "a", encoding="utf-8") as fout:
            for i in range(0, len(prompts), a.gen_batch_size):
                batch = prompts[i:i + a.gen_batch_size]
                L = max(len(x[1]) for x in batch)
                inp = torch.tensor([[tok.pad_token_id] * (L - len(x[1])) + x[1] for x in batch]).cuda()
                att = torch.tensor([[0] * (L - len(x[1])) + [1] * len(x[1]) for x in batch]).cuda()
                with torch.no_grad():
                    gen = model.generate(inp, attention_mask=att, max_new_tokens=D.MAX_NEW, do_sample=False,
                                         pad_token_id=tok.pad_token_id)
                gts = [tok.encode(exs[x[0]]["gt"], add_special_tokens=False)[:P.MAX_TARGET] for x in batch]
                glp = lp_sums(model, tok, [(x[1], g) for x, g in zip(batch, gts)], dev, a.batch_size)
                for (t, _, ct, it, fit), g, lp in zip(batch, gen[:, L:], glp):
                    fout.write(json.dumps({"task_id": t, "pred": tok.decode(g, skip_special_tokens=True),
                                           "gt_lp_sum": round(lp, 4), "cross_tokens": ct, "infile_tokens": it,
                                           "fitted": fit}) + "\n")
                fout.flush()
        print(f"[{cond}] done", flush=True)

# ----------------------------------------------------------------------------------
# analyze (CPU, local)
# ----------------------------------------------------------------------------------


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx and vy else None


def cmd_analyze(a):
    gt = {t: r["gt"] for t, r in P.load_jsonl(os.path.join(a.prev_dir, "gt.jsonl")).items()}
    # saved (T4) Phase-1 EM -> P0 (best fixed on all 1000) and the subset's headroom
    saved = {c: {t: D.simple_score(r["pred"], gt[t])[0]
                 for t, r in P.load_jsonl(os.path.join(a.prev_dir, c, "prediction.jsonl")).items()}
             for c in P.CONDITIONS}
    p0 = max(P.CONDITIONS, key=lambda c: sum(saved[c].values()))
    rows = {c: P.load_jsonl(os.path.join(a.out, c, "prediction.jsonl"))
            for c in CONDITIONS if os.path.exists(os.path.join(a.out, c, "prediction.jsonl"))}
    ids = sorted(set.intersection(*[set(v) for v in rows.values()]))
    n = len(ids)
    em = {c: {t: D.simple_score(rows[c][t]["pred"], gt[t])[0] for t in ids} for c in rows}
    # derived: per sample, the dll_pos budget whose final prompt gives GT the highest likelihood (uses GT)
    pos = [f"dll_pos_{b}" for b in BUDGETS if f"dll_pos_{b}" in rows]
    if pos:
        pick = {t: max(pos, key=lambda c: rows[c][t]["gt_lp_sum"]) for t in ids}
        em["dll_bestB"] = {t: em[pick[t]][t] for t in ids}
    if a.probe_out and os.path.exists(os.path.join(a.probe_out, "probe.jsonl")):
        sig = defaultdict(dict)
        for r in map(json.loads, open(os.path.join(a.probe_out, "probe.jsonl"), encoding="utf-8")):
            sig[r["task_id"]][r["cond"]] = r["gt_lp_mean"]
        order = [p0] + [c for c in P.CONDITIONS if c != p0]
        em["P1_saved"] = {t: saved[max(order, key=lambda c: sig[t][c])][t] for t in ids}
    pct = lambda d: round(100 * sum(d[t] for t in ids) / n, 2)
    p0_saved = pct(saved[p0])
    oracle = round(100 * sum(max(saved[c][t] for c in P.CONDITIONS) for t in ids) / n, 2)
    headroom = oracle - p0_saved
    ref = em.get("regen_" + p0)

    table = {}
    for c, d in em.items():
        row = {"em": pct(d)}
        if ref is not None and c != "regen_" + p0:
            w = sum(d[t] > ref[t] for t in ids)
            l_ = sum(d[t] < ref[t] for t in ids)
            row["vs_regen_P0"] = {"gained": w, "lost": l_, "mcnemar_p": D.mcnemar(w, l_),
                                  "delta_ci95": P.bootstrap_ci([d[t] - ref[t] for t in ids])}
            row["headroom_recovered"] = round((pct(d) - pct(ref)) / headroom, 3) if headroom else None
        if c in rows:
            row["mean_cross_tokens"] = round(sum(rows[c][t]["cross_tokens"] for t in ids) / n, 1)
            row["mean_infile_blocks"] = round(sum(rows[c][t]["fitted"].get("infile", 0) for t in ids) / n, 2)
            row["mean_cross_blocks"] = round(sum(rows[c][t]["fitted"].get("cross", 0) for t in ids) / n, 2)
        table[c] = row
    regen_agree = {c: round(sum(em["regen_" + c][t] == saved[c][t] for t in ids) / n, 3)
                   for c in P.CONDITIONS if "regen_" + c in em}

    # chunk-label statistics
    stats = {}
    labs = [r for r in map(json.loads, open(os.path.join(a.out, "chunks.jsonl"), encoding="utf-8"))
            if r["task_id"] in set(ids)]
    for b in BUDGETS:
        lb = [r for r in labs if r["budget"] == b]
        s = {}
        for src in ("cross", "infile"):
            cs = [c for r in lb for c in r["cands"] if c["src"] == src]
            rho = [spearman([c["dense"] for c in r["cands"] if c["src"] == src],
                            [c["dll"] for c in r["cands"] if c["src"] == src]) for r in lb]
            rho = [x for x in rho if x is not None]
            s[src] = {"n_chunks": len(cs), "mean_per_sample": round(len(cs) / max(len(lb), 1), 2),
                      "pct_dll_pos": round(100 * sum(c["dll"] > 0 for c in cs) / max(len(cs), 1), 1),
                      "mean_dll": round(sum(c["dll"] for c in cs) / max(len(cs), 1), 3),
                      "mean_within_sample_spearman_dense_vs_dll": round(sum(rho) / len(rho), 3) if rho else None}
        both = [spearman([c["dense"] for c in r["cands"]], [c["dll"] for c in r["cands"]]) for r in lb]
        both = [x for x in both if x is not None]
        s["all_spearman_dense_vs_dll"] = round(sum(both) / len(both), 3) if both else None
        s["pct_samples_top_dll_is_infile"] = round(
            100 * sum(max(r["cands"], key=lambda c: c["dll"])["src"] == "infile" for r in lb if r["cands"])
            / max(sum(1 for r in lb if r["cands"]), 1), 1)
        s["pct_samples_no_positive_chunk"] = round(
            100 * sum(all(c["dll"] <= 0 for c in r["cands"]) for r in lb) / max(len(lb), 1), 1)
        stats[f"B{b}"] = s

    prim = table.get(PRIMARY, {})
    rec = prim.get("headroom_recovered")
    decision = (None if rec is None else
                "PASS (this dataset): dll_pos_1536 recovers >= 50% of the subset headroom" if rec >= 0.5 else
                "FAIL (this dataset): dll_pos_1536 recovers < 50% of the subset headroom")
    res = {"dataset": a.dataset, "n": n, "P0": p0, "subset_saved": {"P0_em": p0_saved, "oracle_em": oracle,
                                                                     "headroom": round(headroom, 2)},
           "regen_em_agreement_with_saved": regen_agree, "conditions": table, "chunk_stats": stats,
           "primary": PRIMARY, "decision": decision,
           "notes": ("simple EM. dLL labels and dll_bestB / P1_saved use the ground truth (upper bounds, not "
                     "methods). Comparisons are against regen_P0 (same GPU); headroom = saved-oracle minus saved-P0 "
                     "on the same subset. n is small: read CIs, not p-values.")}
    with open(os.path.join(a.out, "pilot_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--prev_dir", help="probe_inputs/<dataset>")
    ap.add_argument("--probe_out", default=None, help="Phase-1 output dir with probe.jsonl (for P1 on the subset)")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="pilot_out")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_pilot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first k pilot ids only (smoke test)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=8, help="sequences per scoring forward pass")
    ap.add_argument("--gen_batch_size", type=int, default=8)
    a = ap.parse_args()
    D.DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
