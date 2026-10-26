#!/usr/bin/env python3
"""E0: does chunk ORDER matter once the SET of chunks is fixed?

In the Phase-2 pilot (chunk_pilot.py) the pool is sorted and then packed until the budget is full, so "sorted by dLL"
vs "random order" changes WHICH chunks enter the prompt, not only where they sit. E0 removes that confound
(ANALYSIS_2026-10-24_direction.md, step E0).

For each pilot sample and budget B, the fixed set S_B = the chunks of dll_pos_B (dLL > 0, sorted by dLL desc) that fit
WHOLE into B (same rule as joint_retrieval.fitted_sources). Only the order of S_B changes; the prompt is built with
max_cross = B + SLACK so that no permutation truncates a chunk. In build_prompt the first block sits at the TOP of the
prompt (farthest from the cursor) and the last block right above "# file path" + in-file code.

Conditions (per B in 512, 1024, 1536):
    set_desc_B     dLL descending: best chunk farthest from cursor (the pilot's order)
    set_asc_B      dLL ascending: best chunk nearest the cursor (RepoBench "L2H")
    set_rand_B     random permutation (seed a)
    set_rand2_B    random permutation (seed b): rand vs rand2 is the noise floor of re-ordering
    set_infnat_B   cross-file chunks by dLL desc, then in-file chunks in original file order (contiguous with the
                   visible left context)
    repro_dll_pos_1536   chunk_pilot's dll_pos_1536 rebuilt here: must match the saved pilot predictions (anchor)

Decision (fixed before running), primary budget B=1536, per dataset, on samples with |S_B| >= 2:
    ORDER MATTERS  if max EM gap among {desc, asc, rand} >= 3.0 points, McNemar p < 0.05, AND that gap exceeds the
                   rand-vs-rand2 gap by >= 2 points.
    Otherwise ORDER IS AN ABLATION (no position-aware claim). Mean GT log-likelihood differences are reported as a
    secondary, more sensitive signal.

usage:
    python e0_order.py run     --dataset repoeval --cands prediction_with_candidates.jsonl \
                               --labels pilot_out/chunks.jsonl --out e0_out
    python e0_order.py analyze --dataset repoeval --out modal_out/e0_repoeval --prev_dir probe_inputs/repoeval \
                               --pilot_dir modal_out/pilot_repoeval
"""
import argparse
import json
import os
import random
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chunk_pilot as C  # noqa: E402
import dep_oracle as D  # noqa: E402
import joint_retrieval as J  # noqa: E402
import likelihood_probe as P  # noqa: E402

BUDGETS = [512, 1024, 1536]
PRIMARY_B = 1536
SLACK = 32  # extra cross tokens so that joining whole chunks in any order never cuts the last one
ORDERS = ["desc", "asc", "rand", "rand2", "infnat"]
CONDITIONS = [f"set_{o}_{b}" for b in BUDGETS for o in ORDERS] + ["repro_dll_pos_1536"]


def fixed_set(tok, pool, lab, budget):
    """Indices (dLL desc) of the dll_pos chunks that fit whole into `budget`."""
    order = [i for i in sorted(range(len(pool)), key=lambda i: -lab[i]["dll"]) if lab[i]["dll"] > 0]
    used, keep = 0, []
    for i in order:
        n = len(tok.encode(D.block_str(pool[i]), add_special_tokens=False)) + 2
        if used + n > budget:
            break
        used += n
        keep.append(i)
    return keep


def arrange(order_name, keep, pool, lab, t, seed):
    if order_name == "desc":
        return keep
    if order_name == "asc":
        return keep[::-1]
    if order_name in ("rand", "rand2"):
        k = list(keep)
        random.Random(f"{seed}-{order_name}-{t}").shuffle(k)
        return k
    if order_name == "infnat":  # in-file blocks are named "in-file block <k>" in file order
        cross = [i for i in keep if pool[i]["_src"] == "cross"]
        inf = sorted((i for i in keep if pool[i]["_src"] == "infile"),
                     key=lambda i: int(pool[i]["description"].rsplit(" ", 1)[1]))
        return cross + inf
    raise ValueError(order_name)


def cmd_run(a):
    import torch
    from transformers import AutoModelForCausalLM

    cands = D.load_cands(a.cands)
    ids = C.pilot_ids(cands, a.n, a.seed, a.n_pilot)
    labels = {}
    for r in map(json.loads, open(a.labels, encoding="utf-8")):
        labels[(r["task_id"], r["budget"])] = r
    exs = D.load_examples(a.data_dir, a.subset, set(ids))
    ids = [t for t in ids if t in exs and all((t, b) in labels for b in BUDGETS)]
    if a.limit:
        ids = ids[:a.limit]
    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    os.makedirs(a.out, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    dev = model.device

    pools = {}
    for t in ids:
        for b in BUDGETS:
            pool, lab = C.pool_at(tok, exs[t], cands[t]["candidates"], b), labels[(t, b)]["cands"]
            assert len(pool) == len(lab), (t, b, len(pool), len(lab))
            pools[(t, b)] = (pool, lab, fixed_set(tok, pool, lab, b))

    def blocks(cond, t):
        if cond == "repro_dll_pos_1536":  # exactly chunk_pilot's packing (may cut the last chunk)
            pool, lab, _ = pools[(t, 1536)]
            order = [i for i in sorted(range(len(pool)), key=lambda i: -lab[i]["dll"]) if lab[i]["dll"] > 0]
            return [pool[i] for i in order], 1536, order
        _, o, b = cond.split("_")
        b = int(b)
        pool, lab, keep = pools[(t, b)]
        order = arrange(o, keep, pool, lab, t, a.seed)
        return [pool[i] for i in order], b + SLACK, order

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
            bl, mc, order = blocks(cond, t)
            prompt, info = D.build_prompt(tok, exs[t], bl, max_cross=mc)
            prompts.append((t, tok.encode(prompt)[-D.MAX_CTX:], info["cross_tokens"], info["infile_tokens"],
                            [pools[(t, 1536 if cond.startswith("repro") else int(cond.rsplit("_", 1)[1]))][0][i]["_src"]
                             for i in order]))
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
                glp = C.lp_sums(model, tok, [(x[1], g) for x, g in zip(batch, gts)], dev, a.batch_size)
                for (t, _, ct, it, srcs), g, lp in zip(batch, gen[:, L:], glp):
                    fout.write(json.dumps({"task_id": t, "pred": tok.decode(g, skip_special_tokens=True),
                                           "gt_lp_sum": round(lp, 4), "cross_tokens": ct, "infile_tokens": it,
                                           "n_set": len(srcs), "srcs": srcs}) + "\n")
                fout.flush()
        print(f"[{cond}] done", flush=True)


# ----------------------------------------------------------------------------------
# analyze (CPU, local)
# ----------------------------------------------------------------------------------

def mean_ci(diff, iters=2000, seed=0):
    rng = random.Random(seed)
    n = len(diff)
    ms = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return [round(sum(diff) / n, 3), [round(ms[int(0.025 * iters)], 3), round(ms[int(0.975 * iters)], 3)]]


def cmd_analyze(a):
    gt = {t: r["gt"] for t, r in P.load_jsonl(os.path.join(a.prev_dir, "gt.jsonl")).items()}
    rows = {c: P.load_jsonl(os.path.join(a.out, c, "prediction.jsonl"))
            for c in CONDITIONS if os.path.exists(os.path.join(a.out, c, "prediction.jsonl"))}
    ids = sorted(set.intersection(*[set(v) for v in rows.values()]))
    em = {c: {t: D.simple_score(rows[c][t]["pred"], gt[t])[0] for t in ids} for c in rows}
    res = {"dataset": a.dataset, "n": len(ids), "budgets": {}}

    if "repro_dll_pos_1536" in rows and a.pilot_dir:
        saved = P.load_jsonl(os.path.join(a.pilot_dir, "dll_pos_1536", "prediction.jsonl"))
        common = [t for t in ids if t in saved]
        res["anchor_repro_dll_pos_1536"] = {
            "em": round(100 * sum(em["repro_dll_pos_1536"][t] for t in common) / len(common), 2),
            "saved_pilot_em": round(100 * sum(D.simple_score(saved[t]["pred"], gt[t])[0] for t in common)
                                    / len(common), 2),
            "pred_agreement": round(sum(rows["repro_dll_pos_1536"][t]["pred"] == saved[t]["pred"] for t in common)
                                    / len(common), 3)}

    for b in BUDGETS:
        conds = {o: f"set_{o}_{b}" for o in ORDERS if f"set_{o}_{b}" in rows}
        if not conds:
            continue
        sizes = {t: rows[conds["desc"]][t]["n_set"] for t in ids}
        multi = [t for t in ids if sizes[t] >= 2]
        has_inf = [t for t in multi if "infile" in rows[conds["desc"]][t]["srcs"]]
        out = {"mean_set_size": round(sum(sizes.values()) / len(ids), 2), "n_set_ge2": len(multi),
               "n_set_ge2_with_infile": len(has_inf), "all": {}, "set_ge2": {}}
        for name, sub in (("all", ids), ("set_ge2", multi)):
            if not sub:
                continue
            blk = {"em": {o: round(100 * sum(em[c][t] for t in sub) / len(sub), 2) for o, c in conds.items()},
                   "mean_gt_lp": {o: round(sum(rows[c][t]["gt_lp_sum"] for t in sub) / len(sub), 3)
                                  for o, c in conds.items()},
                   "pairs": {}}
            for o1, o2 in combinations(conds, 2):
                c1, c2 = conds[o1], conds[o2]
                w = sum(em[c1][t] > em[c2][t] for t in sub)
                l_ = sum(em[c1][t] < em[c2][t] for t in sub)
                blk["pairs"][f"{o1}-{o2}"] = {
                    "em_delta": round(100 * (w - l_) / len(sub), 2), "w": w, "l": l_, "mcnemar_p": D.mcnemar(w, l_),
                    "gt_lp_delta_mean_ci": mean_ci([rows[c1][t]["gt_lp_sum"] - rows[c2][t]["gt_lp_sum"]
                                                    for t in sub])}
            out[name] = blk
        if b == PRIMARY_B and out["set_ge2"]:
            blk = out["set_ge2"]
            main = {k: v for k, v in blk["pairs"].items()
                    if set(k.split("-")) <= {"desc", "asc", "rand"}}
            k_best = max(main, key=lambda k: abs(main[k]["em_delta"]))
            gap, p = abs(main[k_best]["em_delta"]), main[k_best]["mcnemar_p"]
            noise = abs(blk["pairs"].get("rand-rand2", {}).get("em_delta", 0.0))
            res["decision"] = {
                "largest_pair": k_best, "em_gap": gap, "mcnemar_p": p, "rand_vs_rand2_gap": noise,
                "verdict": ("ORDER MATTERS" if gap >= 3.0 and p < 0.05 and gap - noise >= 2.0
                            else "ORDER IS AN ABLATION (no position-aware claim)")}
        res["budgets"][f"B{b}"] = out
    res["notes"] = ("simple EM; sets come from GT-based dLL labels (upper-bound sets, not a method). Decision rule "
                    "fixed in the docstring before running. n is small: read CIs.")
    with open(os.path.join(a.out, "e0_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--labels", help="chunk_pilot chunks.jsonl (dLL labels)")
    ap.add_argument("--prev_dir", help="probe_inputs/<dataset>")
    ap.add_argument("--pilot_dir", default=None, help="pilot output dir (for the repro anchor)")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="e0_out")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_pilot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
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
