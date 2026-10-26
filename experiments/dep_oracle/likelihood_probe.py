#!/usr/bin/env python3
"""Likelihood probe: does the generator's likelihood of the ground truth tell which prompt construction helps?

See PLAN_likelihood_probe.md. No generation for the main measurement, forward passes only; EM per construction
comes from the predictions saved by joint_dense.py / joint_retrieval.py.

Constructions (same 1000-sample subset, seed 0):
    align1536                AlignCoder saved top-10, budget 1536
    align512                 AlignCoder saved top-10, budget 512
    dense_joint1536          hidden in-file chunks + saved top-10, ranked jointly by AlignRetriever, budget 1536
    dense_infile_then_cross  best hidden in-file chunks (<= 512 tokens) first, then AlignCoder order

Subcommands
    prepare  CPU, local. Copy the saved predictions / prompt_meta / gt of the 4 constructions into
             probe_inputs/<dataset>/ (this folder goes into the Kaggle dataset).
    score    GPU, Kaggle. Rebuild every prompt exactly as before, then
               gate : (1) cross/in-file token counts (and fitted blocks for dense) must match the saved
                          prompt_meta for every sample,
                      (2) teacher-forced greedy agreement: on --gate_n samples per construction, the share of
                          saved-prediction tokens that are still the argmax under the rebuilt prompt must be
                          >= GATE_MIN_AGREE. Fails -> stop (unless --force).
                      Exact greedy re-generation is reported but not required: the saved runs used T4,
                      batch 8, left padding; other GPUs / batch 1 flip near-ties and the text then diverges.
                      Control: the same agreement under align1536 with the cross-file blocks reversed.
               A    : log-probs of the ground-truth tokens after the prompt   (uses GT: probe only)
               B    : log-probs of the saved greedy prediction after the prompt (no GT: self-confidence)
             Writes gate.json and probe.jsonl (resumable).
    analyze  CPU, local. Per-sample selection policies, McNemar vs best fixed construction, recovered share of
             the oracle headroom, pairwise ranking accuracy, bootstrap CI, H1/H2 decision.

usage:
    python likelihood_probe.py prepare --dataset repoeval
    python likelihood_probe.py score   --dataset repoeval --cands prediction_with_candidates.jsonl \
                                       --prev_dir probe_inputs/repoeval --out probe
    python likelihood_probe.py analyze --dataset repoeval --prev_dir probe_inputs/repoeval --out probe
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dep_oracle as D  # noqa: E402
from budget_sweep import first_line_conf  # noqa: E402

CONDITIONS = ["align1536", "align512", "dense_joint1536", "dense_infile_then_cross"]
BUDGET = {"align1536": 1536, "align512": 512, "dense_joint1536": 1536, "dense_infile_then_cross": 1536}
# where each construction's saved run lives (relative to this folder), see HANDOFF.md
PREV_SOURCES = {"align1536": "kaggle_dense_{ds}_out/dense/align1536",
                "align512": "kaggle_joint_{ds}_out/joint/align512",
                "dense_joint1536": "kaggle_dense_{ds}_out/dense/dense_joint1536",
                "dense_infile_then_cross": "kaggle_dense_{ds}_out/dense/dense_infile_then_cross"}
ANCHORS = {"repoeval": {"align1536": 36.8, "align512": 40.5}, "cceval": {"align1536": 28.5, "align512": 22.9}}
MAX_TARGET = 256      # target tokens scored per sequence (GT can be long; flagged when cut)
GATE_MIN_AGREE = 0.97  # mean per-token argmax agreement; near-tie flips across GPUs stay far below 3%
# H1 thresholds, fixed before running (PLAN_likelihood_probe.md)
H1_GO, H1_WEAK, PAIR_GO, PAIR_NO = 0.5, 0.25, 0.65, 0.55


def load_jsonl(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            out[r["task_id"]] = r  # resumed runs may repeat a task: last line wins (as joint_retrieval.analyze)
    return out

# ----------------------------------------------------------------------------------
# prepare (local)
# ----------------------------------------------------------------------------------


def cmd_prepare(a):
    here = os.path.dirname(os.path.abspath(__file__))
    dst = os.path.join(here, "probe_inputs", a.dataset)
    for c in CONDITIONS:
        src = os.path.join(here, PREV_SOURCES[c].format(ds=a.dataset))
        os.makedirs(os.path.join(dst, c), exist_ok=True)
        for fn in ("prediction.jsonl", "prompt_meta.jsonl"):
            shutil.copyfile(os.path.join(src, fn), os.path.join(dst, c, fn))
    shutil.copyfile(os.path.join(here, f"kaggle_dense_{a.dataset}_out/dense/gt.jsonl"), os.path.join(dst, "gt.jsonl"))
    print(f"wrote {dst}")

# ----------------------------------------------------------------------------------
# score (Kaggle, GPU)
# ----------------------------------------------------------------------------------


def blocks_for(cond, tok, ex, saved, scorer, cache):
    import joint_dense as JD
    if cond in ("align1536", "align512"):
        return saved  # untouched, as joint_retrieval.blocks_for
    return JD.blocks_for(cond, tok, ex, saved, scorer, cache)


def target_logprobs(model, tok, seqs, device, with_argmax=False):
    """seqs: list of (prompt_ids, target_ids). Returns per-target-token log-probs for each sequence.

    Right padding keeps real-token positions unchanged; only target positions go through the LM head.
    with_argmax: also return, per target token, whether it is the argmax (teacher-forced greedy agreement).
    """
    import torch
    L = max(len(p) + len(t) for p, t in seqs)
    pad = tok.pad_token_id
    inp = torch.tensor([p + t + [pad] * (L - len(p) - len(t)) for p, t in seqs], device=device)
    att = torch.tensor([[1] * (len(p) + len(t)) + [0] * (L - len(p) - len(t)) for p, t in seqs], device=device)
    with torch.no_grad():
        h = model.base_model(input_ids=inp, attention_mask=att).last_hidden_state
        head = model.get_output_embeddings()
        out = []
        for j, (p, t) in enumerate(seqs):
            if not t:
                out.append(([], []) if with_argmax else [])
                continue
            pos = torch.arange(len(p) - 1, len(p) + len(t) - 1, device=device)  # position i predicts i+1
            lp = torch.log_softmax(head(h[j, pos]).float(), dim=-1)
            tgt = torch.tensor(t, device=device)
            v = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).tolist()
            out.append((v, (lp.argmax(-1) == tgt).tolist()) if with_argmax else v)
    return out


def lp_stats(tok, ids, lps, prefix):
    pieces = [tok.decode([i]) for i in ids]
    n = len(lps)
    return {f"{prefix}_lp_sum": round(sum(lps), 4) if n else 0.0,
            f"{prefix}_lp_mean": round(sum(lps) / n, 4) if n else -10.0,
            f"{prefix}_lp_first": round(first_line_conf(pieces, lps), 4),
            f"n_{prefix}_tokens": n}


def cmd_score(a):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    import joint_dense as JD
    import joint_retrieval as J

    conds = a.conditions.split(",")
    cands = D.load_cands(a.cands)
    ids = D.pick_subset(list(cands), a.n, a.seed)
    exs = D.load_examples(a.data_dir, a.subset, set(ids))
    ids = [t for t in ids if t in exs]
    if a.limit:  # smoke test: first k of the SAME subset, so the gate still compares against saved runs
        ids = ids[:a.limit]
    prev = {c: {"pred": load_jsonl(os.path.join(a.prev_dir, c, "prediction.jsonl")),
                "meta": load_jsonl(os.path.join(a.prev_dir, c, "prompt_meta.jsonl"))} for c in conds}
    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    os.makedirs(a.out, exist_ok=True)

    # 1) rebuild prompts exactly as the saved runs did
    scorer = JD.DenseScorer(snapshot_download(JD.RETRIEVER_REPO, revision=JD.RETRIEVER_REV)) \
        if any(c.startswith("dense") for c in conds) else None
    prompts, mismatch, t0 = {}, defaultdict(list), time.time()
    for k, t in enumerate(ids):
        cache = {}
        for c in conds:
            bl = blocks_for(c, tok, exs[t], cands[t]["candidates"], scorer, cache)
            prompt, info = D.build_prompt(tok, exs[t], bl, max_cross=BUDGET[c])
            prompts[(t, c)] = tok.encode(prompt)[-D.MAX_CTX:]
            m = prev[c]["meta"].get(t)
            if m is None or (m["cross_tokens"], m["infile_tokens"]) != (info["cross_tokens"], info["infile_tokens"]):
                mismatch[c].append(t)
            elif c.startswith("dense") and m.get("fitted") != J.fitted_sources(tok, bl, BUDGET[c]):
                mismatch[c].append(t)
        if k % 100 == 0:
            print(f"prompts {k}/{len(ids)} {time.time() - t0:.0f}s", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    dev = model.device

    # 2) gate
    gate_path = os.path.join(a.out, "gate.json")
    gate = {"n": len(ids), "token_count_mismatch": {c: len(mismatch[c]) for c in conds},
            "token_count_mismatch_examples": {c: mismatch[c][:5] for c in conds},
            "argmax_agreement": {}, "regen_exact": {}, "regen_em_agrees": {}}
    rng = random.Random(a.seed)
    sample = rng.sample(ids, min(a.gate_n, len(ids)))

    def agreement(pairs):
        agree = []
        for i in range(0, len(pairs), a.batch_size):
            for _, m in target_logprobs(model, tok, pairs[i:i + a.batch_size], dev, with_argmax=True):
                agree += m
        return round(sum(agree) / max(len(agree), 1), 4)

    pred_ids = lambda c, t: tok.encode(prev[c]["pred"][t]["pred"], add_special_tokens=False)[:MAX_TARGET]
    for c in conds:
        gate["argmax_agreement"][c] = agreement([(prompts[(t, c)], pred_ids(c, t)) for t in sample])
        exact = em_same = 0
        for t in sample[:a.gate_regen_n]:
            x = torch.tensor([prompts[(t, c)]], device=dev)
            with torch.no_grad():
                g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=D.MAX_NEW,
                                   do_sample=False, pad_token_id=tok.pad_token_id)
            new, old = tok.decode(g[0, x.size(1):], skip_special_tokens=True), prev[c]["pred"][t]["pred"]
            exact += new == old
            em_same += D.simple_score(new, exs[t]["gt"])[0] == D.simple_score(old, exs[t]["gt"])[0]
        k = max(min(a.gate_regen_n, len(sample)), 1)
        gate["regen_exact"][c], gate["regen_em_agrees"][c] = round(exact / k, 3), round(em_same / k, 3)
    if "align1536" in conds:  # sensitivity control: a wrong prompt (cross-file block order reversed)
        ctrl = []
        for t in sample:
            bl = J.saved_blocks(cands[t]["candidates"])[::-1]
            p, _ = D.build_prompt(tok, exs[t], bl, max_cross=BUDGET["align1536"])
            ctrl.append((tok.encode(p)[-D.MAX_CTX:], pred_ids("align1536", t)))
        gate["control_reversed_blocks_align1536"] = agreement(ctrl)
    gate["pass"] = (all(v == 0 for v in gate["token_count_mismatch"].values())
                    and all(v >= GATE_MIN_AGREE for v in gate["argmax_agreement"].values()))
    with open(gate_path, "w") as f:
        json.dump(gate, f, indent=2)
    print(json.dumps(gate, indent=2), flush=True)
    if not gate["pass"] and not a.force:
        sys.exit("gate failed: prompts do not reproduce the saved runs; not scoring (see gate.json)")

    # 3) forward A (GT) and B (saved prediction), one task (all constructions) per step
    out_path = os.path.join(a.out, "probe.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {(r["task_id"], r["cond"]) for r in map(json.loads, open(out_path, encoding="utf-8"))}
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for k, t in enumerate(ids):
            todo = [c for c in conds if (t, c) not in done]
            if not todo:
                continue
            gt_ids = tok.encode(exs[t]["gt"], add_special_tokens=False)
            rows, seqs = [], []
            for c in todo:
                pred_ids = tok.encode(prev[c]["pred"][t]["pred"], add_special_tokens=False)
                rows.append((c, gt_ids[:MAX_TARGET], pred_ids[:MAX_TARGET]))
                seqs += [(prompts[(t, c)], gt_ids[:MAX_TARGET]), (prompts[(t, c)], pred_ids[:MAX_TARGET])]
            lps = []
            for i in range(0, len(seqs), a.batch_size):
                lps += target_logprobs(model, tok, seqs[i:i + a.batch_size], dev)
            for j, (c, g, p) in enumerate(rows):
                r = {"task_id": t, "cond": c, "prompt_len": len(prompts[(t, c)]),
                     "gt_truncated": len(gt_ids) > MAX_TARGET}
                r.update(lp_stats(tok, g, lps[2 * j], "gt"))
                r.update(lp_stats(tok, p, lps[2 * j + 1], "pred"))
                fout.write(json.dumps(r) + "\n")
            fout.flush()
            if k % 50 == 0:
                print(f"forward {k}/{len(ids)} {time.time() - t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------------
# analyze (CPU, local)
# ----------------------------------------------------------------------------------

SIGNALS = {"P1_gt_lp_mean": "gt_lp_mean", "P1b_gt_lp_sum": "gt_lp_sum", "P1c_gt_lp_first": "gt_lp_first",
           "P2_pred_conf_first": "pred_lp_first", "P2b_pred_conf_mean": "pred_lp_mean"}
USES_GT = {"P1_gt_lp_mean", "P1b_gt_lp_sum", "P1c_gt_lp_first", "P3_oracle"}


def bootstrap_ci(diff, iters=2000, seed=0):
    rng = random.Random(seed)
    n = len(diff)
    means = sorted(sum(diff[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return [round(100 * means[int(0.025 * iters)], 2), round(100 * means[int(0.975 * iters)], 2)]


def cmd_analyze(a):
    conds = a.conditions.split(",")
    gt = {t: r["gt"] for t, r in load_jsonl(os.path.join(a.prev_dir, "gt.jsonl")).items()}
    em = {c: {t: D.simple_score(r["pred"], gt[t])[0]
              for t, r in load_jsonl(os.path.join(a.prev_dir, c, "prediction.jsonl")).items() if t in gt}
          for c in conds}
    sig = defaultdict(dict)  # (task, cond) -> row
    for l in open(os.path.join(a.out, "probe.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        sig[r["task_id"]][r["cond"]] = r
    ids = sorted(t for t in gt if all(t in em[c] for c in conds) and all(c in sig.get(t, {}) for c in conds))
    n = len(ids)
    pct = lambda xs: round(100 * sum(xs) / n, 2)

    fixed = {c: pct([em[c][t] for t in ids]) for c in conds}
    p0 = max(conds, key=lambda c: fixed[c])  # ties -> earlier in CONDITIONS
    base = [em[p0][t] for t in ids]
    oracle = [max(em[c][t] for c in conds) for t in ids]
    headroom = sum(oracle) - sum(base)

    def evaluate(choice):
        e = [em[choice[t]][t] for t in ids]
        w = sum(x > y for x, y in zip(e, base))
        l_ = sum(x < y for x, y in zip(e, base))
        mix = defaultdict(int)
        for t in ids:
            mix[choice[t]] += 1
        return {"em": pct(e), "vs_P0": {"gained": w, "lost": l_, "mcnemar_p": D.mcnemar(w, l_),
                                        "delta_ci95": bootstrap_ci([x - y for x, y in zip(e, base)])},
                "headroom_recovered": round((sum(e) - sum(base)) / headroom, 3) if headroom else None,
                "choice_mix": dict(mix)}

    # ties broken towards the best fixed construction, then CONDITIONS order
    order = [p0] + [c for c in conds if c != p0]
    pol = {"P0_fixed_" + p0: evaluate({t: p0 for t in ids})}
    pairwise = {}
    for name, key in SIGNALS.items():
        pol[name] = evaluate({t: max(order, key=lambda c: sig[t][c][key]) for t in ids})
        # pairwise accuracy: among (sample, i<j) with different EM, how often the signal ranks the EM=1 one higher
        right = tot = 0
        by_pair = defaultdict(lambda: [0, 0])
        for t in ids:
            for i, ci in enumerate(conds):
                for cj in conds[i + 1:]:
                    if em[ci][t] == em[cj][t]:
                        continue
                    good, bad = (ci, cj) if em[ci][t] > em[cj][t] else (cj, ci)
                    ok = sig[t][good][key] > sig[t][bad][key]
                    right += ok
                    tot += 1
                    by_pair[f"{ci}|{cj}"][0] += ok
                    by_pair[f"{ci}|{cj}"][1] += 1
        pairwise[name] = {"acc": round(right / tot, 3) if tot else None, "n_pairs": tot,
                          "by_pair": {k: {"acc": round(v[0] / v[1], 3), "n": v[1]} for k, v in by_pair.items()}}
    pol["P3_oracle"] = evaluate({t: max(order, key=lambda c: em[c][t]) for t in ids})

    # decisions (thresholds from PLAN_likelihood_probe.md; one dataset per call, combine both by hand)
    p1, p2 = pol["P1_gt_lp_mean"], pol["P2_pred_conf_first"]
    rec, acc = p1["headroom_recovered"] or 0.0, pairwise["P1_gt_lp_mean"]["acc"] or 0.0
    if rec >= H1_GO and p1["vs_P0"]["mcnemar_p"] < 0.05 and acc >= PAIR_GO:
        h1 = "GO (on this dataset): >=50% headroom, p<0.05, pairwise >=65%"
    elif rec < H1_WEAK or acc <= PAIR_NO:
        h1 = "NO: <25% headroom or pairwise <=55% -> do not train a likelihood reranker"
    else:
        h1 = "WEAK: run the 200-sample chunk-level pilot before deciding"
    h2 = ("P2 within 1 EM of P1 -> self-confidence is the baseline to beat" if p2["em"] >= p1["em"] - 1
          else "P1 > P2 + 1 EM -> GT likelihood carries signal self-confidence lacks")

    anchors = {c: {"expected": v, "got": fixed.get(c), "ok": fixed.get(c) == v}
               for c, v in ANCHORS[a.dataset].items() if c in fixed}
    trunc = sum(sig[t][c].get("gt_truncated", False) for t in ids for c in conds)
    res = {"dataset": a.dataset, "n": n, "anchors": anchors, "fixed_em": fixed, "P0": p0,
           "oracle_headroom_em": round(100 * headroom / n, 2), "policies": pol, "pairwise": pairwise,
           "H1": h1, "H2": h2, "gt_truncated_rows": trunc,
           "notes": ("simple EM (first k lines) of the SAVED predictions; selection signals from probe.jsonl. "
                     "P1* use the ground truth (probe = upper bound for a likelihood-trained selector), "
                     "P2* do not. Ties go to the best fixed construction. Bootstrap CI: 2000 resamples, seed 0.")}
    if not all(v["ok"] for v in anchors.values()):
        res["WARNING"] = "anchor EM does not reproduce: wrong inputs, do not read the results"
    with open(os.path.join(a.out, "probe_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "pairwise"}, indent=2))
    print(json.dumps({k: {"acc": v["acc"], "n_pairs": v["n_pairs"]} for k, v in pairwise.items()}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "score", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--prev_dir", help="probe_inputs/<dataset> (written by `prepare`)")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="probe_out")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    ap.add_argument("--batch_size", type=int, default=4, help="sequences per forward pass")
    ap.add_argument("--gate_n", type=int, default=50, help="samples per construction for argmax agreement")
    ap.add_argument("--gate_regen_n", type=int, default=20, help="of those, re-generated (diagnostic only)")
    ap.add_argument("--limit", type=int, default=0, help="score only the first k ids of the subset (smoke test)")
    ap.add_argument("--force", action="store_true", help="score even if the gate fails (debug only)")
    a = ap.parse_args()
    D.DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    {"prepare": cmd_prepare, "score": cmd_score, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
