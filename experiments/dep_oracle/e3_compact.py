#!/usr/bin/env python3
"""E3: compact the IN-FILE context so the fixed window holds more of it (training-free).

F2: on RepoEval the left context averages ~3.9k tokens but only ~0.46k fit next to 1536 cross-file tokens, and the
cross-file budget crowds it out. Instead of choosing between in-file and cross-file, keep more in-file information
per token. Cross-file part = AlignCoder's saved candidates, exactly as align512 / align1536 (budget_sweep b512/b1536).

Conditions (per B in 512, 1536; 1000-sample dev subset, greedy, as every earlier run):
    base_B        D.build_prompt unchanged (re-generated here for paired comparison)
    strip_B       drop blank lines and full-line '#' comments from the left context, except its last TAIL lines
    skel_B        the in-file window W starts with a skeleton of the HIDDEN part (import lines + def/class header lines,
                  at most W/4 tokens: imports first, then the headers nearest the cursor), the rest is the verbatim tail
    stripskel_B   strip, then skel

Decision (fixed before running), per dataset, vs base_B at the same B:
    a variant HELPS if EM gain >= 1.5 points with McNemar p < 0.05 at some B and no loss >= 1.5 at the other B.

usage:
    python e3_compact.py run     --dataset repoeval --cands prediction_with_candidates.jsonl --out e3
    python e3_compact.py analyze --dataset repoeval --out modal_out/e3_repoeval --prev_dir probe_inputs/repoeval
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dep_oracle as D  # noqa: E402
import likelihood_probe as P  # noqa: E402

BUDGETS = [512, 1536]
VARIANTS = ["base", "strip", "skel", "stripskel"]
TAIL = 10
IMPORT_RE = re.compile(r"^\s*(import|from)\s+\S")
HEADER_RE = re.compile(r"^\s*(async\s+def|def|class)\s+\w")


def strip_left(left):
    lines = left.split("\n")
    head, tail = lines[:-TAIL], lines[-TAIL:]
    head = [l for l in head if l.strip() and not l.lstrip().startswith("#")]
    return "\n".join(head + tail)


def skeleton(hidden, tok, cap):
    lines = hidden.split("\n")
    imps = [l for l in lines if IMPORT_RE.match(l)]
    heads = [(i, l) for i, l in enumerate(lines) if HEADER_RE.match(l)]
    n = lambda s: len(tok.encode(s + "\n", add_special_tokens=False))
    keep_imp, used = [], 0
    for l in imps:
        if used + n(l) > cap // 2:
            break
        keep_imp.append(l)
        used += n(l)
    keep_head = []
    for i, l in reversed(heads):  # nearest the cursor first
        if used + n(l) > cap:
            break
        keep_head.append((i, l))
        used += n(l)
    return "\n".join(keep_imp + [l for _, l in sorted(keep_head)])


def compact_ex(tok, ex, saved, budget, variant):
    left = strip_left(ex["left"]) if variant in ("strip", "stripskel") else ex["left"]
    e2 = dict(ex, left=left)
    if variant in ("skel", "stripskel"):
        _, info = D.build_prompt(tok, e2, saved, max_cross=budget)
        path_ids = tok.encode(f"\n\n# file path: {ex['path']}\n\n", add_special_tokens=False)
        allowed = D.MAX_CTX - (info["cross_tokens"] + len(path_ids) + 10)
        left_ids = tok.encode(left, add_special_tokens=False)
        if len(left_ids) > allowed:  # something is hidden: spend up to W/4 on its skeleton
            sk = skeleton(tok.decode(left_ids[:-allowed]), tok, allowed // 4)
            if sk:
                sk_ids = tok.encode(sk + "\n", add_special_tokens=False)
                tail_ids = left_ids[-(allowed - len(sk_ids)):]
                e2 = dict(ex, left=sk + "\n" + tok.decode(tail_ids))
    return e2


def cmd_run(a):
    import torch
    from transformers import AutoModelForCausalLM

    cands = D.load_cands(a.cands)
    ids = D.pick_subset(list(cands), a.n, a.seed)
    exs = D.load_examples(a.data_dir, a.subset, set(ids))
    ids = [t for t in ids if t in exs]
    if a.limit:
        ids = ids[:a.limit]
    tok = D.get_tokenizer(a.generator)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
    for b in BUDGETS:
        for v in VARIANTS:
            cond = f"{v}_{b}"
            cdir = os.path.join(a.out, cond)
            os.makedirs(cdir, exist_ok=True)
            pred_path = os.path.join(cdir, "prediction.jsonl")
            done = {json.loads(l)["task_id"] for l in open(pred_path)} if os.path.exists(pred_path) else set()
            prompts = []
            for t in ids:
                if t in done:
                    continue
                saved = cands[t]["candidates"]
                e2 = compact_ex(tok, exs[t], saved, b, v)
                prompt, info = D.build_prompt(tok, e2, saved, max_cross=b)
                n_left_lines = len([l for l in info["infile_text"].split("\n") if l.strip()])
                prompts.append((t, tok.encode(prompt)[-D.MAX_CTX:], info["cross_tokens"], info["infile_tokens"],
                                n_left_lines))
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
                    for (t, _, ct, it, nl), g in zip(batch, gen[:, L:]):
                        fout.write(json.dumps({"task_id": t, "pred": tok.decode(g, skip_special_tokens=True),
                                               "cross_tokens": ct, "infile_tokens": it,
                                               "infile_nonblank_lines": nl}) + "\n")
                    fout.flush()
            print(f"[{cond}] done", flush=True)


def cmd_analyze(a):
    gt = {t: r["gt"] for t, r in P.load_jsonl(os.path.join(a.prev_dir, "gt.jsonl")).items()}
    rows = {c: P.load_jsonl(os.path.join(a.out, c, "prediction.jsonl"))
            for c in (f"{v}_{b}" for b in BUDGETS for v in VARIANTS)
            if os.path.exists(os.path.join(a.out, c, "prediction.jsonl"))}
    ids = sorted(set.intersection(*[set(v) for v in rows.values()]) & set(gt))
    em = {c: {t: D.simple_score(r[t]["pred"], gt[t])[0] for t in ids} for c, r in rows.items()}
    res = {"dataset": a.dataset, "n": len(ids), "conditions": {}, "decision": {}}
    for c in rows:
        b = c.rsplit("_", 1)[1]
        ref = em[f"base_{b}"]
        w = sum(em[c][t] > ref[t] for t in ids)
        l_ = sum(em[c][t] < ref[t] for t in ids)
        res["conditions"][c] = {
            "em": round(100 * sum(em[c].values()) / len(ids), 2), "vs_base": [w, l_, round(D.mcnemar(w, l_), 4)],
            "mean_infile_lines": round(sum(rows[c][t]["infile_nonblank_lines"] for t in ids) / len(ids), 1)}
    for v in VARIANTS[1:]:
        d = {b: (res["conditions"][f"{v}_{b}"]["em"] - res["conditions"][f"base_{b}"]["em"],
                 res["conditions"][f"{v}_{b}"]["vs_base"][2]) for b in BUDGETS if f"{v}_{b}" in rows}
        helps = any(g >= 1.5 and p < 0.05 for g, p in d.values()) and not any(g <= -1.5 for g, _ in d.values())
        res["decision"][v] = {"delta_p": d, "verdict": "HELPS" if helps else "no clear gain"}
    with open(os.path.join(a.out, "e3_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--dataset", default="repoeval", choices=["repoeval", "cceval"])
    ap.add_argument("--subset", default=None)
    ap.add_argument("--cands")
    ap.add_argument("--prev_dir")
    ap.add_argument("--data_dir", default="aligncoder_data")
    ap.add_argument("--out", default="e3")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--generator", default=D.GEN_DEFAULT)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    a = ap.parse_args()
    D.DATASET = a.dataset
    if a.subset is None:
        a.subset = "python" if a.dataset == "cceval" else "api_level"
    {"run": cmd_run, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
