#!/usr/bin/env python3
r"""D1 (GPU, ~20 min L4 for n=400/dataset): Sub-line Syntax-Triggered Call-Site Injection.

Tests whether injecting repo Call-Sites (usage examples) of the target line's syntactic trigger
(`callee(` or `receiver.`) into the cross-file context right before the in-file window beats
baseline AlignCoder and API definition injection (`oracle-D`).

Conditions (evaluated at the dataset's best fixed cross-file budget B* and at B=1536):
  1. base      : Saved AlignCoder top-k candidates (reproduces anchor).
  2. oracle_u  : (Ceiling, uses GT trigger) Top-3 repo call-sites of GT's first callee/receiver
                 packed at the END of cross-file context (closest to cursor, L2H / asc order)
                 within the exact same cross-file token budget B (steals 0 in-file tokens!).
  3. self_u    : (Realistic, NO GT!) Uses the saved baseline greedy prediction's first-line
                 syntactic trigger (`line_pfx + pred_line1` -> `callee(` / `receiver.`) to fetch
                 top-3 repo call-sites, packed at the END of cross-file context within budget B.
  4. self_ud   : (Realistic, NO GT!) Top-2 repo call-sites + 1 concise definition/signature of
                 the triggered symbol, packed at the END of cross-file context within budget B.

Decision gate (pre-registered from directions_beyond_budget_2026-10-25.md):
  - GO if `oracle_u - base >= +4.0 EM` on both datasets (McNemar p < 0.05) AND
    `self_u` (or `self_ud`) recovers >= 40% of `oracle_u`'s gain.
  - STOP H1 if `oracle_u - base < +2.0 EM`.

Usage:
  # 1. CPU dry-run / prompt inspection (0 CU):
  .\.venv-revela\Scripts\python.exe experiments/dep_oracle/d1_callsite_gen.py inspect --dataset repoeval --n 400
  # 2. GPU run on Colab L4 (resume-safe JSONL):
  python d1_callsite_gen.py run --dataset repoeval --cands prediction_with_candidates.jsonl --data_dir $DATA --out $D/runs/d1_repoeval --n 400
  python d1_callsite_gen.py run --dataset cceval --cands cceval_python_prediction_with_candidates.jsonl --data_dir $DATA --out $D/runs/d1_cceval --n 400
  # 3. CPU analyze:
  python d1_callsite_gen.py analyze --out modal_out/d1_repoeval --dataset repoeval
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dep_oracle as D  # noqa: E402
import d1_callsite_cpu as D1  # noqa: E402
import e2_confirm as E2  # noqa: E402
import e7_errors as E7  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ["base", "oracle_u", "self_u", "self_ud"]


def build_asc_callsite_prompt(tok, ex, saved_blocks, cs_blocks, max_cross, max_ctx=D.MAX_CTX):
    """Pack `cs_blocks` (Call-Sites) at the END of the cross-file budget (closest to `# file path:`
    and the in-file context), filling the preceding cross-file budget with `saved_blocks`.
    Guarantees total cross-file tokens <= max_cross so in-file context NEVER loses a single token!"""
    filt = [b for b in saved_blocks if b.get("file_path", "") != ""]
    cs_str = "\n\n".join(D.block_str(b) for b in cs_blocks).strip()
    base_str = "\n\n".join(D.block_str(b) for b in filt).strip()

    cs_ids = tok.encode(("\n\n" + cs_str) if cs_str else "", add_special_tokens=False)
    if len(cs_ids) >= max_cross:
        cross_ids = cs_ids[-max_cross:]
    else:
        rem = max_cross - len(cs_ids)
        base_ids = tok.encode(base_str[: rem * 10], add_special_tokens=False)[:rem]
        cross_ids = base_ids + cs_ids

    path_ids = tok.encode(f"\n\n# file path: {ex['path']}\n\n", add_special_tokens=False)
    allowed = max_ctx - (len(cross_ids) + len(path_ids) + 10)
    infile_ids = tok.encode(ex["left"], add_special_tokens=False)[-allowed:]
    prompt = tok.decode(cross_ids + path_ids + infile_ids)
    return prompt, {"cross_tokens": len(cross_ids), "cs_tokens": min(len(cs_ids), max_cross),
                    "infile_tokens": len(infile_ids)}


def get_condition_cs_blocks(cond, ex, draft_pred_line1, cs_idx, defs_idx):
    if cond == "base":
        return []
    line_pfx = "" if ex["left"].endswith("\n") else ex["left"].split("\n")[-1]
    if cond == "oracle_u":
        g1 = E7.first_line(ex["gt"]).strip()
        callees, dots = D1.extract_triggers(g1)
        f_callees, f_dots = D1.extract_triggers(line_pfx + g1)
        all_callees = callees + [x for x in f_callees if x not in callees]
        all_dots = dots + [x for x in f_dots if x not in dots]
        blocks = []
        for _, short in all_callees[:2]:
            blocks.extend(cs_idx.find_callsites(short, max_sites=3))
        for rec, _ in all_dots[:2]:
            blocks.extend(cs_idx.find_member_usages(rec, max_sites=2))
        return blocks

    # Zero-leakage self_u / self_ud using draft_pred_line1
    p1 = draft_pred_line1.strip()
    callees, dots = D1.extract_triggers(p1)
    f_callees, f_dots = D1.extract_triggers(line_pfx + p1)
    all_callees = callees + [x for x in f_callees if x not in callees]
    all_dots = dots + [x for x in f_dots if x not in dots]

    blocks = []
    if cond == "self_ud":
        # Put concise signature first, then call-sites last (so call-sites are closest to cursor!)
        for _, short in all_callees[:2]:
            if defs_idx.where(short) == "repo":
                blocks.extend(defs_idx.blocks_for(short, max_defs=1))
        for rec, attr in all_dots[:2]:
            for sym in (rec, attr):
                if defs_idx.where(sym) == "repo":
                    blocks.extend(defs_idx.blocks_for(sym, max_defs=1))
    for _, short in all_callees[:2]:
        blocks.extend(cs_idx.find_callsites(short, max_sites=3))
    for rec, _ in all_dots[:2]:
        blocks.extend(cs_idx.find_member_usages(rec, max_sites=2))
    return blocks


def cmd_inspect(a):
    subset, cpath = E7.CFG[a.dataset]
    D.DATASET = a.dataset
    cands = D.load_cands(os.path.join(HERE, cpath))
    ids = D.pick_subset(list(cands), a.n, a.seed)
    if a.limit:
        ids = ids[:a.limit]
    exs = D.load_examples(os.path.join(HERE, a.data_dir), subset, set(ids))
    preds_by_b = E7.load_preds(a.dataset)
    best_b = a.budget or E2.DEV_BEST[a.dataset]
    base_preds = preds_by_b[best_b]
    tok = D.get_tokenizer(a.generator)

    n_has_ou, n_has_su, n_has_sud = 0, 0, 0
    cs_tok_ou, cs_tok_su = [], []
    for t in ids:
        ex = exs[t]
        saved = cands[t]["candidates"]
        p1 = E7.first_line(base_preds[t])
        cs_idx = D1.RepoCallSiteIndex(ex)
        defs_idx = D.ExampleDefs(ex)
        ou_b = get_condition_cs_blocks("oracle_u", ex, p1, cs_idx, defs_idx)
        su_b = get_condition_cs_blocks("self_u", ex, p1, cs_idx, defs_idx)
        sud_b = get_condition_cs_blocks("self_ud", ex, p1, cs_idx, defs_idx)
        n_has_ou += bool(ou_b)
        n_has_su += bool(su_b)
        n_has_sud += bool(sud_b)
        _, info_ou = build_asc_callsite_prompt(tok, ex, saved, ou_b, max_cross=best_b)
        _, info_su = build_asc_callsite_prompt(tok, ex, saved, su_b, max_cross=best_b)
        cs_tok_ou.append(info_ou["cs_tokens"])
        cs_tok_su.append(info_su["cs_tokens"])

    n = len(ids)
    print(f"[{a.dataset} B={best_b} n={n}]")
    print(f"  Non-empty Call-Site blocks: oracle_u={n_has_ou} ({100*n_has_ou/n:.1f}%), "
          f"self_u={n_has_su} ({100*n_has_su/n:.1f}%), self_ud={n_has_sud} ({100*n_has_sud/n:.1f}%)")
    print(f"  Avg Call-Site tokens injected: oracle_u={sum(cs_tok_ou)/n:.1f}, self_u={sum(cs_tok_su)/n:.1f}")


def cmd_run(a):
    import torch
    from transformers import AutoModelForCausalLM

    subset, cpath = E7.CFG[a.dataset]
    D.DATASET = a.dataset
    cands_path = a.cands if os.path.isabs(a.cands) else os.path.join(HERE, a.cands)
    cands = D.load_cands(cands_path)
    ids = D.pick_subset(list(cands), a.n, a.seed)
    if a.limit:
        ids = ids[:a.limit]
    exs = D.load_examples(a.data_dir, subset, set(ids))
    preds_by_b = E7.load_preds(a.dataset)
    best_b = a.budget or E2.DEV_BEST[a.dataset]
    base_preds = preds_by_b[best_b]

    tok = D.get_tokenizer(a.generator)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    os.makedirs(a.out, exist_ok=True)
    out_file = os.path.join(a.out, f"d1_b{best_b}_predictions.jsonl")
    done = set()
    if os.path.exists(out_file):
        with open(out_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["task_id"])
    todo = [t for t in ids if t not in done]
    print(f"[{a.dataset} B={best_b}] Total={len(ids)}, Done={len(done)}, Todo={len(todo)}")
    if not todo:
        cmd_analyze(a)
        return

    model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()

    t0 = time.time()
    with open(out_file, "a", encoding="utf-8") as fout:
        for i in range(0, len(todo), a.batch_size):
            batch_ids = todo[i: i + a.batch_size]
            # Build prompts for all conditions for this batch
            prompts = []
            meta = []
            for t in batch_ids:
                ex = exs[t]
                saved = cands[t]["candidates"]
                p1 = E7.first_line(base_preds[t])
                cs_idx = D1.RepoCallSiteIndex(ex)
                defs_idx = D.ExampleDefs(ex)
                for cond in CONDS:
                    cs_b = get_condition_cs_blocks(cond, ex, p1, cs_idx, defs_idx)
                    pr, info = build_asc_callsite_prompt(tok, ex, saved, cs_b, max_cross=best_b)
                    prompts.append(pr)
                    meta.append((t, cond, info))

            # Generate in sub-batches of gen_batch_size
            preds_flat = []
            for j in range(0, len(prompts), a.gen_batch_size):
                sub_p = prompts[j: j + a.gen_batch_size]
                enc = tok(sub_p, return_tensors="pt", padding=True).to(model.device)
                in_len = enc.input_ids.shape[1]
                with torch.no_grad():
                    out_ids = model.generate(
                        **enc, max_new_tokens=D.MAX_NEW, do_sample=False,
                        pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id
                    )
                for k in range(len(sub_p)):
                    gen_txt = tok.decode(out_ids[k, in_len:], skip_special_tokens=True)
                    preds_flat.append(gen_txt)

            # Group by task_id and save
            by_t = defaultdict(dict)
            for (t, cond, info), pred_txt in zip(meta, preds_flat):
                by_t[t][cond] = {"pred": pred_txt, "cs_tokens": info["cs_tokens"]}
            for t in batch_ids:
                row = {"task_id": t, "budget": best_b, "gt": exs[t]["gt"],
                       "saved_base_pred": base_preds[t], "conds": by_t[t]}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            done_now = i + len(batch_ids)
            elapsed = time.time() - t0
            print(f"  [{done_now}/{len(todo)}] elapsed={elapsed:.1f}s ({elapsed/done_now:.2f}s/item)")

    cmd_analyze(a)


def cmd_analyze(a):
    best_b = a.budget or E2.DEV_BEST[a.dataset]
    out_file = os.path.join(a.out, f"d1_b{best_b}_predictions.jsonl")
    rows = [json.loads(l) for l in open(out_file, encoding="utf-8") if l.strip()]
    n = len(rows)
    em_by_c = {c: [D.simple_score(r["conds"][c]["pred"], r["gt"])[0] for r in rows] for c in CONDS}
    saved_em = [D.simple_score(r["saved_base_pred"], r["gt"])[0] for r in rows]
    agree = sum(a == b for a, b in zip(em_by_c["base"], saved_em)) / max(n, 1)

    summary = {"dataset": a.dataset, "budget": best_b, "n": n,
               "saved_base_em": round(100 * sum(saved_em) / max(n, 1), 2),
               "base_regen_em_agreement": round(100 * agree, 2),
               "conditions": {}}
    base_vec = em_by_c["base"]
    for c in CONDS:
        vec = em_by_c[c]
        w = sum(x > y for x, y in zip(vec, base_vec))
        l_ = sum(x < y for x, y in zip(vec, base_vec))
        summary["conditions"][c] = {
            "em": round(100 * sum(vec) / max(n, 1), 2),
            "diff_vs_base": round(100 * (sum(vec) - sum(base_vec)) / max(n, 1), 2),
            "w_l_p": [w, l_, round(D.mcnemar(w, l_), 4)]
        }
    # Oracle over (base, self_u, self_ud)
    ora_self = [max(em_by_c["base"][i], em_by_c["self_u"][i], em_by_c["self_ud"][i]) for i in range(n)]
    summary["oracle_base_plus_self"] = round(100 * sum(ora_self) / max(n, 1), 2)

    sum_path = os.path.join(a.out, f"d1_b{best_b}_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "run", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", choices=["repoeval", "cceval"], required=True)
        p.add_argument("--cands", default="")
        p.add_argument("--data_dir", default="aligncoder_data")
        p.add_argument("--out", default="modal_out/d1")
        p.add_argument("--generator", default=D.GEN_DEFAULT)
        p.add_argument("--budget", type=int, default=0)
        p.add_argument("--n", type=int, default=400)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--batch_size", type=int, default=4)
        p.add_argument("--gen_batch_size", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "inspect":
        cmd_inspect(a)
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "analyze":
        cmd_analyze(a)


if __name__ == "__main__":
    main()
