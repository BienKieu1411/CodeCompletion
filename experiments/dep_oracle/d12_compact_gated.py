#!/usr/bin/env python3
r"""D1.2 (GPU, ~6-8 min L4 total):
Part A: Teacher-Forced Confidence + Static Copy/Trigger Gating on existing D1 predictions (`d1_b512_predictions.jsonl` & `d1_b1536_predictions.jsonl`).
Part B: Compact Non-Displacing Call-Site Injection (`compact_u`, `compact_ud`, <= 96 tokens, 0 eviction of saved_blocks) + Confidence Gating.

Why D1.2?
In D1:
  - CCEval (B=1536): `oracle_u` = 33.00 (+4.75, p=0.0043), `self_ud` = 30.50 (+2.25), `oracle(base, self)` = 34.00 (+5.75).
  - RepoEval (B=512): `base` = 40.50, `self_ud` = 33.00 (-7.50), YET `oracle(base, self)` = 43.25 (+2.75, 11 new wins!).
  - Root cause of RepoEval drop: D1 injected ~421 tokens of verbose Call-Site blocks into B=512, leaving only 91 tokens for AlignCoder's top-1 chunk (evicting 82% of the retrieved chunk that RepoEval copies verbatim!).

D1.2 fixes this via two mechanisms:
  1. Part A (`gate_existing`): Scores existing D1 predictions (`base`, `self_u`, `self_ud`) with teacher-forced
     `first_line_conf` and static copy-match features (`max_line_sim` in `saved_blocks` + `left`), learning/evaluating
     a zero-leakage gate between `base` and `self_ud` (and joint with E2 budgets).
  2. Part B (`run_compact`): Formats ultra-compact call-sites (<= 96 tokens total, 1-2 lines per call) and appends
     them AFTER the full `B` tokens of `saved_blocks` (so `saved_blocks` loses 0 tokens on RepoEval B=512!),
     while computing `first_line_conf` online.
"""
import argparse
import difflib
import json
import math
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dep_oracle as D  # noqa: E402
import d1_callsite_cpu as D1  # noqa: E402
import d1_callsite_gen as D1G  # noqa: E402
import e2_confirm as E2  # noqa: E402
import e7_errors as E7  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def max_line_similarity(cand_line, reference_texts):
    """Compute maximum normalized line similarity between `cand_line` and any non-trivial line
    in `reference_texts` (used to detect when `base` is already backed by a strong copy match)."""
    c = cand_line.strip()
    if len(c) < 6:
        return 0.0
    best = 0.0
    for txt in reference_texts:
        for raw in txt.splitlines():
            s = raw.strip().lstrip("#").strip()
            if len(s) < 6:
                continue
            if s == c:
                return 1.0
            # Fast length ratio filter before SequenceMatcher
            if min(len(s), len(c)) / max(len(s), len(c)) < 0.65:
                continue
            r = difflib.SequenceMatcher(None, c, s).ratio()
            if r > best:
                best = r
    return best


def build_compact_callsite_str(ex, draft_pred_line1, cs_idx, defs_idx, include_def=True, max_lines=6):
    """Build an ultra-compact Call-Site snippet (no redundant headers per block, 1-2 lines per site,
    <= ~96 tokens total) so it never bloats the prompt."""
    line_pfx = "" if ex["left"].endswith("\n") else ex["left"].split("\n")[-1]
    p1 = draft_pred_line1.strip()
    callees, dots = D1.extract_triggers(p1)
    f_callees, f_dots = D1.extract_triggers(line_pfx + p1)
    all_callees = callees + [x for x in f_callees if x not in callees]
    all_dots = dots + [x for x in f_dots if x not in dots]

    lines_out = []
    seen = set()

    if include_def:
        for _, short in all_callees[:2]:
            if defs_idx.where(short) == "repo":
                for b in defs_idx.blocks_for(short, max_defs=1):
                    # Take only the def signature header (up to closing paren / colon, max 2 lines)
                    sig_lines = [l.strip() for l in b["code_content"].splitlines()[:2] if l.strip()]
                    sig = " ".join(sig_lines)
                    if sig and sig not in seen:
                        seen.add(sig)
                        lines_out.append(f"# sig: {sig}")

    for _, short in all_callees[:2]:
        for b in cs_idx.find_callsites(short, max_sites=3):
            # Find the exact call line inside the snippet
            snip_lines = [l.strip() for l in b["code_content"].splitlines() if l.strip()]
            call_lines = [l for l in snip_lines if short in l] or snip_lines
            cl = " ".join(call_lines[:2])
            if cl and cl not in seen:
                seen.add(cl)
                lines_out.append(f"# usage: {cl}")
            if len(lines_out) >= max_lines:
                break

    for rec, _ in all_dots[:2]:
        if len(lines_out) >= max_lines:
            break
        for b in cs_idx.find_member_usages(rec, max_sites=2):
            for l in b["code_content"].splitlines():
                ls = l.strip()
                if ls and ls not in seen:
                    seen.add(ls)
                    lines_out.append(f"# usage: {ls}")
                    break

    return "\n".join(lines_out[:max_lines]).strip()


def build_nondisplacing_compact_prompt(tok, ex, saved_blocks, compact_cs_str, base_cross_budget,
                                       max_cs_tokens=96, max_ctx=D.MAX_CTX):
    """Keep FULL `base_cross_budget` tokens for `saved_blocks` when possible (so RepoEval B=512 loses
    ZERO tokens of AlignCoder's top-1 retrieved chunk!), appending at most `max_cs_tokens` of
    compact Call-Sites right before `# file path:`."""
    filt = [b for b in saved_blocks if b.get("file_path", "") != ""]
    base_str = "\n\n".join(D.block_str(b) for b in filt).strip()
    base_ids = tok.encode(base_str[: base_cross_budget * 10], add_special_tokens=False)[:base_cross_budget]

    if compact_cs_str:
        cs_ids = tok.encode("\n\n" + compact_cs_str, add_special_tokens=False)[:max_cs_tokens]
    else:
        cs_ids = []

    # If base_cross_budget + max_cs_tokens exceeds 1536 (e.g. on CCEval B=1536), trim oldest base_ids slightly
    max_total_cross = max(base_cross_budget, min(1536, base_cross_budget + max_cs_tokens))
    if len(base_ids) + len(cs_ids) > max_total_cross:
        keep_base = max(0, max_total_cross - len(cs_ids))
        base_ids = base_ids[:keep_base]

    cross_ids = base_ids + cs_ids
    path_ids = tok.encode(f"\n\n# file path: {ex['path']}\n\n", add_special_tokens=False)
    allowed = max_ctx - (len(cross_ids) + len(path_ids) + 10)
    infile_ids = tok.encode(ex["left"], add_special_tokens=False)[-allowed:]
    prompt = tok.decode(cross_ids + path_ids + infile_ids)
    return prompt, {
        "base_cross_tokens": len(base_ids),
        "cs_tokens": len(cs_ids),
        "infile_tokens": len(infile_ids),
    }


def teacher_forced_first_line_conf(model, tok, prompt_txt, pred_txt):
    """Compute exact teacher-forced mean log-prob (`first_line_conf`) and token count of the first line
    of `pred_txt` given `prompt_txt`."""
    import torch
    import torch.nn.functional as F

    l1 = E7.first_line(pred_txt)
    if not l1:
        return -10.0, 0
    p_ids = tok.encode(prompt_txt, add_special_tokens=False)
    c_ids = tok.encode(l1 + "\n", add_special_tokens=False)
    if not c_ids:
        return -10.0, 0
    # Keep within 2048 window if needed
    max_p = max(64, 2048 - len(c_ids))
    p_ids = p_ids[-max_p:]
    full_ids = torch.tensor([p_ids + c_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(full_ids).logits[0]
    start = len(p_ids) - 1
    end = len(p_ids) + len(c_ids) - 1
    sub_logits = logits[start:end].float()
    log_probs = F.log_softmax(sub_logits, dim=-1)
    target = torch.tensor(c_ids, dtype=torch.long, device=model.device)
    tok_lp = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return float(tok_lp.mean().item()), len(c_ids)


def evaluate_gating_rules(rows, exs, cands, dataset, best_b):
    """Evaluate zero-leakage gating rules between `base` and Call-Site candidates (`self_u`, `self_ud`,
    and if present `compact_u`, `compact_ud`), using:
      1. Static Copy-Protection + Unresolved-Call Gate (0 extra LLM calls)
      2. First-line confidence (`conf`) comparison (`self_conf - base_conf > margin`)
      3. Copy-Protected Confidence Gate (never override `base` when `base` has high copy similarity
         in `saved_blocks`/`left` OR high confidence; otherwise pick highest confidence / leave-one-out threshold).
    """
    n = len(rows)
    base_em = [D.simple_score(r["conds"]["base"]["pred"], r["gt"])[0] for r in rows]

    cand_keys = [k for k in ("self_u", "self_ud", "compact_u", "compact_ud") if k in rows[0]["conds"]]
    results = {
        "n": n,
        "base_em": round(100 * sum(base_em) / n, 2),
        "raw_candidates": {},
        "gated_policies": {},
    }

    for ck in cand_keys:
        vec = [D.simple_score(r["conds"][ck]["pred"], r["gt"])[0] for r in rows]
        w = sum(x > y for x, y in zip(vec, base_em))
        l_ = sum(x < y for x, y in zip(vec, base_em))
        results["raw_candidates"][ck] = {
            "em": round(100 * sum(vec) / n, 2),
            "diff_vs_base": round(100 * (sum(vec) - sum(base_em)) / n, 2),
            "w_l_p": [w, l_, round(D.mcnemar(w, l_), 4)],
        }

    # Precompute static features per row
    feats = []
    for r in rows:
        t = r["task_id"]
        ex = exs[t]
        saved = cands[t]["candidates"]
        b_pred = r["conds"]["base"]["pred"]
        b_l1 = E7.first_line(b_pred)
        saved_txts = [b.get("code_content", "") for b in saved[:4]]
        left_recent = "\n".join(ex["left"].splitlines()[-80:])
        copy_sim = max_line_similarity(b_l1, saved_txts + [left_recent])
        b_conf = r["conds"]["base"].get("conf", -1.0)
        feats.append({"copy_sim": copy_sim, "b_conf": b_conf, "b_l1": b_l1})

    for ck in cand_keys:
        c_em = [D.simple_score(r["conds"][ck]["pred"], r["gt"])[0] for r in rows]
        c_conf = [r["conds"][ck].get("conf", -1.0) for r in rows]
        cs_toks = [r["conds"][ck].get("cs_tokens", 0) for r in rows]

        # Policy 1: Raw Max-Conf (switch to ck iff c_conf > b_conf and cs_toks > 0)
        p_maxconf = [
            c_em[i] if (cs_toks[i] > 0 and c_conf[i] > feats[i]["b_conf"]) else base_em[i]
            for i in range(n)
        ]

        # Policy 2: Margin + Copy-Protected Confidence Gate (2-fold cross-fitted on margin & copy_sim threshold)
        # Split into fold 0 (even indices) and fold 1 (odd indices) to avoid any overfitting
        p_cv = list(base_em)
        chosen_params = []
        for val_fold in (0, 1):
            tr_idx = [i for i in range(n) if (i % 2) != val_fold]
            va_idx = [i for i in range(n) if (i % 2) == val_fold]
            best_gain = -1e9
            best_cfg = (0.05, 0.85, -0.15)
            for margin in (-0.05, 0.0, 0.02, 0.05, 0.10, 0.15, 0.25):
                for max_copy in (0.75, 0.85, 0.92, 1.01):
                    for max_bconf in (-0.05, -0.15, -0.25, 0.0):
                        gain = 0
                        switches = 0
                        for i in tr_idx:
                            if (cs_toks[i] > 0
                                    and feats[i]["copy_sim"] < max_copy
                                    and feats[i]["b_conf"] < max_bconf
                                    and (c_conf[i] - feats[i]["b_conf"]) > margin):
                                gain += (c_em[i] - base_em[i])
                                switches += 1
                        # Slight regularization preferring fewer switches on ties
                        score = gain - 0.001 * switches
                        if score > best_gain:
                            best_gain = score
                            best_cfg = (margin, max_copy, max_bconf)
            margin, max_copy, max_bconf = best_cfg
            chosen_params.append(best_cfg)
            for i in va_idx:
                if (cs_toks[i] > 0
                        and feats[i]["copy_sim"] < max_copy
                        and feats[i]["b_conf"] < max_bconf
                        and (c_conf[i] - feats[i]["b_conf"]) > margin):
                    p_cv[i] = c_em[i]

        for pname, pvec in ((f"{ck}_maxconf", p_maxconf), (f"{ck}_gated_cv", p_cv)):
            w = sum(x > y for x, y in zip(pvec, base_em))
            l_ = sum(x < y for x, y in zip(pvec, base_em))
            results["gated_policies"][pname] = {
                "em": round(100 * sum(pvec) / n, 2),
                "diff_vs_base": round(100 * (sum(pvec) - sum(base_em)) / n, 2),
                "w_l_p": [w, l_, round(D.mcnemar(w, l_), 4)],
            }
        results["gated_policies"][f"{ck}_gated_cv"]["cv_params"] = chosen_params

    # Oracle across all available zero-GT conditions
    all_zero_gt = ["base"] + cand_keys
    ora = [max(D.simple_score(rows[i]["conds"][k]["pred"], rows[i]["gt"])[0] for k in all_zero_gt) for i in range(n)]
    results["oracle_all_zero_gt"] = round(100 * sum(ora) / n, 2)
    return results


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

    cs_toks, base_kept = [], []
    for t in ids:
        ex = exs[t]
        saved = cands[t]["candidates"]
        p1 = E7.first_line(base_preds[t])
        cs_idx = D1.RepoCallSiteIndex(ex)
        defs_idx = D.ExampleDefs(ex)
        c_str = build_compact_callsite_str(ex, p1, cs_idx, defs_idx, include_def=True)
        _, info = build_nondisplacing_compact_prompt(tok, ex, saved, c_str, base_cross_budget=best_b)
        cs_toks.append(info["cs_tokens"])
        base_kept.append(info["base_cross_tokens"])

    n = len(ids)
    print(f"[D1.2 Compact Inspect: {a.dataset} B={best_b} n={n}]")
    print(f"  Avg Compact Call-Site tokens: {sum(cs_toks)/n:.1f} (vs ~421 in D1)")
    print(f"  Avg Base saved_blocks tokens preserved: {sum(base_kept)/n:.1f} / {best_b}")


def cmd_run_all(a):
    """Runs both Part A (scoring existing D1 predictions with teacher-forced confidence) and
    Part B (generating compact non-displacing `compact_u` & `compact_ud` + confidence gating)."""
    import torch
    from transformers import AutoModelForCausalLM

    subset, cpath = E7.CFG[a.dataset]
    D.DATASET = a.dataset
    cands_path = a.cands if os.path.isabs(a.cands) else os.path.join(HERE, cpath)
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
    d1_pred_file = os.path.join(a.out, f"d1_b{best_b}_predictions.jsonl")
    d12_pred_file = os.path.join(a.out, f"d12_b{best_b}_predictions.jsonl")

    # Load existing D1 rows if available
    d1_by_id = {}
    if os.path.exists(d1_pred_file):
        with open(d1_pred_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    d1_by_id[r["task_id"]] = r

    done_d12 = {}
    if os.path.exists(d12_pred_file):
        with open(d12_pred_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_d12[r["task_id"]] = r

    todo = [t for t in ids if t not in done_d12]
    print(f"[D1.2 {a.dataset} B={best_b}] Total={len(ids)}, Existing D1={len(d1_by_id)}, Done D1.2={len(done_d12)}, Todo={len(todo)}")

    if todo:
        model = AutoModelForCausalLM.from_pretrained(a.generator, torch_dtype=torch.float16).cuda().eval()
        t0 = time.time()
        with open(d12_pred_file, "a", encoding="utf-8") as fout:
            for i in range(0, len(todo), a.batch_size):
                batch_ids = todo[i: i + a.batch_size]
                prompts = []
                meta = []
                for t in batch_ids:
                    ex = exs[t]
                    saved = cands[t]["candidates"]
                    p1 = E7.first_line(base_preds[t])
                    cs_idx = D1.RepoCallSiteIndex(ex)
                    defs_idx = D.ExampleDefs(ex)
                    for cond, inc_def in (("compact_u", False), ("compact_ud", True)):
                        c_str = build_compact_callsite_str(ex, p1, cs_idx, defs_idx, include_def=inc_def)
                        pr, info = build_nondisplacing_compact_prompt(
                            tok, ex, saved, c_str, base_cross_budget=best_b, max_cs_tokens=96
                        )
                        prompts.append(pr)
                        meta.append((t, cond, pr, info))

                # Generate compact_u and compact_ud in batch
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

                by_t = defaultdict(dict)
                for (t, cond, pr, info), pred_txt in zip(meta, preds_flat):
                    conf, n_tok = teacher_forced_first_line_conf(model, tok, pr, pred_txt)
                    by_t[t][cond] = {
                        "pred": pred_txt,
                        "cs_tokens": info["cs_tokens"],
                        "conf": round(conf, 5),
                        "n_line1_tokens": n_tok,
                    }

                # Also score existing D1 conditions (base, self_u, self_ud) with teacher-forced conf
                for t in batch_ids:
                    ex = exs[t]
                    saved = cands[t]["candidates"]
                    p1 = E7.first_line(base_preds[t])
                    cs_idx = D1.RepoCallSiteIndex(ex)
                    defs_idx = D.ExampleDefs(ex)
                    row_conds = dict(by_t[t])

                    if t in d1_by_id:
                        old_c = d1_by_id[t]["conds"]
                        for cond in ("base", "oracle_u", "self_u", "self_ud"):
                            if cond in old_c:
                                cs_b = D1G.get_condition_cs_blocks(cond, ex, p1, cs_idx, defs_idx)
                                pr, info = D1G.build_asc_callsite_prompt(tok, ex, saved, cs_b, max_cross=best_b)
                                pred_txt = old_c[cond]["pred"]
                                conf, n_tok = teacher_forced_first_line_conf(model, tok, pr, pred_txt)
                                row_conds[cond] = {
                                    "pred": pred_txt,
                                    "cs_tokens": old_c[cond].get("cs_tokens", info["cs_tokens"]),
                                    "conf": round(conf, 5),
                                    "n_line1_tokens": n_tok,
                                }
                    else:
                        # Fallback if D1 file wasn't present: use saved base_preds
                        pr, info = D1G.build_asc_callsite_prompt(tok, ex, saved, [], max_cross=best_b)
                        conf, n_tok = teacher_forced_first_line_conf(model, tok, pr, base_preds[t])
                        row_conds["base"] = {
                            "pred": base_preds[t],
                            "cs_tokens": 0,
                            "conf": round(conf, 5),
                            "n_line1_tokens": n_tok,
                        }

                    out_row = {
                        "task_id": t,
                        "budget": best_b,
                        "gt": exs[t]["gt"],
                        "saved_base_pred": base_preds[t],
                        "conds": row_conds,
                    }
                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    done_d12[t] = out_row
                fout.flush()
                done_now = i + len(batch_ids)
                elapsed = time.time() - t0
                print(f"  [{done_now}/{len(todo)}] elapsed={elapsed:.1f}s ({elapsed/done_now:.2f}s/item)")

    rows = [done_d12[t] for t in ids if t in done_d12]
    summary = evaluate_gating_rules(rows, exs, cands, a.dataset, best_b)
    summary["dataset"] = a.dataset
    summary["budget"] = best_b
    sum_path = os.path.join(a.out, f"d12_b{best_b}_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "run"):
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
        p.add_argument("--batch_size", type=int, default=8)
        p.add_argument("--gen_batch_size", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "inspect":
        cmd_inspect(a)
    elif a.cmd == "run":
        cmd_run_all(a)


if __name__ == "__main__":
    main()
