#!/usr/bin/env python3
"""E7: error decomposition -- why do samples fail at EVERY budget? (CPU, tokenizer only, no generation)

For each test sample (full test = dev sweep + hold-out, greedy DeepSeek-Coder-1.3B, budgets 256/512/1024/1536) the
prompt at every budget is rebuilt exactly as in budget_sweep.py. GT identifiers = D.identifiers(gt) (keywords,
builtins, self removed). A GT identifier is "visible" at budget B if it occurs as a word in that prompt.

Classes of samples wrong at all 4 budgets (first match wins):
    usage      every GT identifier is visible at some single budget -> the evidence was in a prompt, generator misused it
    budget_cut a missing identifier occurs in the full left context or the saved AlignCoder candidates, but every
               budget's prompt cut it
    retrieval  a missing identifier occurs only in other repo files (not in left context, not in candidates)
    novel      a missing identifier occurs nowhere before the cursor (left context, candidates, repo files): a new
               name, or only in the right context -- prefix-only retrieval cannot supply it
Same classes are reported for samples solved at some budget, as a reference.

Static-checker headroom (prefix-only verifier idea): fraction of first lines whose identifiers include one that
occurs nowhere in left context + repo files. Measured on wrong predictions AND on GT lines (false-positive control:
GT lines can legitimately use library attributes or new names).

usage: python e7_errors.py   (needs aligncoder_data/ parquet, candidates jsonl, sweep outputs)
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dep_oracle as D  # noqa: E402
import e2_confirm as E2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {"repoeval": ("api_level", "prediction_with_candidates.jsonl"),
       "cceval": ("python", "cceval_python_prediction_with_candidates.jsonl")}
BUDGETS = E2.BUDGETS


def words(text):
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))


def first_line(s):
    for l in s.split("\n"):
        if l.strip():
            return l
    return ""


def load_preds(ds):
    out = {b: {} for b in BUDGETS}
    for root in (os.path.join(HERE, f"kaggle_sweep_{ds}_out", "sweep"), os.path.join(HERE, "modal_out", f"e2hold_{ds}")):
        for b in BUDGETS:
            for r in map(json.loads, open(os.path.join(root, f"b{b}", "prediction.jsonl"), encoding="utf-8")):
                out[b][r["task_id"]] = r["pred"]
    return out


def run(ds, tok):
    subset, cpath = CFG[ds]
    D.DATASET = ds
    cands = D.load_cands(os.path.join(HERE, cpath))
    preds = load_preds(ds)
    ids = sorted(set.intersection(*[set(preds[b]) for b in BUDGETS]) & set(cands))
    exs = D.load_examples(os.path.join(HERE, "aligncoder_data"), subset, set(ids))
    ids = [t for t in ids if t in exs]
    rows = []
    for t in ids:
        ex, saved = exs[t], cands[t]["candidates"]
        gt = ex["gt"]
        em = {b: D.simple_score(preds[b][t], gt)[0] for b in BUDGETS}
        gids = D.identifiers(gt)
        vis = {}
        for b in BUDGETS:
            _, info = D.build_prompt(tok, ex, saved, max_cross=b)
            vis[b] = words(info["cross_text"] + "\n" + info["infile_text"])
        left_w = words(ex["left"])
        cand_w = words("\n".join(c["code_content"] + "\n" + c["description"] for c in saved))
        repo_w = words("\n".join(f["text"] for f in ex["files"]))
        right_w = words(ex["right"])
        missing = [n for n in gids if not any(n in vis[b] for b in BUDGETS)]
        all_vis_one = any(all(n in vis[b] for n in gids) for b in BUDGETS)
        if not gids or all_vis_one:
            cls = "usage"
        elif any(n not in left_w and n not in cand_w and n not in repo_w for n in missing):
            cls = "novel"
        elif any(n not in left_w and n not in cand_w for n in missing):
            cls = "retrieval"
        else:
            cls = "budget_cut"
        if cls == "novel":
            sub = "only_in_right" if all(n in right_w for n in missing
                                         if n not in left_w and n not in cand_w and n not in repo_w) else "new_name"
        else:
            sub = ""
        known = left_w | repo_w | cand_w
        ref_b = E2.DEV_BEST[ds]
        p1 = first_line(preds[ref_b][t])
        rows.append({"task_id": t, "solved_any": max(em.values()), "em_ref": em[ref_b], "cls": cls, "sub": sub,
                     "n_gt_ids": len(gids), "n_missing": len(missing),
                     "pred_unknown_id": any(n not in known for n in D.identifiers(p1)),
                     "gt_unknown_id": any(n not in known for n in gids),
                     "first_line_match": p1.strip() == first_line(gt).strip()})
    return rows


def summarize(rows):
    n = len(rows)
    fail = [r for r in rows if not r["solved_any"]]
    ok = [r for r in rows if r["solved_any"]]
    pct = lambda c, m: round(100 * c / m, 1) if m else 0.0
    cls_f, cls_o = Counter(r["cls"] for r in fail), Counter(r["cls"] for r in ok)
    sub = Counter(r["sub"] for r in fail if r["cls"] == "novel")
    wrong_ref = [r for r in rows if not r["em_ref"]]
    return {
        "n": n, "fail_all_budgets": len(fail), "fail_all_pct": pct(len(fail), n),
        "fail_classes_pct": {k: pct(cls_f[k], len(fail)) for k in ("usage", "budget_cut", "retrieval", "novel")},
        "novel_sub_pct_of_fail": {k: pct(v, len(fail)) for k, v in sub.items()},
        "solved_classes_pct": {k: pct(cls_o[k], len(ok)) for k in ("usage", "budget_cut", "retrieval", "novel")},
        "fail_classes_pct_of_all": {k: pct(cls_f[k], n) for k in ("usage", "budget_cut", "retrieval", "novel")},
        "static_check": {
            "wrong_pred_has_unknown_id_pct": pct(sum(r["pred_unknown_id"] for r in wrong_ref), len(wrong_ref)),
            "gt_has_unknown_id_pct(all)": pct(sum(r["gt_unknown_id"] for r in rows), n),
            "correct_pred_has_unknown_id_pct": pct(sum(r["pred_unknown_id"] for r in rows if r["em_ref"]),
                                                   sum(r["em_ref"] for r in rows)),
            "n_wrong_ref": len(wrong_ref)},
    }


def main():
    tok = D.get_tokenizer(D.GEN_DEFAULT)
    res = {}
    for ds in ("repoeval", "cceval"):
        rows = run(ds, tok)
        with open(os.path.join(HERE, "modal_out", f"e7_rows_{ds}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        res[ds] = summarize(rows)
        print(ds, json.dumps(res[ds], indent=1), flush=True)
    json.dump(res, open(os.path.join(HERE, "modal_out", "e7_errors_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
