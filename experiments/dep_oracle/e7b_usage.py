#!/usr/bin/env python3
"""E7b: what goes wrong in E7's "usage" failures (wrong at every budget although every GT identifier is in a prompt)?

Input: modal_out/e7_rows_{ds}.jsonl from e7_errors.py. For each usage failure, one budget is analysed: the
per-dataset best fixed budget if all GT identifiers are visible there, else the first such budget in 1536/1024/512/256.
Its prompt is rebuilt exactly as in budget_sweep.py.

Evidence (what the prompt holds):
    ids_where      all GT identifiers in the in-file part / need the cross-file part
    defs           GT symbols that have a def/class in left, right or repo files: all / some / none of those defs in
                   the prompt, or "no_def_symbols" (GT only uses locals, attributes, library names)
    gt_literals    string/number literals of the GT first line: none / all in prompt / some missing
    gt_line_sim    max similarity of the GT first line to any prompt line (>= 90: answer nearly copyable)
Error type of the prediction's first line vs the GT first line (first match wins):
    empty | first_line_ok (later GT lines wrong) | prefix (one is a prefix of the other) |
    same_ids_diff_literals (same identifier sequence: literals, operators, formatting differ) |
    missing_ids (pred identifiers subset of GT) | extra_ids (GT identifiers subset of pred) |
    same_start (same first identifier, then diverges) | different_statement
Also: pred first line copied verbatim from a prompt line; pred uses an identifier unknown to left + repo + candidates.

usage: python e7b_usage.py [--examples 6]
"""
import argparse
import difflib
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dep_oracle as D  # noqa: E402
import e2_confirm as E2  # noqa: E402
import e7_errors as E7  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LIT_RE = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'|\b\d+(?:\.\d+)?(?:e-?\d+)?\b")


def norm(s):
    return " ".join(s.strip().split())


def sim(a, b):
    return round(100 * difflib.SequenceMatcher(None, a, b).ratio())


def err_type(p, g, gt_full, pred_full):
    pn, gn = norm(p), norm(g)
    if not pn:
        return "empty"
    if pn == gn:
        return "first_line_ok"
    if gn.startswith(pn) or pn.startswith(gn):
        return "prefix"
    pi, gi = D.identifiers(p), D.identifiers(g)
    if pi == gi:
        return "same_ids_diff_literals"
    if set(pi) < set(gi):
        return "missing_ids"
    if set(gi) < set(pi):
        return "extra_ids"
    if pi and gi and pi[0] == gi[0]:
        return "same_start"
    return "different_statement"


def run(ds, tok, n_ex):
    subset, cpath = E7.CFG[ds]
    D.DATASET = ds
    rows = [r for r in map(json.loads, open(os.path.join(HERE, "modal_out", f"e7_rows_{ds}.jsonl"), encoding="utf-8"))
            if r["cls"] == "usage" and not r["solved_any"]]
    ids = {r["task_id"] for r in rows}
    cands = D.load_cands(os.path.join(HERE, cpath))
    preds = E7.load_preds(ds)
    exs = D.load_examples(os.path.join(HERE, "aligncoder_data"), subset, ids)
    order = [E2.DEV_BEST[ds]] + [b for b in (1536, 1024, 512, 256) if b != E2.DEV_BEST[ds]]
    out = []
    for r in rows:
        t = r["task_id"]
        ex, saved = exs[t], cands[t]["candidates"]
        gids = D.identifiers(ex["gt"])
        for b in order:
            _, info = D.build_prompt(tok, ex, saved, max_cross=b)
            if all(n in E7.words(info["cross_text"] + "\n" + info["infile_text"]) for n in gids):
                break
        prompt = info["cross_text"] + "\n" + info["infile_text"]
        in_w = E7.words(info["infile_text"])
        d = D.ExampleDefs(ex)
        def_syms = [n for n in gids if d.where(n)]
        n_def_in = sum(D.has_def(prompt, n) for n in def_syms)
        defs = "no_def_symbols" if not def_syms else ("all" if n_def_in == len(def_syms) else
                                                      "some" if n_def_in else "none")
        g1, p1 = E7.first_line(ex["gt"]), E7.first_line(preds[b][t])
        plines = [norm(l.lstrip("# ")) for l in prompt.split("\n") if l.strip()]
        gnorm, pnorm = norm(g1), norm(p1)
        gt_line_sim = max((sim(gnorm, l) for l in plines if abs(len(l) - len(gnorm)) < 0.5 * len(gnorm) + 5),
                          default=0)
        lits = re.findall(LIT_RE, g1)
        known = E7.words(ex["left"]) | E7.words("\n".join(f["text"] for f in ex["files"])) | \
            E7.words("\n".join(c["code_content"] for c in saved))
        out.append({"task_id": t, "budget": b,
                    "ids_where": "infile_only" if all(n in in_w for n in gids) else "needs_cross",
                    "defs": defs, "gt_line_sim": gt_line_sim,
                    "err": err_type(p1, g1, ex["gt"], preds[b][t]),
                    "es_first_line": sim(pnorm, gnorm),
                    "pred_copied": bool(pnorm) and pnorm in set(plines),
                    "pred_unknown_id": any(n not in known for n in D.identifiers(p1)),
                    "gt_literals": "none" if not lits else ("all_in_prompt" if all(x in prompt for x in lits)
                                                             else "missing"),
                    "gt": g1.strip(), "pred": p1.strip()})
    return out


def summarize(rows, n_all):
    n = len(rows)
    pct = lambda c: round(100 * c / n, 1)
    c = lambda k: {v: pct(x) for v, x in Counter(r[k] for r in rows).most_common()}
    simb = Counter("ge90" if r["gt_line_sim"] >= 90 else "70_89" if r["gt_line_sim"] >= 70 else "lt70" for r in rows)
    esb = Counter("ge80" if r["es_first_line"] >= 80 else "50_79" if r["es_first_line"] >= 50 else "lt50" for r in rows)
    cross = Counter((r["ids_where"], r["err"]) for r in rows)
    return {"n_usage_fail": n, "pct_of_all_samples": round(100 * n / n_all, 1),
            "ids_where": c("ids_where"), "defs": c("defs"), "err": c("err"),
            "gt_line_sim": {k: pct(v) for k, v in simb.items()},
            "es_first_line": {k: pct(v) for k, v in esb.items()},
            "pred_copied_pct": pct(sum(r["pred_copied"] for r in rows)),
            "pred_unknown_id_pct": pct(sum(r["pred_unknown_id"] for r in rows)),
            "gt_literals_x_err(literal errors)": dict(Counter(r["gt_literals"] for r in rows
                                                          if r["err"] == "same_ids_diff_literals")),
            "defs_all_x_err": {k: pct(v) for k, v in Counter(r["err"] for r in rows if r["defs"] == "all").items()},
            "ids_where_x_err": {f"{a}|{b}": pct(v) for (a, b), v in cross.most_common(8)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=6)
    a = ap.parse_args()
    tok = D.get_tokenizer(D.GEN_DEFAULT)
    res = {}
    rng = random.Random(0)
    for ds in ("repoeval", "cceval"):
        rows = run(ds, tok, a.examples)
        n_all = sum(1 for _ in open(os.path.join(HERE, "modal_out", f"e7_rows_{ds}.jsonl"), encoding="utf-8"))
        with open(os.path.join(HERE, "modal_out", f"e7b_rows_{ds}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        res[ds] = summarize(rows, n_all)
        print(ds, json.dumps(res[ds], indent=1))
        for e in res[ds]["err"]:
            ex = [r for r in rows if r["err"] == e]
            for r in rng.sample(ex, min(a.examples, len(ex))):
                print(f"  [{e}] {r['task_id']} B={r['budget']} defs={r['defs']} sim={r['gt_line_sim']}\n"
                      f"     GT  : {r['gt'][:150]}\n     PRED: {r['pred'][:150]}")
    json.dump(res, open(os.path.join(HERE, "modal_out", "e7b_usage_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
