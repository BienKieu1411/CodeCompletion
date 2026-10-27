#!/usr/bin/env python3
"""
analyze_benchmarks.py
Comprehensive forensic failure analysis across cceval_python, cceval_java, and repoeval_line.
"""

import gzip
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from difflib import SequenceMatcher

try:
    from fuzzywuzzy import fuzz
    def cal_edit_sim(p, g): return float(fuzz.ratio(p.strip(), g.strip()))
except ImportError:
    def cal_edit_sim(p, g): return SequenceMatcher(None, p.strip(), g.strip()).ratio() * 100.0

try:
    import editdistance
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "RepoClone/RLCoder"))
from utils.eval_repoeval import compute_EM as repoeval_EM, compute_ES as repoeval_ES

CADRE_PRED_PATH = Path(__file__).resolve().parent / "results_cadre_consolidated/cadre_consolidated_8004/cadre_consolidated_8004_predictions.jsonl.gz"
BASE_DIR = Path(__file__).resolve().parent / "results_fixed"
TRACE_PATH = ROOT_DIR / "experiments/mb_spt_iterative/rlcoder_full_8004_retrieval_trace.jsonl.gz"


def get_py_statement_line(pred: str, gt: str) -> tuple[str, str]:
    pred = re.sub(r"#.*", "", pred)
    gt = re.sub(r"#.*", "", gt)
    gt_lines = [l.strip() for l in gt.splitlines() if l.strip()]
    n = max(1, len(gt_lines))
    pred_lines = [l.strip() for l in pred.splitlines() if l.strip()][:n]
    return "\n".join(pred_lines), "\n".join(gt_lines)


def get_bracket_stmt(completion: str) -> str:
    end_idx = None
    for i in range(len(completion)):
        if completion[i] in [";", "}", "{"]:
            end_idx = i
            break
    return completion[:end_idx + 1] if end_idx is not None else completion


def eval_single(bm: str, pred: str, gt: str) -> tuple[int, float]:
    if bm.startswith("repoeval"):
        em = int(repoeval_EM(gt, [pred], passk=1))
        es = float(repoeval_ES(gt, [pred], passk=1)) * 100.0
        return em, es
    elif bm == "cceval_java":
        gt_s = get_bracket_stmt(re.sub(r"//.*", "", gt)).strip()
        pred_s = get_bracket_stmt(re.sub(r"//.*", "", pred)).strip()
        em = int("".join(pred_s.split()) == "".join(gt_s.split()) and len(gt_s) > 0)
        es = cal_edit_sim(pred_s, gt_s)
        return em, es
    else:  # cceval_python
        pred_s, gt_s = get_py_statement_line(pred, gt)
        em = int("".join(pred_s.split()) == "".join(gt_s.split()) and len(gt_s) > 0)
        es = cal_edit_sim(pred_s, gt_s)
        return em, es


def load_base_predictions(bm: str):
    pred_file = BASE_DIR / bm / "prediction.jsonl"
    preds = {}
    if pred_file.exists():
        with open(pred_file) as f:
            for line in f:
                d = json.loads(line)
                preds[d["task_id"]] = d.get("pred", d.get("prediction", ""))
    return preds


def load_traces():
    traces = {}
    if TRACE_PATH.exists():
        with gzip.open(TRACE_PATH, "rt") as f:
            for line in f:
                d = json.loads(line)
                traces[d["task_id"]] = d
    return traces


def analyze_benchmark(bm: str, cadre_items, base_preds, traces):
    print(f"\n{'='*80}\nBENCHMARK: {bm} (Total: {len(cadre_items)} tasks)\n{'='*80}")
    
    both_correct = []
    both_wrong = []
    wins = []         # base=0, ours=1
    regressions = []  # base=1, ours=0

    # Decision breakdown
    reg_decisions = Counter()
    win_decisions = Counter()
    both_wrong_decisions = Counter()

    # Pass comparison on regressions
    reg_pass1_raw_em = 0
    reg_pass1_cssi_em = 0
    reg_cadre_spec_em = 0

    # Retrieval presence in both wrong
    both_wrong_retrieval_hits = 0

    for item in cadre_items:
        tid = item["task_id"]
        gt = item["groundtruth"]
        cadre_pred = item["cadre_26_final"]
        base_pred = base_preds.get(tid, "")
        decision = item.get("gate_decision", "unknown")

        cadre_em, cadre_es = eval_single(bm, cadre_pred, gt)
        base_em, base_es = eval_single(bm, base_pred, gt)

        # check alternative arms for this item
        p1_raw_em, _ = eval_single(bm, item.get("pass1_raw", ""), gt)
        p1_cssi_em, _ = eval_single(bm, item.get("pass1_tree_cssi", ""), gt)
        spec_em, _ = eval_single(bm, item.get("cadre_tree_spec_raw", ""), gt)

        entry = {
            "task_id": tid,
            "groundtruth": gt,
            "base_pred": base_pred,
            "cadre_pred": cadre_pred,
            "pass1_raw": item.get("pass1_raw", ""),
            "pass1_tree_cssi": item.get("pass1_tree_cssi", ""),
            "cadre_tree_spec_raw": item.get("cadre_tree_spec_raw", ""),
            "copy_sim": item.get("copy_sim", 0.0),
            "gate_decision": decision,
            "base_em": base_em,
            "cadre_em": cadre_em,
            "base_es": base_es,
            "cadre_es": cadre_es,
            "p1_raw_em": p1_raw_em,
            "p1_cssi_em": p1_cssi_em,
            "spec_em": spec_em,
        }

        if base_em == 1 and cadre_em == 1:
            both_correct.append(entry)
        elif base_em == 0 and cadre_em == 1:
            wins.append(entry)
            win_decisions[decision] += 1
        elif base_em == 1 and cadre_em == 0:
            regressions.append(entry)
            reg_decisions[decision] += 1
            if p1_raw_em == 1: reg_pass1_raw_em += 1
            if p1_cssi_em == 1: reg_pass1_cssi_em += 1
            if spec_em == 1: reg_cadre_spec_em += 1
        else:
            both_wrong.append(entry)
            both_wrong_decisions[decision] += 1
            
            # Check retrieval hit for both wrong
            trace = traces.get(tid)
            if trace:
                top_contexts = trace.get("top_contexts", [])
                gt_clean = gt.strip()
                # Check if distinctive identifier from gt is in top contexts
                tokens = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", gt_clean) if t not in {"self", "true", "false", "none", "return", "import", "class", "public", "private", "void"}]
                if tokens:
                    hit = any(all(tok in ctx.get("text", "") for tok in tokens[:2]) for ctx in top_contexts[:5])
                    if hit:
                        both_wrong_retrieval_hits += 1

    print(f"Quadrant Breakdown:")
    print(f"  - Both Correct:     {len(both_correct):5d} ({len(both_correct)/len(cadre_items)*100:.2f}%)")
    print(f"  - CADRE Wins:        {len(wins):5d} ({len(wins)/len(cadre_items)*100:.2f}%)  [Base 0 -> Ours 1]")
    print(f"  - Regressions:      {len(regressions):5d} ({len(regressions)/len(cadre_items)*100:.2f}%)  [Base 1 -> Ours 0]")
    print(f"  - Both Wrong:       {len(both_wrong):5d} ({len(both_wrong)/len(cadre_items)*100:.2f}%)")
    print(f"  - Net Gain (EM):    +{len(wins) - len(regressions)} samples ({(len(wins) - len(regressions))/len(cadre_items)*100:+.2f} pp)")

    print(f"\nRegressions Decision Distribution ({len(regressions)} total):")
    for dec, count in reg_decisions.most_common():
        print(f"  - {dec}: {count} ({count/len(regressions)*100:.1f}%)")
    print(f"  In Regressions:")
    print(f"    - pass1_raw was EM=1:       {reg_pass1_raw_em}/{len(regressions)} ({reg_pass1_raw_em/max(1,len(regressions))*100:.1f}%)")
    print(f"    - pass1_tree_cssi was EM=1: {reg_pass1_cssi_em}/{len(regressions)} ({reg_pass1_cssi_em/max(1,len(regressions))*100:.1f}%)")
    print(f"    - cadre_tree_spec was EM=1: {reg_cadre_spec_em}/{len(regressions)} ({reg_cadre_spec_em/max(1,len(regressions))*100:.1f}%)")

    print(f"\nWins Decision Distribution ({len(wins)} total):")
    for dec, count in win_decisions.most_common():
        print(f"  - {dec}: {count} ({count/len(wins)*100:.1f}%)")

    print(f"\nBoth Wrong Analysis ({len(both_wrong)} total):")
    print(f"  - Retrieval hit (GT identifier in Top-5 context): {both_wrong_retrieval_hits}/{len(both_wrong)} ({both_wrong_retrieval_hits/max(1,len(both_wrong))*100:.1f}%)")

    return {
        "benchmark": bm,
        "both_correct": both_correct,
        "wins": wins,
        "regressions": regressions,
        "both_wrong": both_wrong,
        "reg_decisions": dict(reg_decisions),
        "win_decisions": dict(win_decisions),
    }


def main():
    print("Loading CADRE consolidated predictions...")
    cadre_by_bm = defaultdict(list)
    with gzip.open(CADRE_PRED_PATH, "rt") as f:
        for line in f:
            d = json.loads(line)
            cadre_by_bm[d["benchmark"]].append(d)

    print("Loading traces...")
    traces = load_traces()

    results = {}
    for bm in ["cceval_python", "cceval_java", "repoeval_line"]:
        base_preds = load_base_predictions(bm)
        results[bm] = analyze_benchmark(bm, cadre_by_bm[bm], base_preds, traces)

    # Dump detailed samples for deeper inspection
    out_file = Path(__file__).resolve().parent / "failure_analysis_summary.json"
    summary_data = {}
    for bm, res in results.items():
        summary_data[bm] = {
            "counts": {
                "total": len(res["both_correct"]) + len(res["wins"]) + len(res["regressions"]) + len(res["both_wrong"]),
                "both_correct": len(res["both_correct"]),
                "wins": len(res["wins"]),
                "regressions": len(res["regressions"]),
                "both_wrong": len(res["both_wrong"]),
            },
            "reg_decisions": res["reg_decisions"],
            "win_decisions": res["win_decisions"],
            "sample_regressions": res["regressions"][:10],
            "sample_wins": res["wins"][:10],
            "sample_both_wrong": res["both_wrong"][:10],
        }

    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nWrote failure analysis summary to {out_file}")


if __name__ == "__main__":
    main()
