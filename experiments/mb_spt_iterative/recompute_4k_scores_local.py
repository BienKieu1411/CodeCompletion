#!/usr/bin/env python3
"""
Recompute Official Scores for CADRE 4K Full Left Context Evaluation (8,004 tasks)
Comparing Arm 1 (RLCoder Base 4k), Arm 2 (CADRE Post-hoc CSSI 4k), Arm 3 (CADRE Pre+Post CSSI 4k).

Protocols computed:
1. AlignCoder Candidate Prefix EM (Official evaluate_with_candidates.py logic)
2. Normalized Statement / Line EM (stripped, comments removed, sliced to target boundary)
3. RepoEval Official Repo-Macro EM
4. Edit Similarity (Levenshtein / fuzz ratio)
5. Identifier Match (id_em, id_f1)
"""

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
import difflib

try:
    from fuzzywuzzy import fuzz
    def cal_edit_sim(p, g):
        return float(fuzz.ratio(p.strip(), g.strip()))
except ImportError:
    def cal_edit_sim(p, g):
        return difflib.SequenceMatcher(None, p.strip(), g.strip()).ratio() * 100.0

STOP_WORDS = {
    "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
    "del", "elif", "else", "except", "finally", "for", "from", "global", "if",
    "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
    "return", "try", "while", "with", "yield", "self", "cls", "True", "False",
    "None", "int", "str", "float", "bool", "list", "dict", "set", "tuple",
    "public", "private", "protected", "static", "final", "void", "new", "this",
    "super", "extends", "implements", "interface", "package", "throws", "throw",
    "catch", "instanceof", "synchronized", "volatile", "transient", "native",
}

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

def extract_identifiers(code: str) -> list[str]:
    words = WORD_RE.findall(code)
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]

def remove_comments(code: str) -> str:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"//.*", "", code)
    return code

def clean_bpe_text(s: str) -> str:
    return s.replace('Ġ', ' ').replace('Ċ', '\n')

def get_bracket_stmt(completion: str) -> str:
    end_idx = None
    for i in range(len(completion)):
        if completion[i] in [";", "}", "{"]:
            end_idx = i
            break
    return completion[:end_idx + 1] if end_idx is not None else completion

def main():
    preds_path = Path("experiments/mb_spt_iterative/cadre_4k_full_left_context_eval_8004_predictions.jsonl.gz")
    if not preds_path.exists():
        print(f"Error: {preds_path} not found.")
        return

    records = []
    with gzip.open(preds_path, "rt", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} prediction records from {preds_path.name}.")

    arms = ["pred_arm1", "pred_arm2", "pred_arm3"]
    arm_labels = {
        "pred_arm1": "Arm 1: RLCoder Baseline 4k",
        "pred_arm2": "Arm 2: CADRE Post-hoc CSSI 4k",
        "pred_arm3": "Arm 3: CADRE Pre+Post CSSI 4k",
    }

    all_results = {}

    for arm in arms:
        by_bench = defaultdict(list)
        for r in records:
            bench = r["benchmark"]
            gt = r["groundtruth"]
            raw = clean_bpe_text(r[arm])

            # Strip comments
            gt_no_comment = remove_comments(gt)
            raw_no_comment = remove_comments(raw)

            # 1. Target and Prediction line tokenization
            gt_lines = [l.strip() for l in gt_no_comment.splitlines() if l.strip()]
            n_target_lines = max(1, len(gt_lines))

            # Slice prediction to target lines (Official RepoEval logic)
            pred_lines = [l.strip() for l in raw_no_comment.splitlines() if l.strip()][:n_target_lines]

            # Collapsed strings (whitespace removed)
            gt_flat = "".join("".join(gt_lines).split())
            pred_flat = "".join("".join(pred_lines).split())

            # Metric A: Exact Prefix Match (AlignCoder evaluate_with_candidates.py)
            # Checks if the prediction starts with the ground truth target prefix
            prefix_match = int(pred_flat[:len(gt_flat)] == gt_flat and len(gt_flat) > 0)

            # Metric B: Java statement match (up to semicolon)
            if bench == "cceval_java":
                gt_br = "".join(get_bracket_stmt(gt_no_comment).split())
                pred_br = "".join(get_bracket_stmt(raw_no_comment).split())
                stmt_match = int(pred_br[:len(gt_br)] == gt_br and len(gt_br) > 0)
            else:
                stmt_match = prefix_match

            # Metric C: Edit Similarity
            es = cal_edit_sim(raw_no_comment, gt_no_comment)

            # Metric D: Identifier Match
            gt_ids = set(extract_identifiers(gt_no_comment))
            pred_ids = set(extract_identifiers(raw_no_comment[:len(gt_no_comment) * 2]))
            if not gt_ids and not pred_ids:
                id_f1 = 100.0
            elif not gt_ids or not pred_ids:
                id_f1 = 0.0
            else:
                p = len(gt_ids & pred_ids) / len(pred_ids)
                rec = len(gt_ids & pred_ids) / len(gt_ids)
                id_f1 = (2 * p * rec / (p + rec)) * 100.0 if (p + rec) > 0 else 0.0

            repo = r["task_id"].split("/")[0]
            by_bench[bench].append({
                "repo": repo,
                "prefix_match": prefix_match,
                "stmt_match": stmt_match,
                "edit_sim": es,
                "id_f1": id_f1,
            })

        arm_summary = {}
        macro_prefix, macro_stmt, macro_es, macro_id_f1 = [], [], [], []

        for b in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
            items = by_bench[b]
            n = len(items)

            # Repo-Macro for RepoEval, Sample mean for CCEval
            if b.startswith("repoeval"):
                repo_groups = defaultdict(list)
                for it in items:
                    repo_groups[it["repo"]].append(it["stmt_match"])
                b_em = sum(sum(v)/len(v) for v in repo_groups.values()) / len(repo_groups) * 100.0
            else:
                b_em = sum(it["stmt_match"] for it in items) / n * 100.0

            b_pref = sum(it["prefix_match"] for it in items) / n * 100.0
            b_es = sum(it["edit_sim"] for it in items) / n
            b_f1 = sum(it["id_f1"] for it in items) / n

            arm_summary[b] = {
                "count": n,
                "aligncoder_em": round(b_em, 2),
                "prefix_em": round(b_pref, 2),
                "edit_sim": round(b_es, 2),
                "id_f1": round(b_f1, 2),
            }
            macro_stmt.append(b_em)
            macro_prefix.append(b_pref)
            macro_es.append(b_es)
            macro_id_f1.append(b_f1)

        all_items = [it for items in by_bench.values() for it in items]
        total_n = len(all_items)

        arm_summary["micro_mean"] = {
            "count": total_n,
            "aligncoder_em": round(sum(it["stmt_match"] for it in all_items) / total_n * 100.0, 2),
            "prefix_em": round(sum(it["prefix_match"] for it in all_items) / total_n * 100.0, 2),
            "edit_sim": round(sum(it["edit_sim"] for it in all_items) / total_n, 2),
            "id_f1": round(sum(it["id_f1"] for it in all_items) / total_n, 2),
        }
        arm_summary["macro_mean"] = {
            "aligncoder_em": round(sum(macro_stmt) / 4.0, 2),
            "prefix_em": round(sum(macro_prefix) / 4.0, 2),
            "edit_sim": round(sum(macro_es) / 4.0, 2),
            "id_f1": round(sum(macro_id_f1) / 4.0, 2),
        }
        all_results[arm] = arm_summary

    # Print Formatted Comparison Table
    print("\n" + "=" * 115)
    print("HEAD-TO-HEAD RECOMPUTED SCORES (ALIGNCODER OFFICIAL EVALUATION STANDARDS, 4,096 CONTEXT BUDGET)")
    print("=" * 115)
    print(f"{'Benchmark':<18} | {'RLCoder Base 4k':<20} | {'CADRE Post-hoc 4k':<20} | {'CADRE Pre+Post 4k':<20} | {'Gain (Arm3 vs 1)':<15}")
    print("-" * 115)

    for b in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        s1 = all_results["pred_arm1"][b]["aligncoder_em"]
        s2 = all_results["pred_arm2"][b]["aligncoder_em"]
        s3 = all_results["pred_arm3"][b]["aligncoder_em"]
        gain = s3 - s1
        print(f"{b:<18} | {s1:>18.2f}% | {s2:>18.2f}% | {s3:>18.2f}% | {gain:>+13.2f} pp")

    print("-" * 115)
    m1 = all_results["pred_arm1"]["macro_mean"]["aligncoder_em"]
    m2 = all_results["pred_arm2"]["macro_mean"]["aligncoder_em"]
    m3 = all_results["pred_arm3"]["macro_mean"]["aligncoder_em"]
    mgain = m3 - m1
    print(f"{'MACRO MEAN':<18} | {m1:>18.2f}% | {m2:>18.2f}% | {m3:>18.2f}% | {mgain:>+13.2f} pp")
    print("=" * 115)

    # Save to local JSON
    out_file = Path("experiments/mb_spt_iterative/cadre_4k_recomputed_official_metrics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved recomputed official metrics to {out_file}")

if __name__ == "__main__":
    main()
