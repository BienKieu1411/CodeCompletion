"""
Evaluates CADRE-TreeSpec predictions on the 400 Pilot Tasks using the official evaluation protocol.
"""

import gzip
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

try:
    from fuzzywuzzy import fuzz
    def cal_edit_sim(p: str, g: str) -> float:
        return float(fuzz.ratio(p.strip(), g.strip()))
except ImportError:
    def cal_edit_sim(p: str, g: str) -> float:
        return SequenceMatcher(None, p.strip(), g.strip()).ratio() * 100.0


def clean_bpe_text(s: str) -> str:
    return s.replace("Ġ", " ").replace("Ċ", "\n")


def remove_comments(code: str) -> str:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"//.*", "", code)
    return code


def get_bracket_stmt(completion: str) -> str:
    end_idx = None
    for i in range(len(completion)):
        if completion[i] in [";", "}", "{"]:
            end_idx = i
            break
    return completion[:end_idx + 1] if end_idx is not None else completion


def evaluate_predictions(records: List[Dict[str, Any]], pred_key: str) -> Dict[str, Any]:
    by_bench = defaultdict(list)
    for r in records:
        bm = r["benchmark"]
        lang = r["language"]
        raw = clean_bpe_text(r[pred_key])
        gt = r["groundtruth"]

        gt_no_comment = remove_comments(gt)
        raw_no_comment = remove_comments(raw)

        gt_lines = [l.strip() for l in gt_no_comment.splitlines() if l.strip()]
        n_target = max(1, len(gt_lines))
        pred_lines = [l.strip() for l in raw_no_comment.splitlines() if l.strip()][:n_target]

        gt_flat = "".join("".join(gt_lines).split())
        pred_flat = "".join("".join(pred_lines).split())
        prefix_match = int(pred_flat[:len(gt_flat)] == gt_flat and len(gt_flat) > 0)

        if bm == "cceval_java" or (lang == "java" and "cceval" in bm):
            gt_br = "".join(get_bracket_stmt(gt_no_comment).split())
            pred_br = "".join(get_bracket_stmt(raw_no_comment).split())
            stmt_match = int(pred_br[:len(gt_br)] == gt_br and len(gt_br) > 0)
        else:
            stmt_match = prefix_match

        es = cal_edit_sim(raw_no_comment, gt_no_comment)
        repo = r["task_id"].split("/")[0]
        by_bench[bm].append({"repo": repo, "em": stmt_match, "es": es})

    summary = {}
    macro_em, macro_es = [], []
    for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        items = by_bench[bm]
        n = len(items)
        if bm.startswith("repoeval"):
            repo_map = defaultdict(list)
            for it in items:
                repo_map[it["repo"]].append(it["em"])
            b_em = sum(sum(v)/len(v) for v in repo_map.values()) / len(repo_map) * 100.0

            repo_es = defaultdict(list)
            for it in items:
                repo_es[it["repo"]].append(it["es"])
            b_es = sum(sum(v)/len(v) for v in repo_es.values()) / len(repo_es)
        else:
            b_em = sum(it["em"] for it in items) / n * 100.0
            b_es = sum(it["es"] for it in items) / n

        summary[bm] = {"em": round(b_em, 2), "es": round(b_es, 2), "count": n}
        macro_em.append(b_em)
        macro_es.append(b_es)

    summary["macro_mean"] = {
        "em": round(sum(macro_em) / 4.0, 2),
        "es": round(sum(macro_es) / 4.0, 2),
    }
    return summary


def main():
    p_path = Path("experiments/mb_spt_iterative/cadre_tree_spec_predictions.jsonl.gz")
    records = []
    with gzip.open(p_path, "rt", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} prediction records.")

    stats_raw = evaluate_predictions(records, "pass1_raw")
    stats_tree_p1 = evaluate_predictions(records, "pass1_tree_cssi")
    stats_final = evaluate_predictions(records, "cadre_tree_spec_final")

    print("\n" + "=" * 88)
    print("=== CADRE-TreeSpec BENCHMARK COMPARISON (400 Pilot Tasks, DeepSeek-1.3B) ===")
    print("=" * 88)
    print(f"{'Method / Configuration':<36} | {'CC-Py':<7} | {'CC-Java':<7} | {'RE-API':<7} | {'RE-Line':<7} | {'Macro EM':<8} | {'Macro ES':<8}")
    print("-" * 88)
    print(f"{'1. Baseline CADRE v3 (Raw Only)':<36} | {stats_raw['cceval_python']['em']:<7.2f} | {stats_raw['cceval_java']['em']:<7.2f} | {stats_raw['repoeval_api']['em']:<7.2f} | {stats_raw['repoeval_line']['em']:<7.2f} | {stats_raw['macro_mean']['em']:<8.2f} | {stats_raw['macro_mean']['es']:<8.2f}")
    print(f"{'2. Pass 1B: Tree-CSSI Decorated':<36} | {stats_tree_p1['cceval_python']['em']:<7.2f} | {stats_tree_p1['cceval_java']['em']:<7.2f} | {stats_tree_p1['repoeval_api']['em']:<7.2f} | {stats_tree_p1['repoeval_line']['em']:<7.2f} | {stats_tree_p1['macro_mean']['em']:<8.2f} | {stats_tree_p1['macro_mean']['es']:<8.2f}")
    print(f"{'3. Pass 2: CADRE-TreeSpec (Final)':<36} | {stats_final['cceval_python']['em']:<7.2f} | {stats_final['cceval_java']['em']:<7.2f} | {stats_final['repoeval_api']['em']:<7.2f} | {stats_final['repoeval_line']['em']:<7.2f} | {stats_final['macro_mean']['em']:<8.2f} | {stats_final['macro_mean']['es']:<8.2f}")
    print("=" * 88)

    # Save metrics JSON
    with open("experiments/mb_spt_iterative/cadre_tree_spec_official_metrics.json", "w") as f:
        json.dump({
            "pass1_raw": stats_raw,
            "pass1_tree_cssi": stats_tree_p1,
            "cadre_tree_spec_final": stats_final,
        }, f, indent=2)

if __name__ == "__main__":
    main()
