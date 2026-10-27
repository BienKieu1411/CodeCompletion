#!/usr/bin/env python3
"""
evaluate_retrieval_impact.py
Evaluate and compare retrieval quality across:
1. Standard RLCoder (Paper Baseline): BM25 20-line tail, UniXcoder 508-token tail.
2. Full Left Context (Ablation): BM25 Full left context, UniXcoder 508-token tail.
3. CSSI-Guided RLCoder (Proposed): BM25 Full left context + CSSI, UniXcoder [256 CSSI + 252 tail].

Metrics evaluated:
- Target Identifier Hit@1, Hit@3, Hit@5, Hit@10 (Does top-k contain >=1 target identifier?)
- Target Identifier Full Hit@1, Hit@5, Hit@10 (Does top-k contain ALL target identifiers?)
- Target Identifier Recall@1, Recall@3, Recall@5, Recall@10 (Average fraction of target identifiers covered)
- Token Jaccard Similarity@1, @5, @10
- Mean Reciprocal Rank (MRR) of first target-bearing snippet
- Breakdown by benchmark: cceval_python, cceval_java, repoeval_api, repoeval_line
"""

import gzip
import json
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "experiments/mb_spt_iterative"
BUNDLE_PATH = DATA_DIR / "full_benchmark_8004_bundle.tar.gz"

TRACE_PATHS = {
    "RLCoder Paper Standard": DATA_DIR / "rlcoder_full_8004_retrieval_trace.jsonl.gz",
    "Full Left Context": DATA_DIR / "rlcoder_full_left_standard_8004_trace.jsonl.gz",
    "CSSI-Guided RLCoder": DATA_DIR / "rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz",
}

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


def extract_identifiers(code: str) -> List[str]:
    clean = re.sub(r"#.*", "", code)
    clean = re.sub(r"//.*", "", clean)
    clean = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'', "", clean)
    words = WORD_RE.findall(clean)
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def compute_retrieval_metrics_for_task(target_ids: Set[str], target_tokens: Set[str], top10_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not target_ids:
        # If target has no identifiers (e.g. only operators or syntax), fallback to target tokens
        eval_ids = target_tokens
    else:
        eval_ids = target_ids

    mrr = 0.0
    first_hit_rank = None

    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    full_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    recalls = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
    jaccards = {1: 0.0, 5: 0.0, 10: 0.0}

    accumulated_chunk_tokens: Set[str] = set()

    for rank, chunk in enumerate(top10_chunks[:10], start=1):
        if chunk.get("is_stop") or not chunk.get("file_path"):
            continue
        c_text = chunk.get("code_content", "") or chunk.get("code", "")
        c_tokens = set(WORD_RE.findall(c_text))
        accumulated_chunk_tokens.update(c_tokens)

        # Check hits
        overlap = eval_ids & accumulated_chunk_tokens
        has_hit = len(overlap) > 0
        has_full_hit = eval_ids.issubset(accumulated_chunk_tokens) if eval_ids else False
        recall = len(overlap) / len(eval_ids) if eval_ids else 0.0

        if first_hit_rank is None and len(eval_ids & c_tokens) > 0:
            first_hit_rank = rank
            mrr = 1.0 / rank

        for k in [1, 3, 5, 10]:
            if rank == k:
                hits[k] = int(has_hit)
                full_hits[k] = int(has_full_hit)
                recalls[k] = recall

        for k in [1, 5, 10]:
            if rank == k:
                union_len = len(target_tokens | accumulated_chunk_tokens)
                jaccards[k] = (len(target_tokens & accumulated_chunk_tokens) / union_len) if union_len > 0 else 0.0

    # Fill any remaining k if fewer than 10 chunks
    for k in [1, 3, 5, 10]:
        if k > len(top10_chunks):
            overlap = eval_ids & accumulated_chunk_tokens
            hits[k] = int(len(overlap) > 0)
            full_hits[k] = int(eval_ids.issubset(accumulated_chunk_tokens) if eval_ids else False)
            recalls[k] = len(overlap) / len(eval_ids) if eval_ids else 0.0
            if k in jaccards:
                union_len = len(target_tokens | accumulated_chunk_tokens)
                jaccards[k] = (len(target_tokens & accumulated_chunk_tokens) / union_len) if union_len > 0 else 0.0

    return {
        "hit@1": hits[1],
        "hit@3": hits[3],
        "hit@5": hits[5],
        "hit@10": hits[10],
        "full_hit@1": full_hits[1],
        "full_hit@5": full_hits[5],
        "full_hit@10": full_hits[10],
        "recall@1": recalls[1],
        "recall@3": recalls[3],
        "recall@5": recalls[5],
        "recall@10": recalls[10],
        "jaccard@1": jaccards[1],
        "jaccard@5": jaccards[5],
        "jaccard@10": jaccards[10],
        "mrr": mrr,
    }


def main():
    print("=" * 90)
    print("EMPIRICAL RETRIEVAL QUALITY EVALUATION (8,004 TASKS)")
    print("Comparing: RLCoder Standard vs Full Left Context vs CSSI-Guided RLCoder")
    print("=" * 90)

    t0 = time.time()
    print("Loading benchmark tasks from bundle...")
    with tarfile.open(BUNDLE_PATH, "r:gz") as tar:
        f_tasks = tar.extractfile("tasks.jsonl.gz")
        with gzip.open(f_tasks, "rt") as gz:
            tasks = [json.loads(line) for line in gz]
    print(f"✓ Loaded {len(tasks)} tasks in {time.time() - t0:.2f}s.")

    # Pre-extract target identifiers and tokens for each task
    task_targets = []
    for t in tasks:
        gt = t["groundtruth"]
        t_ids = set(extract_identifiers(gt))
        t_toks = set(WORD_RE.findall(gt))
        task_targets.append({
            "task_id": t["task_id"],
            "benchmark": t["benchmark"],
            "language": t["language"],
            "target_ids": t_ids,
            "target_tokens": t_toks,
        })

    all_results = {}

    for arm_name, trace_file in TRACE_PATHS.items():
        if not trace_file.exists():
            print(f"⚠ Trace file missing: {trace_file}")
            continue

        print(f"\nProcessing Trace: {arm_name} ({trace_file.name})...")
        t_trace = time.time()

        traces_by_id = {}
        with gzip.open(trace_file, "rt") as f:
            for line in f:
                d = json.loads(line)
                key = (d.get("benchmark"), d["task_id"])
                traces_by_id[key] = d.get("rlcoder_top10") or d.get("top10") or []

        print(f"  Loaded {len(traces_by_id)} traces in {time.time() - t_trace:.2f}s.")

        by_benchmark = defaultdict(lambda: defaultdict(list))
        overall_metrics = defaultdict(list)

        for t_info in task_targets:
            key = (t_info["benchmark"], t_info["task_id"])
            top10 = traces_by_id.get(key, [])
            metrics = compute_retrieval_metrics_for_task(t_info["target_ids"], t_info["target_tokens"], top10)

            bm = t_info["benchmark"]
            for m_key, val in metrics.items():
                by_benchmark[bm][m_key].append(val)
                overall_metrics[m_key].append(val)

        arm_summary = {"per_benchmark": {}, "overall": {}}

        for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
            if bm not in by_benchmark:
                continue
            bm_data = by_benchmark[bm]
            bm_res = {}
            for m_key, vals in bm_data.items():
                bm_res[m_key] = round(sum(vals) / len(vals) * 100.0, 2)
            arm_summary["per_benchmark"][bm] = bm_res

        for m_key, vals in overall_metrics.items():
            arm_summary["overall"][m_key] = round(sum(vals) / len(vals) * 100.0, 2)

        # Macro mean across 4 benchmarks
        macro_res = {}
        for m_key in overall_metrics.keys():
            macro_res[m_key] = round(sum(arm_summary["per_benchmark"][bm][m_key] for bm in arm_summary["per_benchmark"]) / len(arm_summary["per_benchmark"]), 2)
        arm_summary["macro_mean"] = macro_res

        all_results[arm_name] = arm_summary

    # Print comprehensive comparison tables
    print("\n" + "=" * 115)
    print(f"{'RETRIEVAL EVALUATION SUMMARY (8,004 TASKS)':^115}")
    print("=" * 115)

    headers = ["Metric", "RLCoder Paper Std", "Full Left Context", "CSSI-Guided RLCoder", "Δ (CSSI vs Paper)", "Δ (CSSI vs Full Left)"]
    print(f"{headers[0]:<25} | {headers[1]:<18} | {headers[2]:<18} | {headers[3]:<20} | {headers[4]:<18} | {headers[5]:<18}")
    print("-" * 125)

    key_metrics = [
        ("Target Hit@1 (%)", "hit@1"),
        ("Target Hit@3 (%)", "hit@3"),
        ("Target Hit@5 (%)", "hit@5"),
        ("Target Hit@10 (%)", "hit@10"),
        ("Full Target Hit@5 (%)", "full_hit@5"),
        ("Full Target Hit@10 (%)", "full_hit@10"),
        ("Identifier Recall@1 (%)", "recall@1"),
        ("Identifier Recall@5 (%)", "recall@5"),
        ("Identifier Recall@10 (%)", "recall@10"),
        ("Token Jaccard@5 (%)", "jaccard@5"),
        ("Token Jaccard@10 (%)", "jaccard@10"),
        ("MRR (Mean Recip Rank)", "mrr"),
    ]

    for label, m_key in key_metrics:
        val_paper = all_results.get("RLCoder Paper Standard", {}).get("macro_mean", {}).get(m_key, 0.0)
        val_fl = all_results.get("Full Left Context", {}).get("macro_mean", {}).get(m_key, 0.0)
        val_cssi = all_results.get("CSSI-Guided RLCoder", {}).get("macro_mean", {}).get(m_key, 0.0)

        diff_paper = val_cssi - val_paper
        diff_fl = val_cssi - val_fl

        sign_p = "+" if diff_paper >= 0 else ""
        sign_fl = "+" if diff_fl >= 0 else ""

        print(f"{label:<25} | {val_paper:>17.2f}% | {val_fl:>17.2f}% | {val_cssi:>19.2f}% | {sign_p + f'{diff_paper:.2f}%':>17} | {sign_fl + f'{diff_fl:.2f}%':>17}")

    print("=" * 125)

    # Breakdown per benchmark
    print("\n" + "=" * 115)
    print(f"{'BREAKDOWN BY BENCHMARK (Target Hit@10 & Identifier Recall@10)':^115}")
    print("=" * 115)
    bench_hdrs = ["Benchmark", "Metric", "RLCoder Paper Std", "Full Left Context", "CSSI-Guided RLCoder", "Improvement (vs Paper)"]
    print(f"{bench_hdrs[0]:<16} | {bench_hdrs[1]:<20} | {bench_hdrs[2]:<18} | {bench_hdrs[3]:<18} | {bench_hdrs[4]:<20} | {bench_hdrs[5]:<20}")
    print("-" * 125)

    for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        p_hit = all_results.get("RLCoder Paper Standard", {}).get("per_benchmark", {}).get(bm, {}).get("hit@10", 0.0)
        fl_hit = all_results.get("Full Left Context", {}).get("per_benchmark", {}).get(bm, {}).get("hit@10", 0.0)
        c_hit = all_results.get("CSSI-Guided RLCoder", {}).get("per_benchmark", {}).get(bm, {}).get("hit@10", 0.0)
        d_hit = c_hit - p_hit
        s_hit = "+" if d_hit >= 0 else ""

        p_rec = all_results.get("RLCoder Paper Standard", {}).get("per_benchmark", {}).get(bm, {}).get("recall@10", 0.0)
        fl_rec = all_results.get("Full Left Context", {}).get("per_benchmark", {}).get(bm, {}).get("recall@10", 0.0)
        c_rec = all_results.get("CSSI-Guided RLCoder", {}).get("per_benchmark", {}).get(bm, {}).get("recall@10", 0.0)
        d_rec = c_rec - p_rec
        s_rec = "+" if d_rec >= 0 else ""

        print(f"{bm:<16} | {'Target Hit@10':<20} | {p_hit:>17.2f}% | {fl_hit:>17.2f}% | {c_hit:>19.2f}% | {s_hit + f'{d_hit:.2f}%':>19}")
        print(f"{'':<16} | {'Identifier Recall@10':<20} | {p_rec:>17.2f}% | {fl_rec:>17.2f}% | {c_rec:>19.2f}% | {s_rec + f'{d_rec:.2f}%':>19}")
        print("-" * 125)

    # Save detailed json report
    out_json = DATA_DIR / "retrieval_impact_evaluation_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved full retrieval impact report to {out_json}")


if __name__ == "__main__":
    main()
