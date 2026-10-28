#!/usr/bin/env python3
"""
run_experiment.py - Empirical Study on the Impact of Repository Context (CSSI)
on Retrieval Quality and Candidate Reranking under strict <= 512 token query budget.

Evaluates across all 8,004 benchmark tasks:
- Part 1: Retrieval Input Construction (Query Formulation <= 512 tokens)
    Arm A: Tail-only (Standard RLCoder 508 tail tokens)
    Arm B: Naive Full Left Context (BM25 full file)
    Arm C: CSSI-Partitioned Query ([256 tokens CSSI] + [252 tokens Tail])
- Part 2: Rerank Stage with CSSI Structural Alignment (Post-Retrieval Rescoring)
    Rank A: Dense Neural Baseline (Cosine similarity alone)
    Rank B: CSSI Structural Type Reranker
    Rank C: CSSI Type + Interface Signature Agreement Reranker

Metrics: Hit@1, Hit@3, Hit@5, Hit@10, Target Identifier Recall@10, Mean Reciprocal Rank (MRR).
"""

import gzip
import json
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BUNDLE_PATH = ROOT_DIR / "experiments/mb_spt_iterative/full_benchmark_8004_bundle.tar.gz"
OUTPUT_DIR = Path(__file__).resolve().parent

TRACE_PATHS = {
    "standard_tail": ROOT_DIR / "experiments/mb_spt_iterative/rlcoder_full_8004_retrieval_trace.jsonl.gz",
    "full_left": ROOT_DIR / "experiments/mb_spt_iterative/rlcoder_full_left_standard_8004_trace.jsonl.gz",
    "cssi_query": ROOT_DIR / "experiments/mb_spt_iterative/rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz",
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


def extract_identifiers(code: str) -> set[str]:
    clean = re.sub(r"#.*", "", code)
    clean = re.sub(r"//.*", "", clean)
    clean = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'', "", clean)
    words = WORD_RE.findall(clean)
    return set(w for w in words if w not in STOP_WORDS and len(w) > 1)


def extract_cssi_signals(left_context: str, language: str) -> dict:
    lines = left_context.splitlines()
    recent = lines[-35:] if len(lines) >= 35 else lines
    recent_text = "\n".join(recent)
    cur_line = lines[-1] if lines else ""

    receivers = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", cur_line)
    callees = re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", cur_line)

    assigned_types = {}
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", recent_text):
        assigned_types[m.group(1)] = m.group(2)
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", recent_text):
        assigned_types[m.group(1)] = m.group(2)
    # Java types: ClassName var = new ClassName() or ClassName var;
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[;=]", recent_text):
        assigned_types[m.group(2)] = m.group(1)

    imported_symbols = {}
    for m in re.finditer(r"(?:from\s+([A-Za-z0-9_.]+)\s+import\s+([^#\n]+)|import\s+([^#\n]+))", left_context):
        if m.group(1) and m.group(2):
            raw_syms = m.group(2).replace("(", "").replace(")", "").split(",")
            for s in raw_syms:
                s = s.strip()
                if " as " in s:
                    orig, alias = s.split(" as ")
                    imported_symbols[alias.strip()] = orig.strip()
                elif s:
                    imported_symbols[s] = s
        elif m.group(3):
            # Java or Python import module
            for s in m.group(3).split(","):
                s = s.strip().rstrip(";")
                short = s.split(".")[-1]
                imported_symbols[short] = s

    target_types = set()
    for r in receivers:
        if r in assigned_types:
            target_types.add(assigned_types[r])
        elif r in imported_symbols:
            target_types.add(imported_symbols[r])
        elif r and r[0].isupper():
            target_types.add(r)

    # Class definitions within file
    return {
        "receivers": set(receivers),
        "callees": set(callees),
        "target_types": target_types,
        "cur_line_ids": set(WORD_RE.findall(cur_line)) - STOP_WORDS,
    }


def evaluate_chunk_list(target_ids: set[str], chunks: list[dict]) -> tuple[dict[int, int], float, float]:
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr = 0.0
    accumulated_tokens = set()
    first_hit_rank = None

    for rank, c in enumerate(chunks[:10], start=1):
        if c.get("is_stop") or not c.get("file_path"):
            continue
        c_text = c.get("code_content", "") or c.get("code", "")
        c_tokens = set(WORD_RE.findall(c_text))
        accumulated_tokens.update(c_tokens)

        if first_hit_rank is None and len(target_ids & c_tokens) > 0:
            first_hit_rank = rank
            mrr = 1.0 / rank

    # The first match is also a hit at larger K, including when fewer than K
    # non-STOP candidates are present or the kth entry itself is STOP.
    for k in hits:
        hits[k] = int(first_hit_rank is not None and first_hit_rank <= k)

    overlap_10 = len(target_ids & accumulated_tokens)
    recall_10 = (overlap_10 / len(target_ids)) if target_ids else 0.0

    return hits, recall_10, mrr


def main():
    print("=" * 90)
    print("EMPIRICAL STUDY: IMPACT OF REPO CONTEXT (CSSI) ON RETRIEVAL & RERANK (8,004 TASKS)")
    print("Constraint: Query input max 512 tokens for retriever.")
    print("=" * 90)

    # 1. Load tasks
    print("\n[Step 1/3] Loading benchmark groundtruth and metadata...")
    tasks = {}
    with tarfile.open(BUNDLE_PATH) as tar:
        with gzip.open(tar.extractfile("tasks.jsonl.gz"), "rt") as f:
            for line in f:
                t = json.loads(line)
                tasks[(t["benchmark"], t["task_id"])] = t
    print(f"✓ Loaded {len(tasks)} tasks.")

    # 2. Evaluate Part 1: Retrieval Query Formulation (<= 512 Tokens)
    print("\n[Step 2/3] Evaluating Part 1: Query Formulation within 512-Token Constraint...")
    retrieval_results = {}
    for arm_name, path in TRACE_PATHS.items():
        print(f"  Reading trace: {arm_name} ({path.name})...")
        by_bm = defaultdict(lambda: {"h1": [], "h3": [], "h5": [], "h10": [], "rec10": [], "mrr": []})
        with gzip.open(path, "rt") as f:
            for line in f:
                tr = json.loads(line)
                bm = tr["benchmark"]
                tid = tr["task_id"]
                t = tasks.get((bm, tid))
                if not t:
                    continue
                gt_ids = extract_identifiers(t["groundtruth"])
                if not gt_ids:
                    continue
                hits, rec10, mrr = evaluate_chunk_list(gt_ids, tr["rlcoder_top10"])
                for k in [1, 3, 5, 10]:
                    by_bm[bm][f"h{k}"].append(hits[k])
                by_bm[bm]["rec10"].append(rec10)
                by_bm[bm]["mrr"].append(mrr)

        summary = {}
        macro = defaultdict(list)
        for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
            d = by_bm[bm]
            n = len(d["h1"])
            summary[bm] = {
                "hit@1": round(sum(d["h1"]) / n * 100.0, 2),
                "hit@3": round(sum(d["h3"]) / n * 100.0, 2),
                "hit@5": round(sum(d["h5"]) / n * 100.0, 2),
                "hit@10": round(sum(d["h10"]) / n * 100.0, 2),
                "recall@10": round(sum(d["rec10"]) / n * 100.0, 2),
                "mrr": round(sum(d["mrr"]) / n * 100.0, 2),
                "count": n,
            }
            for k in ["hit@1", "hit@3", "hit@5", "hit@10", "recall@10", "mrr"]:
                macro[k].append(summary[bm][k])

        summary["macro_mean"] = {k: round(sum(v) / len(v), 2) for k, v in macro.items()}
        retrieval_results[arm_name] = summary

    # 3. Evaluate Part 2: Rerank Stage with CSSI Structural Alignment
    print("\n[Step 3/3] Evaluating Part 2: Rerank Stage with CSSI Structural Alignment...")
    # Using the standard 508 tail trace as candidate pool
    by_bm_dense = defaultdict(lambda: {"h1": [], "h3": [], "h5": [], "h10": [], "rec10": [], "mrr": []})
    by_bm_cssi_rerank = defaultdict(lambda: {"h1": [], "h3": [], "h5": [], "h10": [], "rec10": [], "mrr": []})

    with gzip.open(TRACE_PATHS["standard_tail"], "rt") as f:
        for line in f:
            tr = json.loads(line)
            bm = tr["benchmark"]
            tid = tr["task_id"]
            t = tasks.get((bm, tid))
            if not t:
                continue
            gt_ids = extract_identifiers(t["groundtruth"])
            if not gt_ids:
                continue

            raw_cands = tr["rlcoder_top10"]

            # Baseline Dense
            h_b, r_b, m_b = evaluate_chunk_list(gt_ids, raw_cands)
            for k in [1, 3, 5, 10]:
                by_bm_dense[bm][f"h{k}"].append(h_b[k])
            by_bm_dense[bm]["rec10"].append(r_b)
            by_bm_dense[bm]["mrr"].append(m_b)

            # CSSI-Guided Structural Reranking
            signals = extract_cssi_signals(t["left_context"], t["language"])
            scored_cands = []
            for rank_idx, c in enumerate(raw_cands):
                base_score = c.get("score", 1.0 / (rank_idx + 1))
                content = c.get("code_content", "") or c.get("code", "")
                fpath = c.get("file_path", "")
                c_ids = set(WORD_RE.findall(content))

                boost = 0.0
                # 1. Target Class / Interface definition boost
                for tt in signals["target_types"]:
                    if f"class {tt}" in content or f"interface {tt}" in content:
                        boost += 0.25
                    elif tt in c_ids:
                        boost += 0.10
                    if tt.lower() in fpath.lower():
                        boost += 0.15

                # 2. Target Callee method definition boost
                for callee in signals["callees"]:
                    if f"def {callee}" in content or f"{callee}(" in content:
                        boost += 0.20

                # 3. Context identifier overlap boost
                overlap_cur = len(signals["cur_line_ids"] & c_ids)
                if overlap_cur:
                    boost += min(0.10, overlap_cur * 0.02)

                scored_cands.append((base_score + boost, c))

            scored_cands.sort(key=lambda x: x[0], reverse=True)
            reranked_chunks = [c for _, c in scored_cands]

            h_r, r_r, m_r = evaluate_chunk_list(gt_ids, reranked_chunks)
            for k in [1, 3, 5, 10]:
                by_bm_cssi_rerank[bm][f"h{k}"].append(h_r[k])
            by_bm_cssi_rerank[bm]["rec10"].append(r_r)
            by_bm_cssi_rerank[bm]["mrr"].append(m_r)

    rerank_summary = {"dense_baseline": {}, "cssi_rerank": {}}
    for setting, store in [("dense_baseline", by_bm_dense), ("cssi_rerank", by_bm_cssi_rerank)]:
        macro = defaultdict(list)
        for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
            d = store[bm]
            n = len(d["h1"])
            rerank_summary[setting][bm] = {
                "hit@1": round(sum(d["h1"]) / n * 100.0, 2),
                "hit@3": round(sum(d["h3"]) / n * 100.0, 2),
                "hit@5": round(sum(d["h5"]) / n * 100.0, 2),
                "hit@10": round(sum(d["h10"]) / n * 100.0, 2),
                "recall@10": round(sum(d["rec10"]) / n * 100.0, 2),
                "mrr": round(sum(d["mrr"]) / n * 100.0, 2),
                "count": n,
            }
            for k in ["hit@1", "hit@3", "hit@5", "hit@10", "recall@10", "mrr"]:
                macro[k].append(rerank_summary[setting][bm][k])
        rerank_summary[setting]["macro_mean"] = {k: round(sum(v) / len(v), 2) for k, v in macro.items()}

    # Print Report
    print("\n" + "=" * 95)
    print("PART 1: RETRIEVAL QUERY FORMULATION UNDER MAX 512-TOKEN BUDGET (MACRO MEAN)")
    print("=" * 95)
    print(f"{'Configuration':<35} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'Hit@10':<8} | {'Rec@10':<8} | {'MRR':<8}")
    print("-" * 95)
    for arm_name, label in [
        ("standard_tail", "Arm 1: Standard 508 Tail Tokens"),
        ("full_left", "Arm 2: Full Left Context (BM25 Dilution)"),
        ("cssi_query", "Arm 3: CSSI [256] + Tail [252] Tokens"),
    ]:
        m = retrieval_results[arm_name]["macro_mean"]
        print(f"{label:<35} | {m['hit@1']:>6.2f}% | {m['hit@3']:>6.2f}% | {m['hit@5']:>6.2f}% | {m['hit@10']:>6.2f}% | {m['recall@10']:>6.2f}% | {m['mrr']:>6.2f}%")

    print("\n" + "=" * 95)
    print("PART 2: RERANK STAGE WITH CSSI STRUCTURAL ALIGNMENT")
    print("=" * 95)
    print(f"{'Benchmark':<18} | {'Dense Hit@1':<12} | {'CSSI Hit@1':<12} | {'Δ Hit@1':<10} | {'Dense MRR':<10} | {'CSSI MRR':<10} | {'Δ MRR':<10}")
    print("-" * 95)
    for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line", "macro_mean"]:
        d_b = rerank_summary["dense_baseline"][bm]
        d_r = rerank_summary["cssi_rerank"][bm]
        dh1, drh1 = d_b["hit@1"], d_r["hit@1"]
        dmrr, drmrr = d_b["mrr"], d_r["mrr"]
        diff_h1 = drh1 - dh1
        diff_mrr = drmrr - dmrr
        name = "Macro Mean" if bm == "macro_mean" else bm
        print(f"{name:<18} | {dh1:>10.2f}% | {drh1:>10.2f}% | {diff_h1:>+8.2f} pp | {dmrr:>8.2f}% | {drmrr:>8.2f}% | {diff_mrr:>+8.2f} pp")

    # Save to JSON
    out_file = OUTPUT_DIR / "study_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "part1_query_formulation_512": retrieval_results,
            "part2_cssi_reranking": rerank_summary,
        }, f, indent=2)
    print(f"\n✓ Saved detailed experiment artifacts to {out_file}.")


if __name__ == "__main__":
    main()
