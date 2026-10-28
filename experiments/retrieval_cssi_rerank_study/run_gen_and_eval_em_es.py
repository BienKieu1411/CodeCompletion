#!/usr/bin/env python3
"""
run_gen_and_eval_em_es.py
Comprehensive End-to-End Generation & Official EM/ES Evaluation Script.

Addresses the core research questions:
1. Impact of candidate sources on final completion:
   - When expanding left context, retrieved top-10 candidates change.
   - When injecting repo context (CSSI) into the 512-token query, candidate quality changes.
   - When using CSSI as a Structural Alignment Prior to Rerank the top candidates.
2. Evaluates the resulting impact on final completion generation metrics:
   - EM (Exact Match) using official benchmark truncation protocols:
     * RepoEval: exact line match with repository-level macro averaging.
     * CCEval Python: Python statement line match.
     * CCEval Java: Java statement / bracket match.
   - ES (Edit Similarity) via Levenshtein / SequenceMatcher.
   - ID-EM and ID-F1 for code identifier accuracy.

Supports:
- Running with DeepSeek-Coder (MPS / CUDA / CPU) on full 8,004 tasks or pilot batches (--limit N).
- Evaluating existing generation traces or saving new predictions.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import json
import os
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Optional imports with fallbacks
try:
    import editdistance
except ImportError:
    editdistance = None

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

# Directories and Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BUNDLE_PATH = ROOT_DIR / "experiments/mb_spt_iterative/full_benchmark_8004_bundle.tar.gz"
OUTPUT_DIR = ROOT_DIR / "experiments/retrieval_cssi_rerank_study"

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


# -------------------------------------------------------------
# 1. Official Evaluation Protocol & Metric Helpers
# -------------------------------------------------------------
def cal_edit_sim(pred: str, gt: str) -> float:
    if fuzz is not None:
        return float(fuzz.ratio(pred.strip(), gt.strip()))
    return difflib.SequenceMatcher(None, pred.strip(), gt.strip()).ratio() * 100.0


def repoeval_es(target: str, pred: str) -> float:
    t_lines = [l.strip() for l in target.splitlines() if l.strip()]
    p_lines = [l.strip() for l in pred.splitlines() if l.strip()][: len(t_lines)]
    t_str = "\n".join(t_lines)
    p_str = "\n".join(p_lines)
    if not t_str and not p_str:
        return 100.0
    max_l = max(len(t_str), len(p_str))
    if max_l == 0:
        return 100.0
    if editdistance is not None:
        return (1.0 - (editdistance.eval(t_str, p_str) / max_l)) * 100.0
    return cal_edit_sim(p_str, t_str)


def get_py_statement_line(pred: str, gt: str) -> Tuple[str, str]:
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
    return completion[: end_idx + 1] if end_idx is not None else completion


def extract_identifiers(code: str) -> Set[str]:
    clean = re.sub(r"#.*", "", code)
    clean = re.sub(r"//.*", "", clean)
    clean = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'', "", clean)
    words = WORD_RE.findall(clean)
    return set(w for w in words if w not in STOP_WORDS and len(w) > 1)


def evaluate_completion(target: str, pred: str, benchmark: str, language: str) -> Tuple[int, float, int, float]:
    """
    Returns (em, es, id_em, id_f1) for a single completion.
    """
    if benchmark.startswith("repoeval"):
        t_lines = [l.strip() for l in target.splitlines() if l.strip()]
        p_lines = [l.strip() for l in pred.splitlines() if l.strip()][: len(t_lines)]
        em = int(len(t_lines) == len(p_lines) and t_lines == p_lines)
        es = repoeval_es(target, pred)
    elif benchmark == "cceval_java":
        gt_s = get_bracket_stmt(re.sub(r"//.*", "", target)).strip()
        pred_s = get_bracket_stmt(re.sub(r"//.*", "", pred)).strip()
        em = int("".join(pred_s.split()) == "".join(gt_s.split()) and len(gt_s) > 0)
        es = cal_edit_sim(pred_s, gt_s)
    else:  # cceval_python
        pred_s, gt_s = get_py_statement_line(pred, target)
        em = int("".join(pred_s.split()) == "".join(gt_s.split()) and len(gt_s) > 0)
        es = cal_edit_sim(pred_s, gt_s)

    # Identifier-level metrics
    gt_ids = extract_identifiers(target)
    pred_ids = extract_identifiers(pred)
    if not gt_ids and not pred_ids:
        id_em = 1
        id_f1 = 100.0
    elif not gt_ids or not pred_ids:
        id_em = 0
        id_f1 = 0.0
    else:
        id_em = int(gt_ids == pred_ids)
        p = len(gt_ids & pred_ids) / len(pred_ids)
        r = len(gt_ids & pred_ids) / len(gt_ids)
        id_f1 = (2 * p * r / (p + r)) * 100.0 if (p + r) > 0 else 0.0

    return em, es, id_em, id_f1


def summarize_benchmark_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_bench = defaultdict(list)
    for r in records:
        by_bench[r["benchmark"]].append(r)

    summary = {}
    macro_em, macro_es, macro_id_em, macro_id_f1 = [], [], [], []

    for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        items = by_bench[bm]
        if not items:
            continue
        n = len(items)

        if bm.startswith("repoeval"):
            # Official benchmark protocol: repository-level macro average
            repo_em = defaultdict(list)
            repo_es = defaultdict(list)
            repo_id_em = defaultdict(list)
            repo_id_f1 = defaultdict(list)
            for it in items:
                repo = it["task_id"].split("/")[0]
                repo_em[repo].append(it["em"])
                repo_es[repo].append(it["es"])
                repo_id_em[repo].append(it["id_em"])
                repo_id_f1[repo].append(it["id_f1"])

            b_em = sum(sum(v) / len(v) for v in repo_em.values()) / len(repo_em) * 100.0
            b_es = sum(sum(v) / len(v) for v in repo_es.values()) / len(repo_es)
            b_id_em = sum(sum(v) / len(v) for v in repo_id_em.values()) / len(repo_id_em) * 100.0
            b_id_f1 = sum(sum(v) / len(v) for v in repo_id_f1.values()) / len(repo_id_f1)
        else:
            b_em = sum(it["em"] for it in items) / n * 100.0
            b_es = sum(it["es"] for it in items) / n
            b_id_em = sum(it["id_em"] for it in items) / n * 100.0
            b_id_f1 = sum(it["id_f1"] for it in items) / n

        summary[bm] = {
            "em": round(b_em, 2),
            "es": round(b_es, 2),
            "id_em": round(b_id_em, 2),
            "id_f1": round(b_id_f1, 2),
            "count": n,
        }
        macro_em.append(b_em)
        macro_es.append(b_es)
        macro_id_em.append(b_id_em)
        macro_id_f1.append(b_id_f1)

    if macro_em:
        summary["macro_mean"] = {
            "em": round(sum(macro_em) / len(macro_em), 2),
            "es": round(sum(macro_es) / len(macro_es), 2),
            "id_em": round(sum(macro_id_em) / len(macro_id_em), 2),
            "id_f1": round(sum(macro_id_f1) / len(macro_id_f1), 2),
        }
    return summary


# -------------------------------------------------------------
# 2. CSSI Structural Alignment & Reranker
# -------------------------------------------------------------
def extract_cssi_signals(left_context: str) -> Dict[str, Any]:
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

    return {
        "receivers": set(receivers),
        "callees": set(callees),
        "target_types": target_types,
        "cur_line_ids": set(WORD_RE.findall(cur_line)) - STOP_WORDS,
    }


def rerank_candidates_with_cssi(
    candidates: List[Dict[str, Any]], left_context: str, language: str
) -> List[Dict[str, Any]]:
    """
    Reranks candidate chunks by combining their initial dense retrieval score
    with a structural alignment score derived from CSSI.
    """
    signals = extract_cssi_signals(left_context)
    receivers = signals["receivers"]
    callees = signals["callees"]
    target_types = signals["target_types"]
    cur_line_ids = signals["cur_line_ids"]

    scored_candidates = []
    for idx, c in enumerate(candidates):
        if c.get("is_stop") or not c.get("file_path"):
            scored_candidates.append((-999.0, idx, c))
            continue

        c_text = c.get("code_content", "") or c.get("code", "")
        fpath = c.get("file_path", "")
        c_tokens = set(WORD_RE.findall(c_text))

        # Initial dense score
        dense_score = c.get("score", 1.0 / (idx + 1))

        # Structural alignment score
        struct_score = 0.0

        # Type/Class match in candidate chunk
        if target_types:
            for tt in target_types:
                if tt in c_tokens:
                    struct_score += 1.8
                if f"class {tt}" in c_text or f"interface {tt}" in c_text:
                    struct_score += 3.5

        # Callee method match
        if callees:
            for cal in callees:
                if cal in c_tokens:
                    struct_score += 1.5
                if f"def {cal}" in c_text or f"{cal}(" in c_text:
                    struct_score += 2.8

        # Receiver match in path or code
        if receivers:
            for r in receivers:
                if r in c_tokens:
                    struct_score += 0.8

        # Line token overlap
        overlap = len(cur_line_ids & c_tokens)
        struct_score += min(overlap * 0.4, 2.0)

        # Combined ranking score
        combined_score = (dense_score * 2.0) + struct_score
        scored_candidates.append((combined_score, idx, c))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for _, _, c in scored_candidates]


# -------------------------------------------------------------
# 3. Context & Prompt Assembly
# -------------------------------------------------------------
def coalesce_chunks_for_prompt(chunks: List[Dict[str, Any]], language: str, max_tokens: int = 1536) -> str:
    comment_sym = "#" if language == "python" else "//"
    chunks_by_file = defaultdict(list)
    for c in chunks:
        fp = c.get("file_path") or c.get("path") or ""
        if not fp or c.get("is_stop"):
            continue
        code = c.get("code_content") or c.get("code") or c.get("text") or ""
        if code.strip():
            chunks_by_file[fp].append(code.strip())

    blocks = []
    for fp, clist in chunks_by_file.items():
        header = f"{comment_sym} file path: {fp}\n"
        body = "\n...\n".join(clist)
        blocks.append(f"{header}{body}\n\n")

    full_text = "".join(blocks)
    words = full_text.split()
    if len(words) > max_tokens:
        full_text = " ".join(words[:max_tokens])
    return full_text


def build_completion_prompt(
    task: Dict[str, Any], candidates: List[Dict[str, Any]], max_context: int = 3072, crossfile_budget: int = 1536
) -> str:
    lang = task["language"]
    cs = "#" if lang == "python" else "//"
    path = task.get("path", "file.py" if lang == "python" else "File.java")

    # Cross-file context block
    cross_text = coalesce_chunks_for_prompt(candidates[:10], lang, max_tokens=crossfile_budget)

    # Local context
    left_context = task["left_context"]
    left_lines = left_context.splitlines()

    # Budgeting: preserve at least 1536 tokens for left context
    allowed_left_words = max_context - len(cross_text.split()) - 30
    left_words = left_context.split()
    if len(left_words) > allowed_left_words and allowed_left_words > 0:
        left_context = " ".join(left_words[-allowed_left_words:])

    prompt = f"{cross_text}\n{cs} file path: {path}\n\n{left_context}"
    return prompt


# -------------------------------------------------------------
# 4. Main Execution Engine
# -------------------------------------------------------------
def run_evaluation_study(
    arms: List[str],
    limit: Optional[int] = None,
    device: str = "mps",
    use_generator: bool = False,
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-base",
) -> Dict[str, Any]:
    if not use_generator:
        raise RuntimeError(
            "Offline synthetic completion was removed: it used ground truth as a prediction. "
            "Use --use-generator or evaluate saved model predictions with a separate scorer."
        )
    if torch is None or AutoModelForCausalLM is None:
        raise RuntimeError("Live generator evaluation requires torch and transformers")
    print("=" * 95)
    print("EMPIRICAL STUDY: EVALUATING FINAL CODE GENERATION (EM & ES) ACROSS CANDIDATE SOURCES")
    print(f"Arms to evaluate: {arms}")
    print(f"Limit: {limit if limit else 'Full 8,004 benchmark tasks'}")
    print("=" * 95)

    # Step 1: Load Benchmark Ground Truth & Tasks
    print("\n[Step 1/4] Loading benchmark groundtruth and metadata from bundle...")
    tasks = {}
    ordered_task_keys = []
    with tarfile.open(BUNDLE_PATH) as tar:
        with gzip.open(tar.extractfile("tasks.jsonl.gz"), "rt") as f:
            for line in f:
                t = json.loads(line)
                key = (t["benchmark"], t["task_id"])
                tasks[key] = t
                ordered_task_keys.append(key)
    print(f"✓ Loaded {len(tasks)} benchmark tasks.")

    if limit is not None:
        ordered_task_keys = ordered_task_keys[:limit]
        print(f"  (Limiting evaluation to first {limit} tasks for rapid testing)")

    # Step 2: Load Traces
    print("\n[Step 2/4] Loading retrieval candidate traces...")
    traces = {}
    for arm_name in ["standard_tail", "full_left", "cssi_query"]:
        path = TRACE_PATHS[arm_name]
        print(f"  Loading trace for {arm_name} from {path.name}...")
        t_map = {}
        with gzip.open(path, "rt") as f:
            for line in f:
                rec = json.loads(line)
                t_map[(rec["benchmark"], rec["task_id"])] = rec["rlcoder_top10"]
        traces[arm_name] = t_map

    # Step 3: Generator setup (if live generation requested)
    model = None
    tokenizer = None
    if use_generator and torch is not None and AutoModelForCausalLM is not None:
        print(f"\n[Step 3/4] Initializing generator model {model_name} on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device in ["mps", "cuda"] else torch.float32,
        ).to(device).eval()
        print("✓ Generator ready for live inference.")
    else:
        print("\n[Step 3/4] Running in offline evaluation mode using CADRE prompt alignment and trace candidate pools...")

    # Step 4: Evaluate Each Arm
    print("\n[Step 4/4] Generating predictions and evaluating official EM/ES...")
    results_by_arm = {}

    for arm in arms:
        print(f"\n---> Evaluating Candidate Pipeline Arm: [{arm}] ...")
        t_arm_start = time.time()
        records = []

        for key in ordered_task_keys:
            t = tasks[key]
            bm = t["benchmark"]
            lang = t["language"]
            gt = t["groundtruth"]

            # Select Candidate Pool
            if arm == "standard_tail":
                cands = traces["standard_tail"].get(key, [])
            elif arm == "full_left":
                cands = traces["full_left"].get(key, [])
            elif arm == "cssi_query":
                cands = traces["cssi_query"].get(key, [])
            elif arm == "cssi_rerank":
                base_cands = traces["standard_tail"].get(key, [])
                cands = rerank_candidates_with_cssi(base_cands, t["left_context"], lang)
            else:
                continue

            # Generate Prediction
            if model is not None and tokenizer is not None:
                prompt_str = build_completion_prompt(t, cands)
                inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=48,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                pred = tokenizer.decode(out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
            else:
                raise RuntimeError("Generator was not initialized")

            em, es, id_em, id_f1 = evaluate_completion(gt, pred, bm, lang)
            records.append({
                "task_id": t["task_id"],
                "benchmark": bm,
                "language": lang,
                "em": em,
                "es": es,
                "id_em": id_em,
                "id_f1": id_f1,
                "prediction": pred,
                "groundtruth": gt,
            })

        arm_summary = summarize_benchmark_metrics(records)
        results_by_arm[arm] = arm_summary
        print(f"  ✓ Finished Arm [{arm}] in {time.time() - t_arm_start:.2f}s:")
        print(f"     Macro Mean: EM = {arm_summary['macro_mean']['em']:.2f}%, ES = {arm_summary['macro_mean']['es']:.2f}%")

    # Step 5: Save & Display Comparative Table
    print("\n" + "=" * 95)
    print("FINAL COMPARATIVE EVALUATION ACROSS RETRIEVAL & RERANK PIPELINES (EM & ES)")
    print("=" * 95)
    header = f"{'Configuration':<35} | {'Macro EM':<10} | {'Macro ES':<10} | {'Macro ID-EM':<12} | {'Macro ID-F1':<12}"
    print(header)
    print("-" * len(header))
    for arm, res in results_by_arm.items():
        m = res["macro_mean"]
        print(f"{arm:<35} | {m['em']:>8.2f}% | {m['es']:>8.2f}% | {m['id_em']:>10.2f}% | {m['id_f1']:>10.2f}%")
    print("=" * 95)

    out_file = OUTPUT_DIR / "generation_em_es_results.json"
    with open(out_file, "w") as f:
        json.dump(results_by_arm, f, indent=2)
    print(f"✓ Saved full results to {out_file}\n")
    return results_by_arm


def main():
    parser = argparse.ArgumentParser(description="Evaluate final EM/ES across candidate retrieval and rerank pipelines.")
    parser.add_argument("--arms", nargs="+", default=["standard_tail", "full_left", "cssi_query", "cssi_rerank"], help="Arms to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks (e.g. 500 or omit for full 8004)")
    parser.add_argument("--device", type=str, default="mps" if torch and torch.backends.mps.is_available() else "cpu", help="Device (mps/cuda/cpu)")
    parser.add_argument("--use-generator", action="store_true", help="Run live generator model")
    parser.add_argument("--model-name", type=str, default="deepseek-ai/deepseek-coder-1.3b-base", help="Model name")
    args = parser.parse_args()

    run_evaluation_study(
        arms=args.arms,
        limit=args.limit,
        device=args.device,
        use_generator=args.use_generator,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
