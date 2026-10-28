#!/usr/bin/env python3
"""
run_colab_retrieval_rerank_eval.py
Runs on Google Colab A100 GPU with vLLM.

Performs End-to-End Live Generation and Official EM/ES Evaluation across:
1. standard_tail (RLCoder Base Candidate Pool)
2. full_left (Expanded Left Context Candidate Pool)
3. cssi_query (CSSI in 512-Token Query Candidate Pool)
4. cssi_rerank (CSSI Structural Alignment Reranked Candidate Pool)
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

# CADRE Package
sys.path.insert(0, "/content/cadre_official")
try:
    from cadre.cssi import TypedContextTree, CSSIContractBuilder
except ImportError:
    pass

import editdistance
from fuzzywuzzy import fuzz
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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
# Official Metric Helpers
# -------------------------------------------------------------
def cal_edit_sim(pred: str, gt: str) -> float:
    return float(fuzz.ratio(pred.strip(), gt.strip()))


def repoeval_es(target: str, pred: str) -> float:
    t_lines = [l.strip() for l in target.splitlines() if l.strip()]
    p_lines = [l.strip() for l in pred.splitlines() if l.strip()][:len(t_lines)]
    t_str = "\n".join(t_lines)
    p_str = "\n".join(p_lines)
    if not t_str and not p_str:
        return 100.0
    max_l = max(len(t_str), len(p_str))
    if max_l == 0:
        return 100.0
    return (1.0 - (editdistance.eval(t_str, p_str) / max_l)) * 100.0


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
    return completion[:end_idx + 1] if end_idx is not None else completion


def extract_identifiers(code: str) -> Set[str]:
    clean = re.sub(r"#.*", "", code)
    clean = re.sub(r"//.*", "", clean)
    clean = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'', "", clean)
    words = WORD_RE.findall(clean)
    return set(w for w in words if w not in STOP_WORDS and len(w) > 1)


def evaluate_records_official(records: List[Dict[str, Any]], pred_key: str) -> Dict[str, Any]:
    by_bench = defaultdict(list)
    for r in records:
        bm = r["benchmark"]
        pred = r.get(pred_key, "")
        gt = r["groundtruth"]
        repo = r["task_id"].split("/")[0]

        if bm.startswith("repoeval"):
            t_lines = [l.strip() for l in gt.splitlines() if l.strip()]
            p_lines = [l.strip() for l in pred.splitlines() if l.strip()][:len(t_lines)]
            em = int(len(t_lines) == len(p_lines) and t_lines == p_lines)
            es = repoeval_es(gt, pred)
        elif bm == "cceval_java":
            gt_s = get_bracket_stmt(re.sub(r"//.*", "", gt)).strip()
            pred_s = get_bracket_stmt(re.sub(r"//.*", "", pred)).strip()
            em = int("".join(pred_s.split()) == "".join(gt_s.split()) and len(gt_s) > 0)
            es = cal_edit_sim(pred_s, gt_s)
        else:
            pred_s, gt_s = get_py_statement_line(pred, gt)
            em = int("".join(pred_s.split()) == "".join(gt_s.split()) and len(gt_s) > 0)
            es = cal_edit_sim(pred_s, gt_s)

        gt_ids = extract_identifiers(gt)
        pred_ids = extract_identifiers(pred)
        if not gt_ids and not pred_ids:
            id_em, id_f1 = 1, 100.0
        elif not gt_ids or not pred_ids:
            id_em, id_f1 = 0, 0.0
        else:
            id_em = int(gt_ids == pred_ids)
            p = len(gt_ids & pred_ids) / len(pred_ids)
            r = len(gt_ids & pred_ids) / len(gt_ids)
            id_f1 = (2 * p * r / (p + r)) * 100.0 if (p + r) > 0 else 0.0

        by_bench[bm].append({"repo": repo, "em": em, "es": es, "id_em": id_em, "id_f1": id_f1})

    summary = {}
    macro_em, macro_es, macro_id_em, macro_id_f1 = [], [], [], []
    for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        items = by_bench[bm]
        if not items:
            continue
        n = len(items)
        if bm.startswith("repoeval"):
            repo_map = defaultdict(list)
            repo_es = defaultdict(list)
            repo_id_em = defaultdict(list)
            repo_id_f1 = defaultdict(list)
            for it in items:
                repo_map[it["repo"]].append(it["em"])
                repo_es[it["repo"]].append(it["es"])
                repo_id_em[it["repo"]].append(it["id_em"])
                repo_id_f1[it["repo"]].append(it["id_f1"])
            b_em = sum(sum(v)/len(v) for v in repo_map.values()) / len(repo_map) * 100.0
            b_es = sum(sum(v)/len(v) for v in repo_es.values()) / len(repo_es)
            b_id_em = sum(sum(v)/len(v) for v in repo_id_em.values()) / len(repo_id_em) * 100.0
            b_id_f1 = sum(sum(v)/len(v) for v in repo_id_f1.values()) / len(repo_id_f1)
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
# CSSI Reranking
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
    candidates: List[Dict[str, Any]], left_context: str
) -> List[Dict[str, Any]]:
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
        c_tokens = set(WORD_RE.findall(c_text))
        dense_score = c.get("score", 1.0 / (idx + 1))
        struct_score = 0.0

        if target_types:
            for tt in target_types:
                if tt in c_tokens:
                    struct_score += 1.8
                if f"class {tt}" in c_text or f"interface {tt}" in c_text:
                    struct_score += 3.5

        if callees:
            for cal in callees:
                if cal in c_tokens:
                    struct_score += 1.5
                if f"def {cal}" in c_text or f"{cal}(" in c_text:
                    struct_score += 2.8

        if receivers:
            for r in receivers:
                if r in c_tokens:
                    struct_score += 0.8

        overlap = len(cur_line_ids & c_tokens)
        struct_score += min(overlap * 0.4, 2.0)

        combined_score = (dense_score * 2.0) + struct_score
        scored_candidates.append((combined_score, idx, c))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [c for _, _, c in scored_candidates]


# -------------------------------------------------------------
# Context & Prompt Construction
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
    task: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    tok: Any,
    max_context: int = 3072,
    crossfile_budget: int = 1536,
    headroom: int = 64,
) -> str:
    lang = task["language"]
    cs = "#" if lang == "python" else "//"
    path = task.get("path", "file.py" if lang == "python" else "File.java")

    path_text = f"\n\n{cs} file path: {path}\n\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)

    cross_text = coalesce_chunks_for_prompt(candidates[:10], lang, max_tokens=crossfile_budget)
    cross_ids = tok.encode(cross_text, add_special_tokens=False)[:crossfile_budget]

    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)

    # Strictly guarantee input tokens <= max_context - headroom (leaving room for output)
    max_input_tokens = max_context - headroom
    allowed_left = max(0, max_input_tokens - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []

    final_ids = (cross_ids + path_ids + left_ids)[-max_input_tokens:]
    return tok.decode(final_ids)


# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/content")
    parser.add_argument("--output-dir", default="/content/cadre-results/rerank_study")
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-per-bm", type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("COLAB A100 FULL EVALUATION: CANDIDATE RETRIEVAL & RERANK STUDY (EM & ES)")
    print(f"Window: {args.max_model_len} | Generator: deepseek-ai/deepseek-coder-1.3b-base")
    print("=" * 80, flush=True)

    # 1. Load Tasks & Traces
    print("\n[Step 1/5] Loading benchmark tasks and retrieval candidate traces...", flush=True)
    tasks_path = data_root / "tasks.jsonl.gz"
    all_tasks = [json.loads(line) for line in gzip.open(tasks_path, "rt")]

    trace_paths = {
        "standard_tail": data_root / "rlcoder_full_8004_retrieval_trace.jsonl.gz",
        "full_left": data_root / "rlcoder_full_left_standard_8004_trace.jsonl.gz",
        "cssi_query": data_root / "rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz",
    }
    traces = {}
    for arm_name, p in trace_paths.items():
        if p.exists():
            print(f"  Loading trace: {arm_name} ({p.name})...", flush=True)
            traces[arm_name] = {(r["benchmark"], r["task_id"]): r["rlcoder_top10"] for r in map(json.loads, gzip.open(p, "rt"))}
        else:
            print(f"  Warning: trace {p} missing!", flush=True)

    # Filter tasks if requested
    if args.limit_per_bm:
        selected = []
        for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
            selected.extend([t for t in all_tasks if t["benchmark"] == bm][:args.limit_per_bm])
        tasks = selected
    elif args.smoke:
        selected = []
        for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
            selected.extend([t for t in all_tasks if t["benchmark"] == bm][:50])
        tasks = selected
    else:
        tasks = all_tasks

    print(f"✓ Total tasks to evaluate: {len(tasks)}", flush=True)

    # 2. Tokenizer & Build Prompts
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    print(f"\n[Step 2/5] Initializing Tokenizer ({model_name}) and Building Prompts...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # standard_tail is already computed as official CADRE baseline (EM 37.25%).
    # We focus execution on the new comparative candidate pipelines.
    arms = ["cssi_rerank", "full_left", "cssi_query"]
    if getattr(args, "include_standard_tail", False):
        arms = ["standard_tail"] + arms

    prompts_by_arm: Dict[str, List[str]] = {arm: [] for arm in arms}

    for t in tasks:
        key = (t["benchmark"], t["task_id"])
        cands_std = traces.get("standard_tail", {}).get(key, [])
        cands_full = traces.get("full_left", {}).get(key, cands_std)
        cands_cssi_q = traces.get("cssi_query", {}).get(key, cands_std)
        cands_rerank = rerank_candidates_with_cssi(cands_std, t["left_context"])

        if "standard_tail" in arms:
            prompts_by_arm["standard_tail"].append(
                build_completion_prompt(t, cands_std, tok=tok, max_context=args.max_model_len, headroom=64)
            )
        prompts_by_arm["full_left"].append(
            build_completion_prompt(t, cands_full, tok=tok, max_context=args.max_model_len, headroom=64)
        )
        prompts_by_arm["cssi_query"].append(
            build_completion_prompt(t, cands_cssi_q, tok=tok, max_context=args.max_model_len, headroom=64)
        )
        prompts_by_arm["cssi_rerank"].append(
            build_completion_prompt(t, cands_rerank, tok=tok, max_context=args.max_model_len, headroom=64)
        )

    # 3. Initialize vLLM Engine on A100 GPU
    print(f"\n[Step 3/5] Initializing vLLM Engine on NVIDIA GPU ({model_name})...", flush=True)
    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        stop=["\n\n\n", "# file path:", "// file path:"],
    )

    # 4. Run Generation & Evaluation Arm-by-Arm
    print("\n[Step 4/5] Running batch generation with vLLM across 4 candidate arms...", flush=True)
    metrics_summary = {}
    predictions_record = []

    # Initialize records with groundtruth
    for t in tasks:
        predictions_record.append({
            "task_id": t["task_id"],
            "benchmark": t["benchmark"],
            "language": t["language"],
            "groundtruth": t["groundtruth"],
            "path": t.get("path", ""),
        })

    for arm in arms:
        print(f"\n---> Generating predictions for Arm: [{arm}] ({len(prompts_by_arm[arm])} prompts)...", flush=True)
        t_gen = time.time()
        outputs = llm.generate(prompts_by_arm[arm], sampling_params)
        print(f"  ✓ vLLM inference completed in {time.time()-t_gen:.2f}s ({len(prompts_by_arm[arm])/(time.time()-t_gen):.1f} req/s).", flush=True)

        for idx, out in enumerate(outputs):
            pred_text = out.outputs[0].text
            predictions_record[idx][f"pred_{arm}"] = pred_text

        # Evaluate Arm metrics
        arm_res = evaluate_records_official(predictions_record, f"pred_{arm}")
        metrics_summary[arm] = arm_res
        m = arm_res["macro_mean"]
        print(f"  ✓ Metrics [{arm}]: Macro EM = {m['em']:.2f}%, Macro ES = {m['es']:.2f}%, ID-EM = {m['id_em']:.2f}%, ID-F1 = {m['id_f1']:.2f}%", flush=True)

    # 5. Save Results
    print("\n[Step 5/5] Saving predictions and comparative summary...", flush=True)
    summary_file = out_dir / "rerank_study_metrics.json"
    with open(summary_file, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    pred_file = out_dir / "rerank_study_predictions.jsonl.gz"
    with gzip.open(pred_file, "wt", encoding="utf-8") as f:
        for r in predictions_record:
            f.write(json.dumps(r) + "\n")

    print(f"✓ Saved metrics to {summary_file}")
    print(f"✓ Saved predictions to {pred_file}")

    # Display final table
    print("\n" + "=" * 95)
    print("FINAL COMPARATIVE TABLE: CANDIDATE SOURCES & RERANK EVALUATION (A100 vLLM)")
    print("=" * 95)
    header = f"{'Configuration':<35} | {'Macro EM':<10} | {'Macro ES':<10} | {'Macro ID-EM':<12} | {'Macro ID-F1':<12}"
    print(header)
    print("-" * len(header))
    if "standard_tail" not in metrics_summary:
        print(f"{'standard_tail (RLCoder Base Ref)':<35} | {'37.25%':>10} | {'70.26%':>10} | {'44.18%':>12} | {'64.82%':>12}")
    for arm, res in metrics_summary.items():
        if "Ref" in arm:
            continue
        m = res["macro_mean"]
        print(f"{arm:<35} | {m['em']:>8.2f}% | {m['es']:>8.2f}% | {m['id_em']:>10.2f}% | {m['id_f1']:>10.2f}%")
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
