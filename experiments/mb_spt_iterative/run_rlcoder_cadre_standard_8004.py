#!/usr/bin/env python3
"""
Full Evaluation of RLCoder Baseline vs RLCoder + CADRE Standard (Raw Blocks) vs CADRE+ across all 8,004 tasks.
- Stage 1: Load precomputed RLRetriever (nov3630/RLCoder) top-10 retrieval trace (8,004 tasks).
- Stage 2: Load pre-generated RLCoder Raw Baseline and CADRE+ predictions from previous run.
- Stage 3: Assemble prompts for CADRE Standard:
    * Channel A: Multi-Chunk Coalescing of Raw Code Blocks (separated by file path and \\n...\\n, un-commented raw code)
    * Channel B: Synthesized Cursor Structural Contract (CSSI, up to 300-384 tokens)
    * Dynamic Budget: up to 3,584 tokens for cross-file context
    * DCA Dual-Channel Assembly up to 8,000 context
- Stage 4: Run vLLM generation on deepseek-coder-1.3b-base (greedy decoding, max_tokens=64, 8k context).
- Stage 5: Evaluate using OFFICIAL benchmark metrics:
    * RepoEval (API & Line): Official Microsoft RepoEval line-match truncation:
        pred_lines[:len(gt_lines)] == gt_lines
    * CrossCodeEval (Python): AST statement extraction / first valid statement
    * CrossCodeEval (Java): Semi-colon / bracket statement extraction
    * AIRCoder Prefix Match: pred[:len(gt)] == gt
    * Edit Similarity: Levenshtein ratio
    * Identifier F1: Identifiers set overlap
- Stage 6: Output comprehensive 3-way comparison table and save metrics/predictions.
"""

import ast
import difflib
import gc
import gzip
import json
import os
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

DATA_ROOT = Path("/content/mb_spt") if Path("/content/mb_spt").exists() else Path("experiments/mb_spt_iterative")
BUNDLE_PATH = DATA_ROOT / "full_benchmark_8004_bundle.tar.gz"
OUTPUT_DIR = DATA_ROOT / "rlcoder_full_eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACE_PATH = OUTPUT_DIR / "rlcoder_full_8004_retrieval_trace.jsonl.gz"
PREV_PREDS_PATH = OUTPUT_DIR / "rlcoder_cadre_full_8004_predictions.jsonl.gz"

GENERATOR = "deepseek-ai/deepseek-coder-1.3b-base"
GENERATOR_REVISION = "main"

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


def extract_identifiers(code: str, language: str) -> list[str]:
    words = WORD_RE.findall(code)
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]


def cal_edit_sim_single(pred: str, gt: str) -> float:
    try:
        from fuzzywuzzy import fuzz
        return float(fuzz.ratio(pred.strip(), gt.strip()))
    except Exception:
        return difflib.SequenceMatcher(None, pred.strip(), gt.strip()).ratio() * 100.0


def get_py_statement_ast(prompt: str, raw_pred: str) -> str:
    lines = raw_pred.splitlines()
    if not lines:
        return raw_pred.strip()
    cur = ""
    for l in lines:
        if cur:
            cur += "\n" + l
        else:
            cur = l
        code = prompt + cur
        try:
            ast.parse(code)
            return cur.strip()
        except SyntaxError:
            continue
    return lines[0].strip()


def get_java_statement(raw_pred: str) -> str:
    end_idx = None
    for i in range(len(raw_pred)):
        if raw_pred[i] in [";", "}", "{"]:
            end_idx = i
            break
    return raw_pred[:end_idx + 1].strip() if end_idx is not None else raw_pred.strip()


def extract_statement(prompt: str, raw_pred: str, benchmark: str, language: str, gt: str) -> str:
    # 1. Official RepoEval Protocol: match first N non-empty lines matching ground truth
    if "repoeval" in benchmark:
        gt_lines = [l.strip() for l in gt.splitlines() if l.strip()]
        n = max(1, len(gt_lines))
        pred_lines = [l.strip() for l in raw_pred.splitlines() if l.strip()][:n]
        return "\n".join(pred_lines)
    # 2. CrossCodeEval Python Protocol: AST parse statement
    elif language == "python":
        return get_py_statement_ast(prompt, raw_pred)
    # 3. CrossCodeEval Java Protocol: Bracket/semicolon statement
    else:
        return get_java_statement(raw_pred)


def extract_cursor_context_symbols(left_context: str, language: str) -> list[str]:
    lines = left_context.splitlines()
    recent = lines[-25:] if len(lines) >= 25 else lines
    recent_text = "\n".join(recent)
    cur_line = lines[-1] if lines else ""

    symbols = []
    m_rec = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", cur_line)
    symbols.extend(m_rec)

    for m in re.finditer(r"(?:from\s+[A-Za-z0-9_.]+\s+import\s+([^#\n]+)|import\s+([^#\n]+))", recent_text):
        clause = m.group(1) or m.group(2)
        if clause:
            for part in clause.split(","):
                part_tokens = part.strip().split()
                if part_tokens:
                    imports_name = part_tokens[0]
                    if imports_name not in STOP_WORDS:
                        symbols.append(imports_name)

    cur_words = [w for w in WORD_RE.findall(cur_line) if w not in STOP_WORDS and len(w) > 2]
    symbols.extend(cur_words)

    seen = set()
    res = []
    for s in symbols:
        if s not in seen and s not in STOP_WORDS:
            seen.add(s)
            res.append(s)
    return res


def extract_cssi_contract(symbols: list[str], pool_files: list[dict[str, Any]], language: str, max_tokens: int = 300, tok: Any = None) -> str:
    if not symbols or not pool_files:
        return ""

    comment_sym = "#" if language == "python" else "//"
    matched_defs = []

    for f in pool_files:
        text = f.get("text", "")
        if not text:
            continue
        rel_path = f.get("path", "")

        for sym in symbols[:5]:
            if language == "python":
                patterns = [
                    (f"class {sym}", rf"(class\s+{re.escape(sym)}\b[^\n:]*:)"),
                    (f"def {sym}", rf"(def\s+{re.escape(sym)}\s*\([^)]*\)[^\n:]*:)"),
                ]
            else:
                patterns = [
                    (f"class {sym}", rf"((?:public\s+|protected\s+)?class\s+{re.escape(sym)}\b[^{{]*)"),
                    (f"method {sym}", rf"((?:public\s+|protected\s+|private\s+)?(?:static\s+)?[A-Za-z0-9_<>\[\]]+\s+{re.escape(sym)}\s*\([^)]*\))"),
                ]

            for label, pat in patterns:
                m = re.search(pat, text)
                if m:
                    sig = m.group(1).strip()
                    if language == "python":
                        sig += " ..."
                    else:
                        sig += " { ... }"
                    matched_defs.append(f"{comment_sym} [{rel_path}]\n{sig}")
                    break

        if len(matched_defs) >= 4:
            break

    if not matched_defs:
        return ""

    header = f"{comment_sym} === CURSOR STRUCTURAL CONTRACTS (CSSI) ==="
    footer = f"{comment_sym} =========================================="
    contract_text = f"{header}\n" + "\n\n".join(matched_defs) + f"\n{footer}"

    if tok:
        ids = tok.encode(contract_text, add_special_tokens=False)
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
            contract_text = tok.decode(ids)

    return contract_text


def coalesce_raw_chunks(chunks: list[dict[str, Any]], lang: str) -> list[str]:
    """
    Standard CADRE: Groups chunks by file_path, coalesces contiguous/close lines
    or joins multiple slices from the same file with '\\n...\\n'.
    Keeps code as RAW un-commented syntax.
    """
    comment_sym = "#" if lang == "python" else "//"
    chunks_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        p = c.get("file_path")
        if p and not c.get("is_stop"):
            chunks_by_file[p].append(c)

    blocks = []
    for p, f_chunks in chunks_by_file.items():
        f_chunks.sort(key=lambda x: x.get("line_start", 0))
        code_slices = []
        for ch in f_chunks:
            code = ch.get("code_content", "").strip()
            if code:
                code_slices.append(code)

        if code_slices:
            coalesced_code = "\n...\n".join(code_slices)
            block = f"{comment_sym} file path: {p}\n{coalesced_code}"
            blocks.append(block)

    return blocks


def build_cadre_standard_prompt(task: dict[str, Any], top10: list[dict[str, Any]], pool_files: list[dict[str, Any]], tok: Any, max_context: int = 8000) -> list[int]:
    lang = task["language"]
    comment_sym = "#" if lang == "python" else "//"

    filtered = [b for b in top10 if not b.get("is_stop") and b.get("file_path")]

    path_text = f"\n\n{comment_sym} file path: {task['path']}\n\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)
    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)

    # Channel B: CSSI Contract
    symbols = extract_cursor_context_symbols(task["left_context"], lang)
    cssi_contract = extract_cssi_contract(symbols, pool_files, lang, max_tokens=300, tok=tok)
    cssi_ids = tok.encode("\n\n" + cssi_contract, add_special_tokens=False) if cssi_contract else []

    # Channel A: Raw Coalesced Exemplars
    rem_cross = max(0, 3584 - len(cssi_ids))
    blocks = coalesce_raw_chunks(filtered, lang)
    exemplar_text = "\n\n".join(blocks)
    exemplar_ids = tok.encode(exemplar_text, add_special_tokens=False)[:rem_cross]

    cross_ids = exemplar_ids + cssi_ids
    allowed_left = max(0, max_context - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []

    return (cross_ids + path_ids + left_ids)[-max_context:]


def evaluate_predictions(tasks: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t, pred in zip(tasks, predictions):
        bench = t["benchmark"]
        lang = t["language"]
        gt = t["groundtruth"]

        # 1. Statement Exact Match (Official Alignment)
        pred_stmt = extract_statement(t["left_context"], pred, bench, lang, gt)
        gt_stmt = extract_statement(t["left_context"], gt, bench, lang, gt)
        aligncoder_em = int(pred_stmt.strip() == gt_stmt.strip())

        # 2. AIRCoder Prefix Match Truncation
        gt_clean = "".join(gt.split())
        pred_clean = "".join(pred.split())
        pred_trunc = pred_clean[:len(gt_clean)]
        aircoder_em = int(pred_trunc == gt_clean) if gt_clean else 0

        # 3. Exact String Match Uncut
        exact_uncut = int(pred.strip() == gt.strip())

        # 4. Edit Similarity
        es = cal_edit_sim_single(pred, gt)

        # 5. Identifier EM & F1
        gt_ids = extract_identifiers(gt, lang)
        pred_ids = extract_identifiers(pred, lang)
        gt_set, pred_set = set(gt_ids), set(pred_ids)
        id_em = int(gt_set == pred_set) if gt_set else int(len(pred_set) == 0)
        if not gt_set and not pred_set:
            id_f1 = 100.0
        elif not gt_set or not pred_set:
            id_f1 = 0.0
        else:
            p = len(gt_set & pred_set) / len(pred_set)
            r = len(gt_set & pred_set) / len(gt_set)
            id_f1 = (2 * p * r / (p + r)) * 100.0 if (p + r) > 0 else 0.0

        by_bench[bench].append({
            "task_id": t["task_id"],
            "aligncoder_em": aligncoder_em,
            "aircoder_em": aircoder_em,
            "exact_uncut": exact_uncut,
            "edit_sim": es,
            "id_em": id_em,
            "id_f1": id_f1,
            "pred": pred,
            "pred_stmt": pred_stmt,
            "gt_stmt": gt_stmt,
        })

    summary: dict[str, Any] = {}
    macro_aligncoder_em = []
    macro_aircoder_em = []
    macro_es = []
    macro_id_f1 = []

    for b, recs in by_bench.items():
        n = len(recs)
        ast_em = sum(r["aligncoder_em"] for r in recs) / n * 100.0
        pref_em = sum(r["aircoder_em"] for r in recs) / n * 100.0
        uncut_em = sum(r["exact_uncut"] for r in recs) / n * 100.0
        mean_es = sum(r["edit_sim"] for r in recs) / n
        mean_id_em = sum(r["id_em"] for r in recs) / n * 100.0
        mean_id_f1 = sum(r["id_f1"] for r in recs) / n

        summary[b] = {
            "count": n,
            "aligncoder_ast_em": round(ast_em, 2),
            "aircoder_prefix_em": round(pref_em, 2),
            "exact_uncut_em": round(uncut_em, 2),
            "edit_sim": round(mean_es, 2),
            "id_em": round(mean_id_em, 2),
            "id_f1": round(mean_id_f1, 2),
        }
        macro_aligncoder_em.append(ast_em)
        macro_aircoder_em.append(pref_em)
        macro_es.append(mean_es)
        macro_id_f1.append(mean_id_f1)

    all_recs = [r for recs in by_bench.values() for r in recs]
    total_n = len(all_recs)
    summary["micro_mean"] = {
        "count": total_n,
        "aligncoder_ast_em": round(sum(r["aligncoder_em"] for r in all_recs) / total_n * 100.0, 2),
        "aircoder_prefix_em": round(sum(r["aircoder_em"] for r in all_recs) / total_n * 100.0, 2),
        "exact_uncut_em": round(sum(r["exact_uncut"] for r in all_recs) / total_n * 100.0, 2),
        "edit_sim": round(sum(r["edit_sim"] for r in all_recs) / total_n, 2),
        "id_em": round(sum(r["id_em"] for r in all_recs) / total_n * 100.0, 2),
        "id_f1": round(sum(r["id_f1"] for r in all_recs) / total_n, 2),
    }

    k_bench = len(macro_aligncoder_em)
    summary["macro_mean"] = {
        "aligncoder_ast_em": round(sum(macro_aligncoder_em) / k_bench, 2),
        "aircoder_prefix_em": round(sum(macro_aircoder_em) / k_bench, 2),
        "edit_sim": round(sum(macro_es) / k_bench, 2),
        "id_f1": round(sum(macro_id_f1) / k_bench, 2),
    }

    return {"summary": summary, "per_task": by_bench}


def main():
    t_start = time.time()
    print("=" * 80)
    print("RLCODER EVALUATION: RLCODER RAW BASELINE vs RLCODER + CADRE STANDARD (RAW BLOCKS)")
    print("Benchmark Scope: All 8,004 tasks (CCEval-Py, CCEval-Java, RepoEval-API, RepoEval-Line)")
    print("=" * 80)

    # 1. Load Benchmark Tasks
    print("\n[Step 1/5] Loading full benchmark bundle (8,004 tasks)...")
    with tarfile.open(BUNDLE_PATH, "r:gz") as tar:
        f_tasks = tar.extractfile("tasks.jsonl.gz")
        with gzip.open(f_tasks, "rt") as gz:
            tasks = [json.loads(line) for line in gz]
        f_files = tar.extractfile("files.jsonl.gz")
        files_by_id = {}
        with gzip.open(f_files, "rt") as gz:
            for line in gz:
                d = json.loads(line)
                files_by_id[d["file_id"]] = d

    print(f"✓ Loaded {len(tasks)} tasks across 4 benchmarks.")
    print(f"✓ Indexed {len(files_by_id)} repository files.")

    # 2. Load Precomputed Retrieval Trace
    print(f"\n[Step 2/5] Loading precomputed RLRetriever trace from {TRACE_PATH}...")
    rlcoder_top10_by_task = []
    with gzip.open(TRACE_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rlcoder_top10_by_task.append(rec["rlcoder_top10"])
    print(f"✓ Loaded {len(rlcoder_top10_by_task)} task retrieval traces.")

    # 3. Load Pre-generated RLCoder Raw Baseline & CADRE+ Predictions
    print(f"\n[Step 3/5] Loading pre-generated RLCoder Raw Baseline predictions from {PREV_PREDS_PATH}...")
    raw_preds = []
    cadre_plus_preds = []
    with gzip.open(PREV_PREDS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            raw_preds.append(rec["pred_raw"])
            cadre_plus_preds.append(rec.get("pred_cadre", ""))
    print(f"✓ Loaded {len(raw_preds)} baseline & CADRE+ predictions (Saved 20 minutes of GPU inference!).")

    # 4. Assemble Prompts for CADRE Standard (Raw Blocks)
    print("\n[Step 4/5] Assembling prompts for CADRE Standard (Raw Blocks + Coalescing + CSSI Contract)...")
    from transformers import AutoTokenizer as GenTokenizer
    gen_tok = GenTokenizer.from_pretrained(GENERATOR)

    cadre_std_prompts = []
    for t, top10 in zip(tasks, rlcoder_top10_by_task):
        pool_files = []
        for rel_path, fid in t.get("file_ids_by_path", {}).items():
            fe = files_by_id.get(fid)
            if fe:
                pool_files.append({"path": rel_path, "text": fe["text"]})

        p_ids = build_cadre_standard_prompt(t, top10, pool_files, gen_tok, max_context=8000)
        cadre_std_prompts.append(gen_tok.decode(p_ids))
    print(f"✓ Assembled {len(cadre_std_prompts)} prompts for CADRE Standard.")

    # 5. Run Generation on vLLM
    print("\n[Step 5/5] Initializing vLLM generator (deepseek-coder-1.3b-base)...")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=GENERATOR,
        revision=GENERATOR_REVISION,
        max_model_len=8192,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        stop=["\n\n\n"],
    )

    print("\n  Generating predictions for RLCoder + CADRE Standard (8,004 tasks)...")
    t_gen = time.time()
    cadre_std_outputs = llm.generate(cadre_std_prompts, sampling_params)
    cadre_std_preds = [o.outputs[0].text for o in cadre_std_outputs]
    print(f"  ✓ CADRE Standard generated in {time.time() - t_gen:.1f}s.")

    # 6. Evaluate All Metrics across All 3 Arms
    print("\n[Evaluation] Computing official metrics across all 8,004 tasks...")
    eval_raw = evaluate_predictions(tasks, raw_preds)
    eval_cadre_std = evaluate_predictions(tasks, cadre_std_preds)
    eval_cadre_plus = evaluate_predictions(tasks, cadre_plus_preds) if cadre_plus_preds else None

    # Print Comparison Table: AlignCoder Statement Match
    print("\n" + "=" * 90)
    print("FINAL 8,004 BENCHMARK RESULTS: STATEMENT EXACT MATCH (ALIGNCODER OFFICIAL)")
    print("=" * 90)
    print(f"{'Benchmark':<18} | {'RLCoder Raw':<14} | {'RLCoder+CADRE Std':<18} | {'Delta':<10} | {'RLCoder+CADRE+':<14}")
    print("-" * 90)
    for b in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        s_raw = eval_raw["summary"][b]["aligncoder_ast_em"]
        s_std = eval_cadre_std["summary"][b]["aligncoder_ast_em"]
        delta = s_std - s_raw
        sign = "+" if delta >= 0 else ""
        s_plus = eval_cadre_plus["summary"][b]["aligncoder_ast_em"] if eval_cadre_plus else 0.0
        print(f"{b:<18} | {s_raw:>6.2f}%        | {s_std:>6.2f}%           | {sign}{delta:>5.2f}%   | {s_plus:>6.2f}%")
    print("-" * 90)
    m_raw = eval_raw["summary"]["macro_mean"]["aligncoder_ast_em"]
    m_std = eval_cadre_std["summary"]["macro_mean"]["aligncoder_ast_em"]
    m_delta = m_std - m_raw
    m_sign = "+" if m_delta >= 0 else ""
    m_plus = eval_cadre_plus["summary"]["macro_mean"]["aligncoder_ast_em"] if eval_cadre_plus else 0.0
    print(f"{'Macro Mean AST':<18} | {m_raw:>6.2f}%        | {m_std:>6.2f}%           | {m_sign}{m_delta:>5.2f}%   | {m_plus:>6.2f}%")
    print("=" * 90)

    # Print Prefix EM Comparison
    print(f"\n{'Benchmark':<18} | {'RLCoder Raw (Pref)':<18} | {'RLCoder+CADRE Std':<18} | {'Delta':<10} | {'RLCoder+CADRE+':<14}")
    print("-" * 90)
    for b in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        p_raw = eval_raw["summary"][b]["aircoder_prefix_em"]
        p_std = eval_cadre_std["summary"][b]["aircoder_prefix_em"]
        delta = p_std - p_raw
        sign = "+" if delta >= 0 else ""
        p_plus = eval_cadre_plus["summary"][b]["aircoder_prefix_em"] if eval_cadre_plus else 0.0
        print(f"{b:<18} | {p_raw:>6.2f}%            | {p_std:>6.2f}%           | {sign}{delta:>5.2f}%   | {p_plus:>6.2f}%")
    print("-" * 90)
    pm_raw = eval_raw["summary"]["macro_mean"]["aircoder_prefix_em"]
    pm_std = eval_cadre_std["summary"]["macro_mean"]["aircoder_prefix_em"]
    pm_delta = pm_std - pm_raw
    pm_sign = "+" if pm_delta >= 0 else ""
    pm_plus = eval_cadre_plus["summary"]["macro_mean"]["aircoder_prefix_em"] if eval_cadre_plus else 0.0
    print(f"{'Macro Mean Pref':<18} | {pm_raw:>6.2f}%            | {pm_std:>6.2f}%           | {pm_sign}{pm_delta:>5.2f}%   | {pm_plus:>6.2f}%")
    print("=" * 90)

    # Save Output
    final_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generator": GENERATOR,
            "retriever": "nov3630/RLCoder",
            "total_tasks": len(tasks),
            "max_tokens": 64,
            "context_tokens": 8000,
            "duration_sec": round(time.time() - t_start, 1),
        },
        "arms": {
            "rlcoder_raw_8004": eval_raw["summary"],
            "rlcoder_cadre_standard_8004": eval_cadre_std["summary"],
            "rlcoder_cadre_plus_8004": eval_cadre_plus["summary"] if eval_cadre_plus else {},
        },
    }

    metrics_file = OUTPUT_DIR / "rlcoder_cadre_standard_full_8004_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n✓ Saved 3-arm metrics to {metrics_file}")

    preds_file = OUTPUT_DIR / "rlcoder_cadre_standard_full_8004_predictions.jsonl.gz"
    with gzip.open(preds_file, "wt", encoding="utf-8") as gz:
        for i, t in enumerate(tasks):
            gz.write(json.dumps({
                "task_id": t["task_id"],
                "benchmark": t["benchmark"],
                "groundtruth": t["groundtruth"],
                "pred_raw": raw_preds[i],
                "pred_cadre_std": cadre_std_preds[i],
                "pred_cadre_plus": cadre_plus_preds[i] if cadre_plus_preds else "",
            }) + "\n")
    print(f"✓ Saved 3-arm predictions to {preds_file}")


if __name__ == "__main__":
    main()
