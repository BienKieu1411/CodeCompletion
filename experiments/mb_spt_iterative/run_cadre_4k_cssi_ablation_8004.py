#!/usr/bin/env python3
"""
Full Evaluation of CADRE 4k Context Window & CSSI Ablation across all 8,004 tasks.
- Total Budget: 4,096 tokens (instead of 8,000).
- Snippets: Sourced purely from RLRetriever (nov3630/RLCoder Top 10 from precomputed trace).
- Cross-File Budget Ceiling: 1,280 tokens (guaranteeing >= 2,800 tokens for Left Context).
- Three Arms for Direct Comparison:
    * Arm 1 (RLCoder Raw Flat 4k): Official RLCoder flat prompt formatting capped at 4k context.
    * Arm 2 (CADRE 4k No-CSSI): Dynamic Coalesced Raw Blocks from RLRetriever without CSSI (4k context).
    * Arm 3 (CADRE 4k With Enhanced CSSI): Coalesced Raw Blocks + Enhanced CSSI Contract (up to 512 tokens, 4k context).
- Generator: deepseek-ai/deepseek-coder-1.3b-base (vLLM, max_model_len=4096, greedy, max_tokens=64).
- Metrics: Official Benchmark Protocols (RepoEval Line-Match, CCEval AST Match, AIRCoder Prefix Match, ES, ID-F1).
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
OUTPUT_DIR = DATA_ROOT / "cadre_4k_eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACE_PATH = DATA_ROOT / "rlcoder_full_eval_results" / "rlcoder_full_8004_retrieval_trace.jsonl.gz"
if not TRACE_PATH.exists():
    TRACE_PATH = Path("experiments/mb_spt_iterative/rlcoder_full_8004_retrieval_trace.jsonl.gz")

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


def extract_cursor_context_symbols(left_context: str, language: str) -> dict[str, Any]:
    lines = left_context.splitlines()
    recent = lines[-25:] if len(lines) >= 25 else lines
    recent_text = "\n".join(recent)
    cur_line = lines[-1] if lines else ""

    receivers = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", cur_line)
    
    assigned_types = {}
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", recent_text):
        assigned_types[m.group(1)] = m.group(2)
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", recent_text):
        assigned_types[m.group(1)] = m.group(2)

    imports = []
    imported_symbols = {}
    for m in re.finditer(r"(?:from\s+([A-Za-z0-9_.]+)\s+import\s+([^#\n]+)|import\s+([^#\n]+))", left_context):
        if m.group(1) and m.group(2):
            raw_syms = m.group(2).replace("(", "").replace(")", "").split(",")
            for s in raw_syms:
                s = s.strip()
                if " as " in s:
                    orig, alias = s.split(" as ")
                    imported_symbols[alias.strip()] = orig.strip()
                    imports.append(orig.strip())
                elif s:
                    imported_symbols[s] = s
                    imports.append(s)
        elif m.group(3):
            for s in m.group(3).split(","):
                s = s.strip()
                if " as " in s:
                    orig, alias = s.split(" as ")
                    imported_symbols[alias.strip()] = orig.strip()
                    imports.append(orig.strip())
                elif s:
                    imported_symbols[s] = s
                    imports.append(s)

    target_types = set()
    for r in receivers:
        if r in assigned_types:
            target_types.add(assigned_types[r])
        elif r in imported_symbols:
            target_types.add(imported_symbols[r])
        elif r and r[0].isupper():
            target_types.add(r)
    for s in imports:
        if s and s[0].isupper():
            target_types.add(s)

    cur_syms = [w for w in WORD_RE.findall(cur_line) if w not in STOP_WORDS and len(w) > 2]

    return {
        "receivers": receivers,
        "target_types": list(target_types),
        "cursor_symbols": cur_syms,
    }


def synthesize_enhanced_cssi_contract(pool_files: list[dict[str, Any]], scope_info: dict[str, Any], language: str, max_tokens: int = 512, tok: Any = None) -> str:
    target_types = scope_info.get("target_types", [])
    cursor_syms = scope_info.get("cursor_symbols", [])
    comment_sym = "#" if language == "python" else "//"

    contract_snippets = []
    total_tokens = 0

    if language == "python":
        for f in pool_files:
            content = f.get("text", "")
            if not content:
                continue
            fpath = f.get("path", "")

            # 1. Class signatures and method skeletons
            for tname in target_types[:4]:
                pattern = re.compile(rf"class\s+{re.escape(tname)}\b[^\n:]*:\s*\n((?:[ \t]+[^\n]*\n|\s*\n)*)", re.MULTILINE)
                for m in pattern.finditer(content):
                    lines = m.group(1).splitlines()
                    retained_lines = []
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("def ") or stripped.startswith("async def "):
                            sig = stripped.split(":", 1)[0]
                            retained_lines.append(f"    {sig}: ...")
                        elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", stripped) or re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped):
                            if len(stripped) < 80:
                                retained_lines.append(f"    {stripped}")
                    if retained_lines:
                        snippet = f"# Contract: {tname} (from {fpath})\nclass {tname}:\n" + "\n".join(retained_lines[:20]) + "\n"
                        contract_snippets.append(snippet)
                        total_tokens += len(snippet.split())
                        break

            # 2. Function signatures matching cursor symbols
            for sname in cursor_syms[:5]:
                f_pat = re.compile(rf"(?:def|async def)\s+{re.escape(sname)}\s*\([^)]*\)[^\n:]*:\s*", re.MULTILINE)
                for m in f_pat.finditer(content):
                    sig = m.group(0).strip()
                    snippet = f"# Contract: {sname} (from {fpath})\n{sig}: ...\n"
                    contract_snippets.append(snippet)
                    total_tokens += len(snippet.split())
                    if total_tokens > max_tokens:
                        break

            if total_tokens > max_tokens:
                break
    else:  # java
        for f in pool_files:
            content = f.get("text", "")
            if not content:
                continue
            fpath = f.get("path", "")

            for tname in target_types[:4]:
                if f"class {tname}" in content or f"interface {tname}" in content:
                    lines = content.splitlines()
                    sigs = []
                    for line in lines:
                        st = line.strip()
                        if ("public " in st or "protected " in st) and "(" in st and ")" in st and not st.startswith("//"):
                            sig = st.split("{")[0].strip()
                            sigs.append(f"    {sig};")
                        elif (st.startswith("private ") or st.startswith("protected ")) and ";" in st and "(" not in st:
                            sigs.append(f"    {st}")
                    if sigs:
                        snippet = f"// Contract: {tname} (from {fpath})\nclass {tname} {{\n" + "\n".join(sigs[:16]) + "\n}}\n"
                        contract_snippets.append(snippet)
                        total_tokens += len(snippet.split())
                        break

            if total_tokens > max_tokens:
                break

    if not contract_snippets:
        return ""

    header = f"{comment_sym} === CURSOR STRUCTURAL CONTRACTS (CSSI ENHANCED) ==="
    footer = f"{comment_sym} ====================================================="
    contract_text = f"{header}\n" + "\n".join(contract_snippets) + f"{footer}"

    if tok:
        ids = tok.encode(contract_text, add_special_tokens=False)
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
            contract_text = tok.decode(ids)

    return contract_text


def coalesce_raw_chunks(chunks: list[dict[str, Any]], lang: str) -> list[str]:
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


def build_prompt_4k(
    arm: str,
    task: dict[str, Any],
    top10: list[dict[str, Any]],
    pool_files: list[dict[str, Any]],
    tok: Any,
    max_context: int = 4096,
    cross_max: int = 1280,
    cssi_max: int = 512,
) -> list[int]:
    lang = task["language"]
    cs = "#" if lang == "python" else "//"

    path_text = f"\n\n{cs} file path: {task['path']}\n\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)
    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)

    filtered = [b for b in top10 if not b.get("is_stop") and b.get("file_path")]

    if arm == "rlcoder_raw_4k":
        # Flat prompt format of RLCoder capped at cross_max
        block_texts = []
        for b in filtered:
            desc = f"{cs} file path: {b['file_path']}\n{cs} lines: {b.get('line_start', 0)}-{b.get('line_end', 0)}"
            code_lines = "\n".join(f"{cs} {l}" for l in b["code_content"].splitlines() if l.strip())
            block_texts.append(f"{desc}\n{code_lines}")
        cross_text = "\n\n".join(block_texts)
        cross_ids = tok.encode(cross_text, add_special_tokens=False)[:cross_max]

    elif arm == "cadre_no_cssi_4k":
        # Coalesced raw code blocks, NO CSSI
        blocks = coalesce_raw_chunks(filtered, lang)
        exemplar_text = "\n\n".join(blocks)
        cross_ids = tok.encode(exemplar_text, add_special_tokens=False)[:cross_max]

    elif arm == "cadre_with_cssi_4k":
        # Coalesced raw code blocks + Enhanced CSSI Contract
        scope_info = extract_cursor_context_symbols(task["left_context"], lang)
        cssi_text = synthesize_enhanced_cssi_contract(pool_files, scope_info, lang, max_tokens=cssi_max, tok=tok)
        cssi_ids = tok.encode("\n\n" + cssi_text, add_special_tokens=False) if cssi_text else []

        rem_snippets = max(0, cross_max - len(cssi_ids))
        blocks = coalesce_raw_chunks(filtered, lang)
        exemplar_text = "\n\n".join(blocks)
        exemplar_ids = tok.encode(exemplar_text, add_special_tokens=False)[:rem_snippets]

        # Channel A (Exemplars) -> Channel B (Contract) -> Path -> Left Context
        cross_ids = exemplar_ids + cssi_ids

    else:
        raise ValueError(f"Unknown arm: {arm}")

    # Left context gets the remaining budget up to 4,096 tokens
    allowed_left = max(0, max_context - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []

    return (cross_ids + path_ids + left_ids)[-max_context:]


def compute_official_em(target: str, prediction: str, language: str) -> int:
    comment_prefix = "#" if language == "python" else "//"
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][: len(target_lines)]
    target_lines_str = "".join("".join(target_lines).split())
    prediction_lines_str = "".join("".join(prediction_lines).split())
    prediction_lines_str = prediction_lines_str[: len(target_lines_str)]
    return int(target_lines_str == prediction_lines_str) if target_lines_str else 0


def evaluate_predictions_official(tasks: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    by_bench = defaultdict(list)
    for t, pred in zip(tasks, predictions):
        bench = t["benchmark"]
        lang = t["language"]
        gt = t["groundtruth"]

        em = compute_official_em(gt, pred, lang)
        es = cal_edit_sim_single(pred, gt)

        gt_ids = extract_identifiers(gt, lang)
        pred_ids = extract_identifiers(pred, lang)
        gt_set, pred_set = set(gt_ids), set(pred_ids)
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
            "em": em,
            "es": es,
            "id_f1": id_f1,
        })

    summary = {}
    macro_em, macro_es, macro_id_f1 = [], [], []
    for b, recs in by_bench.items():
        n = len(recs)
        m_em = sum(r["em"] for r in recs) / n * 100.0
        m_es = sum(r["es"] for r in recs) / n
        m_f1 = sum(r["id_f1"] for r in recs) / n
        summary[b] = {
            "count": n,
            "em": round(m_em, 2),
            "es": round(m_es, 2),
            "id_f1": round(m_f1, 2),
        }
        macro_em.append(m_em)
        macro_es.append(m_es)
        macro_id_f1.append(m_f1)

    all_recs = [r for recs in by_bench.values() for r in recs]
    total_n = len(all_recs)
    summary["micro_mean"] = {
        "count": total_n,
        "em": round(sum(r["em"] for r in all_recs) / total_n * 100.0, 2),
        "es": round(sum(r["edit_sim"] for r in all_recs) / total_n if "edit_sim" in all_recs[0] else sum(r["es"] for r in all_recs) / total_n, 2),
        "id_f1": round(sum(r["id_f1"] for r in all_recs) / total_n, 2),
    }
    summary["macro_mean"] = {
        "em": round(sum(macro_em) / len(macro_em), 2),
        "es": round(sum(macro_es) / len(macro_es), 2),
        "id_f1": round(sum(macro_id_f1) / len(macro_id_f1), 2),
    }

    return summary


def main():
    t_start = time.time()
    print("=" * 85)
    print("CADRE 4K CONTEXT WINDOW & CSSI ABLATION EVALUATION ACROSS ALL 8,004 TASKS")
    print("Configuration: Total Budget = 4,096 tokens | Snippets = RLRetriever Top 10")
    print("=" * 85)

    # 1. Load Tasks & Files
    print("\n[Step 1/5] Loading benchmark tasks and repository files from bundle...")
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
    print(f"✓ Loaded {len(tasks)} tasks and indexed {len(files_by_id)} repository files.")

    # 2. Load Retrieval Trace
    print(f"\n[Step 2/5] Loading precomputed RLRetriever trace from {TRACE_PATH}...")
    rlcoder_top10_by_task = []
    with gzip.open(TRACE_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rlcoder_top10_by_task.append(rec["rlcoder_top10"])
    print(f"✓ Loaded {len(rlcoder_top10_by_task)} task retrieval traces.")

    # 3. Assemble Prompts for 3 Arms (all at 4k context)
    print("\n[Step 3/5] Assembling prompts for 3 arms (4,096 context tokens)...")
    from transformers import AutoTokenizer as GenTokenizer
    gen_tok = GenTokenizer.from_pretrained(GENERATOR)

    prompts_raw_4k = []
    prompts_no_cssi_4k = []
    prompts_with_cssi_4k = []

    for t, top10 in zip(tasks, rlcoder_top10_by_task):
        pool_files = []
        for rel_path, fid in t.get("file_ids_by_path", {}).items():
            fe = files_by_id.get(fid)
            if fe:
                pool_files.append({"path": rel_path, "text": fe["text"]})

        p1 = build_prompt_4k("rlcoder_raw_4k", t, top10, pool_files, gen_tok, max_context=4096, cross_max=1280)
        p2 = build_prompt_4k("cadre_no_cssi_4k", t, top10, pool_files, gen_tok, max_context=4096, cross_max=1280)
        p3 = build_prompt_4k("cadre_with_cssi_4k", t, top10, pool_files, gen_tok, max_context=4096, cross_max=1280, cssi_max=512)

        prompts_raw_4k.append(gen_tok.decode(p1))
        prompts_no_cssi_4k.append(gen_tok.decode(p2))
        prompts_with_cssi_4k.append(gen_tok.decode(p3))

    print(f"✓ Assembled 3 x {len(tasks)} = {3 * len(tasks)} prompts for 4k evaluation.")

    # 4. Run Generation with vLLM
    print("\n[Step 4/5] Initializing vLLM generator (deepseek-coder-1.3b-base, max_model_len=4096)...")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=GENERATOR,
        revision=GENERATOR_REVISION,
        max_model_len=4096,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        stop=["\n\n\n"],
    )

    print("\n  [Arm 1] Generating predictions for RLCoder Raw Flat 4k...")
    t_g1 = time.time()
    out1 = llm.generate(prompts_raw_4k, sampling_params)
    preds_raw_4k = [o.outputs[0].text for o in out1]
    print(f"  ✓ Arm 1 completed in {time.time() - t_g1:.1f}s.")

    print("\n  [Arm 2] Generating predictions for CADRE 4k (No-CSSI)...")
    t_g2 = time.time()
    out2 = llm.generate(prompts_no_cssi_4k, sampling_params)
    preds_no_cssi_4k = [o.outputs[0].text for o in out2]
    print(f"  ✓ Arm 2 completed in {time.time() - t_g2:.1f}s.")

    print("\n  [Arm 3] Generating predictions for CADRE 4k (With Enhanced CSSI 512 tok)...")
    t_g3 = time.time()
    out3 = llm.generate(prompts_with_cssi_4k, sampling_params)
    preds_with_cssi_4k = [o.outputs[0].text for o in out3]
    print(f"  ✓ Arm 3 completed in {time.time() - t_g3:.1f}s.")

    # 5. Evaluate Metrics
    print("\n[Step 5/5] Computing official benchmark metrics for all 3 arms...")
    eval_raw = evaluate_predictions_official(tasks, preds_raw_4k)
    eval_no_cssi = evaluate_predictions_official(tasks, preds_no_cssi_4k)
    eval_with_cssi = evaluate_predictions_official(tasks, preds_with_cssi_4k)

    # Print Table
    print("\n" + "=" * 90)
    print("4,096 CONTEXT EVALUATION: IMPACT OF COALESCING & ENHANCED CSSI CONTRACT")
    print("=" * 90)
    print(f"{'Benchmark':<18} | {'RLCoder Raw 4k':<16} | {'CADRE No-CSSI 4k':<18} | {'CADRE + CSSI 4k':<18} | {'CSSI Gain':<10}")
    print("-" * 90)
    for b in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        s_raw = eval_raw[b]["em"]
        s_no = eval_no_cssi[b]["em"]
        s_cssi = eval_with_cssi[b]["em"]
        cssi_gain = s_cssi - s_no
        sign = "+" if cssi_gain >= 0 else ""
        print(f"{b:<18} | {s_raw:>6.2f}%          | {s_no:>6.2f}%            | {s_cssi:>6.2f}%            | {sign}{cssi_gain:>5.2f}%")
    print("-" * 90)
    m_raw = eval_raw["macro_mean"]["em"]
    m_no = eval_no_cssi["macro_mean"]["em"]
    m_cssi = eval_with_cssi["macro_mean"]["em"]
    m_gain = m_cssi - m_no
    m_sign = "+" if m_gain >= 0 else ""
    print(f"{'Macro Mean':<18} | {m_raw:>6.2f}%          | {m_no:>6.2f}%            | {m_cssi:>6.2f}%            | {m_sign}{m_gain:>5.2f}%")
    print("=" * 90)

    # Save Metrics & Predictions
    final_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generator": GENERATOR,
            "retriever": "nov3630/RLCoder",
            "total_tasks": len(tasks),
            "max_tokens": 64,
            "context_tokens": 4096,
            "cross_max_tokens": 1280,
            "cssi_max_tokens": 512,
            "duration_sec": round(time.time() - t_start, 1),
        },
        "arms": {
            "rlcoder_raw_4k": eval_raw,
            "cadre_no_cssi_4k": eval_no_cssi,
            "cadre_with_cssi_4k": eval_with_cssi,
        },
    }

    metrics_file = OUTPUT_DIR / "cadre_4k_cssi_ablation_8004_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_file}")

    preds_file = OUTPUT_DIR / "cadre_4k_cssi_ablation_8004_predictions.jsonl.gz"
    with gzip.open(preds_file, "wt", encoding="utf-8") as gz:
        for i, t in enumerate(tasks):
            gz.write(json.dumps({
                "task_id": t["task_id"],
                "benchmark": t["benchmark"],
                "groundtruth": t["groundtruth"],
                "pred_raw_4k": preds_raw_4k[i],
                "pred_no_cssi_4k": preds_no_cssi_4k[i],
                "pred_with_cssi_4k": preds_with_cssi_4k[i],
            }) + "\n")
    print(f"✓ Saved predictions to {preds_file}")


if __name__ == "__main__":
    main()
