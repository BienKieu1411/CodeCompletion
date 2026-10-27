#!/usr/bin/env python3
"""
run_cadre_consolidated_full_8004.py
CADRE Consolidated (BẢN HIỆN TẠI) - Full 8,004 Tasks Evaluation on RLCoder.

Architecture (Exact Setting of CADRE Consolidated):
1. Retrieval: Top-10 candidates directly from RLCoder RLRetriever (Zero modification to retriever).
2. Budgets:
   - Exemplars (Channel A): 1,536 tokens (RLCoder cross-file limit).
   - Tree-CSSI (Channel B): 300 tokens (TypedContextTree with inheritance & MRO resolution).
   - Target API Spec (Channel C): 300 tokens (Resolved from Pass 1B speculative draft).
   - Active File Context: 512 tokens (RLCoder in-file left context limit).
   - Max prompt tokens: ~2,680 tokens (Total model len: 4,096 tokens).
3. 2-Pass Generation:
   - Pass 1A: RLCoder Baseline (Raw Top-10 Chunks).
   - Pass 1B: Tree-CSSI Augmented (Raw Chunks + TypedContextTree Decorated Stubs).
   - Pass 2: CADRE-TreeSpec (Refined with Target Spec Resolver from Pass 1B Draft).
4. Copy-Protection Safety Coupling Gate (G_copy >= 0.85):
   - Local verbatim completion -> Retain Pass 1A Raw (Guarantees Zero Negative Transfer).
   - Semantic cross-file completion -> Use Tree-Spec Refined Pass 2.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import tarfile
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Set up CADRE package path
ROOT_DIR = Path(__file__).resolve().parent
CADRE_PKG = ROOT_DIR / "cadre_official"
if not CADRE_PKG.exists():
    CADRE_PKG = Path("/content/cadre_official")
if not CADRE_PKG.exists():
    CADRE_PKG = Path("cadre_official").resolve()
sys.path.insert(0, str(CADRE_PKG))

try:
    from cadre.cssi import TypedContextTree, CSSIContractBuilder
    from cadre.intent_parser import IntentParser, clean_bpe_text, strip_method_affixes
except ImportError:
    # If in root or child dir
    sys.path.insert(0, str(Path.cwd() / "cadre_official"))
    from cadre.cssi import TypedContextTree, CSSIContractBuilder
    from cadre.intent_parser import IntentParser, clean_bpe_text, strip_method_affixes

try:
    from fuzzywuzzy import fuzz
    def cal_edit_sim(p: str, g: str) -> float:
        return float(fuzz.ratio(p.strip(), g.strip()))
except ImportError:
    def cal_edit_sim(p: str, g: str) -> float:
        return SequenceMatcher(None, p.strip(), g.strip()).ratio() * 100.0

import torch
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None


# -------------------------------------------------------------
# Metric Helpers
# -------------------------------------------------------------
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


def evaluate_records(records: List[Dict[str, Any]], pred_key: str) -> Dict[str, Any]:
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
        if not items:
            continue
        n = len(items)
        if bm.startswith("repoeval"):
            repo_map = defaultdict(list)
            for it in items:
                repo_map[it["repo"]].append(it["em"])
            b_em = sum(sum(v)/len(v) for v in repo_map.values()) / len(repo_map) * 100.0

            repo_es = defaultdict(list)
            for it in items:
                repo_es[it["repo"]].append(it["es"])
            b_es = sum(sum(v)/len(v) for v in repo_es.values()) / len(repo_es) * 100.0
        else:
            b_em = sum(it["em"] for it in items) / n * 100.0
            b_es = sum(it["es"] for it in items) / n * 100.0

        summary[bm] = {"em": round(b_em, 2), "es": round(b_es, 2), "count": n}
        macro_em.append(b_em)
        macro_es.append(b_es)

    if macro_em:
        summary["macro_mean"] = {
            "em": round(sum(macro_em) / len(macro_em), 2),
            "es": round(sum(macro_es) / len(macro_es), 2),
        }
    return summary


# -------------------------------------------------------------
# Chunk Coalescing & Prompt Construction
# -------------------------------------------------------------
def coalesce_chunks(chunks: List[Dict[str, Any]], comment_prefix: str) -> List[str]:
    chunks_by_file = defaultdict(list)
    for c in chunks:
        # Filter stop block (empty file path or is_stop flag)
        fp = c.get("file_path") or c.get("path") or ""
        if not fp or c.get("is_stop"):
            break
        code = c.get("code") or c.get("text") or c.get("code_content") or ""
        if fp and code.strip():
            chunks_by_file[fp].append(code.strip())

    blocks = []
    for fp, clist in chunks_by_file.items():
        content = "\n...\n".join(clist)
        blocks.append(f"{comment_prefix} file path: {fp}\n{content}\n\n")
    return blocks


def build_cadre_prompt(
    tokenizer: Any,
    task: Dict[str, Any],
    tr: Dict[str, Any],
    tree_cssi_block: str = "",
    target_spec_block: str = "",
    max_exemplar_tokens: int = 1536,
    max_tree_tokens: int = 300,
    max_spec_tokens: int = 300,
    max_left_tokens: int = 512,
    max_total_tokens: int = 4096,
) -> List[int]:
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    comment_prefix = "#" if lang == "python" else "//"

    path_text = f"{comment_prefix} file path: {cur_path}\n"
    path_ids = tokenizer.encode(path_text, add_special_tokens=False)
    left_ids = tokenizer.encode(left_context, add_special_tokens=False)[-max_left_tokens:]

    tree_ids = tokenizer.encode(tree_cssi_block, add_special_tokens=False)[:max_tree_tokens] if tree_cssi_block else []
    spec_ids = tokenizer.encode(target_spec_block, add_special_tokens=False)[:max_spec_tokens] if target_spec_block else []

    raw_candidates = tr.get("rlcoder_top10") or tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    exemplar_ids = []
    if raw_candidates:
        blocks = coalesce_chunks(raw_candidates, comment_prefix)
        for b in blocks:
            b_ids = tokenizer.encode(b, add_special_tokens=False)
            if len(exemplar_ids) + len(b_ids) > max_exemplar_tokens:
                break
            exemplar_ids.extend(b_ids)

    cross_ids = exemplar_ids + tree_ids + spec_ids
    final_ids = (cross_ids + path_ids + left_ids)[-max_total_tokens:]
    return final_ids


# -------------------------------------------------------------
# Copy-Protection Metric (copy_sim)
# -------------------------------------------------------------
def max_line_similarity(cand_line: str, reference_texts: List[str]) -> float:
    c = cand_line.strip()
    if len(c) < 6:
        return 0.0
    best = 0.0
    for txt in reference_texts:
        for raw in txt.splitlines():
            s = raw.strip().lstrip("#").lstrip("//").strip()
            if len(s) < 6:
                continue
            if s == c:
                return 1.0
            if min(len(s), len(c)) / max(len(s), len(c)) < 0.65:
                continue
            r = SequenceMatcher(None, c, s).ratio()
            if r > best:
                best = r
    return best


# -------------------------------------------------------------
# Target Specification Resolver with Class Hierarchy
# -------------------------------------------------------------
def resolve_target_spec_with_hierarchy(
    task: Dict[str, Any],
    tr: Dict[str, Any],
    pool_files: List[Dict[str, Any]],
    raw_draft: str,
) -> str:
    lang = task["language"]
    left_ctx = task["left_context"]
    comment = "#" if lang == "python" else "//"

    receiver, candidate_callees, candidate_tokens = IntentParser.extract_intent(left_ctx, raw_draft)
    if not candidate_tokens:
        return ""

    raw_candidates = tr.get("rlcoder_top10") or tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    top_chunks_text = " ".join((c.get("code") or c.get("text") or c.get("code_content") or "") for c in raw_candidates[:3])
    primary = candidate_tokens[0]
    if f"def {primary}(" in top_chunks_text or f"def {primary} (" in top_chunks_text:
        return ""

    best_candidate = None
    best_score = -1.0

    for pf in pool_files:
        fpath = pf.get("path", "")
        text = pf.get("text", "")
        lines = text.splitlines()

        for idx, line in enumerate(lines):
            s = line.strip()
            if lang == "python":
                if s.startswith("def ") or s.startswith("async def "):
                    m = re.match(r"(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", s)
                    if m:
                        def_name = m.group(1)
                        score = 0.0
                        for cand in candidate_tokens:
                            if cand == def_name:
                                score = max(score, 2.0)
                            elif strip_method_affixes(cand) and strip_method_affixes(cand) == strip_method_affixes(def_name):
                                score = max(score, 1.5)
                            elif cand.lower() in def_name.lower():
                                score = max(score, 1.0)

                        if score > best_score and score >= 1.0:
                            sig_lines = [s]
                            j = idx
                            while ":" not in sig_lines[-1] and j + 1 < len(lines):
                                j += 1
                                sig_lines.append(lines[j].strip())
                            full_sig = " ".join(" ".join(sig_lines).split())
                            if ":" in full_sig:
                                full_sig = full_sig.split(":")[0] + ": ..."

                            enclosing_class = ""
                            for k in range(idx - 1, max(-1, idx - 60), -1):
                                pl = lines[k].strip()
                                if pl.startswith("class "):
                                    m_cls = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", pl)
                                    if m_cls:
                                        enclosing_class = m_cls.group(1)
                                        break

                            if enclosing_class:
                                full_sig = f"class {enclosing_class}:\n        {full_sig}"

                            doc = ""
                            if j + 1 < len(lines):
                                next_l = lines[j+1].strip()
                                if '"""' in next_l or "'''" in next_l:
                                    doc = next_l.replace('"""', '').replace("'''", '').strip()
                                    if not doc and j + 2 < len(lines):
                                        doc = lines[j+2].replace('"""', '').replace("'''", '').strip()

                            best_score = score
                            best_candidate = {
                                "file": fpath,
                                "name": def_name,
                                "sig": full_sig,
                                "doc": doc,
                            }
            else:  # Java
                if (s.startswith("public ") or s.startswith("protected ") or s.startswith("private ") or s.startswith("static ")) and "(" in s and ")" in s:
                    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", s)
                    if m:
                        def_name = m.group(1)
                        score = 0.0
                        for cand in candidate_tokens:
                            if cand == def_name:
                                score = max(score, 2.0)
                            elif strip_method_affixes(cand) and strip_method_affixes(cand) == strip_method_affixes(def_name):
                                score = max(score, 1.5)
                            elif cand.lower() in def_name.lower():
                                score = max(score, 1.0)

                        if score > best_score and score >= 1.0:
                            sig = s.split("{")[0].strip()
                            if not sig.endswith(";"): sig += ";"

                            enclosing_class = ""
                            for k in range(idx - 1, max(-1, idx - 60), -1):
                                pl = lines[k].strip()
                                if "class " in pl or "interface " in pl:
                                    m_cls = re.search(r"(?:class|interface)\s+([A-Za-z_][A-Za-z0-9_]*)", pl)
                                    if m_cls:
                                        enclosing_class = m_cls.group(1)
                                        break
                            if enclosing_class:
                                sig = f"interface {enclosing_class} {{\n        {sig}\n    }}"

                            best_score = score
                            best_candidate = {
                                "file": fpath,
                                "name": def_name,
                                "sig": sig,
                                "doc": "",
                            }

    if not best_candidate:
        return ""

    banner = [
        f"{comment} ==============================================================================",
        f"{comment} <<< TARGET API SPECIFICATION (RESOLVED FROM DRAFT INTENT) >>>",
        f"{comment} Source Module: {best_candidate['file']}",
        f"{comment} Canonical Signature (L_stub):",
    ]
    for sig_l in best_candidate["sig"].splitlines():
        banner.append(f"{comment}     {sig_l}")
    if best_candidate.get("doc"):
        banner.append(f"{comment} Docstring Summary: \"\"\"{best_candidate['doc']}\"\"\"")
    banner.append(f"{comment} ==============================================================================\n\n")

    return "\n".join(banner)


# -------------------------------------------------------------
# Main Evaluation Pipeline
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run CADRE Consolidated on full 8,004 tasks")
    parser.add_argument("--data-root", default="/content", help="Directory containing bundles")
    parser.add_argument("--output-dir", default="/content/cadre-results/cadre_consolidated_8004", help="Output directory")
    parser.add_argument("--benchmarks", default="cceval_python,cceval_java,repoeval_line,repoeval_api", help="Benchmarks to evaluate")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--generator-batch-size", type=int, default=128)
    parser.add_argument("--limit-per-benchmark", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--trace-path", default=None, help="Path to retrieval trace jsonl.gz")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = data_root / "full_benchmark_8004_bundle.tar.gz"
    if not bundle_path.exists():
        bundle_path = Path("experiments/mb_spt_iterative/full_benchmark_8004_bundle.tar.gz")

    if args.trace_path:
        trace_path = Path(args.trace_path)
    else:
        trace_path = data_root / "rlcoder_full_8004_retrieval_trace.jsonl.gz"
        if not trace_path.exists():
            trace_path = Path("experiments/mb_spt_iterative/rlcoder_full_8004_retrieval_trace.jsonl.gz")

    print(f"=== Starting CADRE Consolidated (BẢN HIỆN TẠI) Full 8,004 Evaluation ===", flush=True)
    print(f"Bundle path: {bundle_path}", flush=True)
    print(f"Trace path:  {trace_path}", flush=True)
    print(f"Output dir:  {output_dir}", flush=True)

    # 1. Extract bundle if needed
    if not (data_root / "tasks.jsonl.gz").exists() and bundle_path.exists():
        print(f"Extracting {bundle_path} to {data_root}...", flush=True)
        with tarfile.open(bundle_path, "r:gz") as t:
            t.extractall(data_root)

    # 2. Load tokenizer
    generator_name = "deepseek-ai/deepseek-coder-1.3b-base"
    generator_rev = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
    local_model_dir = Path("/content/models/deepseek-coder-1.3b-base")
    tokenizer_source = str(local_model_dir) if (local_model_dir / "tokenizer_config.json").is_file() else generator_name

    print(f"Loading tokenizer from {tokenizer_source} (Fast BPE)...", flush=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_source, revision=generator_rev, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # 3. Load tasks, files, traces
    t0 = time.time()
    print("Loading repository pool files and task records...", flush=True)
    files = {r["file_id"]: {"path": r["path"], "text": r["text"]} for r in map(json.loads, gzip.open(data_root / "files.jsonl.gz", "rt"))}
    all_tasks = [json.loads(l) for l in gzip.open(data_root / "tasks.jsonl.gz", "rt")]
    traces = {(r["benchmark"], r["task_id"]): r for r in map(json.loads, gzip.open(trace_path, "rt"))}
    print(f"Loaded {len(files)} repo files, {len(all_tasks)} tasks, {len(traces)} retrieval traces in {time.time()-t0:.2f}s", flush=True)

    requested_bms = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    if args.limit_per_benchmark is not None:
        selected_tasks = []
        for bm in requested_bms:
            bm_tasks = [t for t in all_tasks if t["benchmark"] == bm][:args.limit_per_benchmark]
            selected_tasks.extend(bm_tasks)
        tasks = selected_tasks
    else:
        tasks = [t for t in all_tasks if t["benchmark"] in requested_bms]

    print(f"Selected {len(tasks)} tasks across benchmarks: {requested_bms}", flush=True)

    # Build task pool files
    task_pool_files = []
    for t in tasks:
        pf = [files[f] for f in t.get("file_ids_by_path", {}).values() if f in files]
        task_pool_files.append(pf)
    task_trs = [traces.get((t["benchmark"], t["task_id"]), {}) for t in tasks]

    # 4. Synthesize Tree-CSSI contracts for all tasks
    t_tree_start = time.time()
    print("\n[Tree Construction] Synthesizing TypedContextTree with Inheritance Resolution...", flush=True)
    tree_cssi_blocks = []
    for idx, (t, pf) in enumerate(zip(tasks, task_pool_files)):
        contract = CSSIContractBuilder.synthesize_contract(
            language=t["language"],
            left_context=t["left_context"],
            pool_files=pf,
            mode="decorated_stub",
            max_methods_per_class=6,
        )
        tree_cssi_blocks.append(contract)
        if (idx + 1) % 1000 == 0 or idx == len(tasks) - 1:
            print(f"  Processed {idx + 1}/{len(tasks)} task trees in {time.time()-t_tree_start:.1f}s...", flush=True)

    non_empty_trees = sum(1 for c in tree_cssi_blocks if c)
    print(f"  ✓ Successfully synthesized {non_empty_trees}/{len(tasks)} TypedContextTree contracts ({non_empty_trees/len(tasks)*100:.1f}%).", flush=True)

    # 5. Initialize vLLM Engine
    print(f"\n=== INITIALIZING VLLM ENGINE: {generator_name} ===", flush=True)
    llm = LLM(
        model=generator_name,
        tokenizer=tokenizer_source,
        revision=generator_rev,
        tokenizer_revision=generator_rev,
        dtype="float16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.generator_batch_size,
        max_num_batched_tokens=32768,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=20261026,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sp = SamplingParams(max_tokens=64, temperature=0.0, top_p=1.0)

    all_predictions = []
    for t in tasks:
        all_predictions.append({
            "task_id": t["task_id"],
            "benchmark": t["benchmark"],
            "language": t["language"],
            "groundtruth": t["groundtruth"],
            "path": t.get("path", ""),
        })

    # -------------------------------------------------------------
    # Pass 1A: Baseline CADRE / RLCoder Raw (Raw Chunks only)
    # -------------------------------------------------------------
    print(f"\n[Pass 1A] Running RLCoder Baseline (Raw Top-10 Chunks Only, n={len(tasks)})...", flush=True)
    p0_raw_ids = [build_cadre_prompt(tok, t, tr, tree_cssi_block="", target_spec_block="") for t, tr in zip(tasks, task_trs)]
    out0 = llm.generate([{"prompt_token_ids": p} for p in p0_raw_ids], sampling_params=sp)
    preds_raw = []
    for i, o in enumerate(out0):
        out_token_ids = list(o.outputs[0].token_ids)
        decoded = tok.decode(out_token_ids, skip_special_tokens=True)
        srv_text = o.outputs[0].text
        # Preserve whitespace
        ws_dec = sum(c.isspace() for c in decoded)
        ws_srv = sum(c.isspace() for c in srv_text)
        final_text = decoded if ws_dec >= ws_srv else srv_text
        preds_raw.append(final_text)
        all_predictions[i]["pass1_raw"] = final_text

    stats_raw = evaluate_records(all_predictions, "pass1_raw")
    print("  ✓ Pass 1A (Raw Baseline) Results:", stats_raw, flush=True)

    # -------------------------------------------------------------
    # Pass 1B: Tree-CSSI Augmented (Raw Chunks + TypedContextTree Stubs)
    # -------------------------------------------------------------
    print(f"\n[Pass 1B] Running Tree-CSSI Augmented (Raw Chunks + TypedContextTree Stubs, n={len(tasks)})...", flush=True)
    p0_tree_ids = [build_cadre_prompt(tok, t, tr, tree_cssi_block=tb, target_spec_block="") for t, tr, tb in zip(tasks, task_trs, tree_cssi_blocks)]
    out1b = llm.generate([{"prompt_token_ids": p} for p in p0_tree_ids], sampling_params=sp)
    preds_tree_p1 = []
    for i, o in enumerate(out1b):
        out_token_ids = list(o.outputs[0].token_ids)
        decoded = tok.decode(out_token_ids, skip_special_tokens=True)
        srv_text = o.outputs[0].text
        ws_dec = sum(c.isspace() for c in decoded)
        ws_srv = sum(c.isspace() for c in srv_text)
        final_text = decoded if ws_dec >= ws_srv else srv_text
        preds_tree_p1.append(final_text)
        all_predictions[i]["pass1_tree_cssi"] = final_text

    stats_tree_p1 = evaluate_records(all_predictions, "pass1_tree_cssi")
    print("  ✓ Pass 1B (Tree-CSSI Augmented) Results:", stats_tree_p1, flush=True)

    # -------------------------------------------------------------
    # Pass 2: CADRE-TreeSpec (Refined with Target Spec Resolver from Pass 1B Draft)
    # -------------------------------------------------------------
    print(f"\n[Pass 2] Resolving Target Specifications from Pass 1B Draft (n={len(tasks)})...", flush=True)
    t_p2_res = time.time()
    p2_prompts = []
    p2_indices = []
    p2_all_preds = list(preds_tree_p1)

    for i, (t, tr, pf, draft, tree_block) in enumerate(zip(tasks, task_trs, task_pool_files, preds_tree_p1, tree_cssi_blocks)):
        target_spec = resolve_target_spec_with_hierarchy(t, tr, pf, draft)
        if target_spec:
            p2 = build_cadre_prompt(tok, t, tr, tree_cssi_block=tree_block, target_spec_block=target_spec)
            p2_prompts.append(p2)
            p2_indices.append(i)

    print(f"  ✓ Target specification triggered for {len(p2_prompts)}/{len(tasks)} tasks ({len(p2_prompts)/len(tasks)*100:.1f}%) in {time.time()-t_p2_res:.1f}s.", flush=True)

    if p2_prompts:
        out2 = llm.generate([{"prompt_token_ids": p} for p in p2_prompts], sampling_params=sp)
        for idx_pos, o in zip(p2_indices, out2):
            out_token_ids = list(o.outputs[0].token_ids)
            decoded = tok.decode(out_token_ids, skip_special_tokens=True)
            srv_text = o.outputs[0].text
            ws_dec = sum(c.isspace() for c in decoded)
            ws_srv = sum(c.isspace() for c in srv_text)
            final_text = decoded if ws_dec >= ws_srv else srv_text
            p2_all_preds[idx_pos] = final_text

    for i, p in enumerate(p2_all_preds):
        all_predictions[i]["cadre_tree_spec_raw"] = p

    stats_tree_spec = evaluate_records(all_predictions, "cadre_tree_spec_raw")
    print("  ✓ Pass 2 (CADRE-TreeSpec Unconstrained) Results:", stats_tree_spec, flush=True)

    # -------------------------------------------------------------
    # Step 4: Defensive Safety Coupling Gate (Copy-Protection, copy_sim >= 0.85)
    # -------------------------------------------------------------
    print("\n[Coupling Gate] Applying Copy-Protection Safety Gate (copy_sim >= 0.85)...", flush=True)
    copy_protected_count = 0
    tree_spec_count = 0

    for i, (t, tr, raw_p, tree_spec_p) in enumerate(zip(tasks, task_trs, preds_raw, p2_all_preds)):
        clean_draft = clean_bpe_text(raw_p)
        draft_lines = [l.strip() for l in clean_draft.splitlines() if l.strip()]
        first_line = draft_lines[0] if draft_lines else ""

        raw_candidates = tr.get("rlcoder_top10") or tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
        chunk_texts = [(c.get("code") or c.get("text") or c.get("code_content") or "") for c in raw_candidates[:6]]
        recent_left = t["left_context"].splitlines()[-80:]
        copy_sim = max_line_similarity(first_line, chunk_texts + recent_left)

        all_predictions[i]["copy_sim"] = copy_sim

        if copy_sim >= 0.85:
            # High verbatim overlap with local context -> Retain Pass 1A Raw
            copy_protected_count += 1
            all_predictions[i]["cadre_26_final"] = raw_p
            all_predictions[i]["gate_decision"] = "copy_protected_pass1"
        else:
            # Semantic cross-file completion -> Use Tree-Spec Refined
            tree_spec_count += 1
            all_predictions[i]["cadre_26_final"] = tree_spec_p
            all_predictions[i]["gate_decision"] = "tree_spec_final"

    print(f"  ✓ Gate Decision Breakdown across {len(tasks)} tasks:", flush=True)
    print(f"    - Copy-Protected (Frozen Pass 1A Raw): {copy_protected_count}", flush=True)
    print(f"    - Tree-Spec Refined (Cross-file):      {tree_spec_count}", flush=True)

    # Evaluate Final Coupled Method
    stats_final = evaluate_records(all_predictions, "cadre_26_final")

    print("\n" + "=" * 92)
    print("=== CADRE Consolidated (BẢN HIỆN TẠI) FULL 8,004 BENCHMARK RESULTS ===")
    print("=" * 92)
    print(f"{'Method / Configuration':<38} | {'CC-Py':<7} | {'CC-Java':<7} | {'RE-API':<7} | {'RE-Line':<7} | {'Macro EM':<8} | {'Macro ES':<8}")
    print("-" * 92)
    print(f"{'1. RLCoder Baseline (Raw Only)':<38} | {stats_raw.get('cceval_python', {}).get('em', 0):<7.2f} | {stats_raw.get('cceval_java', {}).get('em', 0):<7.2f} | {stats_raw.get('repoeval_api', {}).get('em', 0):<7.2f} | {stats_raw.get('repoeval_line', {}).get('em', 0):<7.2f} | {stats_raw.get('macro_mean', {}).get('em', 0):<8.2f} | {stats_raw.get('macro_mean', {}).get('es', 0):<8.2f}")
    print(f"{'2. Pass 1B: Tree-CSSI Decorated':<38} | {stats_tree_p1.get('cceval_python', {}).get('em', 0):<7.2f} | {stats_tree_p1.get('cceval_java', {}).get('em', 0):<7.2f} | {stats_tree_p1.get('repoeval_api', {}).get('em', 0):<7.2f} | {stats_tree_p1.get('repoeval_line', {}).get('em', 0):<7.2f} | {stats_tree_p1.get('macro_mean', {}).get('em', 0):<8.2f} | {stats_tree_p1.get('macro_mean', {}).get('es', 0):<8.2f}")
    print(f"{'3. Pass 2: CADRE-TreeSpec (Unconst)':<38} | {stats_tree_spec.get('cceval_python', {}).get('em', 0):<7.2f} | {stats_tree_spec.get('cceval_java', {}).get('em', 0):<7.2f} | {stats_tree_spec.get('repoeval_api', {}).get('em', 0):<7.2f} | {stats_tree_spec.get('repoeval_line', {}).get('em', 0):<7.2f} | {stats_tree_spec.get('macro_mean', {}).get('em', 0):<8.2f} | {stats_tree_spec.get('macro_mean', {}).get('es', 0):<8.2f}")
    print(f"{'4. CADRE Consolidated (BẢN HIỆN TẠI)':<38} | {stats_final.get('cceval_python', {}).get('em', 0):<7.2f} | {stats_final.get('cceval_java', {}).get('em', 0):<7.2f} | {stats_final.get('repoeval_api', {}).get('em', 0):<7.2f} | {stats_final.get('repoeval_line', {}).get('em', 0):<7.2f} | {stats_final.get('macro_mean', {}).get('em', 0):<8.2f} | {stats_final.get('macro_mean', {}).get('es', 0):<8.2f}")
    print("=" * 92)

    # Save outputs
    out_json = output_dir / "cadre_consolidated_8004_metrics.json"
    with open(out_json, "w") as f:
        json.dump({
            "pass1_raw": stats_raw,
            "pass1_tree_cssi": stats_tree_p1,
            "cadre_tree_spec_unconstrained": stats_tree_spec,
            "cadre_26_final": stats_final,
            "tasks_count": len(tasks),
            "copy_protected_count": copy_protected_count,
            "tree_spec_count": tree_spec_count,
        }, f, indent=2)

    # Save predictions per benchmark
    for bm in requested_bms:
        bm_dir = output_dir / bm
        bm_dir.mkdir(parents=True, exist_ok=True)
        bm_preds = [p for p in all_predictions if p["benchmark"] == bm]
        with open(bm_dir / "prediction.jsonl", "w") as f:
            for p in bm_preds:
                f.write(json.dumps({"task_id": p["task_id"], "pred": p["cadre_26_final"]}) + "\n")

    out_preds = output_dir / "cadre_consolidated_8004_predictions.jsonl.gz"
    with gzip.open(out_preds, "wt") as f:
        for r in all_predictions:
            f.write(json.dumps(r) + "\n")

    print(f"\nAll artifacts successfully saved to {output_dir}!", flush=True)


if __name__ == "__main__":
    main()
