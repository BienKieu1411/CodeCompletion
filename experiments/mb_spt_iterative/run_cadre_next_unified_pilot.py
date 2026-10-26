#!/usr/bin/env python3
"""
CADRE-Next Unified Pilot Experiment on 400-Task Benchmark.
Head-to-head comparison of training-free improvements:
  Arm 0: Baseline CADRE v3 (Fixed 3200 Crossfile Budget, Flat CSSI)
  Arm 1: CADRE + Asymmetric Budgeting (RepoEval 512 / CCEval 1280 tokens)
  Arm 2: CADRE + Receiver-Focused Call-Site & Canonical Stub (RFSC + L_stub)
  Arm 3: Unified CADRE-Next (Asymmetric Budget + RFSC Call-Site & Stub)
  Arm 4: CADRE-Next 2-Pass IDGR (Iterative Draft-Guided Retrieval & Refinement)

Official AlignCoder Evaluation Standard across all 4 benchmarks.
"""

from __future__ import annotations

import difflib
import gzip
import json
import math
import os
import random
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_ROOT = Path("/content/mb_spt") if Path("/content/mb_spt").exists() else Path("experiments/mb_spt_iterative")
BUNDLE_PATH = Path("/content/full_benchmark_8004_bundle.tar.gz") if Path("/content/full_benchmark_8004_bundle.tar.gz").exists() else Path("experiments/mb_spt_iterative/full_benchmark_8004_bundle.tar.gz")
OUTPUT_DIR = DATA_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENERATOR = "deepseek-ai/deepseek-coder-1.3b-base"
GENERATOR_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
SEED = 20261026

# Ensure dependencies
def ensure_deps():
    needed = []
    for mod, pkg in [("fuzzywuzzy", "fuzzywuzzy"), ("Levenshtein", "python-Levenshtein"), ("editdistance", "editdistance")]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    try:
        from importlib.metadata import version
        if version("tree-sitter") != "0.20.4":
            needed.append("tree-sitter==0.20.4")
    except Exception:
        needed.append("tree-sitter==0.20.4")
    try:
        import vllm
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
        subprocess.check_call(["uv", "pip", "install", "--system", "vllm", "--torch-backend=auto"])
    try:
        import torchaudio
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"])
    except Exception:
        pass
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])

ensure_deps()

from transformers import AutoTokenizer
from tree_sitter import Language, Parser
from vllm import LLM, SamplingParams
from fuzzywuzzy import fuzz

# Extract bundle if needed
if not (DATA_ROOT / "tasks.jsonl.gz").exists():
    print(f"Extracting bundle from {BUNDLE_PATH}...", flush=True)
    with tarfile.open(BUNDLE_PATH, "r:gz") as t:
        t.extractall(DATA_ROOT)

tok = AutoTokenizer.from_pretrained(GENERATOR, revision=GENERATOR_REVISION, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# Load datasets
all_tasks = [json.loads(l) for l in gzip.open(DATA_ROOT / "tasks.jsonl.gz", "rt")]
traces = {(r["benchmark"], r["task_id"]): r for r in map(json.loads, gzip.open(DATA_ROOT / "retrieval_trace.jsonl.gz", "rt"))}
files = {r["file_id"]: {"path": r["path"], "text": r["text"]} for r in map(json.loads, gzip.open(DATA_ROOT / "files.jsonl.gz", "rt"))}
pools = {}
if (DATA_ROOT / "pools.jsonl.gz").exists():
    pools = {r["pool_id"]: r["file_ids"] for r in map(json.loads, gzip.open(DATA_ROOT / "pools.jsonl.gz", "rt"))}

parsers = {}
for lang in ("python", "java"):
    so_p = DATA_ROOT / "parsers" / f"{lang}-lang-parser.so"
    p = Parser()
    p.set_language(Language(str(so_p), lang))
    parsers[lang] = p

# Load EXACT 400 pilot tasks
pilot_pool_path = Path("/content/rerank_pilot_pool.jsonl.gz")
if not pilot_pool_path.exists():
    pilot_pool_path = DATA_ROOT / "rerank_pilot_pool.jsonl.gz"
if not pilot_pool_path.exists():
    pilot_pool_path = Path("experiments/mb_spt_iterative/rerank_pilot_pool.jsonl.gz")

with gzip.open(pilot_pool_path, "rt") as f:
    pilot_pool_records = [json.loads(l) for l in f]

pilot_keys_set = {(r["benchmark"], r["task_id"]) for r in pilot_pool_records}
tasks = [t for t in all_tasks if (t["benchmark"], t["task_id"]) in pilot_keys_set]
print(f"Loaded {len(tasks)} pilot tasks matching baseline pool.", flush=True)

# Regex helpers for syntactic trigger extraction
CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
DOT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)")
DOT_END_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*$")
STOP_WORDS = {"self", "this", "None", "True", "False", "def", "class", "if", "for", "while", "return", "import", "from", "public", "private", "protected", "static", "void"}

def extract_cursor_triggers(left_context: str) -> tuple[list[str], list[str]]:
    lines = [l.strip() for l in left_context.splitlines() if l.strip()]
    if not lines:
        return [], []
    last_line = lines[-1]
    
    callees = []
    for m in CALL_RE.finditer(last_line):
        full = m.group(1).split(".")[-1]
        if len(full) > 1 and full not in STOP_WORDS:
            callees.append(full)
            
    receivers = []
    for m in DOT_RE.finditer(last_line):
        rec = m.group(1)
        if len(rec) > 1 and rec not in STOP_WORDS:
            receivers.append(rec)
            
    m_end = DOT_END_RE.search(last_line)
    if m_end:
        rec = m_end.group(1)
        if len(rec) > 1 and rec not in STOP_WORDS:
            receivers.append(rec)
            
    return list(dict.fromkeys(callees)), list(dict.fromkeys(receivers))

def build_compact_callsite_contract(
    callees: list[str],
    receivers: list[str],
    sources: list[tuple[str, str]],
    lang: str,
    max_lines: int = 4
) -> str:
    comment = "#" if lang == "python" else "//"
    lines_out = []
    seen = set()
    
    # 1. Search for signatures of callees
    for callee in callees[:2]:
        def_pat = re.compile(rf"^\s*(?:async\s+)?(?:def|public|protected|private|static)?\s*.*?\b{re.escape(callee)}\s*\([^)]*\)", re.MULTILINE)
        for path, text in sources:
            for m in def_pat.finditer(text):
                sig = " ".join(m.group(0).split())
                if sig and sig not in seen:
                    seen.add(sig)
                    lines_out.append(f"{comment} [CSSI Stub]: {sig} -> ...")
                    break
            if len(lines_out) >= max_lines:
                break
            
    # 2. Search for call-sites of callees
    for callee in callees[:2]:
        for path, text in sources:
            for l in text.splitlines():
                if callee in l and not l.strip().startswith(("def ", "class ", "//", "#")):
                    cl = l.strip()
                    if 10 < len(cl) < 120 and cl not in seen:
                        seen.add(cl)
                        lines_out.append(f"{comment} [CSSI Usage]: {cl}")
                        break
            if len(lines_out) >= max_lines:
                break

    # 3. Search for member usages / methods of receivers
    for rec in receivers[:2]:
        for path, text in sources:
            for l in text.splitlines():
                if rec + "." in l and not l.strip().startswith(("def ", "class ", "//", "#")):
                    cl = l.strip()
                    if 10 < len(cl) < 120 and cl not in seen:
                        seen.add(cl)
                        lines_out.append(f"{comment} [CSSI Usage]: {cl}")
                        break
            if len(lines_out) >= max_lines:
                break

    if not lines_out:
        return ""
    
    header = f"{comment} === CURSOR STRUCTURAL CONTRACT ===\n"
    footer = f"\n{comment} ====================================\n"
    return header + "\n".join(lines_out[:max_lines]) + footer

def build_prompt_flexible(
    task: dict[str, Any],
    tr: dict[str, Any],
    pool_files: list[dict[str, Any]],
    cross_budget: int = 1280,
    use_callsite_contract: bool = True,
    extra_callee: str | None = None,
) -> list[int]:
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    comment_label = "#" if lang == "python" else "//"
    
    left_all_ids = tok.encode(left_context)
    path_ids = tok.encode(f"{comment_label} file path: {cur_path}\n")
    
    raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    
    # Group chunks by file
    chunks_by_file: dict[str, list[str]] = defaultdict(list)
    for ch in raw_candidates:
        fp = ch.get("file_path") or ch.get("path", "")
        c = ch.get("text") or ch.get("code", "")
        if fp and c.strip():
            chunks_by_file[fp].append(c)

    exemplar_ids: list[int] = []
    max_exemplar_tokens = max(0, cross_budget - 128)
    for fp, code_list in chunks_by_file.items():
        coalesced = "\n...\n".join(code_list)
        block = f"{comment_label} file path: {fp}\n{coalesced}\n\n"
        b_ids = tok.encode(block)
        if len(exemplar_ids) + len(b_ids) > max_exemplar_tokens:
            break
        exemplar_ids.extend(b_ids)
        
    contract_ids: list[int] = []
    if use_callsite_contract:
        callees, receivers = extract_cursor_triggers(left_context)
        if extra_callee and extra_callee not in callees:
            callees.insert(0, extra_callee)
            
        sources = [(pf["path"], pf["text"]) for pf in pool_files] + [(ch.get("file_path", "chunk"), ch.get("text") or ch.get("code", "")) for ch in raw_candidates]
        contract_text = build_compact_callsite_contract(callees, receivers, sources, lang)
        if contract_text:
            contract_ids = tok.encode(contract_text)[:128]

    # Assembly: Exemplars -> Contract (closest to cursor) -> File Path -> Left Context
    cross_ids = exemplar_ids + contract_ids
    allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []
    
    return (cross_ids + path_ids + left_ids)[-8000:]

# Metrics and postprocessing
def remove_comments(code: str) -> str:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"//.*", "", code)
    return code

def is_parse_valid(parser: Any, code: str) -> bool:
    def syntax_error(node: Any) -> bool:
        if node.type == "ERROR":
            return True
        for child in node.children:
            if syntax_error(child):
                return True
        return False
    try:
        tree = parser.parse(bytes(code, "utf8"))
        return not syntax_error(tree.root_node) if tree else False
    except Exception:
        return False

def get_python_one_statement(prompt: str, completion: str, parser: Any) -> str:
    for i in range(len(completion)):
        code = prompt + completion[: i + 1]
        if not is_parse_valid(parser, code):
            continue
        if i + 1 < len(completion) and completion[i + 1] == "\n":
            return completion[: i + 1].rstrip()
    return completion

def get_bracket_lang_statement(completion: str) -> str:
    end_idx = None
    for i in range(len(completion)):
        if completion[i] in [";", "}", "{"]:
            end_idx = i
            break
    return completion[: end_idx + 1] if end_idx is not None else completion

def postprocess_code_lines(prompt: str, completion: str, parser: Any, lang: str) -> str:
    try:
        if lang == "python":
            return get_python_one_statement(prompt, completion, parser)
        return get_bracket_lang_statement(completion)
    except Exception:
        return completion

def postprocess_repoeval(pred: str, gt: str) -> str:
    gt_lines = [l.strip() for l in gt.splitlines() if l.strip()]
    n = max(1, len(gt_lines))
    pred_lines = [l.strip() for l in pred.splitlines() if l.strip()][:n]
    return "\n".join(pred_lines)

def levenshtein_dist(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]

def evaluate_predictions(tasks: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    by_bench = defaultdict(list)
    for t, raw_pred in zip(tasks, predictions):
        lang = t["language"]
        bm = t["benchmark"]
        gt = t["groundtruth"]
        
        if bm.startswith("cceval"):
            p = parsers.get(lang)
            clean_pred = postprocess_code_lines(t["left_context"], raw_pred, p, lang)
        else:
            clean_pred = postprocess_repoeval(raw_pred, gt)
            
        clean_pred = remove_comments(clean_pred)
        clean_gt = remove_comments(gt)
        
        p_lines = [l.strip() for l in clean_pred.splitlines() if l.strip()]
        g_lines = [l.strip() for l in clean_gt.splitlines() if l.strip()]
        
        if bm.startswith("repoeval"):
            pe = p_lines[:len(g_lines)]
            em = int(len(g_lines) == len(pe) and g_lines == pe)
            t_str, p_str = "\n".join(g_lines), "\n".join(pe)
            max_l = max(len(t_str), len(p_str))
            es = (1.0 - (levenshtein_dist(t_str, p_str) / max_l)) * 100.0 if max_l else 100.0
        else:
            em = int(p_lines == g_lines and len(g_lines) > 0)
            es = float(fuzz.ratio(clean_pred.strip(), clean_gt.strip()))
            
        repo = t["task_id"].split("/")[0]
        by_bench[bm].append({"repo": repo, "em": em, "es": es})

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
            repo_es_map = defaultdict(list)
            for it in items:
                repo_es_map[it["repo"]].append(it["es"])
            b_es = sum(sum(v)/len(v) for v in repo_es_map.values()) / len(repo_es_map)
        else:
            b_em = sum(it["em"] for it in items) / n * 100.0
            b_es = sum(it["es"] for it in items) / n
            
        summary[bm] = {"count": n, "em": round(b_em, 2), "es": round(b_es, 2)}
        macro_em.append(b_em)
        macro_es.append(b_es)
        
    summary["macro_mean"] = {
        "em": round(sum(macro_em) / 4.0, 2),
        "es": round(sum(macro_es) / 4.0, 2)
    }
    return summary

def main():
    print(f"=== INITIALIZING VLLM ENGINE: {GENERATOR} (8k context) ===", flush=True)
    llm = LLM(
        model=GENERATOR,
        tokenizer=GENERATOR,
        revision=GENERATOR_REVISION,
        tokenizer_revision=GENERATOR_REVISION,
        dtype="float16",
        max_model_len=8192,
        max_num_seqs=128,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.85,
        seed=SEED,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sp = SamplingParams(max_tokens=48, temperature=0.0)

    # Prepare pool_files for all tasks
    task_pool_files = []
    for t in tasks:
        if "file_ids_by_path" in t:
            pf = [files[f] for f in t["file_ids_by_path"].values() if f in files]
        else:
            pf = [files[f] for f in pools.get(t.get("pool_id"), []) if f in files]
        task_pool_files.append(pf)

    all_results = {}

    # -------------------------------------------------------------
    # Arm 0: Baseline CADRE v3 (Fixed 3200 crossfile, flat CSSI)
    # -------------------------------------------------------------
    print("\n[Arm 0] Building prompts for Baseline CADRE v3...", flush=True)
    p0_ids = [build_prompt_flexible(t, traces[(t["benchmark"], t["task_id"])], pf, cross_budget=3200, use_callsite_contract=False) for t, pf in zip(tasks, task_pool_files)]
    out0 = llm.generate([{"prompt_token_ids": p} for p in p0_ids], sampling_params=sp)
    preds_arm0 = [o.outputs[0].text for o in out0]
    all_results["Arm 0: Baseline CADRE v3"] = evaluate_predictions(tasks, preds_arm0)
    print("  ✓ Arm 0 Macro EM:", all_results["Arm 0: Baseline CADRE v3"]["macro_mean"]["em"], "%", flush=True)

    # -------------------------------------------------------------
    # Arm 1: Asymmetric Budgeting (RepoEval 512, CCEval 1280)
    # -------------------------------------------------------------
    print("\n[Arm 1] Building prompts for Asymmetric Budgeting (RepoEval 512 / CCEval 1280)...", flush=True)
    p1_ids = []
    for t, pf in zip(tasks, task_pool_files):
        bm = t["benchmark"]
        b = 512 if bm.startswith("repoeval") else 1280
        p1_ids.append(build_prompt_flexible(t, traces[(bm, t["task_id"])], pf, cross_budget=b, use_callsite_contract=False))
    out1 = llm.generate([{"prompt_token_ids": p} for p in p1_ids], sampling_params=sp)
    preds_arm1 = [o.outputs[0].text for o in out1]
    all_results["Arm 1: Asymmetric Budgeting"] = evaluate_predictions(tasks, preds_arm1)
    print("  ✓ Arm 1 Macro EM:", all_results["Arm 1: Asymmetric Budgeting"]["macro_mean"]["em"], "%", flush=True)

    # -------------------------------------------------------------
    # Arm 2: Receiver-Focused Call-Site & Canonical Stub (RFSC + L_stub)
    # -------------------------------------------------------------
    print("\n[Arm 2] Building prompts for RFSC + L_stub (1280 budget)...", flush=True)
    p2_ids = [build_prompt_flexible(t, traces[(t["benchmark"], t["task_id"])], pf, cross_budget=1280, use_callsite_contract=True) for t, pf in zip(tasks, task_pool_files)]
    out2 = llm.generate([{"prompt_token_ids": p} for p in p2_ids], sampling_params=sp)
    preds_arm2 = [o.outputs[0].text for o in out2]
    all_results["Arm 2: RFSC + L_stub"] = evaluate_predictions(tasks, preds_arm2)
    print("  ✓ Arm 2 Macro EM:", all_results["Arm 2: RFSC + L_stub"]["macro_mean"]["em"], "%", flush=True)

    # -------------------------------------------------------------
    # Arm 3: Unified CADRE-Next (Asymmetric Budget + RFSC Call-Site & Stub)
    # -------------------------------------------------------------
    print("\n[Arm 3] Building prompts for Unified CADRE-Next...", flush=True)
    p3_ids = []
    for t, pf in zip(tasks, task_pool_files):
        bm = t["benchmark"]
        b = 512 if bm.startswith("repoeval") else 1280
        p3_ids.append(build_prompt_flexible(t, traces[(bm, t["task_id"])], pf, cross_budget=b, use_callsite_contract=True))
    out3 = llm.generate([{"prompt_token_ids": p} for p in p3_ids], sampling_params=sp)
    preds_arm3 = [o.outputs[0].text for o in out3]
    all_results["Arm 3: Unified CADRE-Next"] = evaluate_predictions(tasks, preds_arm3)
    print("  ✓ Arm 3 Macro EM:", all_results["Arm 3: Unified CADRE-Next"]["macro_mean"]["em"], "%", flush=True)

    # -------------------------------------------------------------
    # Arm 4: 2-Pass Iterative Draft-Guided Retrieval (IDGR on Arm 3)
    # -------------------------------------------------------------
    print("\n[Arm 4] Performing Pass 2 Draft-Guided Refinement...", flush=True)
    p4_ids = []
    pass2_tasks_indices = []
    for i, (t, pf, raw_draft) in enumerate(zip(tasks, task_pool_files, preds_arm3)):
        bm = t["benchmark"]
        b = 512 if bm.startswith("repoeval") else 1280
        # Extract callee from draft
        draft_lines = [l.strip() for l in raw_draft.splitlines() if l.strip()]
        draft_line1 = draft_lines[0] if draft_lines else ""
        draft_callees = [m.group(1).split(".")[-1] for m in CALL_RE.finditer(draft_line1)]
        top_callee = draft_callees[0] if draft_callees else None
        
        p4_ids.append(build_prompt_flexible(t, traces[(bm, t["task_id"])], pf, cross_budget=b, use_callsite_contract=True, extra_callee=top_callee))

    out4 = llm.generate([{"prompt_token_ids": p} for p in p4_ids], sampling_params=sp)
    preds_arm4 = [o.outputs[0].text for o in out4]
    all_results["Arm 4: CADRE-Next + 2-Pass IDGR"] = evaluate_predictions(tasks, preds_arm4)
    print("  ✓ Arm 4 Macro EM:", all_results["Arm 4: CADRE-Next + 2-Pass IDGR"]["macro_mean"]["em"], "%", flush=True)

    # Print Formatted Comparison Table
    print("\n" + "=" * 125, flush=True)
    print(f"{'ARM':<38} | {'cceval_py':<12} | {'cceval_java':<12} | {'repoeval_api':<14} | {'repoeval_line':<14} | {'MACRO EM':<12} | {'MACRO ES':<10}", flush=True)
    print("=" * 125, flush=True)
    for arm_name, r in all_results.items():
        s_py = f"{r['cceval_python']['em']:.1f}%"
        s_java = f"{r['cceval_java']['em']:.1f}%"
        s_api = f"{r['repoeval_api']['em']:.1f}%"
        s_line = f"{r['repoeval_line']['em']:.1f}%"
        m_em = f"{r['macro_mean']['em']:.2f}%"
        m_es = f"{r['macro_mean']['es']:.2f}%"
        print(f"{arm_name:<38} | {s_py:<12} | {s_java:<12} | {s_api:<14} | {s_line:<14} | {m_em:<12} | {m_es:<10}", flush=True)
    print("=" * 125, flush=True)

    # Save metrics and predictions
    out_json = OUTPUT_DIR / "cadre_next_unified_pilot_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Saved full metrics to {out_json}", flush=True)

    out_preds = OUTPUT_DIR / "cadre_next_unified_pilot_predictions.jsonl.gz"
    with gzip.open(out_preds, "wt", encoding="utf-8") as gz:
        for i, t in enumerate(tasks):
            gz.write(json.dumps({
                "task_id": t["task_id"],
                "benchmark": t["benchmark"],
                "groundtruth": t["groundtruth"],
                "arm0": preds_arm0[i],
                "arm1": preds_arm1[i],
                "arm2": preds_arm2[i],
                "arm3": preds_arm3[i],
                "arm4": preds_arm4[i],
            }) + "\n")
    print(f"✓ Saved all predictions to {out_preds}", flush=True)

if __name__ == "__main__":
    main()
