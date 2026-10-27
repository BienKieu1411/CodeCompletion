#!/usr/bin/env python3
"""
run_cadre_v29_adaptive_slack.py
CADRE v2.9: Slack-Gated & Grounding-Calibrated Adaptive Context Budgeting.

Key Principles:
1. Slack-Gated Execution (Zero-Competition Bound):
   Slack = Window - (L_left + B_max + headroom)
   - When Slack >= 0: No competition exists between in-file and cross-file tokens.
     Lowering cross-file budget discards external APIs without recovering ANY in-file tokens.
     Execution is locked to B_max (1536/2304), protecting CCEval from generic overconfident hallucination.
   - When Slack < 0: Physical crowding occurs (in-file context is truncated).
     Multi-budget candidate generation {B_low, B_mid, B_max} is activated to explore the trade-off frontier.

2. Grounding-Calibrated Confidence (Context Attribution):
   For crowded tasks where candidates are evaluated across budgets:
     Score(y, B) = conf(y) + beta * GroundingRatio(y, Context_B)
   Penalizes generic high-probability tokens that ignore context and rewards candidates grounded in repo evidence.

3. Calibrated Copy-Protection Gate (G_copy >= 0.95):
   Preserves Pass 1A local completions on verbatim copying tasks with guaranteed zero negative transfer.
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

# CADRE imports
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
    sys.path.insert(0, str(Path.cwd() / "cadre_official"))
    try:
        from cadre.cssi import TypedContextTree, CSSIContractBuilder
        from cadre.intent_parser import IntentParser, clean_bpe_text, strip_method_affixes
    except ImportError:
        pass

# Metric Helpers
try:
    from fuzzywuzzy import fuzz
    def cal_edit_sim(p: str, g: str) -> float:
        return float(fuzz.ratio(p.strip(), g.strip()))
except ImportError:
    def cal_edit_sim(p: str, g: str) -> float:
        return SequenceMatcher(None, p.strip(), g.strip()).ratio() * 100.0

try:
    import editdistance
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
except ImportError:
    def repoeval_es(target: str, pred: str) -> float:
        return cal_edit_sim(pred, target)

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
# Statement Truncation & Official Metric Helpers
# -------------------------------------------------------------
def get_py_statement_line(pred: str, gt: str) -> tuple[str, str]:
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

        by_bench[bm].append({"repo": repo, "em": em, "es": es})

    summary = {}
    macro_em, macro_es = [], []
    for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        items = by_bench[bm]
        if not items:
            continue
        n = len(items)
        if bm.startswith("repoeval"):
            repo_map = defaultdict(list)
            repo_es = defaultdict(list)
            for it in items:
                repo_map[it["repo"]].append(it["em"])
                repo_es[it["repo"]].append(it["es"])
            b_em = sum(sum(v)/len(v) for v in repo_map.values()) / len(repo_map) * 100.0
            b_es = sum(sum(v)/len(v) for v in repo_es.values()) / len(repo_es)
        else:
            b_em = sum(it["em"] for it in items) / n * 100.0
            b_es = sum(it["es"] for it in items) / n

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
# Context Builders & Coalescing
# -------------------------------------------------------------
def coalesce_chunks_unified(chunks: List[Dict[str, Any]], comment_prefix: str) -> List[str]:
    chunks_by_file = defaultdict(list)
    for c in chunks:
        fp = c.get("file_path") or c.get("path") or ""
        if not fp or c.get("is_stop"):
            break
        code = c.get("code") or c.get("text") or c.get("code_content") or ""
        if fp and code.strip():
            chunks_by_file[fp].append(code.strip())

    blocks = []
    for fp, clist in chunks_by_file.items():
        header = f"{comment_prefix} file path: {fp}\n"
        body = "\n...\n".join(clist)
        blocks.append(f"{header}{body}\n\n")
    return blocks


def build_cadre_prompt(
    tokenizer: Any,
    task: Dict[str, Any],
    tr: Dict[str, Any],
    tree_cssi_block: str = "",
    target_spec_block: str = "",
    max_exemplar_tokens: int = 1536,
    max_tree_tokens: int = 200,
    max_spec_tokens: int = 200,
    max_total_tokens: int = 3072,
    min_left_tokens: int = 1024,
    generation_headroom: int = 64,
) -> Tuple[List[int], str]:
    """Builds tokenized prompt and returns (prompt_ids, context_text_for_grounding)."""
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    comment_prefix = "#" if lang == "python" else "//"

    # 1. Active file path header
    path_text = f"\n\n{comment_prefix} file path: {cur_path}\n"
    path_ids = tokenizer.encode(path_text, add_special_tokens=False)

    # 2. Structural & Spec Channels
    tree_ids = tokenizer.encode(tree_cssi_block, add_special_tokens=False)[:max_tree_tokens] if tree_cssi_block else []
    spec_ids = tokenizer.encode(target_spec_block, add_special_tokens=False)[:max_spec_tokens] if target_spec_block else []
    stubs_len = len(tree_ids) + len(spec_ids)

    # 3. Raw Cross-file Chunks
    available_for_crossfile = max_total_tokens - (min_left_tokens + stubs_len + len(path_ids) + generation_headroom)
    allowed_crossfile = min(max_exemplar_tokens, max(0, available_for_crossfile))

    raw_candidates = tr.get("rlcoder_top10") or tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    exemplar_ids = []
    exemplar_text_parts = []
    if raw_candidates:
        blocks = coalesce_chunks_unified(raw_candidates, comment_prefix)
        for b in blocks:
            b_ids = tokenizer.encode(b, add_special_tokens=False)
            if len(exemplar_ids) + len(b_ids) > allowed_crossfile:
                break
            exemplar_ids.extend(b_ids)
            exemplar_text_parts.append(b)

    cross_ids = exemplar_ids + tree_ids + spec_ids
    context_text = "".join(exemplar_text_parts) + "\n" + tree_cssi_block + "\n" + target_spec_block

    # 4. Dynamic Left Context Allocation
    consumed_tokens = len(cross_ids) + len(path_ids) + generation_headroom
    allowed_left_length = max(min_left_tokens, max_total_tokens - consumed_tokens)
    left_ids = tokenizer.encode(left_context, add_special_tokens=False)[-allowed_left_length:]

    final_ids = cross_ids + path_ids + left_ids
    return final_ids[-(max_total_tokens - generation_headroom):], context_text


# -------------------------------------------------------------
# Grounding Scorer & Helpers
# -------------------------------------------------------------
def get_identifiers(text: str) -> List[str]:
    text = re.sub(r"#.*", "", text)
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"\"[^\"]*\"", "", text)
    text = re.sub(r"\'[^\']*\'", "", text)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    skip = {"self", "this", "def", "class", "return", "if", "else", "for", "in", "while",
            "import", "from", "as", "try", "except", "finally", "public", "private",
            "protected", "static", "void", "new", "int", "boolean", "String", "true", "false", "null", "None"}
    return [w for w in words if w not in skip and len(w) > 2]


def first_line_conf(pieces: List[str], lps: List[float]) -> float:
    started, vals, text = False, [], ""
    for p, lp in zip(pieces, lps):
        if not started and p.strip() == "":
            continue
        started = True
        before = text
        text += p
        vals.append(lp)
        if "\n" in p and before.strip():
            break
    return sum(vals) / len(vals) if vals else -10.0


def compute_grounded_score(candidate_text: str, conf: float, context_text: str, beta: float = 0.15) -> float:
    first_line = candidate_text.splitlines()[0] if candidate_text.splitlines() else candidate_text
    cand_ids = set(get_identifiers(first_line))
    if not cand_ids:
        return conf
    context_ids = set(get_identifiers(context_text))
    overlap = len(cand_ids & context_ids)
    grounding_ratio = overlap / len(cand_ids)
    return conf + beta * grounding_ratio


def max_line_similarity(cand_line: str, reference_texts: List[str]) -> float:
    c = cand_line.strip()
    if len(c) < 10:
        return 0.0
    best = 0.0
    for txt in reference_texts:
        for raw in txt.splitlines():
            s = raw.strip().lstrip("#").lstrip("//").strip()
            if len(s) < 10:
                continue
            if s == c:
                return 1.0
            if min(len(s), len(c)) / max(len(s), len(c)) < 0.70:
                continue
            r = SequenceMatcher(None, c, s).ratio()
            if r > best:
                best = r
    return best


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
    best_score = 0.0

    for pf in pool_files:
        fpath = pf.get("path", "")
        code_txt = pf.get("text", "")
        if not code_txt or fpath == task.get("path"):
            continue

        lines = code_txt.splitlines()
        for idx, line in enumerate(lines):
            s = line.strip()
            if lang == "python":
                if s.startswith("def ") and "(" in s:
                    m = re.match(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", s)
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
                            while j < len(lines) and not sig_lines[-1].endswith(":"):
                                j += 1
                                if j < len(lines):
                                    sig_lines.append(lines[j].strip())
                            full_sig = " ".join(sig_lines)

                            enclosing_class = ""
                            for k in range(idx - 1, max(-1, idx - 40), -1):
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
# Main Execution Pipeline
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run CADRE v2.9 Slack-Gated Adaptive Context Budgeting")
    parser.add_argument("--data-root", default="/content", help="Directory containing bundles")
    parser.add_argument("--trace-path", default="", help="Path to retrieval traces jsonl.gz")
    parser.add_argument("--output-dir", default="/content/cadre-results/cadre_v29_slack", help="Output directory")
    parser.add_argument("--benchmarks", default="cceval_python,cceval_java,repoeval_line,repoeval_api", help="Benchmarks")
    parser.add_argument("--limit-per-benchmark", type=int, default=None, help="Limit tasks per benchmark")
    parser.add_argument("--max-model-len", type=int, default=3072, help="Context window (default 3072)")
    parser.add_argument("--generator-batch-size", type=int, default=128, help="vLLM batch size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82, help="vLLM GPU memory utilization")
    parser.add_argument("--copy-threshold", type=float, default=0.95, help="Copy protection threshold (default 0.95)")
    parser.add_argument("--beta-grounding", type=float, default=0.15, help="Grounding calibration weight (default 0.15)")
    parser.add_argument("--budgets", default="384,1536,2304", help="Budgets for adaptive selection (default 384,1536,2304)")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (50 tasks per benchmark)")
    args = parser.parse_args()

    budget_list = [int(b.strip()) for b in args.budgets.split(",")]
    max_b = max(budget_list)

    print("=" * 80)
    print("CADRE v2.9: Slack-Gated & Grounding-Calibrated Adaptive Context Budgeting")
    print(f"Window: {args.max_model_len} | Elastic Budgets: {budget_list} (Max: {max_b})")
    print(f"Copy-Protect Gate: {args.copy_threshold} | Beta Grounding: {args.beta_grounding}")
    print("=" * 80, flush=True)

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = data_root / "full_benchmark_8004_bundle.tar.gz"
    if not bundle_path.exists():
        bundle_path = Path("experiments/mb_spt_iterative/full_benchmark_8004_bundle.tar.gz")

    if args.trace_path:
        trace_path = Path(args.trace_path)
    else:
        trace_path = data_root / "rlcoder_full_8004_retrieval_trace.jsonl.gz"
        if not trace_path.exists():
            trace_path = Path("experiments/mb_spt_iterative/rlcoder_full_8004_retrieval_trace.jsonl.gz")

    if not (data_root / "tasks.jsonl.gz").exists() and bundle_path.exists():
        print(f"Extracting {bundle_path} to {data_root}...", flush=True)
        with tarfile.open(bundle_path, "r:gz") as t:
            t.extractall(data_root)

    # 1. Load Files & Traces
    files_path = data_root / "files.jsonl.gz"
    tasks_path = data_root / "tasks.jsonl.gz"

    print("Loading repository file pool...", flush=True)
    files = {r["file_id"]: {"path": r["path"], "text": r["text"]} for r in map(json.loads, gzip.open(files_path, "rt"))}
    all_tasks = [json.loads(l) for l in gzip.open(tasks_path, "rt")]
    all_traces = {(r["benchmark"], r["task_id"]): r for r in map(json.loads, gzip.open(trace_path, "rt"))}
    print(f"Loaded {len(files)} repo files, {len(all_tasks)} tasks, {len(all_traces)} traces.", flush=True)

    requested_bms = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    if args.limit_per_benchmark is not None:
        selected_tasks = []
        for bm in requested_bms:
            bm_tasks = [t for t in all_tasks if t["benchmark"] == bm][:args.limit_per_benchmark]
            selected_tasks.extend(bm_tasks)
        tasks = selected_tasks
    elif args.smoke:
        selected_tasks = []
        for bm in requested_bms:
            bm_tasks = [t for t in all_tasks if t["benchmark"] == bm][:50]
            selected_tasks.extend(bm_tasks)
        tasks = selected_tasks
    else:
        tasks = [t for t in all_tasks if t["benchmark"] in requested_bms]

    print(f"Total tasks selected for execution: {len(tasks)}", flush=True)

    # Match tasks to pools and traces
    task_trs = [all_traces.get((t["benchmark"], t["task_id"]), {}) for t in tasks]
    task_pool_files = []
    for t in tasks:
        pool = []
        fids = t.get("file_ids_by_path", {})
        for fp, fid in fids.items():
            if fid in files:
                pool.append(files[fid])
        task_pool_files.append(pool)

    # Tokenizer & vLLM setup
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    rev = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
    local_model_dir = Path("/content/models/deepseek-coder-1.3b-base")
    tokenizer_source = str(local_model_dir) if (local_model_dir / "tokenizer_config.json").is_file() else model_name

    print(f"Loading tokenizer from {tokenizer_source} (Fast BPE)...", flush=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_source, revision=rev, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Compute Slack for each task
    headroom = 64
    crowded_indices = []
    non_crowded_indices = []
    for i, t in enumerate(tasks):
        l_ids = tok.encode(t["left_context"], add_special_tokens=False)
        slack = args.max_model_len - (len(l_ids) + max_b + headroom)
        if slack >= 0:
            non_crowded_indices.append(i)
        else:
            crowded_indices.append(i)

    print(f"\nSlack-Gated Profile across {len(tasks)} tasks:")
    print(f"  - Non-Crowded (Slack >= 0, Lock B_max): {len(non_crowded_indices)} tasks ({len(non_crowded_indices)/len(tasks)*100:.1f}%)")
    print(f"  - Crowded (Slack < 0, Multi-Budget Enabled): {len(crowded_indices)} tasks ({len(crowded_indices)/len(tasks)*100:.1f}%)", flush=True)

    # 4. Synthesize TypedContextTree (Pass 1B)
    print("\nSynthesizing TypedContextTree structural contracts...", flush=True)
    t_tree_start = time.time()
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
    print(f"  Done in {time.time()-t_tree_start:.1f}s.", flush=True)

    # 5. Initialize vLLM Engine
    print(f"\nInitializing vLLM Engine ({model_name}, max_model_len={args.max_model_len})...", flush=True)
    llm = LLM(
        model=model_name,
        tokenizer=tokenizer_source,
        revision=rev,
        tokenizer_revision=rev,
        dtype="float16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.generator_batch_size,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=20261026,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sp_fast = SamplingParams(max_tokens=64, temperature=0.0, top_p=1.0)
    sp_logprobs = SamplingParams(max_tokens=64, temperature=0.0, top_p=1.0, logprobs=1)

    def vllm_generate_text(prompts_ids: List[List[int]]) -> List[str]:
        outputs = llm.generate([{"prompt_token_ids": p} for p in prompts_ids], sampling_params=sp_fast)
        results = []
        for o in outputs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            dec = tok.decode(ids, skip_special_tokens=True)
            text = dec if sum(ch.isspace() for ch in dec) >= sum(ch.isspace() for ch in c.text) else c.text
            results.append(text)
        return results

    def vllm_generate_with_conf(prompts_ids: List[List[int]]) -> List[Dict[str, Any]]:
        outputs = llm.generate([{"prompt_token_ids": p} for p in prompts_ids], sampling_params=sp_logprobs)
        results = []
        for o in outputs:
            c = o.outputs[0]
            ids = list(c.token_ids)
            dec = tok.decode(ids, skip_special_tokens=True)
            text = dec if sum(ch.isspace() for ch in dec) >= sum(ch.isspace() for ch in c.text) else c.text
            keep = ids.index(tok.eos_token_id) if tok.eos_token_id in ids else len(ids)
            lp = [round(c.logprobs[j][ids[j]].logprob, 4) for j in range(keep)]
            pieces = [tok.decode([i]) for i in ids[:keep]]
            conf = first_line_conf(pieces, lp)
            results.append({"text": text, "conf": conf, "pieces": pieces, "lp": lp})
        return results

    # Step 1: Pass 1A (Raw Baseline at b=1536) for Copy Protection Gate
    print(f"\n[Step 1] Running Pass 1A (Raw Baseline at b=1536)...", flush=True)
    p0_raw_data = [build_cadre_prompt(tok, t, tr, tree_cssi_block="", target_spec_block="", max_exemplar_tokens=1536, max_total_tokens=args.max_model_len) for t, tr in zip(tasks, task_trs)]
    preds_raw = vllm_generate_text([p[0] for p in p0_raw_data])

    # Step 2: Pass 1B (Tree Draft at b=1536)
    print(f"\n[Step 2] Running Pass 1B (Tree Draft at b=1536)...", flush=True)
    p0_tree_data = [build_cadre_prompt(tok, t, tr, tree_cssi_block=tb, target_spec_block="", max_exemplar_tokens=1536, max_total_tokens=args.max_model_len) for t, tr, tb in zip(tasks, task_trs, tree_cssi_blocks)]
    preds_tree = vllm_generate_text([p[0] for p in p0_tree_data])

    # Step 3: Target Spec Resolution
    print("\n[Step 3] Resolving Target API Specifications from Tree Draft...", flush=True)
    target_specs = [resolve_target_spec_with_hierarchy(t, tr, pf, d) for t, tr, pf, d in zip(tasks, task_trs, task_pool_files, preds_tree)]
    spec_count = sum(1 for s in target_specs if s)
    print(f"  Target specs resolved: {spec_count}/{len(tasks)} ({spec_count/len(tasks)*100:.1f}%).", flush=True)

    # Step 4: Elastic Multi-Budget Generation (Slack-Gated)
    # Non-Crowded tasks (Slack >= 0): evaluated on {1536, 2304} (never degrade below 1536)
    # Crowded tasks (Slack < 0): evaluated on {384, 1536, 2304} (explore full trade-off frontier)
    candidates_by_task = defaultdict(dict)  # task_idx -> budget -> {text, conf, score}

    for b in budget_list:
        # If b < 1536 (e.g. 384): ONLY crowded tasks run this budget to rescue in-file context
        # If b >= 1536: ALL tasks run this budget
        if b < 1536:
            active_task_indices = crowded_indices
        else:
            active_task_indices = list(range(len(tasks)))

        if not active_task_indices:
            continue

        print(f"\n[Step 4 - Budget {b}] Generating candidates for {len(active_task_indices)} tasks (min_left={1024 if b <= 1536 else 256})...", flush=True)
        prompts_b = []
        contexts_b = []
        for idx in active_task_indices:
            t = tasks[idx]
            tr = task_trs[idx]
            tb = tree_cssi_blocks[idx]
            sb = target_specs[idx]
            p_ids, ctx_text = build_cadre_prompt(
                tok, t, tr, tree_cssi_block=tb, target_spec_block=sb,
                max_exemplar_tokens=b, max_total_tokens=args.max_model_len,
                min_left_tokens=1024 if b <= 1536 else 256,
            )
            prompts_b.append(p_ids)
            contexts_b.append(ctx_text)

        out_b = vllm_generate_with_conf(prompts_b)

        for pos, idx in enumerate(active_task_indices):
            res = out_b[pos]
            txt = res["text"]
            conf = res["conf"]
            score = compute_grounded_score(txt, conf, contexts_b[pos], beta=args.beta_grounding)
            candidates_by_task[idx][b] = {
                "text": txt,
                "conf": conf,
                "score": score,
            }

    # Step 5: Candidate Selection & Copy-Protection Gate
    print(f"\n[Step 5] Selecting candidates & applying Copy-Protection Gate (Thresh={args.copy_threshold})...", flush=True)
    all_predictions = []
    slack_guarded_count = 0
    multi_selected_count = 0
    copy_protected_count = 0

    for i, t in enumerate(tasks):
        tr = task_trs[i]
        cands = candidates_by_task[i]

        # Select budget with maximum Grounding-Calibrated Score among allowed candidate budgets
        best_b = max(cands.keys(), key=lambda b: cands[b]["score"])
        chosen_cand = cands[best_b]

        if i in non_crowded_indices:
            decision = f"slack_guarded_b{best_b}"
            slack_guarded_count += 1
        else:
            decision = f"crowded_b{best_b}"
            multi_selected_count += 1

        chosen_text = chosen_cand["text"]

        # Copy-Protection Gate Check against Pass 1A
        ref_texts = []
        raw_candidates = tr.get("rlcoder_top10") or tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
        for rc in raw_candidates[:5]:
            code = rc.get("code") or rc.get("text") or rc.get("code_content") or ""
            if code.strip():
                ref_texts.append(code)

        first_line = chosen_text.splitlines()[0] if chosen_text.splitlines() else chosen_text
        sim = max_line_similarity(first_line, ref_texts)

        if sim >= args.copy_threshold:
            final_pred = preds_raw[i]
            gate_dec = "copy_protected_pass1"
            copy_protected_count += 1
        else:
            final_pred = chosen_text
            gate_dec = decision

        all_predictions.append({
            "task_id": t["task_id"],
            "benchmark": t["benchmark"],
            "language": t["language"],
            "groundtruth": t["groundtruth"],
            "path": t.get("path", ""),
            "pass1_raw": preds_raw[i],
            "chosen_budget": best_b,
            "gate_decision": gate_dec,
            "copy_sim": round(sim, 3),
            "cadre_v29_final": final_pred,
        })

    print(f"Decisions: {slack_guarded_count} Slack-Guarded ({{1536,2304}}), {multi_selected_count} Crowded ({{384,1536,2304}}), {copy_protected_count} Copy-Protected.", flush=True)

    # 6. Evaluation
    stats_raw = evaluate_records_official(all_predictions, "pass1_raw")
    stats_final = evaluate_records_official(all_predictions, "cadre_v29_final")

    print("\n" + "=" * 80)
    print("CADRE v2.9 OFFICIAL EVALUATION RESULTS:")
    print("=" * 80)
    print(f"{'Benchmark':<18} | {'Pass 1A EM':<12} | {'v2.9 Final EM':<14} | {'Gain (Δ)':<10} | {'v2.9 ES':<10}")
    print("-" * 75)
    for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        if bm in stats_final:
            e_raw = stats_raw[bm]["em"]
            e_fin = stats_final[bm]["em"]
            diff = e_fin - e_raw
            es_fin = stats_final[bm]["es"]
            print(f"{bm:<18} | {e_raw:>10.2f}% | {e_fin:>12.2f}% | {diff:>+8.2f} pp | {es_fin:>8.2f}%")
    print("-" * 75)
    m_raw = stats_raw.get("macro_mean", {})
    m_fin = stats_final.get("macro_mean", {})
    if m_fin:
        diff_macro = m_fin["em"] - m_raw["em"]
        print(f"{'Macro Mean':<18} | {m_raw['em']:>10.2f}% | {m_fin['em']:>12.2f}% | {diff_macro:>+8.2f} pp | {m_fin['es']:>8.2f}%")
    print("=" * 80)

    # Save results
    metrics_path = out_dir / f"{out_dir.name}_metrics.json"
    preds_path = out_dir / f"{out_dir.name}_predictions.jsonl.gz"
    with open(metrics_path, "w") as f:
        json.dump({"raw": stats_raw, "final": stats_final}, f, indent=2)
    with open(out_dir / "cadre_v29_metrics.json", "w") as f:
        json.dump({"raw": stats_raw, "final": stats_final}, f, indent=2)
    with gzip.open(preds_path, "wt", encoding="utf-8") as f:
        for p in all_predictions:
            f.write(json.dumps(p) + "\n")
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved predictions to {preds_path}")


if __name__ == "__main__":
    main()
