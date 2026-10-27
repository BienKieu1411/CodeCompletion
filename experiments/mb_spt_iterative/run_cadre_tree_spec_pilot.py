#!/usr/bin/env python3
"""
run_cadre_tree_spec_pilot.py - Evaluates CADRE-TreeSpec on the 400 Stratified Pilot Tasks.
Integrates:
1. TypedContextTree (T_CSSI) with AST Parsing & Inheritance Resolution (pulling inherited methods).
2. Decorated Stub Text Embedding (in-distribution native PEP 484 / Java syntax with ancestor markers).
3. 2-Pass Refinement (Pass 1 speculative draft -> Intent Parser -> Target Spec Resolver with Class Hierarchy).
4. Official evaluation protocols (EM, Bracket Match for Java, Edit Sim, Repo-Averaging).
"""

import gzip
import json
import os
import re
import sys
import tarfile
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, "/content/cadre_official")
from cadre.cssi import TypedContextTree, CSSIContractBuilder
from cadre.intent_parser import IntentParser, strip_method_affixes

try:
    from fuzzywuzzy import fuzz
    def cal_edit_sim(p: str, g: str) -> float:
        return float(fuzz.ratio(p.strip(), g.strip()))
except ImportError:
    def cal_edit_sim(p: str, g: str) -> float:
        return SequenceMatcher(None, p.strip(), g.strip()).ratio() * 100.0

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DATA_ROOT = Path("/content") if Path("/content").exists() else Path("experiments/mb_spt_iterative")
GENERATOR = "deepseek-ai/deepseek-coder-1.3b-base"
GENERATOR_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
SEED = 20261026

print(f"Loading tokenizer {GENERATOR}...", flush=True)
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

# Load EXACT 400 pilot tasks
pilot_pool_path = DATA_ROOT / "rerank_pilot_pool.jsonl.gz"
if not pilot_pool_path.exists():
    pilot_pool_path = Path("experiments/mb_spt_iterative/rerank_pilot_pool.jsonl.gz")

with gzip.open(pilot_pool_path, "rt") as f:
    pilot_pool_records = [json.loads(l) for l in f]

pilot_keys_set = {(r["benchmark"], r["task_id"]) for r in pilot_pool_records}
tasks = [t for t in all_tasks if (t["benchmark"], t["task_id"]) in pilot_keys_set]
print(f"Loaded {len(tasks)} pilot tasks matching baseline pool exactly.", flush=True)

# Build task pool files
task_pool_files = []
for t in tasks:
    if "file_ids_by_path" in t:
        pf = [files[f] for f in t["file_ids_by_path"].values() if f in files]
    else:
        pf = [files[f] for f in pools.get(t.get("pool_id"), []) if f in files]
    task_pool_files.append(pf)

task_trs = [traces.get((t["benchmark"], t["task_id"]), {}) for t in tasks]


# Metric utilities
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
        raw_pred = r[pred_key]
        gt = r["groundtruth"]

        gt_no_comment = remove_comments(gt)
        raw_no_comment = remove_comments(raw_pred)

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
            b_es = sum(sum(v)/len(v) for v in repo_es.values()) / len(repo_es) * 100.0
        else:
            b_em = sum(it["em"] for it in items) / n * 100.0
            b_es = sum(it["es"] for it in items) / n * 100.0

        summary[bm] = {"em": round(b_em, 2), "es": round(b_es, 2), "count": n}
        macro_em.append(b_em)
        macro_es.append(b_es)

    summary["macro_mean"] = {
        "em": round(sum(macro_em) / 4.0, 2),
        "es": round(sum(macro_es) / 4.0, 2),
    }
    return summary


# Coalescing raw chunks by file
def coalesce_chunks(chunks: List[Dict[str, Any]], comment_prefix: str) -> List[str]:
    chunks_by_file = defaultdict(list)
    for c in chunks:
        fp = c.get("file_path") or c.get("path") or ""
        code = c.get("code") or c.get("text") or ""
        if fp and code.strip():
            chunks_by_file[fp].append(code.strip())

    blocks = []
    for fp, clist in chunks_by_file.items():
        content = "\n...\n".join(clist)
        blocks.append(f"{comment_prefix} file path: {fp}\n{content}\n\n")
    return blocks


# Prompt builder supporting Tree CSSI and Target Spec
def build_cadre_prompt(
    task: Dict[str, Any],
    tr: Dict[str, Any],
    tree_cssi_block: str = "",
    target_spec_block: str = "",
    max_total_tokens: int = 8000,
    max_cross_file_tokens: int = 3200,
) -> List[int]:
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    comment_prefix = "#" if lang == "python" else "//"

    path_text = f"{comment_prefix} file path: {cur_path}\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)
    left_ids = tok.encode(left_context, add_special_tokens=False)

    tree_ids = tok.encode(tree_cssi_block, add_special_tokens=False) if tree_cssi_block else []
    spec_ids = tok.encode(target_spec_block, add_special_tokens=False) if target_spec_block else []

    used_for_contracts = len(tree_ids) + len(spec_ids)
    allowed_raw = max(0, max_cross_file_tokens - used_for_contracts)

    raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    exemplar_ids = []
    if raw_candidates:
        blocks = coalesce_chunks(raw_candidates, comment_prefix)
        for b in blocks:
            b_ids = tok.encode(b, add_special_tokens=False)
            if len(exemplar_ids) + len(b_ids) > allowed_raw:
                break
            exemplar_ids.extend(b_ids)

    cross_ids = exemplar_ids + tree_ids + spec_ids
    allowed_left = max(0, max_total_tokens - len(cross_ids) - len(path_ids))
    truncated_left = left_ids[-allowed_left:] if allowed_left else []

    final_ids = (cross_ids + path_ids + truncated_left)[-max_total_tokens:]
    return final_ids


# Target specification resolver with class hierarchy & literal cleansing
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

    # Gating: if callee is already defined in top 3 chunks, skip
    raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    top_chunks_text = " ".join((c.get("code") or c.get("text") or "") for c in raw_candidates[:3])
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

                            # Search backwards for enclosing class
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


def main():
    print(f"\n=== INITIALIZING VLLM ENGINE: {GENERATOR} (8k context) ===", flush=True)
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

    # 1. Synthesize Tree-CSSI contracts for all tasks
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
        if idx % 100 == 0:
            print(f"  Processed {idx}/{len(tasks)} task trees...", flush=True)

    non_empty_trees = sum(1 for c in tree_cssi_blocks if c)
    print(f"  ✓ Successfully synthesized {non_empty_trees}/{len(tasks)} TypedContextTree contracts.", flush=True)

    all_predictions = []
    for t in tasks:
        all_predictions.append({
            "task_id": t["task_id"],
            "benchmark": t["benchmark"],
            "language": t["language"],
            "groundtruth": t["groundtruth"],
        })

    # Pass 1A: Baseline CADRE v3 (Raw Chunks only, no tree)
    print("\n[Pass 1A] Running Standard Baseline (Raw Chunks Only)...", flush=True)
    p0_raw_ids = [build_cadre_prompt(t, tr, tree_cssi_block="", target_spec_block="") for t, tr in zip(tasks, task_trs)]
    out0 = llm.generate([{"prompt_token_ids": p} for p in p0_raw_ids], sampling_params=sp)
    preds_raw = [o.outputs[0].text for o in out0]
    for i, p in enumerate(preds_raw):
        all_predictions[i]["pass1_raw"] = p
    stats_raw = evaluate_records(all_predictions, "pass1_raw")
    print("  ✓ Pass 1A (Raw Baseline) Macro EM:", stats_raw["macro_mean"]["em"], "% | Macro ES:", stats_raw["macro_mean"]["es"], "%", flush=True)

    # Pass 1B: Tree-CSSI Augmented (Raw Chunks + TypedContextTree decorated stubs)
    print("\n[Pass 1B] Running Tree-CSSI Augmented (Raw Chunks + TypedContextTree Stubs)...", flush=True)
    p0_tree_ids = [build_cadre_prompt(t, tr, tree_cssi_block=tb, target_spec_block="") for t, tr, tb in zip(tasks, task_trs, tree_cssi_blocks)]
    out1b = llm.generate([{"prompt_token_ids": p} for p in p0_tree_ids], sampling_params=sp)
    preds_tree_p1 = [o.outputs[0].text for o in out1b]
    for i, p in enumerate(preds_tree_p1):
        all_predictions[i]["pass1_tree_cssi"] = p
    stats_tree_p1 = evaluate_records(all_predictions, "pass1_tree_cssi")
    print("  ✓ Pass 1B (Tree-CSSI Pass 1) Macro EM:", stats_tree_p1["macro_mean"]["em"], "% | Macro ES:", stats_tree_p1["macro_mean"]["es"], "%", flush=True)

    # Pass 2: CADRE-TreeSpec (Refined with Target Spec Resolver)
    print("\n[Pass 2] Resolving Target Specifications from Pass 1 Draft...", flush=True)
    p2_ids = []
    refined_count = 0
    for i, (t, tr, pf, draft, tree_block) in enumerate(zip(tasks, task_trs, task_pool_files, preds_tree_p1, tree_cssi_blocks)):
        target_spec = resolve_target_spec_with_hierarchy(t, tr, pf, draft)
        if target_spec:
            refined_count += 1
            # In Pass 2, inject both Tree-CSSI base contracts and the focused Target Spec
            p2 = build_cadre_prompt(t, tr, tree_cssi_block=tree_block, target_spec_block=target_spec)
        else:
            p2 = p0_tree_ids[i]
        p2_ids.append(p2)

    print(f"  ✓ Target specification triggered for {refined_count}/{len(tasks)} tasks.", flush=True)
    print("\n[Pass 2] Running CADRE-TreeSpec Final Refinement...", flush=True)
    out2 = llm.generate([{"prompt_token_ids": p} for p in p2_ids], sampling_params=sp)
    preds_final = [o.outputs[0].text for o in out2]
    for i, p in enumerate(preds_final):
        all_predictions[i]["cadre_tree_spec_final"] = p
    stats_final = evaluate_records(all_predictions, "cadre_tree_spec_final")
    print("  ✓ Pass 2 (CADRE-TreeSpec Final) Macro EM:", stats_final["macro_mean"]["em"], "% | Macro ES:", stats_final["macro_mean"]["es"], "%", flush=True)

    # Print Full Comparative Table
    print("\n" + "=" * 80)
    print("=== CADRE-TreeSpec COMPREHENSIVE BENCHMARK RESULTS (400 Pilot Tasks) ===")
    print("=" * 80)
    print(f"{'Method / Configuration':<35} | {'Python':<8} | {'Java':<8} | {'API':<8} | {'Line':<8} | {'Macro EM':<10} | {'Macro ES':<10}")
    print("-" * 80)
    print(f"{'Pass 1A: Raw Baseline':<35} | {stats_raw['cceval_python']['em']:<8.2f} | {stats_raw['cceval_java']['em']:<8.2f} | {stats_raw['repoeval_api']['em']:<8.2f} | {stats_raw['repoeval_line']['em']:<8.2f} | {stats_raw['macro_mean']['em']:<10.2f} | {stats_raw['macro_mean']['es']:<10.2f}")
    print(f"{'Pass 1B: Tree-CSSI Decorated':<35} | {stats_tree_p1['cceval_python']['em']:<8.2f} | {stats_tree_p1['cceval_java']['em']:<8.2f} | {stats_tree_p1['repoeval_api']['em']:<8.2f} | {stats_tree_p1['repoeval_line']['em']:<8.2f} | {stats_tree_p1['macro_mean']['em']:<10.2f} | {stats_tree_p1['macro_mean']['es']:<10.2f}")
    print(f"{'Pass 2: CADRE-TreeSpec (Final)':<35} | {stats_final['cceval_python']['em']:<8.2f} | {stats_final['cceval_java']['em']:<8.2f} | {stats_final['repoeval_api']['em']:<8.2f} | {stats_final['repoeval_line']['em']:<8.2f} | {stats_final['macro_mean']['em']:<10.2f} | {stats_final['macro_mean']['es']:<10.2f}")
    print("=" * 80)

    # Save output files
    out_json = DATA_ROOT / "cadre_tree_spec_metrics.json"
    with open(out_json, "w") as f:
        json.dump({
            "pass1_raw": stats_raw,
            "pass1_tree_cssi": stats_tree_p1,
            "cadre_tree_spec_final": stats_final,
        }, f, indent=2)

    out_preds = DATA_ROOT / "cadre_tree_spec_predictions.jsonl.gz"
    with gzip.open(out_preds, "wt") as f:
        for r in all_predictions:
            f.write(json.dumps(r) + "\n")

    print(f"\nArtifacts saved to {out_json} and {out_preds}", flush=True)


if __name__ == "__main__":
    main()
