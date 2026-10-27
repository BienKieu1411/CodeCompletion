#!/usr/bin/env python3
"""
run_cadre_v2_tree_copysim_pilot.py
CADRE-v2 with Dual-Scope AST Tree (Classes + Module Functions) & Copy-Protection Gate
Evaluated on the 400 Stratified Pilot Tasks (DeepSeek-Coder-1.3B)

Architecture Highlights:
1. Pass 1: Baseline CADRE v3 (100% Raw Chunks, max 3200 cross-file, max 8000 total).
   - Zero tree displacement in Pass 1 to preserve 100% of the RepoEval API 48.04% / Line 60.45% baseline.
2. Copy-Protection Gate (copy_sim):
   - Adopted from Phong's branch (d12_compact_gated.py).
   - If Pass 1 draft's first line has string similarity >= 0.85 with retrieved chunks or local context,
     it is already a confident copy-match -> DO NOT intervene in Pass 2! Keep Pass 1 prediction.
3. Strict Definition-Only Redundancy Gate:
   - Only gates when 'def primary(' is explicitly defined in top-3 chunks (prevents false positives from call-sites).
4. Dual-Scope AST Tree Resolver:
   - Searches BOTH class methods (wrapping in class scope to preserve hierarchy) AND module-level functions.
   - Morphological affix stripping (get_, is_, set_) to bridge syntactic variation.
   - Clean, canonical signatures (L_stub) + Docstring only (no noisy in-repo usage lines with literal constants).
"""

import ast
import difflib
import gzip
import json
import os
import re
import subprocess
import sys
import tarfile
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# 1. Dependency checks
def ensure_deps():
    try:
        import vllm
        return
    except ImportError:
        pass
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
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"])
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])

if "COLAB_GPU" in os.environ or Path("/content").exists():
    ensure_deps()

# Import CADRE Tree modules
CADRE_ROOT = Path("/content/cadre_official")
if not CADRE_ROOT.exists():
    CADRE_ROOT = Path("cadre_official").resolve()
sys.path.insert(0, str(CADRE_ROOT))

from cadre.intent_parser import IntentParser, clean_bpe_text, strip_method_affixes

try:
    from fuzzywuzzy import fuzz
    def cal_edit_sim(p: str, g: str) -> float:
        return float(fuzz.ratio(p.strip(), g.strip()))
except ImportError:
    def cal_edit_sim(p: str, g: str) -> float:
        return SequenceMatcher(None, p.strip(), g.strip()).ratio() * 100.0

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Paths & Configurations
DATA_ROOT = Path("/content") if Path("/content").exists() else Path("experiments/mb_spt_iterative")
BUNDLE_PATH = Path("/content/full_benchmark_8004_bundle.tar.gz") if Path("/content/full_benchmark_8004_bundle.tar.gz").exists() else Path("experiments/mb_spt_iterative/full_benchmark_8004_bundle.tar.gz")
OUTPUT_DIR = DATA_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENERATOR = "deepseek-ai/deepseek-coder-1.3b-base"
GENERATOR_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
SEED = 20261026

print(f"Extracting bundle if needed...", flush=True)
if not (DATA_ROOT / "tasks.jsonl.gz").exists() and BUNDLE_PATH.exists():
    with tarfile.open(BUNDLE_PATH, "r:gz") as t:
        t.extractall(DATA_ROOT)

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
pilot_pool_path = Path("/content/rerank_pilot_pool.jsonl.gz")
if not pilot_pool_path.exists():
    pilot_pool_path = DATA_ROOT / "rerank_pilot_pool.jsonl.gz"
if not pilot_pool_path.exists():
    pilot_pool_path = Path("experiments/mb_spt_iterative/rerank_pilot_pool.jsonl.gz")

pilot_pool_records = [json.loads(l) for l in gzip.open(pilot_pool_path, "rt")]
pilot_keys_set = {(r["benchmark"], r["task_id"]) for r in pilot_pool_records}
tasks = [t for t in all_tasks if (t["benchmark"], t["task_id"]) in pilot_keys_set]
print(f"Loaded {len(tasks)} pilot tasks matching baseline pool exactly.", flush=True)

task_pool_files = []
for t in tasks:
    if "file_ids_by_path" in t:
        pf = [files[f] for f in t["file_ids_by_path"].values() if f in files]
    else:
        pf = [files[f] for f in pools.get(t.get("pool_id"), []) if f in files]
    task_pool_files.append(pf)

task_trs = [traces.get((t["benchmark"], t["task_id"]), {}) for t in tasks]


# 2. Evaluation Utilities
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
        by_bench[bm].append({"repo": repo, "em": stmt_match, "es": es, "task_id": r["task_id"]})

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

        summary[bm] = {"em": round(b_em, 2), "es": round(b_es, 2), "count": n, "details": items}
        macro_em.append(b_em)
        macro_es.append(b_es)

    summary["macro_mean"] = {
        "em": round(sum(macro_em) / 4.0, 2),
        "es": round(sum(macro_es) / 4.0, 2),
    }
    return summary


# 3. Prompt Builder
def build_cadre_prompt(
    task: Dict[str, Any],
    tr: Dict[str, Any],
    spec_block: str = "",
    max_total_tokens: int = 8000,
    max_cross_file_tokens: int = 3200,
) -> List[int]:
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    comment_label = "#" if lang == "python" else "//"

    left_all_ids = tok.encode(left_context)
    path_ids = tok.encode(f"{comment_label} file path: {cur_path}\n")
    spec_ids = tok.encode(spec_block) if spec_block else []

    raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    chunks_by_file: Dict[str, List[str]] = defaultdict(list)
    for ch in raw_candidates:
        fp = ch.get("file_path") or ch.get("path", "")
        c = ch.get("code") or ch.get("text", "")
        if fp and c.strip():
            chunks_by_file[fp].append(c)

    exemplar_ids: List[int] = []
    max_exemplar_tokens = max(0, max_cross_file_tokens - len(spec_ids))
    for fp, code_list in chunks_by_file.items():
        coalesced = "\n...\n".join(code_list)
        block = f"{comment_label} file path: {fp}\n{coalesced}\n\n"
        b_ids = tok.encode(block)
        if len(exemplar_ids) + len(b_ids) > max_exemplar_tokens:
            break
        exemplar_ids.extend(b_ids)

    cross_ids = exemplar_ids + spec_ids
    allowed_left = max(0, max_total_tokens - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []
    return (cross_ids + path_ids + left_ids)[-max_total_tokens:]


# 4. Copy-Protection Metric (copy_sim)
def max_line_similarity(cand_line: str, reference_texts: List[str]) -> float:
    """Compute maximum normalized line similarity between cand_line and any line in reference_texts."""
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


# 5. Dual-Scope AST Target Spec Resolver (Classes + Module Functions)
def resolve_target_spec_dual_scope(
    task: Dict[str, Any],
    tr: Dict[str, Any],
    pool_files: List[Dict[str, Any]],
    raw_draft: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Resolves target specification with Dual-Scope AST parsing:
    - Covers Class Methods (with enclosing Class Scope & inheritance)
    - Covers Module-Level Functions (def get_delivery, def load_remote_lats)
    - Strict Definition-only Redundancy Gate
    - Copy-Protection Gate
    """
    lang = task["language"]
    left_ctx = task["left_context"]
    comment = "#" if lang == "python" else "//"

    # Step A: Copy-Protection Gate check
    clean_draft = clean_bpe_text(raw_draft)
    draft_lines = [l.strip() for l in clean_draft.splitlines() if l.strip()]
    first_draft_line = draft_lines[0] if draft_lines else ""

    raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    chunk_texts = [(c.get("code") or c.get("text") or "") for c in raw_candidates[:6]]
    recent_left = left_ctx.splitlines()[-80:]
    copy_sim = max_line_similarity(first_draft_line, chunk_texts + recent_left)

    gate_info = {
        "copy_sim": copy_sim,
        "copy_protected": False,
        "redundancy_gated": False,
        "tree_resolved": False,
        "target_name": None,
        "target_class": None,
    }

    # If already a confident copy-match (>= 0.85), protect from intervention!
    if copy_sim >= 0.85:
        gate_info["copy_protected"] = True
        return "", gate_info

    # Step B: Extract Intent from left context & draft
    receiver, candidate_callees, candidate_tokens = IntentParser.extract_intent(left_ctx, raw_draft)
    if not candidate_tokens:
        return "", gate_info

    # Step C: Strict Definition-only Redundancy Gate
    top_chunks_text = " ".join(chunk_texts[:3])
    primary = candidate_tokens[0]
    if (f"def {primary}(" in top_chunks_text or f"def {primary} (" in top_chunks_text or
        f"void {primary}(" in top_chunks_text or f"public {primary}(" in top_chunks_text):
        gate_info["redundancy_gated"] = True
        return "", gate_info

    # Step D: Dual-Scope Search across pool files
    best_candidate = None
    best_score = -1.0

    for pf in pool_files:
        fpath = pf.get("path", "")
        text = pf.get("text", "")
        if not text.strip():
            continue
        lines = text.splitlines()

        if lang == "python":
            # Attempt AST parse first
            ast_candidates = []
            try:
                tree = ast.parse(text)
                # 1. Module-level functions
                for item in tree.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fname = item.name
                        if not fname.startswith("__"):
                            sig_parts = []
                            for a in item.args.args:
                                ann = f": {ast.unparse(a.annotation)}" if a.annotation else ""
                                sig_parts.append(f"{a.arg}{ann}")
                            ret = f" -> {ast.unparse(item.returns)}" if item.returns else ""
                            sig = f"def {fname}({', '.join(sig_parts)}){ret}: ..."
                            doc = ast.get_docstring(item)
                            doc_sum = doc.splitlines()[0].strip() if doc else ""
                            ast_candidates.append({
                                "name": fname, "class_name": "", "sig": sig, "doc": doc_sum
                            })

                # 2. Class-scoped methods
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        cname = node.name
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                mname = item.name
                                if not (mname.startswith("__") and mname != "__init__"):
                                    sig_parts = []
                                    for a in item.args.args:
                                        ann = f": {ast.unparse(a.annotation)}" if a.annotation else ""
                                        sig_parts.append(f"{a.arg}{ann}")
                                    ret = f" -> {ast.unparse(item.returns)}" if item.returns else ""
                                    inner_sig = f"def {mname}({', '.join(sig_parts)}){ret}: ..."
                                    sig = f"class {cname}:\n        {inner_sig}"
                                    doc = ast.get_docstring(item)
                                    doc_sum = doc.splitlines()[0].strip() if doc else ""
                                    ast_candidates.append({
                                        "name": mname, "class_name": cname, "sig": sig, "doc": doc_sum
                                    })
            except Exception:
                ast_candidates = []

            # If AST succeeded, score them; otherwise fallback to line scanning
            if ast_candidates:
                for cand_info in ast_candidates:
                    def_name = cand_info["name"]
                    score = 0.0
                    for cand in candidate_tokens:
                        if cand == def_name:
                            score = max(score, 2.5)
                        elif strip_method_affixes(cand) and strip_method_affixes(cand) == strip_method_affixes(def_name):
                            score = max(score, 2.0)
                        elif cand.lower() == def_name.lower():
                            score = max(score, 1.8)
                        elif cand.lower() in def_name.lower():
                            score = max(score, 1.0)

                    if score > best_score and score >= 1.0:
                        best_score = score
                        best_candidate = {
                            "file": fpath,
                            "name": def_name,
                            "class_name": cand_info["class_name"],
                            "sig": cand_info["sig"],
                            "doc": cand_info["doc"],
                        }
            else:
                # Regex line scanning fallback
                for idx, line in enumerate(lines):
                    s = line.strip()
                    if s.startswith("def ") or s.startswith("async def "):
                        m = re.match(r"(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", s)
                        if m:
                            def_name = m.group(1)
                            score = 0.0
                            for cand in candidate_tokens:
                                if cand == def_name:
                                    score = max(score, 2.5)
                                elif strip_method_affixes(cand) and strip_method_affixes(cand) == strip_method_affixes(def_name):
                                    score = max(score, 2.0)
                                elif cand.lower() == def_name.lower():
                                    score = max(score, 1.8)
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

                                best_score = score
                                best_candidate = {
                                    "file": fpath,
                                    "name": def_name,
                                    "class_name": enclosing_class,
                                    "sig": full_sig,
                                    "doc": doc,
                                }
        else:  # Java
            for idx, line in enumerate(lines):
                s = line.strip()
                if (s.startswith("public ") or s.startswith("protected ") or s.startswith("private ") or s.startswith("static ")) and "(" in s and ")" in s:
                    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", s)
                    if m:
                        def_name = m.group(1)
                        if def_name in ("if", "for", "while", "switch", "catch"):
                            continue
                        score = 0.0
                        for cand in candidate_tokens:
                            if cand == def_name:
                                score = max(score, 2.5)
                            elif strip_method_affixes(cand) and strip_method_affixes(cand) == strip_method_affixes(def_name):
                                score = max(score, 2.0)
                            elif cand.lower() == def_name.lower():
                                score = max(score, 1.8)
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
                                sig = f"class {enclosing_class} {{\n        {sig}\n    }}"

                            best_score = score
                            best_candidate = {
                                "file": fpath,
                                "name": def_name,
                                "class_name": enclosing_class,
                                "sig": sig,
                                "doc": "",
                            }

    if not best_candidate:
        return "", gate_info

    gate_info["tree_resolved"] = True
    gate_info["target_name"] = best_candidate["name"]
    gate_info["target_class"] = best_candidate["class_name"]

    # Assemble Distinctive Banner (Clean Canonical Signature only, no noisy usage lines)
    banner = []
    banner.append(f"{comment} ==============================================================================")
    banner.append(f"{comment} <<< TARGET API SPECIFICATION (RESOLVED FROM DRAFT INTENT) >>>")
    banner.append(f"{comment} Source Module: {best_candidate['file']}")
    banner.append(f"{comment} Canonical Signature (L_stub):")
    for sig_l in best_candidate["sig"].splitlines():
        banner.append(f"{comment}     {sig_l}")
    if best_candidate.get("doc"):
        banner.append(f"{comment} Docstring Summary: \"\"\"{best_candidate['doc']}\"\"\"")
    banner.append(f"{comment} ==============================================================================\n\n")

    return "\n".join(banner), gate_info


# 6. Main Pipeline
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

    all_predictions = []
    for t in tasks:
        all_predictions.append({
            "task_id": t["task_id"],
            "benchmark": t["benchmark"],
            "language": t["language"],
            "groundtruth": t["groundtruth"],
        })

    # Pass 1: Baseline CADRE v3 (Full Raw Context)
    print("\n[Pass 1] Running Baseline CADRE v3 (100% Raw Chunks, Zero Displacement)...", flush=True)
    p0_ids = [build_cadre_prompt(t, tr, spec_block="") for t, tr in zip(tasks, task_trs)]
    out0 = llm.generate([{"prompt_token_ids": p} for p in p0_ids], sampling_params=sp)
    preds_pass1 = [o.outputs[0].text for o in out0]
    for i, p in enumerate(preds_pass1):
        all_predictions[i]["pass1_baseline"] = p
    stats0 = evaluate_records(all_predictions, "pass1_baseline")
    print(f"  ✓ Pass 1 Baseline Macro EM: {stats0['macro_mean']['em']}% | Macro ES: {stats0['macro_mean']['es']}%", flush=True)

    # Pass 2: CADRE-v2 with Dual-Scope AST Tree & Copy-Protection Gate
    print("\n[Pass 2] Resolving Dual-Scope AST Specifications & Applying Copy-Protection Gate...", flush=True)
    p2_ids = []
    pass2_triggered = []
    gate_stats = defaultdict(int)

    for i, (t, tr, pf, raw_draft) in enumerate(zip(tasks, task_trs, task_pool_files, preds_pass1)):
        spec_block, gate_info = resolve_target_spec_dual_scope(t, tr, pf, raw_draft)
        all_predictions[i]["gate_info"] = gate_info
        all_predictions[i]["spec_block"] = spec_block

        if gate_info["copy_protected"]:
            gate_stats["copy_protected"] += 1
            p2_ids.append(p0_ids[i])
        elif gate_info["redundancy_gated"]:
            gate_stats["redundancy_gated"] += 1
            p2_ids.append(p0_ids[i])
        elif spec_block:
            gate_stats["tree_spec_injected"] += 1
            pass2_triggered.append(i)
            p2_ids.append(build_cadre_prompt(t, tr, spec_block=spec_block))
        else:
            gate_stats["no_spec"] += 1
            p2_ids.append(p0_ids[i])

    print(f"  ✓ Gate Breakdown across 400 pilot tasks:", flush=True)
    print(f"    - Copy-Protected (copy_sim >= 0.85): {gate_stats['copy_protected']} tasks", flush=True)
    print(f"    - Redundancy-Gated (already in top chunks): {gate_stats['redundancy_gated']} tasks", flush=True)
    print(f"    - Target Spec Injected (Dual-Scope AST): {gate_stats['tree_spec_injected']} tasks ({len(pass2_triggered)/400*100:.1f}%)", flush=True)
    print(f"    - No spec resolved: {gate_stats['no_spec']} tasks", flush=True)

    print("\n[Pass 2] Running CADRE-v2 Final Generation...", flush=True)
    out2 = llm.generate([{"prompt_token_ids": p} for p in p2_ids], sampling_params=sp)
    preds_pass2 = [o.outputs[0].text for o in out2]
    for i, p in enumerate(preds_pass2):
        all_predictions[i]["pass2_final"] = p

    stats2 = evaluate_records(all_predictions, "pass2_final")
    print(f"  ✓ Pass 2 Final Macro EM: {stats2['macro_mean']['em']}% | Macro ES: {stats2['macro_mean']['es']}%", flush=True)

    # 7. Comparison Table
    arms = [
        ("Pass 1: Baseline CADRE v3 (Full Raw Context)", stats0),
        ("Pass 2: CADRE-v2 (Dual-Scope Tree + CopyProtection)", stats2),
    ]

    print("\n" + "=" * 125, flush=True)
    print(f"{'ARM':<60} | {'cceval_py':<10} | {'cceval_ja':<10} | {'repo_api':<10} | {'repo_line':<10} | {'MACRO EM':<10} | {'MACRO ES':<10}", flush=True)
    print("=" * 125, flush=True)
    for name, st in arms:
        print(f"{name:<60} | {st['cceval_python']['em']:<10.2f} | {st['cceval_java']['em']:<10.2f} | {st['repoeval_api']['em']:<10.2f} | {st['repoeval_line']['em']:<10.2f} | {st['macro_mean']['em']:<10.2f} | {st['macro_mean']['es']:<10.2f}", flush=True)
    print("=" * 125 + "\n", flush=True)

    # 8. Detailed Win/Loss Analysis & Error Breakdown
    print("=== DETAILED WIN / LOSS & ERROR BREAKDOWN ===", flush=True)
    wins = []
    losses = []
    still_failed = []

    for r in all_predictions:
        p1 = r["pass1_baseline"]
        p2 = r["pass2_final"]
        gt = r["groundtruth"]
        bm = r["benchmark"]
        tid = r["task_id"]

        gt_nc = remove_comments(gt)
        p1_nc = remove_comments(clean_bpe_text(p1))
        p2_nc = remove_comments(clean_bpe_text(p2))

        gt_lines = [l.strip() for l in gt_nc.splitlines() if l.strip()]
        n_t = max(1, len(gt_lines))
        gt_flat = "".join("".join(gt_lines).split())

        p1_lines = [l.strip() for l in p1_nc.splitlines() if l.strip()][:n_t]
        p1_flat = "".join("".join(p1_lines).split())
        em1 = int(p1_flat[:len(gt_flat)] == gt_flat and len(gt_flat) > 0)

        p2_lines = [l.strip() for l in p2_nc.splitlines() if l.strip()][:n_t]
        p2_flat = "".join("".join(p2_lines).split())
        em2 = int(p2_flat[:len(gt_flat)] == gt_flat and len(gt_flat) > 0)

        if bm == "cceval_java":
            gt_br = "".join(get_bracket_stmt(gt_nc).split())
            p1_br = "".join(get_bracket_stmt(p1_nc).split())
            p2_br = "".join(get_bracket_stmt(p2_nc).split())
            em1 = int(p1_br[:len(gt_br)] == gt_br and len(gt_br) > 0)
            em2 = int(p2_br[:len(gt_br)] == gt_br and len(gt_br) > 0)

        if em1 == 0 and em2 == 1:
            wins.append((bm, tid, gt, p1, p2, r.get("gate_info", {})))
        elif em1 == 1 and em2 == 0:
            losses.append((bm, tid, gt, p1, p2, r.get("gate_info", {})))
        elif em1 == 0 and em2 == 0:
            still_failed.append((bm, tid, gt, p2, r.get("gate_info", {})))

    print(f"Total Wins (Wrong -> Correct): {len(wins)}")
    for bm, tid, gt, p1, p2, gi in wins:
        print(f"  [WIN] {bm} {tid} | Resolved Target: {gi.get('target_class')}.{gi.get('target_name')}")
        print(f"        GT:    {repr(gt[:80])}")
        print(f"        Pass1: {repr(clean_bpe_text(p1)[:80])}")
        print(f"        Pass2: {repr(clean_bpe_text(p2)[:80])}")

    print(f"\nTotal Losses (Correct -> Wrong): {len(losses)}")
    for bm, tid, gt, p1, p2, gi in losses:
        print(f"  [LOSS] {bm} {tid} | Resolved Target: {gi.get('target_class')}.{gi.get('target_name')}")
        print(f"        GT:    {repr(gt[:80])}")
        print(f"        Pass1: {repr(clean_bpe_text(p1)[:80])}")
        print(f"        Pass2: {repr(clean_bpe_text(p2)[:80])}")

    print(f"\nTotal Still Failed: {len(still_failed)}/400")

    # Save output artifacts
    pred_out = DATA_ROOT / "cadre_v2_tree_copysim_predictions.jsonl.gz"
    with gzip.open(pred_out, "wt", encoding="utf-8") as f:
        for r in all_predictions:
            f.write(json.dumps(r) + "\n")

    summary_out = DATA_ROOT / "cadre_v2_tree_copysim_metrics.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump({
            "pass1_baseline": stats0,
            "pass2_final": stats2,
            "gate_stats": gate_stats,
            "wins_count": len(wins),
            "losses_count": len(losses),
            "wins": [{"benchmark": w[0], "task_id": w[1]} for w in wins],
            "losses": [{"benchmark": l[0], "task_id": l[1]} for l in losses],
        }, f, indent=2)

    print(f"\nPredictions saved to {pred_out}")
    print(f"Metrics saved to {summary_out}")

if __name__ == "__main__":
    main()
