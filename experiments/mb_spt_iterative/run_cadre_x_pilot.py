#!/usr/bin/env python3
"""
CADRE-X Unified Empirical Validation on 400 Stratified Pilot Tasks (DeepSeek-Coder-1.3B)

Corrected and refined from empirical findings:
1. Elastic Token Budgeting:
   - Full 3,200 cross-file budget for RepoEval (prevents the 9% EM drop from artificial throttling).
   - Cursor-preserving left-context reserve (>=1024 tokens) for CCEval.
2. Canonical Interface Stub (L_stub) & Usage Call-Sites:
   - Compresses raw loop/try-except bodies into clean PEP 484 / Java interface signatures.
   - Provides 1-2 lines of realistic call-site examples.
3. Selective Gated 2-Pass IDGR:
   - Pass 1 generates greedy draft z^(0).
   - Extracts callee from draft.
   - GATING: Only re-retrieve if callee is NOT in context AND exact definition exists in repo pool.
   - Prevents pulling hallucinated methods or harmful overloads.
"""

import gzip
import json
import os
import re
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any
import difflib

# 1. Environment & Dependencies
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
try:
    from fuzzywuzzy import fuzz
    def cal_edit_sim(p: str, g: str) -> float:
        return float(fuzz.ratio(p.strip(), g.strip()))
except ImportError:
    def cal_edit_sim(p: str, g: str) -> float:
        return difflib.SequenceMatcher(None, p.strip(), g.strip()).ratio() * 100.0

# 2. Paths & Configurations
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

# 3. Load datasets
all_tasks = [json.loads(l) for l in gzip.open(DATA_ROOT / "tasks.jsonl.gz", "rt")]
traces = {(r["benchmark"], r["task_id"]): r for r in map(json.loads, gzip.open(DATA_ROOT / "retrieval_trace.jsonl.gz", "rt"))}
files = {r["file_id"]: {"path": r["path"], "text": r["text"]} for r in map(json.loads, gzip.open(DATA_ROOT / "files.jsonl.gz", "rt"))}
pools = {}
if (DATA_ROOT / "pools.jsonl.gz").exists():
    pools = {r["pool_id"]: r["file_ids"] for r in map(json.loads, gzip.open(DATA_ROOT / "pools.jsonl.gz", "rt"))}

parsers = {}
so_dir = Path("/content/build") if Path("/content/build").exists() else Path("RepoClone/AlignCoder/utils/build")
for lang in ("python", "java"):
    so_p = so_dir / f"{lang}-lang-parser.so"
    if so_p.exists():
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
print(f"Loaded {len(tasks)} pilot tasks matching baseline pool exactly.", flush=True)

# 4. Regex & Syntactic Extraction Helpers
CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
DOT_MEMBER_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)")
DOT_END_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*$")
STOP_WORDS = {
    "self", "this", "None", "True", "False", "def", "class", "if", "for", "while",
    "return", "import", "from", "public", "private", "protected", "static", "void",
    "int", "str", "float", "bool", "list", "dict", "set", "super", "try", "except"
}

def clean_bpe_text(s: str) -> str:
    """Crucial decode-time normalizer for GPT-2 / DeepSeek BPE."""
    return s.replace("Ġ", " ").replace("Ċ", "\n")

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
    for m in DOT_MEMBER_RE.finditer(last_line):
        rec = m.group(1)
        if len(rec) > 1 and rec not in STOP_WORDS:
            receivers.append(rec)
            
    m_end = DOT_END_RE.search(last_line)
    if m_end:
        rec = m_end.group(1)
        if len(rec) > 1 and rec not in STOP_WORDS:
            receivers.append(rec)
            
    return list(dict.fromkeys(callees)), list(dict.fromkeys(receivers))

# 5. Distillation & Canonical Stub (L_stub) Builder
def distill_chunk_to_stubs(text: str, lang: str, max_stubs: int = 8) -> list[str]:
    """Extract Canonical Interface Stubs (L_stub). Drops loop/try bodies."""
    stubs = []
    lines = text.splitlines()
    if lang == "python":
        i = 0
        while i < len(lines):
            l = lines[i]
            s = l.strip()
            if s.startswith("class ") and ":" in s:
                stubs.append(s.split(":")[0] + ": ...")
            elif s.startswith("def ") or s.startswith("async def "):
                def_parts = [s]
                while ":" not in def_parts[-1] and i + 1 < len(lines):
                    i += 1
                    def_parts.append(lines[i].strip())
                sig = " ".join(" ".join(def_parts).split())
                if ":" in sig:
                    sig = sig.split(":")[0] + ": ..."
                stubs.append(sig)
            i += 1
            if len(stubs) >= max_stubs:
                break
    else:  # Java
        for l in lines:
            s = l.strip()
            if (s.startswith("public ") or s.startswith("protected ") or s.startswith("private ")) and "(" in s and ")" in s:
                sig = s.split("{")[0].strip() + ";"
                stubs.append(sig)
            elif (s.startswith("public class ") or s.startswith("class ") or s.startswith("public interface ")) and "{" in s:
                stubs.append(s.split("{")[0].strip() + " { ... }")
            if len(stubs) >= max_stubs:
                break
    return stubs[:max_stubs]

def build_rfsc_contract_block(
    callees: list[str],
    receivers: list[str],
    sources: list[tuple[str, str]],
    lang: str,
    max_lines: int = 5
) -> str:
    comment = "#" if lang == "python" else "//"
    lines_out = []
    seen = set()
    
    # 1. Def signatures of callees
    for callee in callees[:2]:
        def_pat = re.compile(rf"^\s*(?:async\s+)?(?:def|public|protected|private|static)?\s*.*?\b{re.escape(callee)}\s*\([^)]*\)", re.MULTILINE)
        for path, text in sources:
            for m in def_pat.finditer(text):
                sig = " ".join(m.group(0).split())
                if sig and sig not in seen:
                    seen.add(sig)
                    lines_out.append(f"{comment} [Stub] {sig} -> ...")
                    break
            if len(lines_out) >= max_lines:
                break
                
    # 2. Usage call-sites
    target_names = set(callees[:2] + receivers[:2])
    for path, text in sources:
        for l in text.splitlines():
            s = l.strip()
            if not s or s.startswith(("#", "//", "def ", "class ", "import ", "from ", "package ", "public ", "private ")):
                continue
            for name in target_names:
                if (name + "(" in s or name + "." in s) and 10 < len(s) < 120 and s not in seen:
                    seen.add(s)
                    lines_out.append(f"{comment} [Usage] {s}")
                    break
        if len(lines_out) >= max_lines:
            break

    if not lines_out:
        return ""
    header = f"{comment} [CSSI Interface Contract: {lang.title()}]\n"
    return header + "\n".join(lines_out[:max_lines]) + "\n"

# 6. Prompt Assembly Strategies
def build_prompt_arm0_baseline(task: dict[str, Any], tr: dict[str, Any], pool_files: list[dict[str, Any]]) -> list[int]:
    """Arm 0: Baseline CADRE v3 (Fixed 3200 tok cross-file, raw chunks)."""
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    comment_label = "#" if lang == "python" else "//"
    
    left_all_ids = tok.encode(left_context)
    path_ids = tok.encode(f"{comment_label} file path: {cur_path}\n")
    
    raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    chunks_by_file: dict[str, list[str]] = defaultdict(list)
    for ch in raw_candidates:
        fp = ch.get("file_path") or ch.get("path", "")
        c = ch.get("text") or ch.get("code", "")
        if fp and c.strip():
            chunks_by_file[fp].append(c)

    exemplar_ids: list[int] = []
    max_exemplar_tokens = 3200
    for fp, code_list in chunks_by_file.items():
        coalesced = "\n...\n".join(code_list)
        block = f"{comment_label} file path: {fp}\n{coalesced}\n\n"
        b_ids = tok.encode(block)
        if len(exemplar_ids) + len(b_ids) > max_exemplar_tokens:
            break
        exemplar_ids.extend(b_ids)
        
    allowed_left = max(0, 8000 - len(exemplar_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []
    return (exemplar_ids + path_ids + left_ids)[-8000:]

def build_prompt_cadre_x(
    task: dict[str, Any],
    tr: dict[str, Any],
    pool_files: list[dict[str, Any]],
    extra_callee: str | None = None,
    use_distilled: bool = True,
) -> list[int]:
    """
    CADRE-X Prompt Builder:
    - Elastic Budgeting: Full 3,200 cross-file budget (RepoEval never choked!).
    - Guarantees minimum 1,024 tokens tail for left-context.
    - Uses L_stub distillation for background chunks + RFSC contract.
    """
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    comment_label = "#" if lang == "python" else "//"
    
    left_all_ids = tok.encode(left_context)
    path_ids = tok.encode(f"{comment_label} file path: {cur_path}\n")
    
    # 8k context window support
    total_window = 8000
    min_left_reserve = min(len(left_all_ids), 1024)
    cross_file_budget = min(3200, total_window - min_left_reserve - len(path_ids) - 64)
    
    raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    
    # 1. Distill or keep chunks
    chunks_by_file: dict[str, list[str]] = defaultdict(list)
    for ch in raw_candidates:
        fp = ch.get("file_path") or ch.get("path", "")
        c = ch.get("text") or ch.get("code", "")
        if fp and c.strip():
            chunks_by_file[fp].append(c)

    exemplar_ids: list[int] = []
    for fp, code_list in chunks_by_file.items():
        if use_distilled:
            stubs = []
            for c in code_list:
                stubs.extend(distill_chunk_to_stubs(c, lang, max_stubs=5))
            if stubs:
                block = f"{comment_label} [Distilled: {fp}]\n" + "\n".join(stubs[:8]) + "\n\n"
            else:
                coalesced = "\n...\n".join(code_list[:2])
                block = f"{comment_label} file path: {fp}\n{coalesced[:600]}\n\n"
        else:
            coalesced = "\n...\n".join(code_list)
            block = f"{comment_label} file path: {fp}\n{coalesced}\n\n"
            
        b_ids = tok.encode(block)
        if len(exemplar_ids) + len(b_ids) > cross_file_budget - 160:
            break
        exemplar_ids.extend(b_ids)

    # 2. Extract and format RFSC Structural Contract
    callees, receivers = extract_cursor_triggers(left_context)
    if extra_callee and extra_callee not in callees:
        callees.insert(0, extra_callee)
        
    sources = [(pf["path"], pf["text"]) for pf in pool_files] + [(ch.get("file_path", "chunk"), ch.get("text") or ch.get("code", "")) for ch in raw_candidates]
    contract_text = build_rfsc_contract_block(callees, receivers, sources, lang, max_lines=4)
    contract_ids = tok.encode(contract_text)[:160] if contract_text else []
    
    cross_ids = exemplar_ids + contract_ids
    available_left = max(0, total_window - len(cross_ids) - len(path_ids) - 64)
    left_ids = left_all_ids[-available_left:] if available_left else []
    
    return (cross_ids + path_ids + left_ids)[-8000:]

# 7. Evaluation & Metrics
def evaluate_records(records: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    by_bench = defaultdict(list)
    for r in records:
        bm = r["benchmark"]
        gt = r["groundtruth"]
        raw = clean_bpe_text(r[pred_key])
        
        gt_no_comment = remove_comments(gt)
        raw_no_comment = remove_comments(raw)
        
        gt_lines = [l.strip() for l in gt_no_comment.splitlines() if l.strip()]
        n_target = max(1, len(gt_lines))
        
        pred_lines = [l.strip() for l in raw_no_comment.splitlines() if l.strip()][:n_target]
        
        gt_flat = "".join("".join(gt_lines).split())
        pred_flat = "".join("".join(pred_lines).split())
        
        prefix_match = int(pred_flat[:len(gt_flat)] == gt_flat and len(gt_flat) > 0)
        
        if bm == "cceval_java":
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

# 8. Main Pipeline
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

    # Prepare pool_files for all tasks
    task_pool_files = []
    for t in tasks:
        if "file_ids_by_path" in t:
            pf = [files[f] for f in t["file_ids_by_path"].values() if f in files]
        else:
            pf = [files[f] for f in pools.get(t.get("pool_id"), []) if f in files]
        task_pool_files.append(pf)
        
    task_trs = [traces.get((t["benchmark"], t["task_id"]), {}) for t in tasks]

    all_predictions = []
    for t in tasks:
        all_predictions.append({
            "task_id": t["task_id"],
            "benchmark": t["benchmark"],
            "language": t["language"],
            "groundtruth": t["groundtruth"],
        })

    # -------------------------------------------------------------
    # Arm 0: Baseline CADRE v3 (Reference)
    # -------------------------------------------------------------
    print("\n[Arm 0] Running Baseline CADRE v3 (Full 3200 tok budget)...", flush=True)
    p0_ids = [build_prompt_arm0_baseline(t, tr, pf) for t, tr, pf in zip(tasks, task_trs, task_pool_files)]
    out0 = llm.generate([{"prompt_token_ids": p} for p in p0_ids], sampling_params=sp)
    preds_arm0 = [o.outputs[0].text for o in out0]
    for i, p in enumerate(preds_arm0):
        all_predictions[i]["arm0"] = p
    stats0 = evaluate_records(all_predictions, "arm0")
    print("  ✓ Arm 0 Macro EM:", stats0["macro_mean"]["em"], "% | Macro ES:", stats0["macro_mean"]["es"], "%", flush=True)

    # -------------------------------------------------------------
    # Arm 1: Elastic Budget + Distilled L_stub + RFSC
    # -------------------------------------------------------------
    print("\n[Arm 1] Running Elastic Budget + Distilled L_stub + RFSC...", flush=True)
    p1_ids = [build_prompt_cadre_x(t, tr, pf, use_distilled=True) for t, tr, pf in zip(tasks, task_trs, task_pool_files)]
    out1 = llm.generate([{"prompt_token_ids": p} for p in p1_ids], sampling_params=sp)
    preds_arm1 = [o.outputs[0].text for o in out1]
    for i, p in enumerate(preds_arm1):
        all_predictions[i]["arm1"] = p
    stats1 = evaluate_records(all_predictions, "arm1")
    print("  ✓ Arm 1 Macro EM:", stats1["macro_mean"]["em"], "% | Macro ES:", stats1["macro_mean"]["es"], "%", flush=True)

    # -------------------------------------------------------------
    # Arm 2: Selective Gated 2-Pass IDGR (on Baseline)
    # -------------------------------------------------------------
    print("\n[Arm 2] Running Selective Gated 2-Pass IDGR (on Baseline)...", flush=True)
    p2_ids = []
    gated_tasks = 0
    for i, (t, tr, pf, raw_draft) in enumerate(zip(tasks, task_trs, task_pool_files, preds_arm0)):
        left_ctx = t["left_context"]
        clean_draft = clean_bpe_text(raw_draft)
        draft_lines = [l.strip() for l in clean_draft.splitlines() if l.strip()]
        first_line = draft_lines[0] if draft_lines else ""
        
        draft_callees = [m.group(1).split(".")[-1] for m in CALL_RE.finditer(first_line)]
        candidate_callee = None
        for c in draft_callees:
            if c not in STOP_WORDS and len(c) > 2:
                # GATING 1: Is this callee already in context?
                if c in left_ctx:
                    continue
                # GATING 2: Does repo pool have this exact definition?
                target_def_pat = re.compile(rf"\b(?:def|class|public|protected|private|void)\s+{re.escape(c)}\b")
                found_in_pool = any(target_def_pat.search(p["text"]) for p in pf)
                if found_in_pool:
                    candidate_callee = c
                    break
                    
        if candidate_callee:
            gated_tasks += 1
            p2_ids.append(build_prompt_cadre_x(t, tr, pf, extra_callee=candidate_callee, use_distilled=False))
        else:
            p2_ids.append(p0_ids[i])
            
    print(f"  -> Gating triggered for {gated_tasks}/400 tasks.", flush=True)
    out2 = llm.generate([{"prompt_token_ids": p} for p in p2_ids], sampling_params=sp)
    preds_arm2 = [o.outputs[0].text for o in out2]
    for i, p in enumerate(preds_arm2):
        all_predictions[i]["arm2"] = p
    stats2 = evaluate_records(all_predictions, "arm2")
    print("  ✓ Arm 2 Macro EM:", stats2["macro_mean"]["em"], "% | Macro ES:", stats2["macro_mean"]["es"], "%", flush=True)

    # -------------------------------------------------------------
    # Arm 3: Full CADRE-X (Elastic + Distilled L_stub + Gated IDGR)
    # -------------------------------------------------------------
    print("\n[Arm 3] Running Full CADRE-X (Elastic + Distilled + Gated IDGR)...", flush=True)
    p3_ids = []
    gated_arm3 = 0
    for i, (t, tr, pf, raw_draft) in enumerate(zip(tasks, task_trs, task_pool_files, preds_arm1)):
        left_ctx = t["left_context"]
        clean_draft = clean_bpe_text(raw_draft)
        draft_lines = [l.strip() for l in clean_draft.splitlines() if l.strip()]
        first_line = draft_lines[0] if draft_lines else ""
        
        draft_callees = [m.group(1).split(".")[-1] for m in CALL_RE.finditer(first_line)]
        candidate_callee = None
        for c in draft_callees:
            if c not in STOP_WORDS and len(c) > 2:
                if c in left_ctx:
                    continue
                target_def_pat = re.compile(rf"\b(?:def|class|public|protected|private|void)\s+{re.escape(c)}\b")
                found_in_pool = any(target_def_pat.search(p["text"]) for p in pf)
                if found_in_pool:
                    candidate_callee = c
                    break
                    
        if candidate_callee:
            gated_arm3 += 1
            p3_ids.append(build_prompt_cadre_x(t, tr, pf, extra_callee=candidate_callee, use_distilled=True))
        else:
            p3_ids.append(p1_ids[i])
            
    print(f"  -> Gating triggered for {gated_arm3}/400 tasks in Arm 3.", flush=True)
    out3 = llm.generate([{"prompt_token_ids": p} for p in p3_ids], sampling_params=sp)
    preds_arm3 = [o.outputs[0].text for o in out3]
    for i, p in enumerate(preds_arm3):
        all_predictions[i]["arm3"] = p
    stats3 = evaluate_records(all_predictions, "arm3")
    print("  ✓ Arm 3 Macro EM:", stats3["macro_mean"]["em"], "% | Macro ES:", stats3["macro_mean"]["es"], "%", flush=True)

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    arms = [
        ("Arm 0: Baseline CADRE v3 (Reference)", stats0),
        ("Arm 1: CADRE-X (Elastic + Distilled L_stub)", stats1),
        ("Arm 2: CADRE Baseline + Gated IDGR", stats2),
        ("Arm 3: Full CADRE-X (Elastic + L_stub + Gated IDGR)", stats3),
    ]

    print("\n" + "=" * 125, flush=True)
    print(f"{'ARM':<55} | {'cceval_py':<10} | {'cceval_ja':<10} | {'repo_api':<10} | {'repo_line':<10} | {'MACRO EM':<10} | {'MACRO ES':<10}", flush=True)
    print("=" * 125, flush=True)
    for name, st in arms:
        print(f"{name:<55} | {st['cceval_python']['em']:>8.2f}% | {st['cceval_java']['em']:>8.2f}% | {st['repoeval_api']['em']:>8.2f}% | {st['repoeval_line']['em']:>8.2f}% | {st['macro_mean']['em']:>8.2f}% | {st['macro_mean']['es']:>8.2f}%", flush=True)
    print("=" * 125, flush=True)

    # Save metrics and predictions
    out_metrics = OUTPUT_DIR / "cadre_x_pilot_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump({name: st for name, st in arms}, f, indent=2)
    print(f"✓ Saved full metrics to {out_metrics}", flush=True)

    out_preds = OUTPUT_DIR / "cadre_x_pilot_predictions.jsonl.gz"
    with gzip.open(out_preds, "wt") as f:
        for p in all_predictions:
            f.write(json.dumps(p) + "\n")
    print(f"✓ Saved all predictions to {out_preds}", flush=True)

if __name__ == "__main__":
    main()
