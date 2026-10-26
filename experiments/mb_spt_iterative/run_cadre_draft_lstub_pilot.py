#!/usr/bin/env python3
"""
CADRE-DraftStub: 2-Pass Draft-Guided Retrieval with Targeted L_stub Interface Refinement
Evaluated on the 400 Stratified Pilot Tasks (DeepSeek-Coder-1.3B)

Mechanism:
- Pass 1: Run Baseline CADRE v3 (Full 3,200 token raw chunks).
- Draft Intention Detection: Extract function / API calls from draft z^(0).
- Targeted Repo Query: If callee was missing from retrieved chunks, scan repo pool for its module.
- L_stub Canonical Projection: Format the discovered class/module into a clean PEP 484 / Java interface stub.
- Pass 2 Refinement: Re-decode with targeted L_stub injected, keeping all base context intact.
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

# 4. Syntactic Extraction Helpers
CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
STOP_WORDS = {
    "self", "this", "None", "True", "False", "def", "class", "if", "for", "while",
    "return", "import", "from", "public", "private", "protected", "static", "void",
    "int", "str", "float", "bool", "list", "dict", "set", "super", "try", "except"
}

def clean_bpe_text(s: str) -> str:
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

# 5. Targeted L_stub Extraction
def extract_targeted_lstub_for_callee(pool_files: list[dict[str, Any]], callee: str, lang: str) -> str:
    """
    Scans pool_files for the class or module defining `callee`,
    and extracts a clean Canonical Interface Stub (L_stub).
    """
    comment = "#" if lang == "python" else "//"
    target_pattern = re.compile(rf"\b(?:def|class|interface|public|protected|private|void)\s+.*?\b{re.escape(callee)}\b")
    
    matching_file = None
    for pf in pool_files:
        text = pf.get("text", "")
        if target_pattern.search(text):
            matching_file = pf
            break
            
    if not matching_file:
        return ""
        
    fpath = matching_file.get("path", "module")
    text = matching_file.get("text", "")
    lines = text.splitlines()
    stubs = []
    
    if lang == "python":
        i = 0
        in_target_class = False
        class_name = ""
        while i < len(lines):
            l = lines[i]
            s = l.strip()
            if s.startswith("class ") and ":" in s:
                class_name = s.split(":")[0].strip()
                # Check if this class contains the callee
                class_body = "\n".join(lines[i:i+60])
                if callee in class_body:
                    in_target_class = True
                    stubs.append(f"{class_name}:")
                else:
                    in_target_class = False
            elif in_target_class and (s.startswith("def ") or s.startswith("async def ")):
                def_parts = [s]
                while ":" not in def_parts[-1] and i + 1 < len(lines):
                    i += 1
                    def_parts.append(lines[i].strip())
                sig = " ".join(" ".join(def_parts).split())
                if ":" in sig:
                    sig = sig.split(":")[0] + ": ..."
                stubs.append(f"    {sig}")
            elif not in_target_class and (s.startswith("def ") or s.startswith("async def ")) and callee in s:
                def_parts = [s]
                while ":" not in def_parts[-1] and i + 1 < len(lines):
                    i += 1
                    def_parts.append(lines[i].strip())
                sig = " ".join(" ".join(def_parts).split())
                if ":" in sig:
                    sig = sig.split(":")[0] + ": ..."
                stubs.append(sig)
            i += 1
            if len(stubs) >= 8:
                break
    else:  # Java
        for l in lines:
            s = l.strip()
            if (s.startswith("public ") or s.startswith("protected ") or s.startswith("private ")) and "(" in s and ")" in s and callee in s:
                sig = s.split("{")[0].strip() + ";"
                stubs.append(f"    {sig}")
            elif (s.startswith("public class ") or s.startswith("class ") or s.startswith("public interface ")) and "{" in s and callee in s:
                stubs.append(s.split("{")[0].strip() + " { ... }")
            if len(stubs) >= 8:
                break

    if not stubs:
        return ""
        
    header = f"{comment} [CSSI Interface Stub (L_stub): {fpath}]\n"
    return header + "\n".join(stubs[:8]) + "\n\n"

# 6. Prompt Assembly
def build_prompt_with_optional_lstub(
    task: dict[str, Any],
    tr: dict[str, Any],
    pool_files: list[dict[str, Any]],
    lstub_block: str = "",
) -> list[int]:
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    comment_label = "#" if lang == "python" else "//"
    
    left_all_ids = tok.encode(left_context)
    path_ids = tok.encode(f"{comment_label} file path: {cur_path}\n")
    lstub_ids = tok.encode(lstub_block) if lstub_block else []
    
    raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    chunks_by_file: dict[str, list[str]] = defaultdict(list)
    for ch in raw_candidates:
        fp = ch.get("file_path") or ch.get("path", "")
        c = ch.get("text") or ch.get("code", "")
        if fp and c.strip():
            chunks_by_file[fp].append(c)

    exemplar_ids: list[int] = []
    max_exemplar_tokens = max(0, 3200 - len(lstub_ids))
    for fp, code_list in chunks_by_file.items():
        coalesced = "\n...\n".join(code_list)
        block = f"{comment_label} file path: {fp}\n{coalesced}\n\n"
        b_ids = tok.encode(block)
        if len(exemplar_ids) + len(b_ids) > max_exemplar_tokens:
            break
        exemplar_ids.extend(b_ids)

    # Cross context: Exemplars (raw chunks) + Targeted L_stub Interface
    cross_ids = exemplar_ids + lstub_ids
    allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []
    return (cross_ids + path_ids + left_ids)[-8000:]

# 7. Evaluation Logic
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
    # Pass 1: Baseline CADRE v3
    # -------------------------------------------------------------
    print("\n[Pass 1] Running Baseline CADRE v3 (Full Raw Chunks)...", flush=True)
    p0_ids = [build_prompt_with_optional_lstub(t, tr, pf, lstub_block="") for t, tr, pf in zip(tasks, task_trs, task_pool_files)]
    out0 = llm.generate([{"prompt_token_ids": p} for p in p0_ids], sampling_params=sp)
    preds_pass1 = [o.outputs[0].text for o in out0]
    for i, p in enumerate(preds_pass1):
        all_predictions[i]["pass1_baseline"] = p
    stats0 = evaluate_records(all_predictions, "pass1_baseline")
    print("  ✓ Pass 1 Baseline Macro EM:", stats0["macro_mean"]["em"], "% | Macro ES:", stats0["macro_mean"]["es"], "%", flush=True)

    # -------------------------------------------------------------
    # Pass 2: Targeted L_stub Refinement guided by Pass 1 Draft
    # -------------------------------------------------------------
    print("\n[Pass 2] Analyzing Draft Intentions & Injecting Targeted L_stub...", flush=True)
    p2_ids = []
    pass2_indices = []
    for i, (t, tr, pf, raw_draft) in enumerate(zip(tasks, task_trs, task_pool_files, preds_pass1)):
        lang = t["language"]
        left_ctx = t["left_context"]
        
        # Check raw retrieved chunks text
        raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
        chunk_text = " ".join((c.get("code") or c.get("text") or "") for c in raw_candidates)
        
        clean_draft = clean_bpe_text(raw_draft)
        draft_lines = [l.strip() for l in clean_draft.splitlines() if l.strip()]
        first_line = draft_lines[0] if draft_lines else ""
        
        draft_callees = [m.group(1).split(".")[-1] for m in CALL_RE.finditer(first_line)]
        
        target_lstub = ""
        for c in draft_callees:
            if c not in STOP_WORDS and len(c) > 2:
                # Gating: Only if missing from context!
                if c not in left_ctx and c not in chunk_text:
                    stub = extract_targeted_lstub_for_callee(pf, c, lang)
                    if stub:
                        target_lstub = stub
                        break
                        
        if target_lstub:
            pass2_indices.append(i)
            p2_ids.append(build_prompt_with_optional_lstub(t, tr, pf, lstub_block=target_lstub))
        else:
            p2_ids.append(p0_ids[i])
            
    print(f"  -> Targeted L_stub triggered for {len(pass2_indices)}/400 tasks.", flush=True)
    out2 = llm.generate([{"prompt_token_ids": p} for p in p2_ids], sampling_params=sp)
    preds_pass2 = [o.outputs[0].text for o in out2]
    for i, p in enumerate(preds_pass2):
        all_predictions[i]["pass2_lstub"] = p
    stats2 = evaluate_records(all_predictions, "pass2_lstub")
    print("  ✓ Pass 2 (CADRE-DraftStub) Macro EM:", stats2["macro_mean"]["em"], "% | Macro ES:", stats2["macro_mean"]["es"], "%", flush=True)

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    arms = [
        ("Pass 1: Baseline CADRE v3 (Full Context)", stats0),
        ("Pass 2: CADRE-DraftStub (Targeted L_stub Refined)", stats2),
    ]

    print("\n" + "=" * 125, flush=True)
    print(f"{'ARM':<60} | {'cceval_py':<10} | {'cceval_ja':<10} | {'repo_api':<10} | {'repo_line':<10} | {'MACRO EM':<10} | {'MACRO ES':<10}", flush=True)
    print("=" * 125, flush=True)
    for name, st in arms:
        print(f"{name:<60} | {st['cceval_python']['em']:>8.2f}% | {st['cceval_java']['em']:>8.2f}% | {st['repoeval_api']['em']:>8.2f}% | {st['repoeval_line']['em']:>8.2f}% | {st['macro_mean']['em']:>8.2f}% | {st['macro_mean']['es']:>8.2f}%", flush=True)
    print("=" * 125, flush=True)

    # Save outputs
    out_metrics = OUTPUT_DIR / "cadre_draft_lstub_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump({name: st for name, st in arms}, f, indent=2)
    print(f"✓ Saved full metrics to {out_metrics}", flush=True)

    out_preds = OUTPUT_DIR / "cadre_draft_lstub_predictions.jsonl.gz"
    with gzip.open(out_preds, "wt") as f:
        for p in all_predictions:
            f.write(json.dumps(p) + "\n")
    print(f"✓ Saved all predictions to {out_preds}", flush=True)

if __name__ == "__main__":
    main()
