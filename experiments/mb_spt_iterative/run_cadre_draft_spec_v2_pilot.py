#!/usr/bin/env python3
"""
CADRE-DraftSpec v2: Target API Specification with Class Hierarchy & Literal Cleansing
Evaluated on the 400 Stratified Pilot Tasks (DeepSeek-Coder-1.3B)

Key Improvements over v1:
1. Class Hierarchy Preservation:
   - When extracting method signatures, always look up the enclosing class (`class ClassName:`).
   - Prevents model from confusing member methods (e.g. self.prob_output_layer.log_prob) with outer class methods (self.log_prob).
2. Literal Cleansing on Call-Sites:
   - Discards usage examples containing hardcoded string literals (e.g. "main.qss") to prevent negative transfer.
3. Redundancy Gating:
   - If the exact definition already exists in the top retrieved chunks, do not override with redundant stubs.
"""

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
from typing import Any

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
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torchaudio"])
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])

ensure_deps()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
try:
    from fuzzywuzzy import fuzz
    def cal_edit_sim(p: str, g: str) -> float:
        return float(fuzz.ratio(p.strip(), g.strip()))
except ImportError:
    def cal_edit_sim(p: str, g: str) -> float:
        return SequenceMatcher(None, p.strip(), g.strip()).ratio() * 100.0

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
RECEIVER_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\.\s*$")
STOP_WORDS = {
    "self", "this", "None", "True", "False", "def", "class", "if", "for", "while",
    "return", "import", "from", "public", "private", "protected", "static", "void",
    "int", "str", "float", "bool", "list", "dict", "set", "super", "try", "except",
    "print", "len", "range", "enumerate", "zip", "open", "close", "read", "write",
    "assert", "in", "is", "not", "and", "or", "new", "null", "case", "break", "continue"
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

def strip_method_affixes(name: str) -> str:
    s = re.sub(r"^(?:is_|get_|set_|has_|check_|do_)", "", name, flags=re.IGNORECASE)
    s = re.sub(r"^(?:is|get|set|has|check)", "", s)
    return s.lower()

# 5. Targeted Interface & Usage Spec Extractor v2
def extract_enriched_specification_v2(
    task: dict[str, Any],
    tr: dict[str, Any],
    pool_files: list[dict[str, Any]],
    raw_draft: str,
) -> str:
    lang = task["language"]
    left_ctx = task["left_context"]
    comment = "#" if lang == "python" else "//"
    
    last_line = left_ctx.splitlines()[-1] if left_ctx.splitlines() else ""
    clean_draft = clean_bpe_text(raw_draft)
    draft_lines = [l.strip() for l in clean_draft.splitlines() if l.strip()]
    first_draft_line = draft_lines[0] if draft_lines else ""
    combined_intent = last_line + " " + first_draft_line

    m_rec = RECEIVER_RE.search(last_line)
    receiver = m_rec.group(1) if m_rec else ""
    
    calls = [m.group(1).split(".")[-1] for m in CALL_RE.finditer(combined_intent)]
    
    candidate_tokens = []
    for c in calls:
        if c not in STOP_WORDS and len(c) >= 3:
            candidate_tokens.append(c)
    if receiver and receiver not in STOP_WORDS and len(receiver) >= 3:
        candidate_tokens.append(receiver)
        
    draft_words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", first_draft_line)
    for w in draft_words:
        if len(w) >= 4 and w not in STOP_WORDS and w not in candidate_tokens:
            candidate_tokens.append(w)

    if not candidate_tokens:
        return ""

    # Gating 1: If primary callee is already defined in top-3 chunks, skip injection!
    raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    top_chunks_text = " ".join((c.get("code") or c.get("text") or "") for c in raw_candidates[:3])
    primary_callee = candidate_tokens[0] if candidate_tokens else ""
    if primary_callee and (f"def {primary_callee}(" in top_chunks_text or f"def {primary_callee} (" in top_chunks_text):
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
                                
                            # Search backwards for enclosing class (Class Hierarchy Preservation)
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
                                "doc": doc[:120],
                            }
                elif s.startswith("class "):
                    m = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", s)
                    if m:
                        cls_name = m.group(1)
                        if receiver and receiver.lower() in cls_name.lower():
                            cls_methods = []
                            for k in range(idx + 1, min(len(lines), idx + 50)):
                                ml = lines[k].strip()
                                if ml.startswith("class "):
                                    break
                                if ml.startswith("def "):
                                    m_sig = ml.split(":")[0] + ": ..."
                                    cls_methods.append(f"    {m_sig}")
                                    if len(cls_methods) >= 6:
                                        break
                            if cls_methods and best_score < 1.2:
                                best_score = 1.2
                                best_candidate = {
                                    "file": fpath,
                                    "name": cls_name,
                                    "sig": f"class {cls_name}:\n" + "\n".join(cls_methods),
                                    "doc": "",
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
                            
                            # Search backwards for enclosing class / interface
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

    # 3. Find real in-repo usage with literal cleansing
    target_name = best_candidate["name"]
    usage_example = ""
    for pf in pool_files:
        p_text = pf.get("text", "")
        for line in p_text.splitlines():
            s = line.strip()
            if (f".{target_name}(" in s or f"{target_name}(" in s) and not s.startswith("def ") and not s.startswith("class ") and not s.startswith("public ") and not s.startswith("//") and not s.startswith("#"):
                if 10 <= len(s) <= 110:
                    # Literal Cleansing: if call-site contains string literals ("..."), skip it to prevent literal copying!
                    if not re.search(r'["\'][^"\']+["\']', s):
                        usage_example = s
                        break
        if usage_example:
            break

    # 4. Assemble distinctive attention banner
    banner = []
    banner.append(f"{comment} ==============================================================================")
    banner.append(f"{comment} <<< TARGET API SPECIFICATION (RESOLVED FROM DRAFT INTENT) >>>")
    banner.append(f"{comment} Source Module: {best_candidate['file']}")
    banner.append(f"{comment} Canonical Signature (L_stub):")
    for sig_l in best_candidate["sig"].splitlines():
        banner.append(f"{comment}     {sig_l}")
    if best_candidate.get("doc"):
        banner.append(f"{comment} Docstring Summary: \"\"\"{best_candidate['doc']}\"\"\"")
    if usage_example:
        banner.append(f"{comment} Verified In-Repo Usage Call-site: {usage_example}")
    banner.append(f"{comment} ==============================================================================\n\n")

    return "\n".join(banner)

# 6. Prompt Assembly
def build_prompt_with_optional_spec(
    task: dict[str, Any],
    tr: dict[str, Any],
    spec_block: str = "",
) -> list[int]:
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    comment_label = "#" if lang == "python" else "//"
    
    left_all_ids = tok.encode(left_context)
    path_ids = tok.encode(f"{comment_label} file path: {cur_path}\n")
    spec_ids = tok.encode(spec_block) if spec_block else []
    
    raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    chunks_by_file: dict[str, list[str]] = defaultdict(list)
    for ch in raw_candidates:
        fp = ch.get("file_path") or ch.get("path", "")
        c = ch.get("code") or ch.get("text", "")
        if fp and c.strip():
            chunks_by_file[fp].append(c)

    exemplar_ids: list[int] = []
    max_exemplar_tokens = max(0, 3200 - len(spec_ids))
    for fp, code_list in chunks_by_file.items():
        coalesced = "\n...\n".join(code_list)
        block = f"{comment_label} file path: {fp}\n{coalesced}\n\n"
        b_ids = tok.encode(block)
        if len(exemplar_ids) + len(b_ids) > max_exemplar_tokens:
            break
        exemplar_ids.extend(b_ids)

    cross_ids = exemplar_ids + spec_ids
    allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []
    return (cross_ids + path_ids + left_ids)[-8000:]

# 7. Official Metrics Evaluation
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

    # Pass 1: Baseline CADRE v3
    print("\n[Pass 1] Running Baseline CADRE v3 (Full Raw Chunks)...", flush=True)
    p0_ids = [build_prompt_with_optional_spec(t, tr, spec_block="") for t, tr in zip(tasks, task_trs)]
    out0 = llm.generate([{"prompt_token_ids": p} for p in p0_ids], sampling_params=sp)
    preds_pass1 = [o.outputs[0].text for o in out0]
    for i, p in enumerate(preds_pass1):
        all_predictions[i]["pass1_baseline"] = p
    stats0 = evaluate_records(all_predictions, "pass1_baseline")
    print("  ✓ Pass 1 Baseline Macro EM:", stats0["macro_mean"]["em"], "% | Macro ES:", stats0["macro_mean"]["es"], "%", flush=True)

    # Pass 2: CADRE-DraftSpec v2 (with Class Hierarchy & Literal Cleansing)
    print("\n[Pass 2] Resolving Enriched Specifications v2...", flush=True)
    p2_ids = []
    pass2_indices = []
    enriched_specs = []
    
    for i, (t, tr, pf, raw_draft) in enumerate(zip(tasks, task_trs, task_pool_files, preds_pass1)):
        spec_block = extract_enriched_specification_v2(t, tr, pf, raw_draft)
        enriched_specs.append(spec_block)
        if spec_block:
            pass2_indices.append(i)
            p2_ids.append(build_prompt_with_optional_spec(t, tr, spec_block=spec_block))
        else:
            p2_ids.append(p0_ids[i])
            
    print(f"  -> Target API Specification v2 resolved for {len(pass2_indices)}/400 tasks ({len(pass2_indices)/400*100:.1f}%).", flush=True)
    out2 = llm.generate([{"prompt_token_ids": p} for p in p2_ids], sampling_params=sp)
    preds_pass2 = [o.outputs[0].text for o in out2]
    for i, p in enumerate(preds_pass2):
        all_predictions[i]["pass2_draft_spec_v2"] = p
        all_predictions[i]["spec_block_v2"] = enriched_specs[i]
        
    stats2 = evaluate_records(all_predictions, "pass2_draft_spec_v2")
    print("  ✓ Pass 2 (CADRE-DraftSpec v2) Macro EM:", stats2["macro_mean"]["em"], "% | Macro ES:", stats2["macro_mean"]["es"], "%", flush=True)

    arms = [
        ("Pass 1: Baseline CADRE v3 (Full Raw Context)", stats0),
        ("Pass 2: CADRE-DraftSpec v2 (Class Hierarchy + Clean Usage)", stats2),
    ]

    print("\n" + "=" * 125, flush=True)
    print(f"{'ARM':<60} | {'cceval_py':<10} | {'cceval_ja':<10} | {'repo_api':<10} | {'repo_line':<10} | {'MACRO EM':<10} | {'MACRO ES':<10}", flush=True)
    print("=" * 125, flush=True)
    for name, st in arms:
        print(f"{name:<60} | {st['cceval_python']['em']:>8.2f}% | {st['cceval_java']['em']:>8.2f}% | {st['repoeval_api']['em']:>8.2f}% | {st['repoeval_line']['em']:>8.2f}% | {st['macro_mean']['em']:>8.2f}% | {st['macro_mean']['es']:>8.2f}%", flush=True)
    print("=" * 125, flush=True)

    # Save outputs
    out_metrics = OUTPUT_DIR / "cadre_draft_spec_v2_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump({name: st for name, st in arms}, f, indent=2)
    print(f"✓ Saved full metrics to {out_metrics}", flush=True)

    out_preds = OUTPUT_DIR / "cadre_draft_spec_v2_predictions.jsonl.gz"
    with gzip.open(out_preds, "wt") as f:
        for p in all_predictions:
            f.write(json.dumps(p) + "\n")
    print(f"✓ Saved all predictions to {out_preds}", flush=True)

if __name__ == "__main__":
    main()
