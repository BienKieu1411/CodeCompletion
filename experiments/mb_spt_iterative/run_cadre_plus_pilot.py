#!/usr/bin/env python3
"""
CADRE-Plus Experimental Run on 400 Pilot Tasks (DeepSeek-Coder-1.3B)

Architecture of CADRE-Plus:
1. Full Context Retention:
   - Preserves complete raw chunks (up to 3,200 tokens) with full implementation bodies.
   - Sibling function implementations are retained intact for pattern copying.
2. Call-Site Usage Injection (RFSC v2):
   - Extracts real usage lines (# [CSSI Usage]: res = obj.method(...)) for receiver and callee triggers.
   - Injected as compact contract lines without displacing implementation chunks.
3. Targeted 2-Pass IDGR:
   - For Python CCEval: draft-guided retrieval pulls missing callee modules into Top-1 rank.
   - For Java & RepoEval: single-pass with full context to avoid overload pull.
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

# 4. Syntactic Trigger & Call-site Extraction
CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*\(")
DOT_MEMBER_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)")
DOT_END_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*$")
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

def extract_callsite_usages(
    callees: list[str],
    receivers: list[str],
    sources: list[tuple[str, str]],
    lang: str,
    max_lines: int = 3
) -> str:
    comment = "#" if lang == "python" else "//"
    target_names = set(callees[:2] + receivers[:2])
    if not target_names:
        return ""
        
    usages = []
    seen = set()
    for path, text in sources:
        for l in text.splitlines():
            s = l.strip()
            if not s or s.startswith(("#", "//", "def ", "class ", "import ", "from ", "package ", "public ", "private ")):
                continue
            for name in target_names:
                if (name + "(" in s or name + "." in s) and 10 < len(s) < 120 and s not in seen:
                    seen.add(s)
                    usages.append(f"{comment} [CSSI Usage]: {s}")
                    break
            if len(usages) >= max_lines:
                break
        if len(usages) >= max_lines:
            break
            
    if not usages:
        return ""
    return "\n".join(usages) + "\n"

# 5. Prompt Construction
def build_prompt_raw_chunks(
    task: dict[str, Any],
    tr: dict[str, Any],
    pool_files: list[dict[str, Any]],
    use_callsite: bool = False,
    extra_callee: str | None = None,
) -> list[int]:
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    comment_label = "#" if lang == "python" else "//"
    
    left_all_ids = tok.encode(left_context)
    path_ids = tok.encode(f"{comment_label} file path: {cur_path}\n")
    
    raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    
    ordered_chunks = list(raw_candidates)
    if extra_callee:
        promoted = []
        rest = []
        for ch in ordered_chunks:
            c = ch.get("text") or ch.get("code", "")
            if extra_callee in c:
                promoted.append(ch)
            else:
                rest.append(ch)
        if not promoted:
            for pf in pool_files:
                if extra_callee in pf["text"]:
                    promoted.append({"file_path": pf["path"], "text": pf["text"][:1200]})
                    break
        ordered_chunks = promoted + rest

    chunks_by_file: dict[str, list[str]] = defaultdict(list)
    for ch in ordered_chunks:
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

    contract_ids: list[int] = []
    if use_callsite:
        callees, receivers = extract_cursor_triggers(left_context)
        if extra_callee and extra_callee not in callees:
            callees.insert(0, extra_callee)
        sources = [(pf["path"], pf["text"]) for pf in pool_files] + [(ch.get("file_path", "chunk"), ch.get("text") or ch.get("code", "")) for ch in raw_candidates]
        callsite_text = extract_callsite_usages(callees, receivers, sources, lang, max_lines=3)
        if callsite_text:
            contract_ids = tok.encode(callsite_text)[:120]

    cross_ids = exemplar_ids + contract_ids
    allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []
    return (cross_ids + path_ids + left_ids)[-8000:]

# 6. Evaluation Logic
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

# 7. Main Pipeline
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
    # Arm 0: Baseline CADRE v3 (Reference Baseline)
    # -------------------------------------------------------------
    print("\n[Arm 0] Running Baseline CADRE v3...", flush=True)
    p0_ids = [build_prompt_raw_chunks(t, tr, pf, use_callsite=False) for t, tr, pf in zip(tasks, task_trs, task_pool_files)]
    out0 = llm.generate([{"prompt_token_ids": p} for p in p0_ids], sampling_params=sp)
    preds_arm0 = [o.outputs[0].text for o in out0]
    for i, p in enumerate(preds_arm0):
        all_predictions[i]["arm0"] = p
    stats0 = evaluate_records(all_predictions, "arm0")
    print("  ✓ Arm 0 Macro EM:", stats0["macro_mean"]["em"], "% | Macro ES:", stats0["macro_mean"]["es"], "%", flush=True)

    # -------------------------------------------------------------
    # Arm 1: CADRE + Call-Site Usage Injection (Ablation)
    # -------------------------------------------------------------
    print("\n[Arm 1] Running CADRE + Call-Site Usage Injection...", flush=True)
    p1_ids = [build_prompt_raw_chunks(t, tr, pf, use_callsite=True) for t, tr, pf in zip(tasks, task_trs, task_pool_files)]
    out1 = llm.generate([{"prompt_token_ids": p} for p in p1_ids], sampling_params=sp)
    preds_arm1 = [o.outputs[0].text for o in out1]
    for i, p in enumerate(preds_arm1):
        all_predictions[i]["arm1"] = p
    stats1 = evaluate_records(all_predictions, "arm1")
    print("  ✓ Arm 1 Macro EM:", stats1["macro_mean"]["em"], "% | Macro ES:", stats1["macro_mean"]["es"], "%", flush=True)

    # -------------------------------------------------------------
    # Arm 2: Full CADRE-Plus (Raw Chunks + Call-Sites + Targeted IDGR for Python)
    # -------------------------------------------------------------
    print("\n[Arm 2] Running CADRE-Plus (Targeted IDGR for Python)...", flush=True)
    p2_ids = []
    py_gated_count = 0
    for i, (t, tr, pf, raw_draft) in enumerate(zip(tasks, task_trs, task_pool_files, preds_arm1)):
        bm = t["benchmark"]
        if bm == "cceval_python":
            left_ctx = t["left_context"]
            clean_draft = clean_bpe_text(raw_draft)
            draft_lines = [l.strip() for l in clean_draft.splitlines() if l.strip()]
            first_line = draft_lines[0] if draft_lines else ""
            draft_callees = [m.group(1).split(".")[-1] for m in CALL_RE.finditer(first_line)]
            top_callee = None
            for c in draft_callees:
                if c not in STOP_WORDS and len(c) > 2 and c not in left_ctx:
                    top_callee = c
                    break
            if top_callee:
                py_gated_count += 1
                p2_ids.append(build_prompt_raw_chunks(t, tr, pf, use_callsite=True, extra_callee=top_callee))
            else:
                p2_ids.append(p1_ids[i])
        else:
            # Java and RepoEval use Arm 1 (Full Context + Call-sites)
            p2_ids.append(p1_ids[i])

    print(f"  -> Targeted IDGR triggered for {py_gated_count}/100 Python tasks.", flush=True)
    out2 = llm.generate([{"prompt_token_ids": p} for p in p2_ids], sampling_params=sp)
    preds_arm2 = [o.outputs[0].text for o in out2]
    for i, p in enumerate(preds_arm2):
        all_predictions[i]["arm2"] = p
    stats2 = evaluate_records(all_predictions, "arm2")
    print("  ✓ Arm 2 (CADRE-Plus) Macro EM:", stats2["macro_mean"]["em"], "% | Macro ES:", stats2["macro_mean"]["es"], "%", flush=True)

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    arms = [
        ("Arm 0: Baseline CADRE v3 (Reference)", stats0),
        ("Arm 1: CADRE + Call-Site Usage Injection", stats1),
        ("Arm 2: CADRE-Plus (Raw Chunks + Call-Sites + Targeted IDGR)", stats2),
    ]

    print("\n" + "=" * 125, flush=True)
    print(f"{'ARM':<60} | {'cceval_py':<10} | {'cceval_ja':<10} | {'repo_api':<10} | {'repo_line':<10} | {'MACRO EM':<10} | {'MACRO ES':<10}", flush=True)
    print("=" * 125, flush=True)
    for name, st in arms:
        print(f"{name:<60} | {st['cceval_python']['em']:>8.2f}% | {st['cceval_java']['em']:>8.2f}% | {st['repoeval_api']['em']:>8.2f}% | {st['repoeval_line']['em']:>8.2f}% | {st['macro_mean']['em']:>8.2f}% | {st['macro_mean']['es']:>8.2f}%", flush=True)
    print("=" * 125, flush=True)

    # Save outputs
    out_metrics = OUTPUT_DIR / "cadre_plus_pilot_metrics.json"
    with open(out_metrics, "w") as f:
        json.dump({name: st for name, st in arms}, f, indent=2)
    print(f"✓ Saved full metrics to {out_metrics}", flush=True)

    out_preds = OUTPUT_DIR / "cadre_plus_pilot_predictions.jsonl.gz"
    with gzip.open(out_preds, "wt") as f:
        for p in all_predictions:
            f.write(json.dumps(p) + "\n")
    print(f"✓ Saved all predictions to {out_preds}", flush=True)

if __name__ == "__main__":
    main()
