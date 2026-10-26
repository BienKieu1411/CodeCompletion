#!/usr/bin/env python3
"""
CADRE 4K Full Left Context Evaluation across all 8,004 tasks.
- Official AlignCoder Evaluation Standard:
    * Tree-Sitter AST Statement truncation for Python (cceval_python, repoeval).
    * Semicolon/bracket statement truncation for Java (cceval_java).
    * RepoEval official Repo-Macro average and line truncation.
    * 100% space and newline preservation via vLLM prompt_token_ids payload.
- Caches / Traces loaded directly:
    * Arm 1 & Arm 2: rlcoder_full_left_standard_8004_trace.jsonl.gz
    * Arm 3: rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz
- Context Budget: 4,096 tokens total.
    * Cross-file ceiling: 1,280 tokens.
    * CSSI Contract ceiling: 512 tokens.
    * Left Context: >= 2,800 tokens.
- Three Arms evaluated head-to-head:
    * Arm 1 (RLCoder Baseline 4k): Standard retrieval, RLCoder flat prompt, NO CSSI.
    * Arm 2 (CADRE Post-hoc CSSI 4k): Standard retrieval, Coalesced Raw Blocks + CSSI in prompt.
    * Arm 3 (CADRE Pre+Post CSSI 4k): CSSI-guided retrieval, Coalesced Raw Blocks + CSSI in prompt.
- Generator: deepseek-ai/deepseek-coder-1.3b-base (vLLM, max_model_len=4096, greedy, max_tokens=64).
"""

from __future__ import annotations

import difflib
import gzip
import json
import multiprocessing as mp
import os
import re
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


import subprocess
import sys

def ensure_deps():
    needed = []
    for pkg, imp in [('vllm', 'vllm'), ('transformers', 'transformers'), ('tree-sitter', 'tree_sitter'), ('tree-sitter-python', 'tree_sitter_python'), ('tree-sitter-java', 'tree_sitter_java'), ('fuzzywuzzy', 'fuzzywuzzy'), ('editdistance', 'editdistance')]:
        try:
            __import__(imp)
        except ImportError:
            needed.append(pkg)
    if needed:
        print(f'Installing missing dependencies: {needed}...', flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + needed)
        print('Dependencies installed successfully!', flush=True)

ensure_deps()

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

DATA_ROOT = Path("/content/mb_spt") if Path("/content/mb_spt").exists() else Path("experiments/mb_spt_iterative")
BUNDLE_PATH = DATA_ROOT / "full_benchmark_8004_bundle.tar.gz"
OUTPUT_DIR = DATA_ROOT / "cadre_4k_eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACE_STD_PATH = DATA_ROOT / "rlcoder_full_left_standard_8004_trace.jsonl.gz"
if not TRACE_STD_PATH.exists():
    TRACE_STD_PATH = Path("experiments/mb_spt_iterative/rlcoder_full_left_standard_8004_trace.jsonl.gz")

TRACE_GUIDED_PATH = DATA_ROOT / "rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz"
if not TRACE_GUIDED_PATH.exists():
    TRACE_GUIDED_PATH = Path("experiments/mb_spt_iterative/rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz")

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

    symbols = []
    m_rec = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", cur_line)
    symbols.extend(m_rec)

    if language == "python":
        for m in re.finditer(r"(?:from\s+[A-Za-z0-9_.]+\s+import\s+([^#\n]+)|import\s+([^#\n]+))", recent_text):
            clause = m.group(1) or m.group(2)
            if clause:
                for part in clause.split(","):
                    p_tok = part.strip().split()
                    if p_tok and p_tok[0] not in STOP_WORDS:
                        symbols.append(p_tok[0])
    else:
        for m in re.finditer(r"import\s+([A-Za-z0-9_.]+);", recent_text):
            imp = m.group(1)
            parts = imp.split(".")
            if parts and parts[-1] not in STOP_WORDS:
                symbols.append(parts[-1])

    m_init = re.findall(r"\b([A-Z][A-Za-z0-9_]*)\s*\(", recent_text)
    symbols.extend(m_init)

    m_ann = re.findall(r":\s*([A-Za-z0-9_]+)", recent_text)
    symbols.extend(m_ann)

    m_new = re.findall(r"\bnew\s+([A-Za-z0-9_]+)", recent_text)
    symbols.extend(m_new)

    seen = set()
    dedup = []
    for s in symbols:
        if s and s not in STOP_WORDS and len(s) > 1 and s not in seen:
            seen.add(s)
            dedup.append(s)

    types = [s for s in dedup if s[0].isupper()]
    funcs_or_vars = [s for s in dedup if not s[0].isupper()]
    return {
        "symbols": dedup[:12],
        "types": types[:8],
        "funcs": funcs_or_vars[:8],
    }


def synthesize_enhanced_cssi_contract(
    pool_files: list[dict[str, Any]],
    scope_info: dict[str, Any],
    language: str,
    max_tokens: int = 512,
    tok: Any = None,
) -> tuple[str, str]:
    cursor_types = scope_info.get("types", [])
    cursor_syms = scope_info.get("symbols", [])
    if not cursor_types and not cursor_syms:
        return "", ""

    comment_sym = "#" if language == "python" else "//"
    contract_snippets = []
    signatures_for_query = []
    total_tokens = 0

    for pf in pool_files:
        fpath = pf["path"]
        content = pf["text"]
        if not content:
            continue

        if language == "python":
            for tname in cursor_types[:4]:
                c_pat = re.compile(rf"^class\s+{re.escape(tname)}(?:\([^)]*\))?:", re.MULTILINE)
                m = c_pat.search(content)
                if m:
                    idx = m.start()
                    sub = content[idx : idx + 4000]
                    lines = sub.splitlines()
                    retained_lines = [lines[0]]
                    for line in lines[1:]:
                        st = line.strip()
                        if (st.startswith("def ") or st.startswith("async def ")) and not st.startswith("def _"):
                            sig = st.split(":")[0].strip()
                            retained_lines.append(f"    {sig}: ...")
                        elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", st) or re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", st):
                            retained_lines.append(f"    {st.split('=')[0].strip()}")
                        if len(retained_lines) >= 20:
                            break
                    if len(retained_lines) > 1:
                        snippet = f"# Contract: {tname} (from {fpath})\nclass {tname}:\n" + "\n".join(retained_lines[:20]) + "\n"
                        contract_snippets.append(snippet)
                        signatures_for_query.append(f"class {tname}")
                        total_tokens += len(snippet.split())
                        break

            for sname in cursor_syms[:5]:
                f_pat = re.compile(rf"(?:def|async def)\s+{re.escape(sname)}\s*\([^)]*\)[^\n:]*:\s*", re.MULTILINE)
                for m in f_pat.finditer(content):
                    sig = m.group(0).strip()
                    snippet = f"# Contract: {sname} (from {fpath})\n{sig}: ...\n"
                    contract_snippets.append(snippet)
                    signatures_for_query.append(sig)
                    total_tokens += len(snippet.split())
                    if len(signatures_for_query) >= 8:
                        break

            if total_tokens > max_tokens:
                break
        else:
            for tname in cursor_types[:4]:
                if f"class {tname}" in content or f"interface {tname}" in content:
                    lines = content.splitlines()
                    sigs = []
                    for line in lines:
                        st = line.strip()
                        if ("public " in st or "protected " in st) and "(" in st and ")" in st and not st.startswith("//"):
                            sig = st.split("{")[0].strip()
                            sigs.append(f"    {sig};")
                            signatures_for_query.append(sig)
                        if len(sigs) >= 16:
                            break
                    if sigs:
                        snippet = f"// Contract: {tname} (from {fpath})\nclass {tname} {{\n" + "\n".join(sigs[:16]) + "\n}}\n"
                        contract_snippets.append(snippet)
                        signatures_for_query.append(f"class {tname}")
                        total_tokens += len(snippet.split())
                        break

            if total_tokens > max_tokens:
                break

    if not contract_snippets:
        return "", ""

    header = f"{comment_sym} === CURSOR STRUCTURAL CONTRACTS (CSSI ENHANCED) ==="
    footer = f"{comment_sym} ====================================================="
    contract_text = f"{header}\n" + "\n".join(contract_snippets) + f"{footer}"

    if tok:
        ids = tok.encode(contract_text, add_special_tokens=False)
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
            contract_text = tok.decode(ids)

    query_sigs = " ".join(signatures_for_query[:15])
    return contract_text, query_sigs


def coalesce_raw_chunks(blocks: list[dict[str, Any]], language: str) -> list[str]:
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for b in blocks:
        by_file[b["file_path"]].append(b)

    cs = "#" if language == "python" else "//"
    coalesced_blocks: list[str] = []

    for fpath, fblocks in by_file.items():
        fblocks.sort(key=lambda x: x.get("line_start", 0))
        merged_chunks = []
        cur_start, cur_end, cur_lines = None, None, []

        for b in fblocks:
            bs = b.get("line_start", 0)
            be = b.get("line_end", bs + len(b["code_content"].splitlines()))
            blines = b["code_content"].splitlines()

            if cur_end is not None and bs <= cur_end + 3:
                overlap = max(0, cur_end - bs)
                cur_lines.extend(blines[overlap:])
                cur_end = max(cur_end, be)
            else:
                if cur_lines:
                    merged_chunks.append((cur_start, cur_end, cur_lines))
                cur_start, cur_end = bs, be
                cur_lines = list(blines)

        if cur_lines:
            merged_chunks.append((cur_start, cur_end, cur_lines))

        for s, e, lines in merged_chunks:
            header = f"{cs} file path: {fpath}\n{cs} lines: {s}-{e}"
            raw_body = "\n".join(lines)
            coalesced_blocks.append(f"{header}\n{raw_body}")

    return coalesced_blocks


def build_prompt_4k_ids(
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
        block_texts = []
        for b in filtered:
            desc = f"{cs} file path: {b['file_path']}\n{cs} lines: {b.get('line_start', 0)}-{b.get('line_end', 0)}"
            code_lines = "\n".join(f"{cs} {l}" for l in b["code_content"].splitlines() if l.strip())
            block_texts.append(f"{desc}\n{code_lines}")
        cross_text = "\n\n".join(block_texts)
        cross_ids = tok.encode(cross_text, add_special_tokens=False)[:cross_max]

    elif arm in ("cadre_posthoc_cssi_4k", "cadre_pre_and_post_cssi_4k"):
        scope_info = extract_cursor_context_symbols(task["left_context"], lang)
        cssi_text, _ = synthesize_enhanced_cssi_contract(pool_files, scope_info, lang, max_tokens=cssi_max, tok=tok)
        cssi_ids = tok.encode("\n\n" + cssi_text, add_special_tokens=False) if cssi_text else []

        rem_snippets = max(0, cross_max - len(cssi_ids))
        blocks = coalesce_raw_chunks(filtered, lang)
        exemplar_text = "\n\n".join(blocks)
        exemplar_ids = tok.encode(exemplar_text, add_special_tokens=False)[:rem_snippets]
        cross_ids = exemplar_ids + cssi_ids

    else:
        raise ValueError(f"Unknown arm: {arm}")

    allowed_left = max(0, max_context - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []
    return (cross_ids + path_ids + left_ids)[-max_context:]


# ==========================================
# ALIGNCODER OFFICIAL AST & REPOEVAL METRICS
# ==========================================

def get_bracket_lang_statement(completion: str) -> str:
    end_idx = None
    for i in range(len(completion)):
        if completion[i] in [";", "}", "{"]:
            end_idx = i
            break
    return completion[:end_idx + 1] if end_idx is not None else completion


def get_python_one_statement(prompt: str, completion: str, parser: Any) -> str:
    code = prompt + completion
    tree = parser.parse(code.encode("utf8"))
    root = tree.root_node
    cursor = root.walk()
    cursor_byte = len(prompt.encode("utf8"))
    while True:
        node = cursor.node
        if node.start_byte >= cursor_byte:
            if node.type in [
                "simple_statement", "expression_statement", "assert_statement",
                "return_statement", "raise_statement", "pass_statement",
                "break_statement", "continue_statement", "import_statement",
                "import_from_statement", "global_statement", "nonlocal_statement",
                "type_alias_statement",
            ]:
                text = code.encode("utf8")[cursor_byte : node.end_byte].decode("utf8", errors="ignore")
                return text.splitlines()[0] if "\n" in text else text
            elif node.type in [
                "if_statement", "for_statement", "while_statement", "try_statement",
                "with_statement", "function_definition", "class_definition",
                "decorated_definition", "match_statement",
            ]:
                text = code.encode("utf8")[cursor_byte : node.end_byte].decode("utf8", errors="ignore")
                first_line = text.splitlines()[0] if "\n" in text else text
                if ":" in first_line:
                    return first_line[: first_line.index(":") + 1]
                return first_line
        if cursor.goto_first_child():
            continue
        while not cursor.goto_next_sibling():
            if not cursor.goto_parent():
                lines = completion.splitlines()
                return lines[0] if lines else completion
    return completion.splitlines()[0] if completion.splitlines() else completion


def remove_comments(code: str) -> str:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"//.*", "", code)
    return code


def postprocess_repoeval(pred: str, gt: str) -> str:
    gt_lines = [l.strip() for l in gt.splitlines() if l.strip()]
    n = max(1, len(gt_lines))
    pred_lines = [l.strip() for l in pred.splitlines() if l.strip()][:n]
    return "\n".join(pred_lines)


def evaluate_arm_aligncoder_official(
    tasks: list[dict[str, Any]],
    predictions: list[str],
    py_parser: Any,
) -> dict[str, Any]:
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for t, raw_pred in zip(tasks, predictions):
        b = t["benchmark"]
        lang = t["language"]
        gt = t["groundtruth"]

        if b.startswith("repoeval"):
            clean_pred = postprocess_repoeval(raw_pred, gt)
        elif lang == "python":
            clean_pred = get_python_one_statement(t["left_context"], raw_pred, py_parser).strip()
        else:
            clean_pred = get_bracket_lang_statement(raw_pred).strip()

        clean_pred = remove_comments(clean_pred)
        gt_clean = remove_comments(gt)

        p_lines = [l.strip() for l in clean_pred.splitlines() if l.strip()]
        g_lines = [l.strip() for l in gt_clean.splitlines() if l.strip()]
        em_exact = 1 if p_lines == g_lines and g_lines else 0

        # Whitespace-collapsed prefix match (for comparison)
        p_flat = "".join("".join(p_lines).split())
        g_flat = "".join("".join(g_lines).split())
        em_prefix = 1 if p_flat[:len(g_flat)] == g_flat and g_flat else 0

        es = cal_edit_sim_single(clean_pred, gt_clean)
        gt_ids = extract_identifiers(gt_clean, lang)
        pred_ids = extract_identifiers(clean_pred, lang)
        gt_set, pred_set = set(gt_ids), set(pred_ids)
        if not gt_set and not pred_set:
            id_f1 = 100.0
        elif not gt_set or not pred_set:
            id_f1 = 0.0
        else:
            p = len(gt_set & pred_set) / len(pred_set)
            r = len(gt_set & pred_set) / len(gt_set)
            id_f1 = (2 * p * r / (p + r)) * 100.0 if (p + r) > 0 else 0.0

        repo = t["task_id"].split("/")[0]
        by_bench[b].append({
            "task_id": t["task_id"],
            "repo": repo,
            "em_exact": em_exact,
            "em_prefix": em_prefix,
            "es": es,
            "id_f1": id_f1,
        })

    summary: dict[str, Any] = {}
    macro_exact, macro_prefix, macro_es, macro_id_f1 = [], [], [], []

    for b in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        recs = by_bench[b]
        n = len(recs)
        if b.startswith("repoeval"):
            # Official Repo-Macro EM
            repo_scores = defaultdict(list)
            for r in recs:
                repo_scores[r["repo"]].append(r["em_exact"])
            b_em = sum(sum(v)/len(v) for v in repo_scores.values()) / len(repo_scores) * 100.0
        else:
            b_em = sum(r["em_exact"] for r in recs) / n * 100.0

        b_pref = sum(r["em_prefix"] for r in recs) / n * 100.0
        b_es = sum(r["es"] for r in recs) / n
        b_f1 = sum(r["id_f1"] for r in recs) / n

        summary[b] = {
            "count": n,
            "aligncoder_ast_em": round(b_em, 2),
            "prefix_em": round(b_pref, 2),
            "edit_sim": round(b_es, 2),
            "id_f1": round(b_f1, 2),
        }
        macro_exact.append(b_em)
        macro_prefix.append(b_pref)
        macro_es.append(b_es)
        macro_id_f1.append(b_f1)

    all_recs = [r for recs in by_bench.values() for r in recs]
    total_n = len(all_recs)
    summary["micro_mean"] = {
        "count": total_n,
        "aligncoder_ast_em": round(sum(r["em_exact"] for r in all_recs) / total_n * 100.0, 2),
        "prefix_em": round(sum(r["em_prefix"] for r in all_recs) / total_n * 100.0, 2),
        "edit_sim": round(sum(r["es"] for r in all_recs) / total_n, 2),
        "id_f1": round(sum(r["id_f1"] for r in all_recs) / total_n, 2),
    }
    summary["macro_mean"] = {
        "aligncoder_ast_em": round(sum(macro_exact) / 4.0, 2),
        "prefix_em": round(sum(macro_prefix) / 4.0, 2),
        "edit_sim": round(sum(macro_es) / 4.0, 2),
        "id_f1": round(sum(macro_id_f1) / 4.0, 2),
    }
    return summary


def main():
    print("=" * 100)
    print("CADRE 4K FULL LEFT CONTEXT EVALUATION - OFFICIAL ALIGNCODER AST STANDARD")
    print("=" * 100, flush=True)

    t_start = time.time()

    # 1. Load benchmark bundle
    print("\n[Step 1/5] Loading full benchmark bundle (8,004 tasks)...", flush=True)
    tasks: list[dict[str, Any]] = []
    files_by_id: dict[int, dict[str, Any]] = {}

    with tarfile.open(BUNDLE_PATH, "r:gz") as tar:
        tf = tar.extractfile("tasks.jsonl.gz")
        with gzip.open(tf, "rt", encoding="utf-8") as gz:
            for line in gz:
                tasks.append(json.loads(line))

        ff = tar.extractfile("files.jsonl.gz")
        with gzip.open(ff, "rt", encoding="utf-8") as gz:
            for line in gz:
                f_obj = json.loads(line)
                files_by_id[f_obj["file_id"]] = f_obj

    print(f"✓ Loaded {len(tasks)} tasks and {len(files_by_id)} repository files.", flush=True)

    # 2. Load pre-computed retrieval traces
    print("\n[Step 2/5] Loading pre-computed dual retrieval traces...", flush=True)
    std_trace_map: dict[str, list[dict[str, Any]]] = {}
    with gzip.open(TRACE_STD_PATH, "rt", encoding="utf-8") as gz:
        for line in gz:
            rec = json.loads(line)
            key = f"{rec['benchmark']}::{rec['task_id']}"
            std_trace_map[key] = rec.get("rlcoder_top10", rec.get("top10", []))

    guided_trace_map: dict[str, list[dict[str, Any]]] = {}
    with gzip.open(TRACE_GUIDED_PATH, "rt", encoding="utf-8") as gz:
        for line in gz:
            rec = json.loads(line)
            key = f"{rec['benchmark']}::{rec['task_id']}"
            guided_trace_map[key] = rec.get("rlcoder_top10", rec.get("top10", []))

    print(f"✓ Loaded {len(std_trace_map)} standard traces and {len(guided_trace_map)} guided traces.", flush=True)

    # 3. Assemble prompt token IDs (4k budget)
    print("\n[Step 3/5] Building prompt token IDs (guaranteed 100% space/newline preservation)...", flush=True)
    gen_tok = AutoTokenizer.from_pretrained(GENERATOR)

    prompts_arm1_ids: list[list[int]] = []
    prompts_arm2_ids: list[list[int]] = []
    prompts_arm3_ids: list[list[int]] = []

    for idx, t in enumerate(tasks):
        pool_files = []
        for rel_path, fid in t.get("file_ids_by_path", {}).items():
            fe = files_by_id.get(fid)
            if fe:
                pool_files.append({"path": rel_path, "text": fe["text"]})

        key = f"{t['benchmark']}::{t['task_id']}"
        top10_std = std_trace_map.get(key, [])
        top10_guided = guided_trace_map.get(key, [])

        p1_ids = build_prompt_4k_ids("rlcoder_raw_4k", t, top10_std, pool_files, gen_tok, max_context=4096, cross_max=1280)
        p2_ids = build_prompt_4k_ids("cadre_posthoc_cssi_4k", t, top10_std, pool_files, gen_tok, max_context=4096, cross_max=1280, cssi_max=512)
        p3_ids = build_prompt_4k_ids("cadre_pre_and_post_cssi_4k", t, top10_guided, pool_files, gen_tok, max_context=4096, cross_max=1280, cssi_max=512)

        prompts_arm1_ids.append(p1_ids)
        prompts_arm2_ids.append(p2_ids)
        prompts_arm3_ids.append(p3_ids)

    print(f"✓ Built prompt token IDs for 3 arms x {len(tasks)} tasks = {3 * len(tasks)} prompts.", flush=True)

    # 4. Initialize vLLM and generate completions
    print("\n[Step 4/5] Initializing vLLM generator (deepseek-ai/deepseek-coder-1.3b-base)...", flush=True)
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

    print("\n  [Arm 1] Generating completions for RLCoder Baseline 4k (prompt_token_ids)...", flush=True)
    t_g1 = time.time()
    payload1 = [{"prompt_token_ids": p} for p in prompts_arm1_ids]
    out1 = llm.generate(payload1, sampling_params)
    preds_arm1 = [gen_tok.decode(o.outputs[0].token_ids, skip_special_tokens=True) for o in out1]
    print(f"  ✓ Arm 1 completed in {time.time() - t_g1:.1f}s.", flush=True)

    print("\n  [Arm 2] Generating completions for CADRE Post-hoc CSSI 4k (prompt_token_ids)...", flush=True)
    t_g2 = time.time()
    payload2 = [{"prompt_token_ids": p} for p in prompts_arm2_ids]
    out2 = llm.generate(payload2, sampling_params)
    preds_arm2 = [gen_tok.decode(o.outputs[0].token_ids, skip_special_tokens=True) for o in out2]
    print(f"  ✓ Arm 2 completed in {time.time() - t_g2:.1f}s.", flush=True)

    print("\n  [Arm 3] Generating completions for CADRE Pre+Post CSSI 4k (prompt_token_ids)...", flush=True)
    t_g3 = time.time()
    payload3 = [{"prompt_token_ids": p} for p in prompts_arm3_ids]
    out3 = llm.generate(payload3, sampling_params)
    preds_arm3 = [gen_tok.decode(o.outputs[0].token_ids, skip_special_tokens=True) for o in out3]
    print(f"  ✓ Arm 3 completed in {time.time() - t_g3:.1f}s.", flush=True)

    # 5. Evaluate Metrics with Tree-Sitter AST & RepoEval Official
    print("\n[Step 5/5] Computing official AlignCoder AST & RepoEval metrics...", flush=True)
    import tree_sitter_python
    from tree_sitter import Language, Parser
    py_lang = Language(tree_sitter_python.language())
    py_parser = Parser(py_lang)

    eval_arm1 = evaluate_arm_aligncoder_official(tasks, preds_arm1, py_parser)
    eval_arm2 = evaluate_arm_aligncoder_official(tasks, preds_arm2, py_parser)
    eval_arm3 = evaluate_arm_aligncoder_official(tasks, preds_arm3, py_parser)

    # Print Table
    print("\n" + "=" * 115, flush=True)
    print("HEAD-TO-HEAD COMPARISON ACROSS ALL 8,004 TASKS (ALIGNCODER OFFICIAL PROTOCOL, 4,096 CONTEXT BUDGET)")
    print("=" * 115, flush=True)
    print(f"{'Benchmark':<18} | {'RLCoder Base 4k':<20} | {'CADRE Post-hoc 4k':<20} | {'CADRE Pre+Post 4k':<20} | {'Gain (Arm3 vs 1)':<15}")
    print("-" * 115, flush=True)

    for b in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        s1 = eval_arm1[b]["aligncoder_ast_em"]
        s2 = eval_arm2[b]["aligncoder_ast_em"]
        s3 = eval_arm3[b]["aligncoder_ast_em"]
        gain = s3 - s1
        print(f"{b:<18} | {s1:>18.2f}% | {s2:>18.2f}% | {s3:>18.2f}% | {gain:>+13.2f} pp")

    print("-" * 115, flush=True)
    m1 = eval_arm1["macro_mean"]["aligncoder_ast_em"]
    m2 = eval_arm2["macro_mean"]["aligncoder_ast_em"]
    m3 = eval_arm3["macro_mean"]["aligncoder_ast_em"]
    mgain = m3 - m1
    print(f"{'MACRO MEAN':<18} | {m1:>18.2f}% | {m2:>18.2f}% | {m3:>18.2f}% | {mgain:>+13.2f} pp")
    print("=" * 115, flush=True)

    final_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generator": GENERATOR,
            "evaluation_standard": "AlignCoder Official AST & RepoEval Repo-Macro",
            "total_tasks": len(tasks),
            "max_tokens": 64,
            "context_tokens": 4096,
            "cross_max_tokens": 1280,
            "cssi_max_tokens": 512,
            "duration_sec": round(time.time() - t_start, 1),
        },
        "arms": {
            "rlcoder_base_4k": eval_arm1,
            "cadre_posthoc_cssi_4k": eval_arm2,
            "cadre_pre_and_post_cssi_4k": eval_arm3,
        },
    }

    metrics_file = OUTPUT_DIR / "cadre_4k_official_aligncoder_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n✓ Saved official metrics to {metrics_file}", flush=True)

    preds_file = OUTPUT_DIR / "cadre_4k_official_aligncoder_predictions.jsonl.gz"
    with gzip.open(preds_file, "wt", encoding="utf-8") as gz:
        for i, t in enumerate(tasks):
            gz.write(json.dumps({
                "task_id": t["task_id"],
                "benchmark": t["benchmark"],
                "groundtruth": t["groundtruth"],
                "pred_arm1": preds_arm1[i],
                "pred_arm2": preds_arm2[i],
                "pred_arm3": preds_arm3[i],
            }) + "\n")
    print(f"✓ Saved official predictions to {preds_file}", flush=True)


if __name__ == "__main__":
    main()
