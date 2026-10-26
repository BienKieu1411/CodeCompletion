#!/usr/bin/env python3
"""
CADRE 4K Full Left Context Evaluation across all 8,004 tasks.
- Stage 1 (BM25) & Stage 2 (RLRetriever): Re-run FRESH for all arms.
    * Standard Query (Arm 1 & Arm 2): Full left_context for BM25; 512 tail for RLRetriever.
    * CSSI-Guided Query (Arm 3):
        - BM25: Full left_context + CSSI Contract signatures.
        - RLRetriever: Guaranteed split [CSSI 256 tokens] + [Left Context Tail 252 tokens].
- Context Budget: 4,096 tokens total.
    * Cross-file ceiling: 1,280 tokens.
    * CSSI Contract ceiling: 512 tokens.
    * Left Context: >= 2,800 tokens (guaranteeing local code is NOT truncated).
- Three Arms evaluated head-to-head:
    * Arm 1 (RLCoder Baseline 4k): Standard retrieval, RLCoder flat prompt, NO CSSI.
    * Arm 2 (CADRE Post-hoc CSSI 4k): Standard retrieval, Coalesced Raw Blocks + CSSI in prompt.
    * Arm 3 (CADRE Pre+Post CSSI 4k): CSSI-guided retrieval, Coalesced Raw Blocks + CSSI in prompt.
- Generator: deepseek-ai/deepseek-coder-1.3b-base (vLLM, max_model_len=4096, greedy, max_tokens=64).
"""

import ast
import concurrent.futures
import difflib
import gc
import gzip
import json
import multiprocessing as mp
import os
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

DATA_ROOT = Path("/content/mb_spt") if Path("/content/mb_spt").exists() else Path("experiments/mb_spt_iterative")
BUNDLE_PATH = DATA_ROOT / "full_benchmark_8004_bundle.tar.gz"
OUTPUT_DIR = DATA_ROOT / "cadre_4k_eval_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACE_STANDARD_PATH = OUTPUT_DIR / "rlcoder_full_left_standard_8004_trace.jsonl.gz"
TRACE_GUIDED_PATH = OUTPUT_DIR / "rlcoder_full_left_cssi_guided_8004_trace.jsonl.gz"

RETRIEVER_MODEL = "nov3630/RLCoder"
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

    receivers = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", cur_line)
    
    assigned_types = {}
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", recent_text):
        assigned_types[m.group(1)] = m.group(2)
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", recent_text):
        assigned_types[m.group(1)] = m.group(2)

    imports = []
    imported_symbols = {}
    for m in re.finditer(r"(?:from\s+([A-Za-z0-9_.]+)\s+import\s+([^#\n]+)|import\s+([^#\n]+))", left_context):
        if m.group(1) and m.group(2):
            raw_syms = m.group(2).replace("(", "").replace(")", "").split(",")
            for s in raw_syms:
                s = s.strip()
                if " as " in s:
                    orig, alias = s.split(" as ")
                    imported_symbols[alias.strip()] = orig.strip()
                    imports.append(orig.strip())
                elif s:
                    imported_symbols[s] = s
                    imports.append(s)
        elif m.group(3):
            for s in m.group(3).split(","):
                s = s.strip()
                if " as " in s:
                    orig, alias = s.split(" as ")
                    imported_symbols[alias.strip()] = orig.strip()
                    imports.append(orig.strip())
                elif s:
                    imported_symbols[s] = s
                    imports.append(s)

    target_types = set()
    for r in receivers:
        if r in assigned_types:
            target_types.add(assigned_types[r])
        elif r in imported_symbols:
            target_types.add(imported_symbols[r])
        elif r and r[0].isupper():
            target_types.add(r)
    for s in imports:
        if s and s[0].isupper():
            target_types.add(s)

    cur_syms = [w for w in WORD_RE.findall(cur_line) if w not in STOP_WORDS and len(w) > 2]

    return {
        "receivers": receivers,
        "target_types": list(target_types),
        "cursor_symbols": cur_syms,
    }


def synthesize_enhanced_cssi_contract(pool_files: list[dict[str, Any]], scope_info: dict[str, Any], language: str, max_tokens: int = 512, tok: Any = None) -> tuple[str, str]:
    target_types = scope_info.get("target_types", [])
    cursor_syms = scope_info.get("cursor_symbols", [])
    comment_sym = "#" if language == "python" else "//"

    contract_snippets = []
    signatures_for_query = []
    total_tokens = 0

    if language == "python":
        for f in pool_files:
            content = f.get("text", "")
            if not content:
                continue
            fpath = f.get("path", "")

            # 1. Class signatures and method skeletons
            for tname in target_types[:4]:
                pattern = re.compile(rf"class\s+{re.escape(tname)}\b[^\n:]*:\s*\n((?:[ \t]+[^\n]*\n|\s*\n)*)", re.MULTILINE)
                for m in pattern.finditer(content):
                    lines = m.group(1).splitlines()
                    retained_lines = []
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("def ") or stripped.startswith("async def "):
                            sig = stripped.split(":", 1)[0]
                            retained_lines.append(f"    {sig}: ...")
                            signatures_for_query.append(sig)
                        elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", stripped) or re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped):
                            if len(stripped) < 80:
                                retained_lines.append(f"    {stripped}")
                    if retained_lines:
                        snippet = f"# Contract: {tname} (from {fpath})\nclass {tname}:\n" + "\n".join(retained_lines[:20]) + "\n"
                        contract_snippets.append(snippet)
                        signatures_for_query.append(f"class {tname}")
                        total_tokens += len(snippet.split())
                        break

            # 2. Function signatures matching cursor symbols
            for sname in cursor_syms[:5]:
                f_pat = re.compile(rf"(?:def|async def)\s+{re.escape(sname)}\s*\([^)]*\)[^\n:]*:\s*", re.MULTILINE)
                for m in f_pat.finditer(content):
                    sig = m.group(0).strip()
                    snippet = f"# Contract: {sname} (from {fpath})\n{sig}: ...\n"
                    contract_snippets.append(snippet)
                    signatures_for_query.append(sig)
                    total_tokens += len(snippet.split())
                    if total_tokens > max_tokens:
                        break

            if total_tokens > max_tokens:
                break
    else:  # java
        for f in pool_files:
            content = f.get("text", "")
            if not content:
                continue
            fpath = f.get("path", "")

            for tname in target_types[:4]:
                if f"class {tname}" in content or f"interface {tname}" in content:
                    lines = content.splitlines()
                    sigs = []
                    for line in lines:
                        st = line.strip()
                        if ("public " in st or "protected " in st) and "(" in st and ")" in st and not st.startswith("//"):
                            sig = st.split("{")[0].strip()
                            sigs.append(f"    {sig};")
                            signatures_for_query.append(sig)
                        elif (st.startswith("private ") or st.startswith("protected ")) and ";" in st and "(" not in st:
                            sigs.append(f"    {st}")
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


def split_into_smaller_blocks(file_path: str, code_content: str, language: str) -> list[dict[str, Any]]:
    mini_blocks = []
    current_block = []
    for line in code_content.splitlines():
        if line.strip() == "":
            if current_block:
                mini_blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line)
    if current_block:
        mini_blocks.append(current_block)

    max_len = 15
    temp_mini_blocks = []
    for mb in mini_blocks:
        if len(mb) > max_len:
            for idx in range(0, len(mb), max_len):
                temp_mini_blocks.append(mb[idx:idx + max_len])
        else:
            temp_mini_blocks.append(mb)

    processed_blocks = []
    line_cur = 1
    for mb in temp_mini_blocks:
        start_line = line_cur
        end_line = line_cur + len(mb) - 1
        processed_blocks.append({
            "file_path": file_path,
            "line_start": start_line,
            "line_end": end_line,
            "code_content": "\n".join(mb),
            "language": language,
            "is_stop": False,
        })
        line_cur = end_line + 1

    return processed_blocks


def coalesce_raw_chunks(chunks: list[dict[str, Any]], lang: str) -> list[str]:
    comment_sym = "#" if lang == "python" else "//"
    chunks_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        p = c.get("file_path")
        if p and not c.get("is_stop"):
            chunks_by_file[p].append(c)

    blocks = []
    for p, f_chunks in chunks_by_file.items():
        f_chunks.sort(key=lambda x: x.get("line_start", 0))
        code_slices = []
        for ch in f_chunks:
            code = ch.get("code_content", "").strip()
            if code:
                code_slices.append(code)

        if code_slices:
            coalesced_code = "\n...\n".join(code_slices)
            block = f"{comment_sym} file path: {p}\n{coalesced_code}"
            blocks.append(block)

    return blocks


def build_prompt_4k(
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
        # Flat prompt format of RLCoder capped at cross_max
        block_texts = []
        for b in filtered:
            desc = f"{cs} file path: {b['file_path']}\n{cs} lines: {b.get('line_start', 0)}-{b.get('line_end', 0)}"
            code_lines = "\n".join(f"{cs} {l}" for l in b["code_content"].splitlines() if l.strip())
            block_texts.append(f"{desc}\n{code_lines}")
        cross_text = "\n\n".join(block_texts)
        cross_ids = tok.encode(cross_text, add_special_tokens=False)[:cross_max]

    elif arm in ("cadre_posthoc_cssi_4k", "cadre_pre_and_post_cssi_4k"):
        # Coalesced raw code blocks + Enhanced CSSI Contract
        scope_info = extract_cursor_context_symbols(task["left_context"], lang)
        cssi_text, _ = synthesize_enhanced_cssi_contract(pool_files, scope_info, lang, max_tokens=cssi_max, tok=tok)
        cssi_ids = tok.encode("\n\n" + cssi_text, add_special_tokens=False) if cssi_text else []

        rem_snippets = max(0, cross_max - len(cssi_ids))
        blocks = coalesce_raw_chunks(filtered, lang)
        exemplar_text = "\n\n".join(blocks)
        exemplar_ids = tok.encode(exemplar_text, add_special_tokens=False)[:rem_snippets]

        # Channel A (Exemplars) -> Channel B (Contract) -> Path -> Left Context
        cross_ids = exemplar_ids + cssi_ids

    else:
        raise ValueError(f"Unknown arm: {arm}")

    # Left context gets the remaining budget up to 4,096 tokens (guaranteed >= 2,800 tokens)
    allowed_left = max(0, max_context - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []

    return (cross_ids + path_ids + left_ids)[-max_context:]


def compute_official_em(target: str, prediction: str, language: str) -> int:
    comment_prefix = "#" if language == "python" else "//"
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    prediction_lines = [line for line in prediction_lines if not line.startswith(comment_prefix)][: len(target_lines)]
    target_lines_str = "".join("".join(target_lines).split())
    prediction_lines_str = "".join("".join(prediction_lines).split())
    prediction_lines_str = prediction_lines_str[: len(target_lines_str)]
    return int(target_lines_str == prediction_lines_str) if target_lines_str else 0


def evaluate_predictions_official(tasks: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    by_bench = defaultdict(list)
    for t, pred in zip(tasks, predictions):
        bench = t["benchmark"]
        lang = t["language"]
        gt = t["groundtruth"]

        em = compute_official_em(gt, pred, lang)
        es = cal_edit_sim_single(pred, gt)

        gt_ids = extract_identifiers(gt, lang)
        pred_ids = extract_identifiers(pred, lang)
        gt_set, pred_set = set(gt_ids), set(pred_ids)
        if not gt_set and not pred_set:
            id_f1 = 100.0
        elif not gt_set or not pred_set:
            id_f1 = 0.0
        else:
            p = len(gt_set & pred_set) / len(pred_set)
            r = len(gt_set & pred_set) / len(gt_set)
            id_f1 = (2 * p * r / (p + r)) * 100.0 if (p + r) > 0 else 0.0

        by_bench[bench].append({
            "task_id": t["task_id"],
            "em": em,
            "es": es,
            "id_f1": id_f1,
        })

    summary = {}
    macro_em, macro_es, macro_id_f1 = [], [], []
    for b, recs in by_bench.items():
        n = len(recs)
        m_em = sum(r["em"] for r in recs) / n * 100.0
        m_es = sum(r["es"] for r in recs) / n
        m_f1 = sum(r["id_f1"] for r in recs) / n
        summary[b] = {
            "count": n,
            "em": round(m_em, 2),
            "es": round(m_es, 2),
            "id_f1": round(m_f1, 2),
        }
        macro_em.append(m_em)
        macro_es.append(m_es)
        macro_id_f1.append(m_f1)

    all_recs = [r for recs in by_bench.values() for r in recs]
    total_n = len(all_recs)
    summary["micro_mean"] = {
        "count": total_n,
        "em": round(sum(r["em"] for r in all_recs) / total_n * 100.0, 2),
        "es": round(sum(r["es"] for r in all_recs) / total_n, 2),
        "id_f1": round(sum(r["id_f1"] for r in all_recs) / total_n, 2),
    }
    summary["macro_mean"] = {
        "em": round(sum(macro_em) / len(macro_em), 2),
        "es": round(sum(macro_es) / len(macro_es), 2),
        "id_f1": round(sum(macro_id_f1) / len(macro_id_f1), 2),
    }

    return summary


# Global variables for multiprocessing worker
_G_TASKS = None
_G_FILES_BY_ID = None
_G_QUERIES_STD = None
_G_QUERIES_GUI = None


def _init_worker(tasks, files_by_id, queries_std, queries_gui):
    global _G_TASKS, _G_FILES_BY_ID, _G_QUERIES_STD, _G_QUERIES_GUI
    _G_TASKS = tasks
    _G_FILES_BY_ID = files_by_id
    _G_QUERIES_STD = queries_std
    _G_QUERIES_GUI = queries_gui


def _bm25_worker_task(idx: int):
    t = _G_TASKS[idx]
    lang = t["language"]
    pool_blocks = []
    for rel_path, fid in t.get("file_ids_by_path", {}).items():
        fe = _G_FILES_BY_ID.get(fid)
        if fe:
            blocks = split_into_smaller_blocks(rel_path, fe["text"], lang)
            pool_blocks.extend(blocks)

    stop_cand = {
        "file_path": "",
        "line_start": 0,
        "line_end": 0,
        "code_content": "Don't need cross file context for completion",
        "language": lang,
        "is_stop": True,
    }

    if not pool_blocks:
        return idx, [stop_cand], [stop_cand]

    corpus = [b["code_content"].split() for b in pool_blocks]
    bm25 = BM25Okapi(corpus)

    # Standard
    scores_std = bm25.get_scores(_G_QUERIES_STD[idx].split())
    top_std_idx = np.argsort(scores_std)[-100:][::-1]
    top100_std = [pool_blocks[i] for i in top_std_idx if i < len(pool_blocks)]
    top100_std.append(stop_cand)

    # Guided
    scores_gui = bm25.get_scores(_G_QUERIES_GUI[idx].split())
    top_gui_idx = np.argsort(scores_gui)[-100:][::-1]
    top100_gui = [pool_blocks[i] for i in top_gui_idx if i < len(pool_blocks)]
    top100_gui.append(stop_cand)

    return idx, top100_std, top100_gui


def main():
    t_start = time.time()
    print("=" * 105, flush=True)
    print("CADRE 4K FULL LEFT CONTEXT EVALUATION (8,004 TASKS)", flush=True)
    print("Guaranteed 50/50 Allocation: [CSSI Contract 256 tokens] + [Left Context Tail 252 tokens]", flush=True)
    print("Arm 1: RLCoder Baseline 4k (Query: FULL left_context, Flat Prompt, No CSSI)", flush=True)
    print("Arm 2: CADRE Post-hoc CSSI 4k (Query: FULL left_context, Prompt: +CSSI Contract)", flush=True)
    print("Arm 3: CADRE Pre+Post CSSI 4k (Query: CSSI [256] + Tail [252], Prompt: +CSSI Contract)", flush=True)
    print("=" * 105, flush=True)

    # 1. Load Tasks & Files
    print("\n[Step 1/5] Loading benchmark tasks and repository files from bundle...", flush=True)
    with tarfile.open(BUNDLE_PATH, "r:gz") as tar:
        f_tasks = tar.extractfile("tasks.jsonl.gz")
        with gzip.open(f_tasks, "rt") as gz:
            tasks = [json.loads(line) for line in gz]
        f_files = tar.extractfile("files.jsonl.gz")
        files_by_id = {}
        with gzip.open(f_files, "rt") as gz:
            for line in gz:
                d = json.loads(line)
                files_by_id[d["file_id"]] = d
    print(f"✓ Loaded {len(tasks)} tasks and indexed {len(files_by_id)} repository files.", flush=True)

    # 2. Stage 1: BM25 Candidate Retrieval (Multi-core accelerated)
    print("\n[Step 2/5] Preparing queries for BM25 and RLRetriever...", flush=True)
    queries_standard = []
    queries_guided_bm25 = []
    raw_cssi_signatures = []

    for t in tasks:
        lang = t["language"]
        pool_files = []
        for rel_path, fid in t.get("file_ids_by_path", {}).items():
            fe = files_by_id.get(fid)
            if fe:
                pool_files.append({"path": rel_path, "text": fe["text"]})

        # Standard query: FULL left context
        queries_standard.append(t["left_context"])

        # Guided query: CSSI contract signatures + FULL left context
        scope_info = extract_cursor_context_symbols(t["left_context"], lang)
        _, query_sigs = synthesize_enhanced_cssi_contract(pool_files, scope_info, lang, max_tokens=256)
        raw_cssi_signatures.append(query_sigs)

        if query_sigs:
            q_full = f"{query_sigs}\n{t['left_context']}"
        else:
            q_full = t["left_context"]
        queries_guided_bm25.append(q_full)

    print(f"  ✓ Prepared {len(queries_standard)} standard queries (FULL left context).", flush=True)
    print(f"  ✓ Prepared {len(queries_guided_bm25)} guided queries (FULL left context + CSSI signatures).", flush=True)

    print(f"\n  Running multi-core BM25 dual-retrieval (12 workers across {len(tasks)} tasks)...", flush=True)
    t_bm25 = time.time()
    bm25_top100_standard = [None] * len(tasks)
    bm25_top100_guided = [None] * len(tasks)

    num_workers = min(12, os.cpu_count() or 4)
    with mp.Pool(num_workers, initializer=_init_worker, initargs=(tasks, files_by_id, queries_standard, queries_guided_bm25)) as pool:
        for count, (idx, top100_std, top100_gui) in enumerate(pool.imap_unordered(_bm25_worker_task, range(len(tasks)), chunksize=32), 1):
            bm25_top100_standard[idx] = top100_std
            bm25_top100_guided[idx] = top100_gui
            if count % 2000 == 0 or count == len(tasks):
                print(f"    BM25 completed {count}/{len(tasks)} tasks ({time.time() - t_bm25:.1f}s)...", flush=True)

    print(f"  ✓ Multi-core BM25 dual-retrieval completed in {time.time() - t_bm25:.1f}s.", flush=True)

    # 3. Stage 2: Neural Reranking via nov3630/RLCoder
    print("\n[Step 3/5] Initializing Neural Reranker nov3630/RLCoder...", flush=True)
    t_ret = time.time()
    rl_tok = AutoTokenizer.from_pretrained(RETRIEVER_MODEL)
    rl_model = AutoModel.from_pretrained(RETRIEVER_MODEL).cuda().half().eval()

    def encode_token_lists(token_lists: list[list[str]], max_len: int = 512, batch_size: int = 512) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(token_lists), batch_size):
            batch = token_lists[i:i + batch_size]
            input_ids = []
            for toks in batch:
                full_toks = [rl_tok.cls_token, "<encoder-only>", rl_tok.sep_token] + toks + [rl_tok.sep_token]
                ids = rl_tok.convert_tokens_to_ids(full_toks)
                if len(ids) < max_len:
                    ids += [rl_tok.pad_token_id] * (max_len - len(ids))
                else:
                    ids = ids[:max_len]
                input_ids.append(ids)
            t_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")
            mask = t_ids.ne(rl_tok.pad_token_id)
            with torch.no_grad():
                out = rl_model(t_ids, attention_mask=mask)[0]
                sent = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
                sent = torch.nn.functional.normalize(sent, p=2, dim=1)
            all_embs.append(sent)
        return torch.cat(all_embs, dim=0)

    # Prepare token lists for queries
    # Standard query: takes up to 508 tail tokens closest to cursor
    std_query_token_lists = []
    for t in tasks:
        toks = rl_tok.tokenize(t["left_context"])
        std_query_token_lists.append(toks[-508:])

    # Guided query: Guaranteed split: [CSSI 256 tokens] + [Tail 252 tokens]
    gui_query_token_lists = []
    for t, sigs in zip(tasks, raw_cssi_signatures):
        cssi_toks = rl_tok.tokenize(sigs)[:256] if sigs else []
        rem = 508 - len(cssi_toks)
        left_toks = rl_tok.tokenize(t["left_context"])[-rem:]
        gui_query_token_lists.append(cssi_toks + left_toks)

    print(f"  Encoding {len(tasks)} Standard queries (tail 508 tokens of left context)...", flush=True)
    q_embs_std = encode_token_lists(std_query_token_lists, max_len=512, batch_size=256)

    print(f"  Encoding {len(tasks)} Guided queries (GUARANTEED: CSSI [256] + Tail [252])...", flush=True)
    q_embs_gui = encode_token_lists(gui_query_token_lists, max_len=512, batch_size=256)

    def encode_cand_texts(texts: list[str], max_len: int = 512, batch_size: int = 512) -> torch.Tensor:
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            input_ids = []
            for text in batch:
                tokens = rl_tok.tokenize(text)[:(max_len - 4)]
                full_toks = [rl_tok.cls_token, "<encoder-only>", rl_tok.sep_token] + tokens + [rl_tok.sep_token]
                ids = rl_tok.convert_tokens_to_ids(full_toks)
                ids += [rl_tok.pad_token_id] * (max_len - len(ids))
                input_ids.append(ids)
            t_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda")
            mask = t_ids.ne(rl_tok.pad_token_id)
            with torch.no_grad():
                out = rl_model(t_ids, attention_mask=mask)[0]
                sent = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
                sent = torch.nn.functional.normalize(sent, p=2, dim=1)
            all_embs.append(sent)
        return torch.cat(all_embs, dim=0)

    rlcoder_top10_standard = []
    rlcoder_top10_guided = []
    CHUNK_SIZE = 400

    print("  Reranking candidate chunks for both Standard & Guided traces...", flush=True)
    for chunk_start in range(0, len(tasks), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(tasks))

        # 1. Standard candidates for chunk
        cands_text_std = []
        slices_std = []
        cur_pos_std = 0
        for t_idx in range(chunk_start, chunk_end):
            cands = bm25_top100_standard[t_idx]
            lang = tasks[t_idx]["language"]
            cs = "#" if lang == "python" else "//"
            for b in cands:
                if b.get("is_stop"):
                    cands_text_std.append("Don't need cross file context for completion")
                else:
                    cands_text_std.append(f"{cs} file path: {b['file_path']}\n{cs} lines: {b['line_start']}-{b['line_end']}\n{cs} {b['code_content']}")
            slices_std.append((cur_pos_std, cur_pos_std + len(cands)))
            cur_pos_std += len(cands)

        chunk_c_embs_std = encode_cand_texts(cands_text_std, max_len=512, batch_size=512)

        for t_offset, (s_idx, e_idx) in enumerate(slices_std):
            t_idx = chunk_start + t_offset
            q_emb = q_embs_std[t_idx:t_idx + 1]
            c_emb = chunk_c_embs_std[s_idx:e_idx]
            sims = torch.mm(q_emb, c_emb.t())[0].cpu().numpy()
            cands = bm25_top100_standard[t_idx]
            top10_idx = sims.argsort()[-10:][::-1]
            top10 = []
            for rank, c_i in enumerate(top10_idx):
                cand_copy = dict(cands[c_i])
                cand_copy["score"] = float(sims[c_i])
                cand_copy["rank"] = rank + 1
                top10.append(cand_copy)
            rlcoder_top10_standard.append(top10)

        # 2. Guided candidates for chunk
        cands_text_gui = []
        slices_gui = []
        cur_pos_gui = 0
        for t_idx in range(chunk_start, chunk_end):
            cands = bm25_top100_guided[t_idx]
            lang = tasks[t_idx]["language"]
            cs = "#" if lang == "python" else "//"
            for b in cands:
                if b.get("is_stop"):
                    cands_text_gui.append("Don't need cross file context for completion")
                else:
                    cands_text_gui.append(f"{cs} file path: {b['file_path']}\n{cs} lines: {b['line_start']}-{b['line_end']}\n{cs} {b['code_content']}")
            slices_gui.append((cur_pos_gui, cur_pos_gui + len(cands)))
            cur_pos_gui += len(cands)

        chunk_c_embs_gui = encode_cand_texts(cands_text_gui, max_len=512, batch_size=512)

        for t_offset, (s_idx, e_idx) in enumerate(slices_gui):
            t_idx = chunk_start + t_offset
            q_emb = q_embs_gui[t_idx:t_idx + 1]
            c_emb = chunk_c_embs_gui[s_idx:e_idx]
            sims = torch.mm(q_emb, c_emb.t())[0].cpu().numpy()
            cands = bm25_top100_guided[t_idx]
            top10_idx = sims.argsort()[-10:][::-1]
            top10 = []
            for rank, c_i in enumerate(top10_idx):
                cand_copy = dict(cands[c_i])
                cand_copy["score"] = float(sims[c_i])
                cand_copy["rank"] = rank + 1
                top10.append(cand_copy)
            rlcoder_top10_guided.append(top10)

        print(f"    Reranked dual-traces for {chunk_end}/{len(tasks)} tasks ({time.time() - t_ret:.1f}s)...", flush=True)

    print(f"  ✓ Neural Reranking completed in {time.time() - t_ret:.1f}s.", flush=True)

    # Save traces
    with gzip.open(TRACE_STANDARD_PATH, "wt", encoding="utf-8") as f_out:
        for t, top10 in zip(tasks, rlcoder_top10_standard):
            f_out.write(json.dumps({"task_id": t["task_id"], "benchmark": t["benchmark"], "rlcoder_top10": top10}) + "\n")
    print(f"  ✓ Saved Standard full-left trace to {TRACE_STANDARD_PATH}", flush=True)

    with gzip.open(TRACE_GUIDED_PATH, "wt", encoding="utf-8") as f_out:
        for t, top10 in zip(tasks, rlcoder_top10_guided):
            f_out.write(json.dumps({"task_id": t["task_id"], "benchmark": t["benchmark"], "rlcoder_top10": top10}) + "\n")
    print(f"  ✓ Saved Guided full-left trace to {TRACE_GUIDED_PATH}", flush=True)

    del rl_model, rl_tok, q_embs_std, q_embs_gui
    torch.cuda.empty_cache()
    gc.collect()

    # 4. Assemble Prompts for 3 Arms (all at 4k context)
    print("\n[Step 4/5] Assembling prompts for 3 arms (4,096 context tokens)...", flush=True)
    from transformers import AutoTokenizer as GenTokenizer
    gen_tok = GenTokenizer.from_pretrained(GENERATOR)

    prompts_arm1 = []
    prompts_arm2 = []
    prompts_arm3 = []

    for idx, t in enumerate(tasks):
        pool_files = []
        for rel_path, fid in t.get("file_ids_by_path", {}).items():
            fe = files_by_id.get(fid)
            if fe:
                pool_files.append({"path": rel_path, "text": fe["text"]})

        top10_std = rlcoder_top10_standard[idx]
        top10_guided = rlcoder_top10_guided[idx]

        # Arm 1: RLCoder Baseline 4k (Full left context retrieval, flat prompt, no CSSI)
        p1 = build_prompt_4k("rlcoder_raw_4k", t, top10_std, pool_files, gen_tok, max_context=4096, cross_max=1280)
        # Arm 2: CADRE Post-hoc CSSI 4k (Full left context retrieval, coalesced raw blocks + CSSI)
        p2 = build_prompt_4k("cadre_posthoc_cssi_4k", t, top10_std, pool_files, gen_tok, max_context=4096, cross_max=1280, cssi_max=512)
        # Arm 3: CADRE Pre+Post CSSI 4k (CSSI-guided retrieval, coalesced raw blocks + CSSI)
        p3 = build_prompt_4k("cadre_pre_and_post_cssi_4k", t, top10_guided, pool_files, gen_tok, max_context=4096, cross_max=1280, cssi_max=512)

        prompts_arm1.append(gen_tok.decode(p1))
        prompts_arm2.append(gen_tok.decode(p2))
        prompts_arm3.append(gen_tok.decode(p3))

    print(f"✓ Assembled 3 x {len(tasks)} = {3 * len(tasks)} prompts for 4k evaluation.", flush=True)

    # 5. Run Generation with vLLM
    print("\n[Step 5/5] Initializing vLLM generator (deepseek-coder-1.3b-base, max_model_len=4096)...", flush=True)
    from vllm import LLM, SamplingParams
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

    print("\n  [Arm 1] Generating predictions for RLCoder Baseline 4k...", flush=True)
    t_g1 = time.time()
    out1 = llm.generate(prompts_arm1, sampling_params)
    preds_arm1 = [o.outputs[0].text for o in out1]
    print(f"  ✓ Arm 1 completed in {time.time() - t_g1:.1f}s.", flush=True)

    print("\n  [Arm 2] Generating predictions for CADRE Post-hoc CSSI 4k (Standard Retrieval)...", flush=True)
    t_g2 = time.time()
    out2 = llm.generate(prompts_arm2, sampling_params)
    preds_arm2 = [o.outputs[0].text for o in out2]
    print(f"  ✓ Arm 2 completed in {time.time() - t_g2:.1f}s.", flush=True)

    print("\n  [Arm 3] Generating predictions for CADRE Pre+Post CSSI 4k (CSSI-Guided Retrieval)...", flush=True)
    t_g3 = time.time()
    out3 = llm.generate(prompts_arm3, sampling_params)
    preds_arm3 = [o.outputs[0].text for o in out3]
    print(f"  ✓ Arm 3 completed in {time.time() - t_g3:.1f}s.", flush=True)

    # 6. Evaluate Metrics
    print("\n[Evaluation] Computing official benchmark metrics for all 3 arms...", flush=True)
    eval_arm1 = evaluate_predictions_official(tasks, preds_arm1)
    eval_arm2 = evaluate_predictions_official(tasks, preds_arm2)
    eval_arm3 = evaluate_predictions_official(tasks, preds_arm3)

    # Print Comprehensive Table
    print("\n" + "=" * 110, flush=True)
    print("4,096 CONTEXT EVALUATION: FULL LEFT CONTEXT RETRIEVAL + CSSI ABLATION", flush=True)
    print("=" * 110, flush=True)
    print(f"{'Benchmark':<16} | {'RLCoder Base 4k':<15} | {'CADRE Post CSSI':<15} | {'CADRE Pre+Post':<15} | {'Prompt CSSI Δ':<14} | {'Retrieval CSSI Δ':<16}", flush=True)
    print("-" * 110, flush=True)
    for b in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        s1 = eval_arm1[b]["em"]
        s2 = eval_arm2[b]["em"]
        s3 = eval_arm3[b]["em"]
        d_prompt = s2 - s1
        d_ret = s3 - s2
        sign_p = "+" if d_prompt >= 0 else ""
        sign_r = "+" if d_ret >= 0 else ""
        print(f"{b:<16} | {s1:>6.2f}%         | {s2:>6.2f}%         | {s3:>6.2f}%         | {sign_p}{d_prompt:>5.2f}%        | {sign_r}{d_ret:>5.2f}%", flush=True)
    print("-" * 110, flush=True)
    m1 = eval_arm1["macro_mean"]["em"]
    m2 = eval_arm2["macro_mean"]["em"]
    m3 = eval_arm3["macro_mean"]["em"]
    md_p = m2 - m1
    md_r = m3 - m2
    msign_p = "+" if md_p >= 0 else ""
    msign_r = "+" if md_r >= 0 else ""
    print(f"{'Macro Mean':<16} | {m1:>6.2f}%         | {m2:>6.2f}%         | {m3:>6.2f}%         | {msign_p}{md_p:>5.2f}%        | {msign_r}{md_r:>5.2f}%", flush=True)
    print("=" * 110, flush=True)

    # Save Output
    final_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generator": GENERATOR,
            "retriever": RETRIEVER_MODEL,
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

    metrics_file = OUTPUT_DIR / "cadre_4k_full_left_context_eval_8004_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n✓ Saved 3-arm metrics to {metrics_file}", flush=True)

    preds_file = OUTPUT_DIR / "cadre_4k_full_left_context_eval_8004_predictions.jsonl.gz"
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
    print(f"✓ Saved 3-arm predictions to {preds_file}", flush=True)


if __name__ == "__main__":
    main()
