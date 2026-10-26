#!/usr/bin/env python3
"""
Full Evaluation of CADRE 4k Context Window & CSSI-Guided Retrieval across all 8,004 tasks.
- Total Budget: 4,096 tokens (instead of 8,000).
- Cross-File Budget Ceiling: 1,280 tokens (guaranteeing >= 2,800 tokens for Left Context).
- Three Arms for Direct Comparison:
    * Arm 1 (RLCoder Baseline 4k): Standard retrieval (query: left_context only), flat prompt, NO CSSI, 4k budget.
    * Arm 2 (CADRE Post-hoc CSSI 4k): Standard retrieval (query: left_context only), Coalesced Raw Blocks + CSSI in prompt, 4k budget.
    * Arm 3 (CADRE Pre+Post CSSI 4k): CSSI-Guided retrieval (query: left_context + CSSI signatures), Coalesced Raw Blocks + CSSI in prompt, 4k budget.
- Generator: deepseek-ai/deepseek-coder-1.3b-base (vLLM, max_model_len=4096, greedy, max_tokens=64).
- Metrics: Official Benchmark Protocols (RepoEval Line-Match, CCEval Statement Match, AIRCoder Prefix Match, ES, ID-F1).
"""

import ast
import difflib
import gc
import gzip
import json
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

TRACE_ARM2_PATH = DATA_ROOT / "rlcoder_full_eval_results" / "rlcoder_full_8004_retrieval_trace.jsonl.gz"
if not TRACE_ARM2_PATH.exists():
    TRACE_ARM2_PATH = Path("experiments/mb_spt_iterative/rlcoder_full_8004_retrieval_trace.jsonl.gz")

TRACE_ARM3_PATH = OUTPUT_DIR / "rlcoder_cssi_guided_full_8004_retrieval_trace.jsonl.gz"

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
    """
    Returns:
        contract_text: Full formatted contract for the prompt.
        query_signatures: Condensed signature string for augmenting the retrieval query.
    """
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

    # Left context gets the remaining budget up to 4,096 tokens
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


def main():
    t_start = time.time()
    print("=" * 95)
    print("CADRE 4K CONTEXT WINDOW & CSSI-GUIDED RETRIEVAL EVALUATION (8,004 TASKS)")
    print("Arm 1: RLCoder Baseline 4k (Query: Left Context only, No CSSI)")
    print("Arm 2: CADRE Post-hoc CSSI 4k (Query: Left Context only, Prompt: +CSSI Contract)")
    print("Arm 3: CADRE Pre+Post CSSI 4k (Query: Left Context + CSSI, Prompt: +CSSI Contract)")
    print("=" * 95)

    # 1. Load Tasks & Files
    print("\n[Step 1/6] Loading benchmark tasks and repository files from bundle...")
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
    print(f"✓ Loaded {len(tasks)} tasks and indexed {len(files_by_id)} repository files.")

    # 2. Load Arm 2 Precomputed Retrieval Trace (Query: Left Context only)
    print(f"\n[Step 2/6] Loading precomputed Standard RLRetriever trace from {TRACE_ARM2_PATH}...")
    rlcoder_top10_standard = []
    with gzip.open(TRACE_ARM2_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rlcoder_top10_standard.append(rec["rlcoder_top10"])
    print(f"✓ Loaded {len(rlcoder_top10_standard)} standard retrieval traces for Arm 1 & Arm 2.")

    # 3. Execute Arm 3 Retrieval (CSSI-Guided Retrieval)
    if TRACE_ARM3_PATH.exists():
        print(f"\n[Step 3/6] Found cached Arm 3 CSSI-guided retrieval trace at {TRACE_ARM3_PATH}. Loading...")
        rlcoder_top10_guided = []
        with gzip.open(TRACE_ARM3_PATH, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rlcoder_top10_guided.append(rec["rlcoder_top10"])
        print(f"✓ Loaded {len(rlcoder_top10_guided)} CSSI-guided retrieval traces.")
    else:
        print("\n[Step 3/6] Running Stage 1 (BM25) & Stage 2 (RLRetriever) with CSSI-Guided Queries...")
        # Prepare queries for Arm 3
        print("  Extracting CSSI contract signatures and building augmented queries...")
        queries_guided = []
        for t in tasks:
            lang = t["language"]
            pool_files = []
            for rel_path, fid in t.get("file_ids_by_path", {}).items():
                fe = files_by_id.get(fid)
                if fe:
                    pool_files.append({"path": rel_path, "text": fe["text"]})
            scope_info = extract_cursor_context_symbols(t["left_context"], lang)
            _, query_sigs = synthesize_enhanced_cssi_contract(pool_files, scope_info, lang, max_tokens=256)
            qlines = [l for l in t["left_context"].splitlines() if l.strip()][-15:]
            q_code = "\n".join(qlines)
            if query_sigs:
                q_full = f"{query_sigs}\n{q_code}"
            else:
                q_full = q_code
            queries_guided.append(q_full)

        print(f"  ✓ Prepared {len(queries_guided)} CSSI-guided queries.")

        # Stage 1: BM25 candidate selection with CSSI-guided queries
        print("  Running Stage 1: BM25 candidate retrieval (Top 100 blocks)...")
        t_bm25 = time.time()
        bm25_top100_guided = []
        for idx, t in enumerate(tasks):
            lang = t["language"]
            pool_blocks = []
            for rel_path, fid in t.get("file_ids_by_path", {}).items():
                fe = files_by_id.get(fid)
                if fe:
                    blocks = split_into_smaller_blocks(rel_path, fe["text"], lang)
                    pool_blocks.extend(blocks)

            if not pool_blocks:
                top100 = []
            else:
                corpus = [b["code_content"].split() for b in pool_blocks]
                bm25 = BM25Okapi(corpus)
                q_toks = queries_guided[idx].split()
                scores = bm25.get_scores(q_toks)
                top_indices = np.argsort(scores)[-100:][::-1]
                top100 = [pool_blocks[i] for i in top_indices if i < len(pool_blocks)]

            top100.append({
                "file_path": "",
                "line_start": 0,
                "line_end": 0,
                "code_content": "Don't need cross file context for completion",
                "language": lang,
                "is_stop": True,
            })
            bm25_top100_guided.append(top100)
            if (idx + 1) % 2000 == 0 or (idx + 1) == len(tasks):
                print(f"    BM25 processed {idx + 1}/{len(tasks)} tasks ({time.time() - t_bm25:.1f}s)...")

        print(f"  ✓ Stage 1 BM25 completed in {time.time() - t_bm25:.1f}s.")

        # Stage 2: RLRetriever neural reranking with CSSI-guided queries
        print("\n  Running Stage 2: Neural Reranking via nov3630/RLCoder...")
        t_ret = time.time()
        rl_tok = AutoTokenizer.from_pretrained(RETRIEVER_MODEL)
        rl_model = AutoModel.from_pretrained(RETRIEVER_MODEL).cuda().half().eval()

        def encode_texts(texts: list[str], is_query: bool, max_len: int, batch_size: int = 512) -> torch.Tensor:
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                input_ids = []
                for text in batch:
                    tokens = rl_tok.tokenize(text)
                    if is_query:
                        tokens = tokens[-(max_len - 4):]
                    else:
                        tokens = tokens[:(max_len - 4)]
                    tokens = [rl_tok.cls_token, "<encoder-only>", rl_tok.sep_token] + tokens + [rl_tok.sep_token]
                    ids = rl_tok.convert_tokens_to_ids(tokens)
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

        print(f"  Encoding {len(queries_guided)} CSSI-guided queries...")
        q_embs = encode_texts(queries_guided, is_query=True, max_len=256, batch_size=256)

        rlcoder_top10_guided = []
        CHUNK_SIZE = 400
        for chunk_start in range(0, len(tasks), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(tasks))
            chunk_cands_text = []
            chunk_slices = []
            cur_pos = 0

            for t_idx in range(chunk_start, chunk_end):
                cands = bm25_top100_guided[t_idx]
                lang = tasks[t_idx]["language"]
                cs = "#" if lang == "python" else "//"
                for b in cands:
                    if b.get("is_stop"):
                        chunk_cands_text.append("Don't need cross file context for completion")
                    else:
                        chunk_cands_text.append(f"{cs} file path: {b['file_path']}\n{cs} lines: {b['line_start']}-{b['line_end']}\n{cs} {b['code_content']}")
                chunk_slices.append((cur_pos, cur_pos + len(cands)))
                cur_pos += len(cands)

            chunk_c_embs = encode_texts(chunk_cands_text, is_query=False, max_len=512, batch_size=512)

            for t_offset, (s_idx, e_idx) in enumerate(chunk_slices):
                t_idx = chunk_start + t_offset
                q_emb = q_embs[t_idx:t_idx + 1]
                c_emb = chunk_c_embs[s_idx:e_idx]
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

            print(f"    Reranked {chunk_end}/{len(tasks)} tasks ({time.time() - t_ret:.1f}s)...")

        print(f"  ✓ Stage 2 RLRetriever completed in {time.time() - t_ret:.1f}s.")

        # Save trace
        with gzip.open(TRACE_ARM3_PATH, "wt", encoding="utf-8") as f_out:
            for t, top10 in zip(tasks, rlcoder_top10_guided):
                f_out.write(json.dumps({
                    "task_id": t["task_id"],
                    "benchmark": t["benchmark"],
                    "rlcoder_top10": top10
                }) + "\n")
        print(f"  ✓ Saved CSSI-guided retrieval trace to {TRACE_ARM3_PATH}.")

        del rl_model, rl_tok, q_embs
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Assemble Prompts for 3 Arms (all at 4k context)
    print("\n[Step 4/6] Assembling prompts for 3 arms (4,096 context tokens)...")
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

        # Arm 1: RLCoder Baseline 4k (Standard retrieval, flat prompt, no CSSI)
        p1 = build_prompt_4k("rlcoder_raw_4k", t, top10_std, pool_files, gen_tok, max_context=4096, cross_max=1280)
        # Arm 2: CADRE Post-hoc CSSI 4k (Standard retrieval, coalesced raw blocks + CSSI)
        p2 = build_prompt_4k("cadre_posthoc_cssi_4k", t, top10_std, pool_files, gen_tok, max_context=4096, cross_max=1280, cssi_max=512)
        # Arm 3: CADRE Pre+Post CSSI 4k (CSSI-guided retrieval, coalesced raw blocks + CSSI)
        p3 = build_prompt_4k("cadre_pre_and_post_cssi_4k", t, top10_guided, pool_files, gen_tok, max_context=4096, cross_max=1280, cssi_max=512)

        prompts_arm1.append(gen_tok.decode(p1))
        prompts_arm2.append(gen_tok.decode(p2))
        prompts_arm3.append(gen_tok.decode(p3))

    print(f"✓ Assembled 3 x {len(tasks)} = {3 * len(tasks)} prompts for 4k evaluation.")

    # 5. Run Generation with vLLM
    print("\n[Step 5/6] Initializing vLLM generator (deepseek-coder-1.3b-base, max_model_len=4096)...")
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

    print("\n  [Arm 1] Generating predictions for RLCoder Baseline 4k...")
    t_g1 = time.time()
    out1 = llm.generate(prompts_arm1, sampling_params)
    preds_arm1 = [o.outputs[0].text for o in out1]
    print(f"  ✓ Arm 1 completed in {time.time() - t_g1:.1f}s.")

    print("\n  [Arm 2] Generating predictions for CADRE Post-hoc CSSI 4k (Standard Retrieval)...")
    t_g2 = time.time()
    out2 = llm.generate(prompts_arm2, sampling_params)
    preds_arm2 = [o.outputs[0].text for o in out2]
    print(f"  ✓ Arm 2 completed in {time.time() - t_g2:.1f}s.")

    print("\n  [Arm 3] Generating predictions for CADRE Pre+Post CSSI 4k (CSSI-Guided Retrieval)...")
    t_g3 = time.time()
    out3 = llm.generate(prompts_arm3, sampling_params)
    preds_arm3 = [o.outputs[0].text for o in out3]
    print(f"  ✓ Arm 3 completed in {time.time() - t_g3:.1f}s.")

    # 6. Evaluate Metrics
    print("\n[Step 6/6] Computing official benchmark metrics for all 3 arms...")
    eval_arm1 = evaluate_predictions_official(tasks, preds_arm1)
    eval_arm2 = evaluate_predictions_official(tasks, preds_arm2)
    eval_arm3 = evaluate_predictions_official(tasks, preds_arm3)

    # Print Comprehensive Table
    print("\n" + "=" * 105)
    print("4,096 CONTEXT EVALUATION: IMPACT OF CSSI AT GENERATION vs CSSI AT RETRIEVAL + GENERATION")
    print("=" * 105)
    print(f"{'Benchmark':<16} | {'RLCoder Base 4k':<15} | {'CADRE Post CSSI':<15} | {'CADRE Pre+Post':<15} | {'Prompt CSSI Δ':<14} | {'Retrieval CSSI Δ':<16}")
    print("-" * 105)
    for b in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        s1 = eval_arm1[b]["em"]
        s2 = eval_arm2[b]["em"]
        s3 = eval_arm3[b]["em"]
        d_prompt = s2 - s1
        d_ret = s3 - s2
        sign_p = "+" if d_prompt >= 0 else ""
        sign_r = "+" if d_ret >= 0 else ""
        print(f"{b:<16} | {s1:>6.2f}%         | {s2:>6.2f}%         | {s3:>6.2f}%         | {sign_p}{d_prompt:>5.2f}%        | {sign_r}{d_ret:>5.2f}%")
    print("-" * 105)
    m1 = eval_arm1["macro_mean"]["em"]
    m2 = eval_arm2["macro_mean"]["em"]
    m3 = eval_arm3["macro_mean"]["em"]
    md_p = m2 - m1
    md_r = m3 - m2
    msign_p = "+" if md_p >= 0 else ""
    msign_r = "+" if md_r >= 0 else ""
    print(f"{'Macro Mean':<16} | {m1:>6.2f}%         | {m2:>6.2f}%         | {m3:>6.2f}%         | {msign_p}{md_p:>5.2f}%        | {msign_r}{md_r:>5.2f}%")
    print("=" * 105)

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

    metrics_file = OUTPUT_DIR / "cadre_4k_cssi_retrieval_full_8004_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n✓ Saved 3-arm metrics to {metrics_file}")

    preds_file = OUTPUT_DIR / "cadre_4k_cssi_retrieval_full_8004_predictions.jsonl.gz"
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
    print(f"✓ Saved 3-arm predictions to {preds_file}")


if __name__ == "__main__":
    main()
