#!/usr/bin/env python3
"""Full 8,004 Benchmark Evaluation: RLCoder Raw Baseline vs RLCoder + CADRE+.

Workflow:
1. Extract candidate blocks from repo files for each task using RLCoder's natural mini-block partition (<=15 lines).
2. BM25 indexes candidate blocks per task and retrieves Top 100 blocks.
3. RLCoder Retriever (nov3630/RLCoder) reranks BM25 Top 100 blocks + stop block to select Top 10.
4. Prompt Construction:
   - Arm 1 (rlcoder_raw_8004): RLCoder official prompt (flat snippets, stop-block filtering, 8k context).
   - Arm 2 (rlcoder_cadre_plus_8004): CADRE+ prompt (LC-Coalescing, Param-Aligned CSSI Channel B, DCA 2-channel, 8k context).
5. Generation via vLLM:
   - deepseek-ai/deepseek-coder-1.3b-base, greedy decoding (T=0.0), max_tokens=64.
6. Metric Evaluation across all 8,004 tasks (CCEval Python, CCEval Java, RepoEval API, RepoEval Line):
   - AlignCoder Tree-sitter statement match (strict AST)
   - AIRCoder prefix truncation match [:len(target)]
   - Exact string match uncut
   - Edit similarity (SequenceMatcher ratio * 100)
   - Identifier EM & Identifier F1
"""

from __future__ import annotations

import difflib
import gc
import gzip
import json
import keyword
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
from transformers import AutoTokenizer, AutoModel

ROOT_DIR = Path("/content/mb_spt")
GENERATOR = "deepseek-ai/deepseek-coder-1.3b-base"
GENERATOR_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
RETRIEVER_MODEL = "nov3630/RLCoder"
BUNDLE_PATH = ROOT_DIR / "full_benchmark_8004_bundle.tar.gz"
OUTPUT_DIR = ROOT_DIR / "rlcoder_full_eval_results"

ID_RE = re.compile(r"\b[_A-Za-z][_A-Za-z0-9]*\b")
WORD_RE = re.compile(r"\w+")

STOP_SYMS = {
    "self", "this", "None", "True", "False", "null", "return", "args", "kwargs",
    "init", "__init__", "super", "cls", "pass", "break", "continue", "from",
    "import", "class", "def", "void", "public", "private", "protected", "static", "final",
    "else", "if", "for", "in", "with", "try", "except", "finally", "while",
}

_KW_CACHE: dict[str, set[str]] = {}

def get_keywords(language: str) -> set[str]:
    if language not in _KW_CACHE:
        if language == "python":
            _KW_CACHE[language] = set(keyword.kwlist) - {"True", "False"}
        else:
            kw_path = ROOT_DIR / "aligncoder" / "utils" / "keywords" / f"{language}.txt"
            if kw_path.exists():
                _KW_CACHE[language] = {
                    l.strip() for l in kw_path.read_text(encoding="utf-8").splitlines() if l.strip()
                }
            else:
                _KW_CACHE[language] = set()
    return _KW_CACHE[language]


def extract_identifiers(source_code: str, language: str) -> list[str]:
    clean = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'', "", source_code)
    kws = get_keywords(language)
    return [
        t for t in WORD_RE.findall(clean)
        if re.match(r"[_a-zA-Z][_a-zA-Z0-9]*", t) and t not in kws
    ]


def cal_edit_sim_single(pred: str, gt: str) -> float:
    try:
        from fuzzywuzzy import fuzz
        return fuzz.ratio(pred.strip(), gt.strip())
    except Exception:
        return difflib.SequenceMatcher(None, pred.strip(), gt.strip()).ratio() * 100.0


def is_parse_valid(parser: Any, code: str) -> bool:
    try:
        tree = parser.parse(bytes(code, "utf-8"))
        if tree is None:
            return False
        def syntax_error(node: Any) -> bool:
            if getattr(node, "type", "") == "ERROR" or getattr(node, "is_error", False):
                return True
            try:
                for child in getattr(node, "children", []):
                    if syntax_error(child):
                        return True
            except RecursionError:
                return True
            return False
        return not syntax_error(tree.root_node)
    except Exception:
        return False


def get_python_one_statement(prompt: str, completion: str, parser: Any) -> str:
    for i in range(len(completion)):
        code = prompt + completion[:i + 1]
        if "\n" in completion[:i + 1] and is_parse_valid(parser, code):
            return completion[:i + 1]
    return completion


def get_java_one_statement(completion: str, parser: Any) -> str:
    for i in range(len(completion)):
        if completion[i] == ";" and is_parse_valid(parser, "class Dummy { void dummy() { " + completion[:i + 1] + " } }"):
            return completion[:i + 1]
    lines = completion.splitlines(keepends=True)
    return lines[0] if lines else completion


def truncate_completion(prompt: str, completion: str, language: str, parser: Any) -> str:
    if parser is None:
        lines = completion.splitlines(keepends=True)
        return lines[0] if lines else completion
    try:
        if language == "python":
            return get_python_one_statement(prompt, completion, parser)
        else:
            return get_java_one_statement(completion, parser)
    except Exception:
        lines = completion.splitlines(keepends=True)
        return lines[0] if lines else completion


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
    mini_blocks = temp_mini_blocks

    smaller_blocks = []
    current_content = []
    total_lines = 0
    for block in mini_blocks:
        if total_lines >= 5000:
            break
        if len(current_content) + len(block) <= 15:
            current_content.extend(block)
            total_lines += len(block)
        else:
            if current_content:
                smaller_blocks.append({
                    "file_path": file_path,
                    "line_start": total_lines - len(current_content) + 1,
                    "line_end": total_lines,
                    "code_content": "\n".join(current_content),
                    "language": language,
                })
            current_content = block
            total_lines += len(block)
    if current_content:
        smaller_blocks.append({
            "file_path": file_path,
            "line_start": total_lines - len(current_content) + 1,
            "line_end": total_lines,
            "code_content": "\n".join(current_content),
            "language": language,
        })
    return smaller_blocks


def extract_cursor_context_symbols(left_context: str, language: str) -> dict[str, Any]:
    tail = left_context[-1200:]
    lines = tail.splitlines()
    nonempty = [l for l in lines if l.strip()]

    receivers = []
    for m in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)?$", tail):
        receivers.append(m.group(1))

    constructors = []
    for m in re.finditer(r"(?:new\s+|=)\s*([A-Z][a-zA-Z0-9_]*)\s*\(", tail):
        constructors.append(m.group(1))

    bases = []
    for line in nonempty[-30:]:
        m = re.search(r"class\s+[A-Za-z0-9_]+\s*\((.*?)\):", line)
        if m:
            for b in m.group(1).split(","):
                b = b.strip()
                if b: bases.append(b)
        m_java = re.search(r"class\s+[A-Za-z0-9_]+\s+(?:extends|implements)\s+([^{]+)", line)
        if m_java:
            for b in re.split(r"[, ]+", m_java.group(1)):
                b = b.strip()
                if b and b not in ("extends", "implements"): bases.append(b)

    imports = []
    if language == "python":
        for m in re.finditer(r"(?:from\s+[a-zA-Z0-9_.]+\s+)?import\s+([a-zA-Z0-9_,\s]+)", left_context):
            for part in m.group(1).split(","):
                part_tokens = part.strip().split()
                if part_tokens: imports.append(part_tokens[0])
    elif language == "java":
        for m in re.finditer(r"import\s+([A-Za-z0-9_.]+);", left_context):
            parts = m.group(1).split(".")
            if parts: imports.append(parts[-1])

    var_types: dict[str, str] = {}
    for line in nonempty[-25:]:
        m = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:new\s+)?([A-Z][a-zA-Z0-9_]*)\s*\(", line)
        if m: var_types[m.group(1)] = m.group(2)

    return {
        "receivers": receivers,
        "constructors": constructors,
        "bases": bases,
        "imports": imports,
        "var_types": var_types,
    }


def extract_cssi_contract(symbols: dict[str, Any], pool_files: list[dict[str, Any]], language: str, max_tokens: int = 300, tok: Any = None) -> str:
    target_names: set[str] = set()
    for r in symbols["receivers"]:
        if r in symbols["var_types"]: target_names.add(symbols["var_types"][r])
        elif r in ("self", "this"): target_names.update(symbols["bases"])
        else: target_names.add(r)
    target_names.update(symbols["constructors"])
    target_names.update(symbols["bases"])
    target_names.update(symbols["imports"])
    target_names = {t for t in target_names if t not in STOP_SYMS and len(t) >= 3}
    if not target_names: return ""

    stubs = []
    comment_sym = "#" if language == "python" else "//"

    for pf in pool_files:
        text = pf.get("text", "") or pf.get("content", "")
        if not text: continue
        p = pf.get("path", "")
        stem = Path(p).stem

        matched = {t for t in target_names if t == stem or f"class {t}" in text or f"interface {t}" in text}
        if not matched: continue

        lines = text.splitlines()
        extracted_lines = []
        in_class = False

        for line in lines:
            if not in_class:
                m_cls = re.search(r"^(\s*)(?:public\s+|abstract\s+)*(?:class|interface)\s+([A-Za-z0-9_]+)", line)
                if m_cls and m_cls.group(2) in target_names:
                    in_class = True
                    extracted_lines.append(line.rstrip())
                    continue
            else:
                if language == "python":
                    m_attr = re.match(r"^(\s{4,8})([A-Z_0-9]{3,}|self\.[a-z_0-9]+)\s*=", line)
                    if m_attr and len(extracted_lines) < 8:
                        extracted_lines.append(line.rstrip())
                    m_def = re.match(r"^(\s{4,8})(?:async\s+)?def\s+([a-zA-Z_0-9]+)\s*\((.*?)\):", line)
                    if m_def:
                        sig_name = m_def.group(2)
                        params = m_def.group(3)
                        extracted_lines.append(f"{m_def.group(1)}def {sig_name}({params}): ...")
                elif language == "java":
                    m_mth = re.match(r"^(\s{4,8})(?:public\s+|protected\s+|static\s+|final\s+)*([\w<>\[\]]+)\s+([a-zA-Z_0-9]+)\s*\((.*?)\)", line)
                    if m_mth:
                        ret_type = m_mth.group(2)
                        sig_name = m_mth.group(3)
                        params = m_mth.group(4)
                        extracted_lines.append(f"{m_mth.group(1)}{ret_type} {sig_name}({params});")

                if len(extracted_lines) >= 15: break

        if extracted_lines:
            stub_text = f"{comment_sym} [CADRE-CSSI Contract: {p}]\n" + "\n".join(
                f"{comment_sym} {l}" for l in extracted_lines
            )
            stubs.append(stub_text)
            if len(stubs) >= 2: break

    if not stubs: return ""
    contract = "\n\n".join(stubs)
    if tok is not None:
        c_ids = tok.encode(contract, add_special_tokens=False)
        if len(c_ids) > max_tokens:
            c_ids = c_ids[:max_tokens]
            contract = tok.decode(c_ids, skip_special_tokens=True)
    return contract


def coalesce_commented_chunks(chunks: list[dict[str, Any]], language: str) -> list[str]:
    comment_sym = "#" if language == "python" else "//"
    file_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        p = c.get("path") or c.get("file_path", "")
        if not p or c.get("is_stop") or p == "file_path": continue
        file_chunks[p].append(c)

    blocks = []
    for p, f_chunk_list in file_chunks.items():
        f_chunk_list.sort(key=lambda x: int(x.get("line_start", 0)))
        merged_lines = []
        last_end = -1
        range_start = -1

        for c in f_chunk_list:
            s = int(c.get("line_start", 0))
            e = int(c.get("line_end", 0))
            code_str = c.get("code") or c.get("code_content", "") or c.get("text", "")
            c_lines = code_str.splitlines()
            if last_end != -1 and s <= last_end + 3:
                merged_lines.extend(c_lines)
                last_end = max(last_end, e)
            else:
                if merged_lines:
                    hdr = f"{comment_sym} file path: {p}\n{comment_sym} lines: {range_start}-{last_end}"
                    body = "\n".join(f"{comment_sym} {l}" for l in merged_lines if l.strip())
                    blocks.append(f"{hdr}\n{body}")
                range_start = s
                last_end = e
                merged_lines = list(c_lines)

        if merged_lines:
            hdr = f"{comment_sym} file path: {p}\n{comment_sym} lines: {range_start}-{last_end}"
            body = "\n".join(f"{comment_sym} {l}" for l in merged_lines if l.strip())
            blocks.append(f"{hdr}\n{body}")

    return blocks


def build_rlcoder_raw_prompt(task: dict[str, Any], top10: list[dict[str, Any]], tok: Any, max_context: int = 8000) -> list[int]:
    lang = task["language"]
    cs = "#" if lang == "python" else "//"
    
    # Filter stop block
    filtered = []
    for b in top10:
        if b.get("is_stop") or not b.get("file_path"):
            break
        filtered.append(b)

    # Format flat blocks
    block_texts = []
    for b in filtered:
        desc = f"{cs} file path: {b['file_path']}\n{cs} lines: {b.get('line_start', 0)}-{b.get('line_end', 0)}"
        code_lines = "\n".join(f"{cs} {l}" for l in b["code_content"].splitlines() if l.strip())
        block_texts.append(f"{desc}\n{code_lines}")

    cross_text = "\n\n".join(block_texts)
    cross_ids = tok.encode(cross_text, add_special_tokens=False)[:3584]

    path_text = f"\n\n{cs} file path: {task['path']}\n\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)

    allowed_left = max(0, max_context - len(cross_ids) - len(path_ids))
    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []

    return (cross_ids + path_ids + left_ids)[-max_context:]


def build_cadre_plus_prompt(task: dict[str, Any], top10: list[dict[str, Any]], pool_files: list[dict[str, Any]], tok: Any, max_context: int = 8000) -> list[int]:
    lang = task["language"]
    comment_sym = "#" if lang == "python" else "//"

    # Filter stop block
    filtered = [b for b in top10 if not b.get("is_stop") and b.get("file_path")]

    path_text = f"\n\n{comment_sym} file path: {task['path']}\n\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)
    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)

    # Channel B: CSSI Contract
    symbols = extract_cursor_context_symbols(task["left_context"], lang)
    cssi_contract = extract_cssi_contract(symbols, pool_files, lang, max_tokens=300, tok=tok)
    cssi_ids = tok.encode("\n\n" + cssi_contract, add_special_tokens=False) if cssi_contract else []

    # Channel A: Coalesced Exemplars
    rem_cross = max(0, 3584 - len(cssi_ids))
    blocks = coalesce_commented_chunks(filtered, lang)
    exemplar_text = "\n\n".join(blocks)
    exemplar_ids = tok.encode(exemplar_text, add_special_tokens=False)[:rem_cross]

    cross_ids = exemplar_ids + cssi_ids
    allowed_left = max(0, max_context - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []

    return (cross_ids + path_ids + left_ids)[-max_context:]


def evaluate_predictions(tasks: list[dict[str, Any]], predictions: list[str], py_parser: Any, java_parser: Any) -> dict[str, Any]:
    by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t, pred in zip(tasks, predictions):
        bench = t["benchmark"]
        lang = t["language"]
        parser = py_parser if lang == "python" else java_parser
        gt = t["groundtruth"]

        # 1. AlignCoder AST statement match
        pred_stmt = truncate_completion(t["left_context"], pred, lang, parser)
        gt_stmt = truncate_completion(t["left_context"], gt, lang, parser)
        aligncoder_em = int(pred_stmt.strip() == gt_stmt.strip())

        # 2. AIRCoder Prefix Match
        gt_flat = "".join(gt.split())
        pred_flat = "".join(pred.split())
        min_len = min(len(gt_flat), len(pred_flat))
        if min_len == 0:
            aircoder_em = int(len(gt_flat) == len(pred_flat))
        else:
            aircoder_em = int(pred_flat[:min_len] == gt_flat[:min_len])

        # 3. Exact String Match Uncut
        exact_uncut = int(pred.strip() == gt.strip())

        # 4. Edit Similarity
        es = cal_edit_sim_single(pred, gt)

        # 5. Identifier EM & F1
        gt_ids = extract_identifiers(gt, lang)
        pred_ids = extract_identifiers(pred, lang)
        gt_set, pred_set = set(gt_ids), set(pred_ids)
        id_em = int(gt_set == pred_set) if gt_set else int(len(pred_set) == 0)
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
            "aligncoder_em": aligncoder_em,
            "aircoder_em": aircoder_em,
            "exact_uncut": exact_uncut,
            "edit_sim": es,
            "id_em": id_em,
            "id_f1": id_f1,
            "pred": pred,
            "pred_stmt": pred_stmt,
            "gt_stmt": gt_stmt,
        })

    summary: dict[str, Any] = {}
    macro_aligncoder_em = []
    macro_aircoder_em = []
    macro_es = []
    macro_id_f1 = []

    for b_name in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        items = by_bench[b_name]
        n = len(items)
        if n == 0: continue
        ac_em = sum(x["aligncoder_em"] for x in items) / n * 100.0
        air_em = sum(x["aircoder_em"] for x in items) / n * 100.0
        ex_un = sum(x["exact_uncut"] for x in items) / n * 100.0
        es = sum(x["edit_sim"] for x in items) / n
        id_em = sum(x["id_em"] for x in items) / n * 100.0
        id_f1 = sum(x["id_f1"] for x in items) / n

        summary[b_name] = {
            "samples": n,
            "aligncoder_ast_em": round(ac_em, 2),
            "aircoder_prefix_em": round(air_em, 2),
            "exact_uncut": round(ex_un, 2),
            "edit_sim": round(es, 2),
            "id_em": round(id_em, 2),
            "id_f1": round(id_f1, 2),
        }
        macro_aligncoder_em.append(ac_em)
        macro_aircoder_em.append(air_em)
        macro_es.append(es)
        macro_id_f1.append(id_f1)

    summary["macro_mean"] = {
        "aligncoder_ast_em": round(sum(macro_aligncoder_em) / len(macro_aligncoder_em), 2),
        "aircoder_prefix_em": round(sum(macro_aircoder_em) / len(macro_aircoder_em), 2),
        "edit_sim": round(sum(macro_es) / len(macro_es), 2),
        "id_f1": round(sum(macro_id_f1) / len(macro_id_f1), 2),
    }

    return {"summary": summary, "per_sample": by_bench}


def main():
    print("=" * 80)
    print("FULL 8,004 BENCHMARK EVALUATION: RLCODER RAW BASELINE vs RLCODER + CADRE+")
    print("=" * 80)
    t_start = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Parsers
    from tree_sitter import Language, Parser
    py_parser, java_parser = None, None
    try:
        py_lang = Language(str(ROOT_DIR / "parsers" / "python-lang-parser.so"), "python")
        py_parser = Parser()
        py_parser.set_language(py_lang)
        java_lang = Language(str(ROOT_DIR / "parsers" / "java-lang-parser.so"), "java")
        java_parser = Parser()
        java_parser.set_language(java_lang)
        print("✓ Tree-sitter parsers loaded successfully.")
    except Exception as e:
        print(f"⚠ Warning: Could not load tree-sitter parsers ({e}). Falling back to line truncation.")

    # 2. Load Bundle
    print("\n[Step 1/6] Loading full benchmark bundle (8,004 tasks)...")
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

    print(f"✓ Loaded {len(tasks)} tasks across 4 benchmarks.")
    print(f"✓ Indexed {len(files_by_id)} repository files.")

    trace_path = OUTPUT_DIR / "rlcoder_full_8004_retrieval_trace.jsonl.gz"
    if trace_path.exists():
        print(f"\n[Step 2 & 3] Found precomputed retrieval trace at {trace_path}! Loading directly...")
        rlcoder_top10_by_task = []
        with gzip.open(trace_path, "rt", encoding="utf-8") as f_in:
            for line in f_in:
                d = json.loads(line)
                rlcoder_top10_by_task.append(d["rlcoder_top10"])
        assert len(rlcoder_top10_by_task) == len(tasks), f"Trace length {len(rlcoder_top10_by_task)} != {len(tasks)}"
        print(f"✓ Loaded RLRetriever top-10 candidates for all {len(tasks)} tasks.")
    else:
        # 3. Stage 1: BM25 Top 100 Candidate Extraction
        print("\n[Step 2/6] Running Stage 1: Task-Specific BM25 (Top 100 blocks per task)...")
        t_bm25 = time.time()
        bm25_top100_by_task: list[list[dict[str, Any]]] = []

        for idx, t in enumerate(tasks):
            lang = t["language"]
            blocks = []
            for rel_path, fid in t.get("file_ids_by_path", {}).items():
                f_entry = files_by_id.get(fid)
                if not f_entry: continue
                file_blocks = split_into_smaller_blocks(rel_path, f_entry["text"], lang)
                blocks.extend(file_blocks)

            if not blocks:
                # Fallback if repo has no other files
                blocks = [{
                    "file_path": t["path"],
                    "line_start": 1,
                    "line_end": 10,
                    "code_content": t["left_context"][-200:],
                    "language": lang,
                }]

            corpus = [b["code_content"].lower().split() for b in blocks]
            bm25 = BM25Okapi(corpus)
            query_lines = [l for l in t["left_context"].splitlines() if l.strip()][-20:]
            query_tokens = "\n".join(query_lines).lower().split()
            scores = bm25.get_scores(query_tokens)
            top100_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:100]
            top100 = [blocks[i] for i in top100_idx]

            # Add RLCoder stop block
            top100.append({
                "file_path": "",
                "line_start": 0,
                "line_end": 0,
                "code_content": "Don't need cross file context for completion",
                "language": lang,
                "is_stop": True,
            })
            bm25_top100_by_task.append(top100)

            if (idx + 1) % 2000 == 0 or (idx + 1) == len(tasks):
                print(f"  Processed {idx + 1}/{len(tasks)} tasks with BM25 ({time.time() - t_bm25:.1f}s)")

        print(f"✓ Stage 1 BM25 completed in {time.time() - t_bm25:.1f}s.")

        # 4. Stage 2: RLRetriever Neural Reranking via nov3630/RLCoder
        print("\n[Step 3/6] Running Stage 2: RLRetriever (nov3630/RLCoder) Neural Reranking...")
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

        # Encode all queries
        queries = []
        for t in tasks:
            qlines = [l for l in t["left_context"].splitlines() if l.strip()][-20:]
            queries.append("\n".join(qlines))

        print(f"  Encoding {len(queries)} task queries...")
        q_embs = encode_texts(queries, is_query=True, max_len=256, batch_size=256)

        # Process tasks in chunks to rerank candidates
        rlcoder_top10_by_task = []
        CHUNK_SIZE = 400
        for chunk_start in range(0, len(tasks), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(tasks))
            chunk_cands_text = []
            chunk_slices = []
            cur_pos = 0

            for t_idx in range(chunk_start, chunk_end):
                cands = bm25_top100_by_task[t_idx]
                lang = tasks[t_idx]["language"]
                cs = "#" if lang == "python" else "//"
                for b in cands:
                    if b.get("is_stop"):
                        chunk_cands_text.append("Don't need cross file context for completion")
                    else:
                        chunk_cands_text.append(f"{cs} file path: {b['file_path']}\n{cs} lines: {b['line_start']}-{b['line_end']}\n{cs} {b['code_content']}")
                chunk_slices.append((cur_pos, cur_pos + len(cands)))
                cur_pos += len(cands)

            # Batch encode candidates
            chunk_c_embs = encode_texts(chunk_cands_text, is_query=False, max_len=512, batch_size=512)

            for t_offset, (s_idx, e_idx) in enumerate(chunk_slices):
                t_idx = chunk_start + t_offset
                q_emb = q_embs[t_idx:t_idx + 1] # [1, D]
                c_emb = chunk_c_embs[s_idx:e_idx] # [K, D]
                sims = torch.mm(q_emb, c_emb.t())[0].cpu().numpy()

                cands = bm25_top100_by_task[t_idx]
                top10_idx = sims.argsort()[-10:][::-1]
                top10 = []
                for rank, c_i in enumerate(top10_idx):
                    cand_copy = dict(cands[c_i])
                    cand_copy["score"] = float(sims[c_i])
                    cand_copy["rank"] = rank + 1
                    top10.append(cand_copy)
                rlcoder_top10_by_task.append(top10)

            print(f"  Reranked {chunk_end}/{len(tasks)} tasks ({time.time() - t_ret:.1f}s)...")

        print(f"✓ Stage 2 RLRetriever completed in {time.time() - t_ret:.1f}s.")

        # Save retrieval trace
        print(f"  Saving retrieval trace to {trace_path}...")
        with gzip.open(trace_path, "wt", encoding="utf-8") as f_out:
            for t, top10 in zip(tasks, rlcoder_top10_by_task):
                f_out.write(json.dumps({
                    "task_id": t["task_id"],
                    "benchmark": t["benchmark"],
                    "rlcoder_top10": top10
                }) + "\n")

        # Clean up retriever model from GPU
        del rl_model, rl_tok, q_embs
        torch.cuda.empty_cache()
        gc.collect()
        print("✓ Cleared RLRetriever from GPU memory.")

    # 5. Build Prompts for Both Arms
    print("\n[Step 4/6] Assembling prompts for Arm 1 (RLCoder Raw) and Arm 2 (RLCoder + CADRE+)...")
    from transformers import AutoTokenizer as GenTokenizer
    gen_tok = GenTokenizer.from_pretrained(GENERATOR)

    raw_prompts = []
    cadre_prompts = []

    for t, top10 in zip(tasks, rlcoder_top10_by_task):
        # Pool files for CSSI
        pool_files = []
        for rel_path, fid in t.get("file_ids_by_path", {}).items():
            fe = files_by_id.get(fid)
            if fe: pool_files.append({"path": rel_path, "text": fe["text"]})

        raw_ids = build_rlcoder_raw_prompt(t, top10, gen_tok, max_context=8000)
        raw_prompts.append(gen_tok.decode(raw_ids))

        cadre_ids = build_cadre_plus_prompt(t, top10, pool_files, gen_tok, max_context=8000)
        cadre_prompts.append(gen_tok.decode(cadre_ids))

    print(f"✓ Assembled {len(raw_prompts)} prompts for RLCoder Raw Baseline.")
    print(f"✓ Assembled {len(cadre_prompts)} prompts for RLCoder + CADRE+.")

    # 6. Initialize vLLM Generator
    print("\n[Step 5/6] Initializing vLLM generator (deepseek-coder-1.3b-base, max_model_len=8192)...")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=GENERATOR,
        revision=GENERATOR_REVISION,
        max_model_len=8192,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        stop=["\n\n\n"],
    )

    # Generate Arm 1: RLCoder Raw Baseline
    print("\n  [Arm 1] Generating predictions for RLCoder Raw Baseline (8,004 tasks)...")
    t_gen_raw = time.time()
    raw_outputs = llm.generate(raw_prompts, sampling_params)
    raw_preds = [o.outputs[0].text for o in raw_outputs]
    print(f"  ✓ RLCoder Raw Baseline generated in {time.time() - t_gen_raw:.1f}s.")

    # Generate Arm 2: RLCoder + CADRE+
    print("\n  [Arm 2] Generating predictions for RLCoder + CADRE+ (8,004 tasks)...")
    t_gen_cadre = time.time()
    cadre_outputs = llm.generate(cadre_prompts, sampling_params)
    cadre_preds = [o.outputs[0].text for o in cadre_outputs]
    print(f"  ✓ RLCoder + CADRE+ generated in {time.time() - t_gen_cadre:.1f}s.")

    # 7. Evaluate Metrics
    print("\n[Step 6/6] Computing evaluation metrics for both arms across all 8,004 tasks...")
    eval_raw = evaluate_predictions(tasks, raw_preds, py_parser, java_parser)
    eval_cadre = evaluate_predictions(tasks, cadre_preds, py_parser, java_parser)

    # Print Comparison Table
    print("\n" + "=" * 80)
    print("FINAL 8,004 BENCHMARK RESULTS: RLCODER RAW vs RLCODER + CADRE+")
    print("=" * 80)
    print(f"{'Benchmark':<18} | {'RLCoder Raw (AST EM)':<22} | {'RLCoder + CADRE+ (AST EM)':<26} | {'Delta':<10}")
    print("-" * 80)
    for b in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        s_raw = eval_raw["summary"][b]["aligncoder_ast_em"]
        s_cadre = eval_cadre["summary"][b]["aligncoder_ast_em"]
        delta = s_cadre - s_raw
        sign = "+" if delta >= 0 else ""
        print(f"{b:<18} | {s_raw:>6.2f}%                 | {s_cadre:>6.2f}%                   | {sign}{delta:>5.2f}%")
    print("-" * 80)
    m_raw = eval_raw["summary"]["macro_mean"]["aligncoder_ast_em"]
    m_cadre = eval_cadre["summary"]["macro_mean"]["aligncoder_ast_em"]
    m_delta = m_cadre - m_raw
    m_sign = "+" if m_delta >= 0 else ""
    print(f"{'Macro Mean':<18} | {m_raw:>6.2f}%                 | {m_cadre:>6.2f}%                   | {m_sign}{m_delta:>5.2f}%")
    print("=" * 80)

    # Print AIRCoder Prefix EM Comparison
    print(f"\n{'Benchmark':<18} | {'RLCoder Raw (Prefix EM)':<24} | {'RLCoder + CADRE+ (Prefix EM)':<28} | {'Delta':<10}")
    print("-" * 80)
    for b in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        p_raw = eval_raw["summary"][b]["aircoder_prefix_em"]
        p_cadre = eval_cadre["summary"][b]["aircoder_prefix_em"]
        p_delta = p_cadre - p_raw
        sign = "+" if p_delta >= 0 else ""
        print(f"{b:<18} | {p_raw:>6.2f}%                   | {p_cadre:>6.2f}%                     | {sign}{p_delta:>5.2f}%")
    print("-" * 80)
    pm_raw = eval_raw["summary"]["macro_mean"]["aircoder_prefix_em"]
    pm_cadre = eval_cadre["summary"]["macro_mean"]["aircoder_prefix_em"]
    pm_delta = pm_cadre - pm_raw
    pm_sign = "+" if pm_delta >= 0 else ""
    print(f"{'Macro Mean':<18} | {pm_raw:>6.2f}%                   | {pm_cadre:>6.2f}%                     | {pm_sign}{pm_delta:>5.2f}%")
    print("=" * 80)

    # Save Metrics & Predictions
    final_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "generator": GENERATOR,
            "retriever": RETRIEVER_MODEL,
            "total_tasks": len(tasks),
            "max_tokens": 64,
            "context_tokens": 8000,
            "duration_sec": round(time.time() - t_start, 1),
        },
        "arms": {
            "rlcoder_raw_8004": eval_raw["summary"],
            "rlcoder_cadre_plus_8004": eval_cadre["summary"],
        },
    }

    metrics_file = OUTPUT_DIR / "rlcoder_cadre_full_8004_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n✓ Saved metrics to {metrics_file}")

    preds_file = OUTPUT_DIR / "rlcoder_cadre_full_8004_predictions.jsonl.gz"
    with gzip.open(preds_file, "wt", encoding="utf-8") as gz:
        for t, raw_p, cadre_p in zip(tasks, raw_preds, cadre_preds):
            gz.write(json.dumps({
                "task_id": t["task_id"],
                "benchmark": t["benchmark"],
                "groundtruth": t["groundtruth"],
                "pred_raw": raw_p,
                "pred_cadre": cadre_p,
            }) + "\n")
    print(f"✓ Saved predictions to {preds_file}")

    print(f"\nTotal execution time: {time.time() - t_start:.1f}s")
    print("Done!")

if __name__ == "__main__":
    main()
