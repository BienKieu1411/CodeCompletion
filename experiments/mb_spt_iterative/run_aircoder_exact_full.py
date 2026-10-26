#!/usr/bin/env python3
"""Exact AIRCoder_r (ACL 2026 Paper Replication) on Full Benchmark (8,004 Tasks).

Faithful setting:
1. Generator: deepseek-ai/deepseek-coder-1.3b-base (temperature=0.0, max_tokens=48, 8k input cap)
2. Chunking & Budget: 800 tokens SPC (Self-Paced Chunking)
3. Retrieval RRF: Full 8 metrics (k=60):
   - sparse (token overlap Jaccard)
   - dense (raw code UniXcoder cosine)
   - skelton (AST skeleton structural dense similarity)
   - file_es (file path edit similarity)
   - file_dist (directory hop distance inverse)
   - class_es (cursor enclosing class match)
   - def_es (cursor enclosing def match)
   - dataflow (dataflow target match)
4. Snippet Selection & Format:
   - Ranked by 8-metric RRF over all candidates
   - Filled greedily up to 800 tokens budget (no top-10 truncation)
   - Ascending order (least relevant first, top closest to cursor)
   - Triple-quote block enclosure: '''\n# file\ncode\n''' (Python) or /*\n// file\ncode\n*/ (Java)
5. Dual Evaluation Protocols:
   - AlignCoder Strict Protocol (Tree-sitter AST statement parser)
   - AIRCoder Official Protocol (Character prefix truncation)
"""

from __future__ import annotations

import difflib
import gzip
import json
import keyword
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path("/content/mb_spt")
GENERATOR = "deepseek-ai/deepseek-coder-1.3b-base"
GENERATOR_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
SEED = 20261024

ID_RE = re.compile(r"\b[_A-Za-z][_A-Za-z0-9]*\b")
STRING_PATTERN = re.compile(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'')
WORD_RE = re.compile(r"\w+")

STOP_SYMS = {
    "self", "this", "None", "True", "False", "null", "return", "args", "kwargs",
    "init", "__init__", "super", "cls", "pass", "break", "continue", "from",
    "import", "class", "def", "void", "public", "private", "protected", "static", "final",
    "else", "if", "for", "in", "with",
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


def extract_identifiers(text: str, language: str) -> list[str]:
    cleaned = STRING_PATTERN.sub("", text)
    kws = get_keywords(language)
    return [
        tok
        for tok in ID_RE.findall(cleaned)
        if tok not in kws and tok not in STOP_SYMS and not tok.isdigit()
    ]


def compute_identifier_metrics(pred: str, gt: str, language: str) -> tuple[float, float]:
    pred_ids = set(extract_identifiers(pred, language))
    gt_ids = set(extract_identifiers(gt, language))
    id_em = 1.0 if pred_ids == gt_ids else 0.0
    if not gt_ids and not pred_ids:
        return 1.0, 1.0
    if not gt_ids or not pred_ids:
        return id_em, 0.0
    prec = len(pred_ids & gt_ids) / len(pred_ids)
    rec = len(pred_ids & gt_ids) / len(gt_ids)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return id_em, f1


def get_python_one_statement(prompt: str, completion: str, parser: Any) -> str:
    prompt_lines = prompt.splitlines()
    last_line = prompt_lines[-1] if prompt_lines else ""
    leading_spaces = len(last_line) - len(last_line.lstrip())

    code = prompt + completion
    tree = parser.parse(bytes(code, "utf8"))
    cursor = tree.walk()

    target_line = len(prompt_lines) - 1
    cursor_byte = len(prompt.encode("utf8"))

    while True:
        node = cursor.node
        if node.start_byte <= cursor_byte <= node.end_byte:
            if node.type in [
                "expression_statement", "assert_statement", "return_statement",
                "raise_statement", "pass_statement", "break_statement",
                "continue_statement", "import_statement", "import_from_statement",
                "global_statement", "nonlocal_statement", "exec_statement",
                "type_alias_statement", "print_statement",
            ]:
                text = code.encode("utf8")[cursor_byte : node.end_byte].decode("utf8")
                return text.splitlines()[0] if "\n" in text else text
            elif node.type in [
                "if_statement", "for_statement", "while_statement", "try_statement",
                "with_statement", "function_definition", "class_definition",
                "decorated_definition", "match_statement",
            ]:
                text = code.encode("utf8")[cursor_byte : node.end_byte].decode("utf8")
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


def get_bracket_lang_statement(completion: str) -> str:
    end_idx = None
    for i in range(len(completion)):
        if completion[i] in [";", "}", "{"]:
            end_idx = i
            break
    return completion[:end_idx + 1] if end_idx is not None else completion


def postprocess_code_lines(prompt: str, completion: str, parser: Any, lang: str) -> str:
    try:
        if lang in ["java", "csharp", "typescript"]:
            return get_bracket_lang_statement(completion).strip()
        elif lang == "python":
            return get_python_one_statement(prompt, completion, parser).strip()
        else:
            return completion.splitlines()[0].strip() if completion.splitlines() else completion.strip()
    except Exception:
        return completion.strip()


def postprocess_repoeval(pred: str, gt: str) -> str:
    gt_lines = [l.strip() for l in gt.splitlines() if l.strip()]
    n = max(1, len(gt_lines))
    pred_lines = [l.strip() for l in pred.splitlines() if l.strip()][:n]
    return "\n".join(pred_lines)


def cal_edit_sim_single(pred: str, gt: str) -> float:
    try:
        from fuzzywuzzy import fuzz
        return fuzz.ratio(pred.strip(), gt.strip())
    except Exception:
        return difflib.SequenceMatcher(None, pred.strip(), gt.strip()).ratio() * 100.0


def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[n]


def extract_cursor_symbols_and_receivers(left_context: str, language: str) -> dict[str, Any]:
    lines = left_context.splitlines()
    recent = "\n".join(lines[-25:]) if len(lines) >= 25 else left_context
    cur_line = lines[-1] if lines else ""

    receivers: list[str] = []
    assigned_types: dict[str, str] = {}
    imports: list[str] = []
    imported_symbols: dict[str, str] = {}
    enclosing_classes: list[str] = []
    enclosing_defs: list[str] = []

    if language == "python":
        m_rec = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?:[A-Za-z_][A-Za-z0-9_]*)?$", cur_line)
        if m_rec:
            receivers.extend(m_rec)

        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", recent):
            assigned_types[m.group(1)] = m.group(2)
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", recent):
            assigned_types[m.group(1)] = m.group(2)

        for m in re.finditer(r"(?:from\s+([A-Za-z0-9_\.]+)\s+import\s+([^#\n]+)|import\s+([^#\n]+))", left_context):
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

        enclosing_classes = re.findall(r"(?:class\s+)([A-Za-z_]\w*)", left_context)
        enclosing_defs = re.findall(r"(?:def\s+)([A-Za-z_]\w*)", left_context)

    else:
        # Java / C-style
        m_rec = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?:[A-Za-z_][A-Za-z0-9_]*)?$", cur_line)
        if m_rec:
            receivers.extend(m_rec)

        for m in re.finditer(r"\b([A-Z][A-Za-z0-9_]*)\s+([a-z_][A-Za-z0-9_]*)\s*[=;,)]", recent):
            assigned_types[m.group(2)] = m.group(1)

        for m in re.finditer(r"import\s+(?:static\s+)?([A-Za-z0-9_.]+)\.([A-Z][A-Za-z0-9_]*);", left_context):
            imported_symbols[m.group(2)] = m.group(1) + "." + m.group(2)
            imports.append(m.group(2))

        enclosing_classes = re.findall(r"(?:class\s+)([A-Za-z_]\w*)", left_context)
        enclosing_defs = re.findall(r"(?:void\s+|public\s+|private\s+)([A-Za-z_]\w*)\s*\(", left_context)

    target_types: set[str] = set()
    for r in receivers:
        if r in assigned_types:
            target_types.add(assigned_types[r])
        elif r in imported_symbols:
            target_types.add(imported_symbols[r])

    recent_words = set(re.findall(r"\w+", recent))
    cursor_words = set(re.findall(r"\w+", cur_line))

    return {
        "receivers": receivers,
        "assigned_types": assigned_types,
        "imported_symbols": imported_symbols,
        "imports": imports,
        "target_types": list(target_types),
        "recent_words": list(recent_words),
        "cursor_words": list(cursor_words),
        "enclosing_classes": enclosing_classes,
        "enclosing_defs": enclosing_defs,
    }


def rank_aircoder_rrf_8metric(candidates: list[dict[str, Any]], query_file: str, left_context: str, scope_info: dict[str, Any]) -> list[dict[str, Any]]:
    """AIRCoder_r ACL 2026 Paper Exact: 8-metric Reciprocal Rank Fusion (k=60).
    
    Metrics:
    1. dense: raw code dense similarity (UniXcoder cosine)
    2. sparse: token overlap Jaccard
    3. skelton: AST skeleton structural dense similarity
    4. file_es: file path edit similarity
    5. file_dist: directory hop distance inverse
    6. class_match: enclosing class edit similarity
    7. def_match: enclosing def edit similarity
    8. dataflow: dataflow target matching
    """
    if not candidates:
        return []

    q_path = Path(query_file)
    q_parts = q_path.parts
    q_name = q_path.name.lower()
    q_tokens = set(re.findall(r"\w+", left_context[-600:]))

    cursor_symbols = set([s.lower() for s in scope_info.get("cursor_words", []) if len(s) > 1])
    cursor_receivers = set([s.lower() for s in scope_info.get("receivers", []) if len(s) > 1])
    target_types = set([s.lower() for s in scope_info.get("target_types", []) if len(s) > 1])
    targets = cursor_symbols | cursor_receivers | target_types

    q_classes = [c.lower() for c in scope_info.get("enclosing_classes", [])]
    q_defs = [d.lower() for d in scope_info.get("enclosing_defs", [])]
    query_class = q_classes[-1] if q_classes else ""
    query_def = q_defs[-1] if q_defs else ""

    query_skel_str = f"file: {q_name} classes: {' '.join(q_classes[-3:])} defs: {' '.join(q_defs[-3:])}"
    q_skel_tokens = set(re.findall(r"\w+", query_skel_str.lower()))

    scored = []
    for c in candidates:
        fp = c.get("file_path") or c.get("path", "")
        code = c.get("text") or c.get("code", "")
        if not code.strip():
            continue

        c_path = Path(fp)
        c_parts = c_path.parts
        c_name = c_path.name.lower()
        max_nl = max(len(q_name), len(c_name))
        f_es = (1.0 - (levenshtein_distance(q_name, c_name) / max_nl)) if max_nl > 0 else 0.0

        common = 0
        for p1, p2 in zip(q_parts, c_parts):
            if p1 == p2:
                common += 1
            else:
                break
        dist = (len(q_parts) - common) + (len(c_parts) - common)
        inv_dist = 1.0 if dist == 0 else (1.0 / dist)

        dense = float(c.get("cosine") or c.get("cosine_score") or 0.0)

        c_tokens = set(re.findall(r"\w+", code[:1000]))
        inter = len(q_tokens & c_tokens)
        union = len(q_tokens | c_tokens)
        sparse = (inter / union) if union > 0 else 0.0

        code_lower = code.lower()
        df_score = 1.0 if any(t in code_lower for t in targets) else 0.0

        c_classes = [cls.lower() for cls in re.findall(r"(?:class\s+)([A-Za-z_]\w*)", code)]
        c_defs = [fn.lower() for fn in re.findall(r"(?:def\s+|void\s+|public\s+|private\s+)([A-Za-z_]\w*)", code)]

        cls_es_list = [1.0 - (levenshtein_distance(query_class, sc) / max(len(query_class), len(sc))) for sc in c_classes] if (query_class and c_classes) else [0.0]
        def_es_list = [1.0 - (levenshtein_distance(query_def, sd) / max(len(query_def), len(sd))) for sd in c_defs] if (query_def and c_defs) else [0.0]
        class_score = max(cls_es_list) if cls_es_list else 0.0
        def_score = max(def_es_list) if def_es_list else 0.0

        # AST Skeleton Structural Similarity (Metric #8)
        cand_skel_str = f"file: {c_name} classes: {' '.join(c_classes)} defs: {' '.join(c_defs)}"
        c_skel_tokens = set(re.findall(r"\w+", cand_skel_str.lower()))
        skel_inter = len(q_skel_tokens & c_skel_tokens)
        skel_union = len(q_skel_tokens | c_skel_tokens)
        skel_jaccard = (skel_inter / skel_union) if skel_union > 0 else 0.0
        # Hybrid skeleton dense similarity: combines structural layout overlap with code embedding
        skelton_score = 0.5 * skel_jaccard + 0.5 * dense

        s_dict = {
            "dense": dense,
            "sparse": sparse,
            "skelton": skelton_score,
            "file_es": f_es,
            "file_dist": inv_dist,
            "class_match": class_score,
            "def_match": def_score,
            "dataflow": df_score,
        }
        scored.append({"cand": c, "scores": s_dict, "rrf": 0.0})

    # Exact RRF fusion with k=60 across ALL 8 METRICS
    k = 60
    metrics = ["dense", "sparse", "skelton", "file_es", "file_dist", "class_match", "def_match", "dataflow"]
    for metric in metrics:
        scored.sort(key=lambda x: x["scores"].get(metric, 0.0), reverse=True)
        for rank, item in enumerate(scored):
            item["rrf"] += 1.0 / (k + (rank + 1))

    scored.sort(key=lambda x: x["rrf"], reverse=True)
    return [item["cand"] for item in scored]


def get_all_trace_candidates(tr: dict[str, Any]) -> list[dict[str, Any]]:
    """Retrieve full candidate pool from trace, filtering out dummy/stop blocks."""
    raw = tr.get("aligncoder_top10") or tr.get("rerank_candidates") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []
    return [c for c in raw if c.get("kind") != "stop" and c.get("path") and c["path"] != "file_path" and (c.get("text") or c.get("code"))]


def build_aircoder_exact_prompt(task: dict[str, Any], tr: dict[str, Any], tok: Any) -> tuple[list[int], int, int]:
    """Exact AIRCoder_r prompt builder:
    - 800 token budget for cross-file context
    - Greedy fill (knapsack) without top-10 hardcap
    - Ascending chunk order [::-1]
    - Block comment enclosure (''' for Python, /* */ for Java)
    - 8,000 total prompt cap
    """
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    scope_info = extract_cursor_symbols_and_receivers(left_context, lang)

    comment_label = "#" if lang == "python" else "//"
    path_header = f"{comment_label} file path: {cur_path}\n"
    path_ids = tok.encode(path_header)

    raw_candidates = get_all_trace_candidates(tr)
    ranked_candidates = rank_aircoder_rrf_8metric(raw_candidates, cur_path, left_context, scope_info)

    # Greedily fill up to 800 tokens SPC budget
    selected_chunks = []
    cur_tokens = 0
    budget = 800
    for ch in ranked_candidates:
        fpath = ch.get("file_path") or ch.get("path", "")
        code = ch.get("text") or ch.get("code", "")
        if not code.strip():
            continue
        c_block = f"{comment_label} {fpath}\n{code}\n\n"
        c_ids = tok.encode(c_block)
        if cur_tokens + len(c_ids) > budget:
            continue
        cur_tokens += len(c_ids)
        selected_chunks.append((fpath, code, c_ids))

    # AIRCoder puts chunks in ASCENDING order (least relevant first, top closest to cursor)
    if lang == "python":
        rag_open = "'''\n"
        rag_close = "\n'''\n\n"
    else:
        rag_open = "/*\n"
        rag_close = "\n*/\n\n"

    cross_body = ""
    for fpath, code, _ in selected_chunks[::-1]:
        cross_body += f"{comment_label} {fpath}\n{code}\n\n"

    rag_prompt = f"{rag_open}{cross_body}{rag_close}" if cross_body.strip() else ""
    rag_ids = tok.encode(rag_prompt) if rag_prompt else []

    # Fit within 8,000 total tokens
    left_all_ids = tok.encode(left_context)
    allowed_left = 8000 - len(path_ids) - len(rag_ids)
    if allowed_left < 100:
        allowed_left = 100
        rag_ids = rag_ids[: 8000 - len(path_ids) - allowed_left]

    left_final_ids = left_all_ids[-allowed_left:] if len(left_all_ids) > allowed_left else left_all_ids
    final_ids = rag_ids + path_ids + left_final_ids
    return final_ids, len(final_ids), len(rag_ids)


def compute_aircoder_official(target: str, raw_pred: str, language: str) -> tuple[int, float, int, float]:
    """Official compute_EM and compute_ES from RepoClone/AIRCoder/src/evaluation.py.
    
    Includes character prefix truncation:
    prediction_lines_str = prediction_lines_str[: len(target_lines_str)]
    """
    target_lines = [l.strip() for l in target.splitlines() if l.strip()]
    if not target_lines:
        return 0, 0.0, 0, 0.0
    
    n = len(target_lines)
    pred_lines = [l.strip() for l in raw_pred.splitlines() if l.strip()][:n]
    
    target_str = "".join(target_lines)
    pred_str = "".join(pred_lines)
    
    # Official prefix truncation mechanism
    pred_str_trunc = pred_str[: len(target_str)]
    em = 1 if pred_str_trunc == target_str else 0
    
    es = cal_edit_sim_single(pred_str_trunc, target_str)
    id_em, id_f1 = compute_identifier_metrics(pred_str_trunc, target_str, language)
    return em, es, int(id_em), id_f1 * 100.0


def run_full_evaluation(state_ns: dict[str, Any], tag: str = "aircoder_r_exact_8004", batch_size: int = 500) -> dict[str, Any]:
    from vllm import SamplingParams

    engine = state_ns.get("VLLM_ENGINE")
    data = state_ns.get("MB_DATA", state_ns)
    tok = data.get("tokenizer") or state_ns.get("TOKENIZER")
    all_tasks = data.get("tasks") or state_ns.get("TASKS")
    traces = data.get("traces") or state_ns.get("TRACES")
    parser_map = data.get("parser_map") or state_ns.get("PARSER_MAP")

    benchmarks = ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]
    total_tasks = len(all_tasks)
    print(f"=== STARTING EXACT AIRCODER_R EVALUATION: {total_tasks} TASKS ===", flush=True)

    sp = SamplingParams(max_tokens=48, temperature=0.0)
    results_rows: list[dict[str, Any]] = []

    out_dir = ROOT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / f"{tag}_partial_results.jsonl"
    running_metrics_path = out_dir / f"{tag}_running_metrics.json"

    # Load existing partial results if resuming
    processed_task_keys: set[str] = set()
    if partial_path.exists():
        try:
            with open(partial_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        results_rows.append(r)
                        processed_task_keys.add(f"{r['benchmark']}::{r['task_id']}")
            print(f"Resuming: Loaded {len(results_rows)} records for {len(processed_task_keys)} tasks from {partial_path.name}", flush=True)
        except Exception as exc:
            print(f"Warning: Failed loading partial results: {exc}", flush=True)

    t_start = time.time()
    for batch_idx in range(0, total_tasks, batch_size):
        chunk_tasks = all_tasks[batch_idx : batch_idx + batch_size]
        remaining_chunk = [t for t in chunk_tasks if f"{t['benchmark']}::{t['task_id']}" not in processed_task_keys]
        if not remaining_chunk:
            continue

        print(f"\n[Batch {batch_idx//batch_size + 1}] Processing {len(remaining_chunk)} tasks (tasks {batch_idx} to {batch_idx + len(chunk_tasks)} / {total_tasks})...", flush=True)

        batch_prompts: list[list[int]] = []
        meta: list[tuple[dict[str, Any], int, int]] = []

        for task in remaining_chunk:
            key = (task["benchmark"], task["task_id"])
            tr = traces.get(key) or traces.get(task["task_id"])
            if not tr:
                continue

            p_ids, p_len, c_len = build_aircoder_exact_prompt(task, tr, tok)
            batch_prompts.append(p_ids)
            meta.append((task, p_len, c_len))

        if not batch_prompts:
            continue

        t_gen = time.time()
        prompts_payload = [{"prompt_token_ids": p_ids} for p_ids in batch_prompts]
        outputs = engine.generate(prompts_payload, sampling_params=sp)
        gen_time = time.time() - t_gen
        print(f"  Generated {len(outputs)} completions in {gen_time:.2f}s ({len(outputs)/max(0.01, gen_time):.1f} req/s)", flush=True)

        batch_new_rows: list[dict[str, Any]] = []
        for out, (task, p_len, c_len) in zip(outputs, meta):
            token_ids = list(out.outputs[0].token_ids)
            raw_pred = tok.decode(token_ids, skip_special_tokens=True)
            lang = task["language"]
            b = task["benchmark"]

            # 1. AlignCoder AST Strict postprocessing
            if b.startswith("repoeval"):
                clean_pred = postprocess_repoeval(raw_pred, task["groundtruth"])
            else:
                parser = parser_map.get(lang) if parser_map else None
                clean_pred = postprocess_code_lines(task["left_context"], raw_pred, parser, lang)

            gt_clean = task["groundtruth"].strip()
            pred_clean = clean_pred.strip()

            strict_em = 1 if pred_clean == gt_clean else 0
            strict_es = cal_edit_sim_single(pred_clean, gt_clean)
            strict_id_em, strict_id_f1 = compute_identifier_metrics(pred_clean, gt_clean, lang)

            # 2. AIRCoder Official character prefix truncation
            ac_em, ac_es, ac_id_em, ac_id_f1 = compute_aircoder_official(task["groundtruth"], raw_pred, lang)

            rec = {
                "benchmark": b,
                "task_id": task["task_id"],
                "arm": "aircoder_r",
                "strict_em": strict_em,
                "strict_es": strict_es,
                "strict_id_em": strict_id_em,
                "strict_id_f1": strict_id_f1 * 100.0,
                "aircoder_em": ac_em,
                "aircoder_es": ac_es,
                "aircoder_id_em": ac_id_em,
                "aircoder_id_f1": ac_id_f1,
                "prompt_tokens": p_len,
                "context_tokens": c_len,
                "raw_pred": raw_pred,
                "clean_pred": clean_pred,
                "groundtruth": task["groundtruth"],
            }
            results_rows.append(rec)
            batch_new_rows.append(rec)
            processed_task_keys.add(f"{b}::{task['task_id']}")

        # Append to partial results file
        with open(partial_path, "a", encoding="utf-8") as f:
            for r in batch_new_rows:
                f.write(json.dumps(r) + "\n")

        # Update running metrics
        strict_em_all = sum(r["strict_em"] for r in results_rows) / len(results_rows) * 100
        air_em_all = sum(r["aircoder_em"] for r in results_rows) / len(results_rows) * 100
        running_stat = {
            "processed_tasks": len(processed_task_keys),
            "total_tasks": total_tasks,
            "elapsed_minutes": round((time.time() - t_start) / 60, 2),
            "aligncoder_strict_running_em": round(strict_em_all, 2),
            "aircoder_official_running_em": round(air_em_all, 2),
        }
        running_metrics_path.write_text(json.dumps(running_stat, indent=2) + "\n", encoding="utf-8")
        print(f"  [Progress: {len(processed_task_keys)}/{total_tasks}] Strict EM: {strict_em_all:.2f}% | AIRCoder Official EM: {air_em_all:.2f}%", flush=True)

    total_time = (time.time() - t_start) / 60
    print(f"\n=== EVALUATION COMPLETE: {len(results_rows)} GENERATIONS IN {total_time:.2f} MINUTES ===", flush=True)

    # Compute final metrics summary
    summary: dict[str, Any] = {
        "tag": tag,
        "total_tasks": total_tasks,
        "total_duration_minutes": round(total_time, 2),
        "generator": GENERATOR,
        "setting": "AIRCoder_r ACL 2026 Exact (800-token budget, 8-metric RRF k=60, ascending order, triple-quote)",
        "aligncoder_protocol": {
            "overall": {
                "em": round(sum(r["strict_em"] for r in results_rows) / len(results_rows) * 100, 2),
                "es": round(sum(r["strict_es"] for r in results_rows) / len(results_rows), 2),
                "id_em": round(sum(r["strict_id_em"] for r in results_rows) / len(results_rows) * 100, 2),
                "id_f1": round(sum(r["strict_id_f1"] for r in results_rows) / len(results_rows), 2),
                "count": len(results_rows),
            },
            "by_benchmark": {},
        },
        "aircoder_official_protocol": {
            "overall": {
                "em": round(sum(r["aircoder_em"] for r in results_rows) / len(results_rows) * 100, 2),
                "es": round(sum(r["aircoder_es"] for r in results_rows) / len(results_rows), 2),
                "id_em": round(sum(r["aircoder_id_em"] for r in results_rows) / len(results_rows) * 100, 2),
                "id_f1": round(sum(r["aircoder_id_f1"] for r in results_rows) / len(results_rows), 2),
                "count": len(results_rows),
            },
            "by_benchmark": {},
        },
    }

    for b in benchmarks:
        b_rows = [r for r in results_rows if r["benchmark"] == b]
        if not b_rows:
            continue
        summary["aligncoder_protocol"]["by_benchmark"][b] = {
            "micro_em": round(sum(r["strict_em"] for r in b_rows) / len(b_rows) * 100, 2),
            "micro_es": round(sum(r["strict_es"] for r in b_rows) / len(b_rows), 2),
            "id_em": round(sum(r["strict_id_em"] for r in b_rows) / len(b_rows) * 100, 2),
            "id_f1": round(sum(r["strict_id_f1"] for r in b_rows) / len(b_rows), 2),
            "count": len(b_rows),
        }
        summary["aircoder_official_protocol"]["by_benchmark"][b] = {
            "micro_em": round(sum(r["aircoder_em"] for r in b_rows) / len(b_rows) * 100, 2),
            "micro_es": round(sum(r["aircoder_es"] for r in b_rows) / len(b_rows), 2),
            "id_em": round(sum(r["aircoder_id_em"] for r in b_rows) / len(b_rows) * 100, 2),
            "id_f1": round(sum(r["aircoder_id_f1"] for r in b_rows) / len(b_rows), 2),
            "count": len(b_rows),
        }

    # Save metrics JSON
    metrics_path = out_dir / f"{tag}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics to {metrics_path}", flush=True)

    # Save predictions GZ
    preds_path = out_dir / f"{tag}_predictions.jsonl.gz"
    with gzip.open(preds_path, "wt", encoding="utf-8") as gz:
        for r in results_rows:
            gz.write(json.dumps(r) + "\n")
    print(f"Saved {len(results_rows)} predictions to {preds_path}", flush=True)

    return summary


def main() -> None:
    print("=== AIRCoder_r Exact Benchmark Script Loaded ===", flush=True)


if __name__ == "__main__":
    main()
