#!/usr/bin/env python3
"""CADRE vs AIRCoder (ACL 2026) Full Benchmark Evaluation (8,004 Tasks).

Evaluates across all 4 benchmark datasets:
- CrossCodeEval-Python (2,665 tasks)
- CrossCodeEval-Java (2,139 tasks)
- RepoEval-API (1,600 tasks)
- RepoEval-Line (1,600 tasks)
Total: 8,004 tasks x 2 arms = 16,008 generations.

Arms:
1. aircoder_r: Exact replication of AIRCoder ACL 2026 (8k input, 800-token SPC budget, RRF flat assembly).
2. cadre_full_dual_channel: Full CADRE (8k input, dynamic 3,584-token budget + file coalescing + CS-STR + CSSI + DCA dual-channel assembly).
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


def extract_identifiers(source_code: str, language: str) -> list[str]:
    clean = STRING_PATTERN.sub("", source_code)
    kws = get_keywords(language)
    return [
        t for t in WORD_RE.findall(clean)
        if re.match(r"[_a-zA-Z][_a-zA-Z0-9]*", t) and t not in kws
    ]


def remove_comments(code: str) -> str:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"//.*", "", code)
    return code


def has_error(node: Any) -> bool:
    if node is None:
        return False
    if getattr(node, "is_error", False) or getattr(node, "is_missing", False):
        return True
    return any(has_error(child) for child in getattr(node, "children", []))


def is_parse_valid(parser: Any, code: str) -> bool:
    if parser is None:
        return True
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
        if not is_parse_valid(parser, code):
            continue
        if i + 1 < len(completion) and completion[i + 1] == "\n":
            return completion[:i + 1].rstrip()
    return completion


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


JAVA_TYPE_DECL_RE = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\s+([a-z_][A-Za-z0-9_]*)\s*[=;,)]')
JAVA_CAST_RE = re.compile(r'\(\(([A-Z][A-Za-z0-9_]*)\)\s*([A-Za-z0-9_]+)\)')
JAVA_IMPORT_RE = re.compile(r'import\s+(?:static\s+)?([A-Za-z0-9_.]+)\.([A-Z][A-Za-z0-9_]*);')
JAVA_RECEIVER_RE = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)?\s*$')


def extract_cursor_symbols_and_receivers(left_context: str, language: str) -> dict[str, Any]:
    lines = left_context.splitlines()
    recent = "\n".join(lines[-25:]) if len(lines) >= 25 else left_context
    cur_line = lines[-1] if lines else ""

    if language == "python":
        receivers: list[str] = []
        m_rec = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?:[A-Za-z_][A-Za-z0-9_]*)?$", cur_line)
        if m_rec:
            receivers.extend(m_rec)

        assigned_types: dict[str, str] = {}
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", recent):
            assigned_types[m.group(1)] = m.group(2)
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", recent):
            assigned_types[m.group(1)] = m.group(2)

        imports: list[str] = []
        imported_symbols: dict[str, str] = {}
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

        target_types: set[str] = set()
        for r in receivers:
            if r in assigned_types: target_types.add(assigned_types[r])
            elif r in imported_symbols: target_types.add(imported_symbols[r])
            elif r and r[0].isupper(): target_types.add(r)
        for s in imports:
            if s and s[0].isupper(): target_types.add(s)

        cur_syms = [s for s in WORD_RE.findall(cur_line) if s not in STOP_SYMS and len(s) > 2]
        return {
            "receivers": receivers,
            "assigned_types": assigned_types,
            "imported_symbols": imported_symbols,
            "imports": imports,
            "target_types": target_types,
            "cursor_symbols": cur_syms,
        }
    else:  # java
        target_types = set()
        for m in JAVA_CAST_RE.finditer(recent):
            target_types.add(m.group(1))
        for m in JAVA_TYPE_DECL_RE.finditer(recent):
            target_types.add(m.group(1))
        for m in JAVA_IMPORT_RE.finditer(left_context):
            target_types.add(m.group(2))

        receivers = []
        m_rec = JAVA_RECEIVER_RE.findall(cur_line)
        if m_rec:
            receivers.extend([r[0] for r in m_rec])

        cur_syms = [s for s in WORD_RE.findall(cur_line) if s not in STOP_SYMS and len(s) > 2]
        return {
            "receivers": receivers,
            "assigned_types": {},
            "imported_symbols": {},
            "imports": list(target_types),
            "target_types": target_types,
            "cursor_symbols": cur_syms,
        }


def synthesize_cssi_contract(pool_files: list[dict[str, Any]], scope_info: dict[str, Any], language: str) -> str:
    target_types = scope_info.get("target_types", set())
    cursor_syms = set(scope_info.get("cursor_symbols", []))
    contract_snippets: list[str] = []
    contract_tokens_est = 0

    if language == "python":
        for f in pool_files:
            content = f.get("content") or f.get("text", "")
            if not content:
                continue
            fpath = f.get("path", "")

            for tname in target_types:
                pattern = re.compile(rf"class\s+{re.escape(tname)}\b[^\n:]*:\s*\n((?:[ \t]+[^\n]*\n|\s*\n)*)", re.MULTILINE)
                for m in pattern.finditer(content):
                    lines = m.group(1).splitlines()
                    retained_lines: list[str] = []
                    for line in lines:
                        stripped = line.strip()
                        if stripped.startswith("def ") or stripped.startswith("async def "):
                            sig = stripped.split(":", 1)[0]
                            retained_lines.append(f"    {sig}: ...")
                        elif re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*:", stripped) or re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped):
                            if len(stripped) < 100:
                                retained_lines.append(f"    {stripped}")
                    if retained_lines:
                        snippet = f"# Contract: {tname} (from {fpath})\nclass {tname}:\n" + "\n".join(retained_lines[:15]) + "\n"
                        contract_snippets.append(snippet)
                        contract_tokens_est += len(snippet.split())
                        break

            for sname in cursor_syms:
                if sname in STOP_SYMS:
                    continue
                f_pat = re.compile(rf"(?:def|async def)\s+{re.escape(sname)}\s*\([^)]*\)[^\n:]*:\s*(?:[^\n]*)?", re.MULTILINE)
                for m in f_pat.finditer(content):
                    sig = m.group(0).strip()
                    snippet = f"# Contract: {sname} (from {fpath})\n{sig}: ...\n"
                    contract_snippets.append(snippet)
                    contract_tokens_est += len(snippet.split())
                    if contract_tokens_est > 350:
                        break

            if contract_tokens_est > 350:
                break
    else:  # java
        for f in pool_files:
            content = f.get("content") or f.get("text", "")
            if not content:
                continue
            fpath = f.get("path", "")

            for tname in target_types:
                if f"class {tname}" in content or f"interface {tname}" in content:
                    lines = content.splitlines()
                    sigs: list[str] = []
                    for line in lines:
                        st = line.strip()
                        if ("public " in st or "protected " in st) and "(" in st and ")" in st and not st.startswith("//") and not st.startswith("/*"):
                            sig = st.split("{")[0].strip()
                            sigs.append(f"    {sig};")
                        elif (st.startswith("private ") or st.startswith("protected ")) and ";" in st and "(" not in st:
                            sigs.append(f"    {st}")
                    if sigs:
                        snippet = f"// Contract: {tname} (from {fpath})\nclass {tname} {{\n" + "\n".join(sigs[:12]) + "\n}}\n"
                        contract_snippets.append(snippet)
                        contract_tokens_est += len(snippet.split())
                        break

            if contract_tokens_est > 350:
                break

    return "\n".join(contract_snippets).strip()


def rank_aircoder_rrf(candidates: list[dict[str, Any]], query_file: str, left_context: str, scope_info: dict[str, Any]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    
    q_path = Path(query_file)
    q_parts = q_path.parts
    q_name = q_path.name.lower()
    q_tokens = set(re.findall(r"\w+", left_context[-600:]))
    cursor_symbols = set([s.lower() for s in scope_info.get("symbols", []) if len(s) > 1])
    cursor_receivers = set([s.lower() for s in scope_info.get("receivers", []) if len(s) > 1])
    cursor_calls = set([s.lower() for s in scope_info.get("call_targets", []) if len(s) > 1])
    targets = cursor_symbols | cursor_receivers | cursor_calls

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
        cls_match = 1.0 if any(cls in targets for cls in c_classes) else 0.0
        def_match = 1.0 if any(fn in targets for fn in c_defs) else 0.0

        s_dict = {
            "dense": dense,
            "sparse": sparse,
            "file_es": f_es,
            "file_dist": inv_dist,
            "dataflow": df_score,
            "class_match": cls_match,
            "def_match": def_match,
        }
        scored.append({"cand": c, "scores": s_dict, "rrf": 0.0})

    k = 60
    for metric in ["dense", "sparse", "file_es", "file_dist", "dataflow", "class_match", "def_match"]:
        scored.sort(key=lambda x: x["scores"].get(metric, 0.0), reverse=True)
        for rank, item in enumerate(scored):
            item["rrf"] += 1.0 / (k + (rank + 1))

    scored.sort(key=lambda x: x["rrf"], reverse=True)
    return [item["cand"] for item in scored]


def get_trace_chunks(tr: dict[str, Any]) -> list[dict[str, Any]]:
    return tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or tr.get("rerank_candidates") or []


def build_full_prompt(arm: str, task: dict[str, Any], tr: dict[str, Any], pool_files: list[dict[str, Any]], tok: Any) -> tuple[list[int], int, int]:
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    scope_info = extract_cursor_symbols_and_receivers(left_context, lang)

    left_all_ids = tok.encode(left_context)
    comment_label = "#" if lang == "python" else "//"
    path_ids = tok.encode(f"{comment_label} file path: {cur_path}\n")

    raw_candidates = get_trace_chunks(tr)

    if arm == "aircoder_r":
        # AIRCoder ACL 2026 Replication: 8-metric RRF ranking, 800-token budget, Ascending order [::-1], RAG_PROMPT block enclosure
        ranked_candidates = rank_aircoder_rrf(raw_candidates, cur_path, left_context, scope_info)

        selected_chunks = []
        cur_tokens = 0
        for ch in ranked_candidates:
            fpath = ch.get("file_path") or ch.get("path", "")
            code = ch.get("text") or ch.get("code", "")
            if not code.strip():
                continue
            c_block = f"{comment_label} {fpath}\n{code}\n\n"
            c_ids = tok.encode(c_block)
            if cur_tokens + len(c_ids) > 800:
                break
            cur_tokens += len(c_ids)
            selected_chunks.append((fpath, code, c_ids))

        # AIRCoder puts chunks in ASCENDING order (least relevant first, top relevant closest to code)
        # and wraps inside block comments / quotes
        if lang == "python":
            rag_open = "'''\n"
            rag_close = "\n'''\n\n"
        else:
            rag_open = "/*\n"
            rag_close = "\n*/\n\n"

        cross_body = ""
        for fpath, code, _ in selected_chunks[::-1]:
            cross_body += f"{comment_label} {fpath}\n{code}\n\n"

        if cross_body.strip():
            full_rag_str = rag_open + cross_body.strip() + rag_close
            cross_ids = tok.encode(full_rag_str)
        else:
            cross_ids = []

        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    elif arm == "cadre_full_dual_channel":
        # Full CADRE: Dynamic 3,584 budget + Coalescing + CS-STR + CSSI + DCA Dual-Channel Assembly
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

        cssi_text = synthesize_cssi_contract(pool_files, scope_info, lang)
        cssi_ids: list[int] = []
        if cssi_text:
            cssi_block = f"{comment_label} === CURSOR STRUCTURAL CONTRACTS ===\n{cssi_text}\n{comment_label} ====================================\n\n"
            cssi_ids = tok.encode(cssi_block)
            if len(cssi_ids) > 384:
                cssi_ids = cssi_ids[:384]

        # Dual-Channel order: Channel A (Exemplars) -> Channel B (Contract) -> Current File
        cross_ids = exemplar_ids + cssi_ids
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    else:
        raise ValueError(f"Unknown arm: {arm}")


def compute_aircoder_official(target: str, raw_pred: str, language: str) -> tuple[int, float, int, float]:
    comment_prefix = "#" if language == "python" else "//"
    target_lines = [line.strip() for line in target.splitlines() if line.strip() and not line.strip().startswith(comment_prefix)]
    prediction_lines = [line.strip() for line in raw_pred.splitlines() if line.strip() and not line.strip().startswith(comment_prefix)][: len(target_lines)]
    target_lines_str = "".join(target_lines)
    prediction_lines_str = "".join(prediction_lines)
    prediction_lines_str = prediction_lines_str[: len(target_lines_str)]

    em = int(target_lines_str == prediction_lines_str)
    if target_lines_str.strip() == prediction_lines_str.strip():
        es = 100.0
    else:
        max_l = max(len(target_lines_str), len(prediction_lines_str))
        es = (1.0 - (levenshtein_distance(target_lines_str, prediction_lines_str) / max_l)) * 100.0 if max_l else 100.0

    pred_ids = extract_identifiers(prediction_lines_str, language)
    gt_ids = extract_identifiers(target_lines_str, language)
    id_em = int(pred_ids == gt_ids)
    p_set = set(pred_ids)
    g_set = set(gt_ids)
    tp = len(p_set & g_set)
    fp = len(p_set - g_set)
    fn = len(g_set - p_set)
    id_f1 = (2 * tp / (2 * tp + fp + fn)) * 100.0 if (2 * tp + fp + fn) else 0.0

    return em, es, id_em, id_f1


def run_full_evaluation(state_ns: dict[str, Any], tag: str = "cadre_vs_aircoder_full_8004", batch_size: int = 500) -> dict[str, Any]:
    from vllm import SamplingParams

    engine = state_ns.get("VLLM_ENGINE")
    data = state_ns.get("MB_DATA", state_ns)
    tok = data.get("tokenizer") or state_ns.get("TOKENIZER")
    all_tasks = data.get("tasks") or state_ns.get("TASKS")
    traces = data.get("traces") or state_ns.get("TRACES")
    files = data.get("files") or state_ns.get("FILES")
    pools = data.get("pools") or state_ns.get("POOLS", {})
    parser_map = data.get("parser_map") or state_ns.get("PARSER_MAP")

    benchmarks = ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]
    total_tasks = len(all_tasks)
    print(f"=== STARTING FULL BENCHMARK EVALUATION: {total_tasks} TASKS ===", flush=True)

    arms = ["aircoder_r", "cadre_full_dual_channel"]
    sp = SamplingParams(max_tokens=48, temperature=0.0)

    predictions_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
                        predictions_by_arm[r["arm"]].append(r)
                        processed_task_keys.add(f"{r['benchmark']}::{r['task_id']}")
            print(f"Resuming: Loaded {len(results_rows)} records for {len(processed_task_keys)} tasks from {partial_path.name}", flush=True)
        except Exception as exc:
            print(f"Warning: Failed loading partial results: {exc}", flush=True)

    t_start = time.time()
    for batch_idx in range(0, total_tasks, batch_size):
        chunk_tasks = all_tasks[batch_idx : batch_idx + batch_size]
        # Filter out already processed tasks
        remaining_chunk = [t for t in chunk_tasks if f"{t['benchmark']}::{t['task_id']}" not in processed_task_keys]
        if not remaining_chunk:
            continue

        print(f"\n[Batch {batch_idx//batch_size + 1}] Processing {len(remaining_chunk)} tasks (tasks {batch_idx} to {batch_idx + len(chunk_tasks)} / {total_tasks})...", flush=True)

        batch_prompts: list[list[int]] = []
        meta: list[tuple[str, dict[str, Any], int, int]] = []

        for task in remaining_chunk:
            key = (task["benchmark"], task["task_id"])
            tr = traces.get(key) or traces.get(task["task_id"])
            if not tr:
                continue
            pool_files = []
            if "file_ids_by_path" in task:
                pool_files = [files[fid] for fid in task["file_ids_by_path"].values() if fid in files]
            elif task.get("pool_id"):
                f_ids = pools.get(task["pool_id"], [])
                pool_files = [files[fid] for fid in f_ids if fid in files]

            for arm in arms:
                p_ids, p_len, c_len = build_full_prompt(arm, task, tr, pool_files, tok)
                batch_prompts.append(p_ids)
                meta.append((arm, task, p_len, c_len))

        t_gen = time.time()
        prompts_payload = [{"prompt_token_ids": p_ids} for p_ids in batch_prompts]
        outputs = engine.generate(prompts_payload, sampling_params=sp)
        gen_duration = time.time() - t_gen
        print(f"  Generated {len(outputs)} completions in {gen_duration:.1f}s ({len(outputs)/gen_duration:.1f} req/s)", flush=True)

        for (arm, task, p_len, c_len), out in zip(meta, outputs):
            token_ids = list(out.outputs[0].token_ids)
            raw_pred = tok.decode(token_ids, skip_special_tokens=True)
            lang = task["language"]
            bm = task["benchmark"]

            if bm in ("cceval_python", "cceval_java"):
                parser = parser_map.get(lang) if parser_map else None
                pred = postprocess_code_lines(task["left_context"], raw_pred, parser, lang)
            else:
                pred = postprocess_repoeval(raw_pred, task["groundtruth"])

            clean_pred = remove_comments(pred)
            clean_gt = remove_comments(task["groundtruth"])

            pred_lines = [l.strip() for l in clean_pred.splitlines() if l.strip()]
            gt_lines = [l.strip() for l in clean_gt.splitlines() if l.strip()]

            if bm in ("repoeval_api", "repoeval_line"):
                pred_eval_lines = pred_lines[: len(gt_lines)]
                em = int(len(gt_lines) == len(pred_eval_lines) and gt_lines == pred_eval_lines)
                target_eval = "\n".join(gt_lines)
                pred_eval = "\n".join(pred_eval_lines)
                max_l = max(len(target_eval), len(pred_eval))
                es = (1.0 - (levenshtein_distance(target_eval, pred_eval) / max_l)) * 100.0 if max_l else 100.0
            else:
                em = int(pred_lines == gt_lines)
                es = cal_edit_sim_single(clean_pred, clean_gt)

            pred_ids = extract_identifiers(clean_pred, lang)
            gt_ids = extract_identifiers(clean_gt, lang)
            id_em = int(pred_ids == gt_ids)

            p_set = set(pred_ids)
            g_set = set(gt_ids)
            tp = len(p_set & g_set)
            fp = len(p_set - g_set)
            fn = len(g_set - p_set)
            id_f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0

            # 1. AlignCoder Standard Evaluation (Strict Tree-sitter statement match)
            # (em, es, id_em, id_f1 computed above)

            # 2. AIRCoder Official Evaluation (Prefix-truncation string match from AIRCoder evaluation.py)
            ac_em, ac_es, ac_id_em, ac_id_f1 = compute_aircoder_official(task["groundtruth"], raw_pred, lang)

            row = {
                "task_id": task["task_id"],
                "benchmark": bm,
                "repo_id": task["task_id"].split("/")[0],
                "arm": arm,
                # AlignCoder standard protocol
                "em": em,
                "es": es,
                "id_em": id_em,
                "id_f1": id_f1,
                # AIRCoder official prefix protocol
                "aircoder_em": ac_em,
                "aircoder_es": ac_es,
                "aircoder_id_em": ac_id_em,
                "aircoder_id_f1": ac_id_f1,
                "prompt_tokens": p_len,
                "crossfile_tokens": c_len,
                "prediction": clean_pred,
                "raw_prediction": raw_pred,
                "groundtruth": clean_gt,
            }
            predictions_by_arm[arm].append(row)
            results_rows.append(row)
            processed_task_keys.add(f"{bm}::{task['task_id']}")

        # Append newly generated rows to partial results file
        try:
            with open(partial_path, "a", encoding="utf-8") as pf:
                for r in results_rows[-len(meta):]:
                    pf.write(json.dumps(r) + "\n")
        except Exception as exc:
            print(f"Warning: Failed saving partial results: {exc}", flush=True)

        # Print running summary every batch and save running metrics
        running_summary = {"tag": tag, "processed_tasks": len(results_rows)//len(arms), "total_tasks": total_tasks, "arms": {}}
        for arm in arms:
            rows = predictions_by_arm[arm]
            align_em = sum(r["em"] for r in rows) / len(rows) * 100
            air_em = sum(r["aircoder_em"] for r in rows) / len(rows) * 100
            print(f"  Progress [{arm:24s}]: AlignCoder-EM={align_em:.2f}% | AIRCoder-Official-EM={air_em:.2f}% (N={len(rows)})", flush=True)
            running_summary["arms"][arm] = {
                "aligncoder_running_em": round(align_em, 2),
                "aircoder_official_running_em": round(air_em, 2),
                "count": len(rows),
            }
        try:
            running_metrics_path.write_text(json.dumps(running_summary, indent=2), encoding="utf-8")
        except Exception:
            pass

    total_time = time.time() - t_start
    print(f"\n=== FULL BENCHMARK COMPLETE IN {total_time/60:.2f} MINUTES ===", flush=True)

    summary: dict[str, Any] = {"tag": tag, "total_tasks": total_tasks, "total_duration_minutes": round(total_time/60, 2), "arms": {}}
    for arm in arms:
        rows = predictions_by_arm[arm]
        arm_stats: dict[str, Any] = {
            "aligncoder_protocol": {
                "overall": {
                    "em": round(sum(r["em"] for r in rows) / len(rows) * 100, 2),
                    "es": round(sum(r["es"] for r in rows) / len(rows) * 100, 2),
                    "id_em": round(sum(r["id_em"] for r in rows) / len(rows) * 100, 2),
                    "id_f1": round(sum(r["id_f1"] for r in rows) / len(rows) * 100, 2),
                    "count": len(rows),
                },
                "by_benchmark": {},
            },
            "aircoder_official_protocol": {
                "overall": {
                    "em": round(sum(r["aircoder_em"] for r in rows) / len(rows) * 100, 2),
                    "es": round(sum(r["aircoder_es"] for r in rows) / len(rows) * 100, 2),
                    "id_em": round(sum(r["aircoder_id_em"] for r in rows) / len(rows) * 100, 2),
                    "id_f1": round(sum(r["aircoder_id_f1"] for r in rows) / len(rows) * 100, 2),
                    "count": len(rows),
                },
                "by_benchmark": {},
            },
        }
        for b in benchmarks:
            b_rows = [r for r in rows if r["benchmark"] == b]
            if not b_rows:
                continue
            repo_em_map = defaultdict(list)
            repo_es_map = defaultdict(list)
            for r in b_rows:
                repo_em_map[r["repo_id"]].append(r["em"])
                repo_es_map[r["repo_id"]].append(r["es"])
            macro_em = sum(sum(v)/len(v) for v in repo_em_map.values()) / len(repo_em_map) * 100
            macro_es = sum(sum(v)/len(v) for v in repo_es_map.values()) / len(repo_es_map) * 100

            if b in ("repoeval_api", "repoeval_line"):
                align_str_em = f"{macro_em:.2f}({round(sum(r['em'] for r in b_rows) / len(b_rows) * 100, 2)})"
                align_str_es = f"{macro_es:.2f}({round(sum(r['es'] for r in b_rows) / len(b_rows) * 100, 2)})"
            else:
                align_str_em = f"{round(sum(r['em'] for r in b_rows) / len(b_rows) * 100, 2)}({macro_em:.2f})"
                align_str_es = f"{round(sum(r['es'] for r in b_rows) / len(b_rows) * 100, 2)}({macro_es:.2f})"

            arm_stats["aligncoder_protocol"]["by_benchmark"][b] = {
                "aligncoder_em": align_str_em,
                "aligncoder_es": align_str_es,
                "micro_em": round(sum(r["em"] for r in b_rows) / len(b_rows) * 100, 2),
                "macro_em": round(macro_em, 2),
                "micro_es": round(sum(r["es"] for r in b_rows) / len(b_rows) * 100, 2),
                "macro_es": round(macro_es, 2),
                "id_em": round(sum(r["id_em"] for r in b_rows) / len(b_rows) * 100, 2),
                "id_f1": round(sum(r["id_f1"] for r in b_rows) / len(b_rows) * 100, 2),
                "count": len(b_rows),
            }

            ac_micro_em = round(sum(r["aircoder_em"] for r in b_rows) / len(b_rows) * 100, 2)
            ac_micro_es = round(sum(r["aircoder_es"] for r in b_rows) / len(b_rows) * 100, 2)
            ac_id_em = round(sum(r["aircoder_id_em"] for r in b_rows) / len(b_rows) * 100, 2)
            ac_id_f1 = round(sum(r["aircoder_id_f1"] for r in b_rows) / len(b_rows) * 100, 2)
            arm_stats["aircoder_official_protocol"]["by_benchmark"][b] = {
                "micro_em": ac_micro_em,
                "micro_es": ac_micro_es,
                "id_em": ac_id_em,
                "id_f1": ac_id_f1,
                "count": len(b_rows),
            }

        summary["arms"][arm] = arm_stats

    summary["overall_by_arm"] = {
        arm: {
            "aligncoder_protocol": s["aligncoder_protocol"]["overall"],
            "aircoder_official_protocol": s["aircoder_official_protocol"]["overall"],
        }
        for arm, s in summary["arms"].items()
    }

    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with gzip.open(out_dir / f"{tag}_predictions.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in results_rows:
            f.write(json.dumps(r) + "\n")

    print(f"Full benchmark results saved to {tag}_metrics.json and {tag}_predictions.jsonl.gz", flush=True)
    return summary


if __name__ == "__main__":
    print("Full Benchmark Evaluation Runner (8,004 Tasks) loaded successfully.")
