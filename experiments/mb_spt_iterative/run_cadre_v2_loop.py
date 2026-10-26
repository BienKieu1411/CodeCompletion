#!/usr/bin/env python3
"""CADRE-X Exploration & Evaluation Loop on 256-Task Pilot Benchmark.

Investigating Core Architectural Improvements:
1. Arm 1: aircoder_r (Baseline ACL 2026 Replication, 800-token fixed SPC, RRF, Ascending)
2. Arm 2: cadre_v1 (Previous Full CADRE, 3584 tokens, First-encounter file grouping, top-down)
3. Arm 3: cadre_v2_aro (+Ascending Relevance Ordering: placing highest-ranked file closest to cursor)
4. Arm 4: cadre_v2_mes (+Multi-line Enclosing Scope Analysis for CSSI contract extraction)
5. Arm 5: cadre_v2_cacr (+Cursor-Affinity Coalesced Reranking for Channel A)
6. Arm 6: cadre_x (Unified Architecture: MES + CACR + ARO + Dynamic Adaptive Budgeting)
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
    "else", "if", "for", "in", "with", "try", "except", "finally", "while",
}

JAVA_TYPE_DECL_RE = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\s+([a-z_][A-Za-z0-9_]*)\s*[=;,)]')
JAVA_CAST_RE = re.compile(r'\(\(([A-Z][A-Za-z0-9_]*)\)\s*([A-Za-z0-9_]+)\)')
JAVA_IMPORT_RE = re.compile(r'import\s+(?:static\s+)?([A-Za-z0-9_.]+)\.([A-Z][A-Za-z0-9_]*);')
JAVA_RECEIVER_RE = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)?\s*$')

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


def cal_edit_sim_single(pred: str, gt: str) -> float:
    try:
        from fuzzywuzzy import fuzz
        return fuzz.ratio(pred.strip(), gt.strip())
    except Exception:
        return difflib.SequenceMatcher(None, pred.strip(), gt.strip()).ratio() * 100.0


def has_error(node: Any) -> bool:
    if node.type == "ERROR":
        return True
    for child in node.children:
        if has_error(child):
            return True
    return False


def get_python_one_statement(prompt: str, completion: str, parser: Any) -> str:
    all_lines = completion.splitlines(True)
    valid_lines: list[str] = []
    tree = parser.parse(bytes(prompt, "utf-8"))
    init_err = has_error(tree.root_node)

    for line in all_lines:
        valid_lines.append(line)
        code = prompt + "".join(valid_lines)
        tree = parser.parse(bytes(code, "utf-8"))
        if not init_err and not has_error(tree.root_node):
            break

    return "".join(valid_lines) if valid_lines else completion


def get_bracket_lang_statement(completion: str) -> str:
    end_idx = None
    for i in range(len(completion)):
        if completion[i] in [";", "}", "{"]:
            end_idx = i
            break
    return completion[:end_idx + 1] if end_idx is not None else completion


def postprocess_code(prompt: str, completion: str, parser: Any, lang: str, benchmark: str, gt: str) -> str:
    try:
        if "repoeval" in benchmark:
            gt_lines = [l.strip() for l in gt.splitlines() if l.strip()]
            n = max(1, len(gt_lines))
            pred_lines = [l.strip() for l in completion.splitlines() if l.strip()][:n]
            return "\n".join(pred_lines)
        elif lang == "python":
            return get_python_one_statement(prompt, completion, parser).strip()
        else:
            return get_bracket_lang_statement(completion).strip()
    except Exception:
        return completion.strip()


# =========================================================================
# SCOPE EXTRACTION MODULES (v1 vs MES v2)
# =========================================================================

def extract_scope_v1(left_context: str, language: str) -> dict[str, Any]:
    lines = left_context.splitlines()
    recent = "\n".join(lines[-25:]) if len(lines) >= 25 else left_context
    cur_line = lines[-1] if lines else ""

    if language == "python":
        receivers = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?:[A-Za-z_][A-Za-z0-9_]*)?$", cur_line)
        assigned_types = {}
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", recent):
            assigned_types[m.group(1)] = m.group(2)
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", recent):
            assigned_types[m.group(1)] = m.group(2)

        imports = []
        imported_symbols = {}
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

        target_types = set()
        for r in receivers:
            if r in assigned_types: target_types.add(assigned_types[r])
            elif r in imported_symbols: target_types.add(imported_symbols[r])
            elif r and r[0].isupper(): target_types.add(r)
        for s in imports:
            if s and s[0].isupper(): target_types.add(s)

        cur_syms = [s for s in WORD_RE.findall(cur_line) if s not in STOP_SYMS and len(s) > 2]
        return {
            "receivers": receivers,
            "target_types": target_types,
            "cursor_symbols": cur_syms,
            "focal_vars": [],
            "focal_calls": [],
        }
    else:  # java
        target_types = set()
        for m in JAVA_CAST_RE.finditer(recent): target_types.add(m.group(1))
        for m in JAVA_TYPE_DECL_RE.finditer(recent): target_types.add(m.group(1))
        for m in JAVA_IMPORT_RE.finditer(left_context): target_types.add(m.group(2))
        receivers = [r[0] for r in JAVA_RECEIVER_RE.findall(cur_line)]
        cur_syms = [s for s in WORD_RE.findall(cur_line) if s not in STOP_SYMS and len(s) > 2]
        return {
            "receivers": receivers,
            "target_types": target_types,
            "cursor_symbols": cur_syms,
            "focal_vars": [],
            "focal_calls": [],
        }


def extract_scope_mes(left_context: str, language: str) -> dict[str, Any]:
    lines = left_context.splitlines()
    non_empty = [l for l in lines if l.strip()]
    recent_5 = non_empty[-5:] if len(non_empty) >= 5 else non_empty
    cur_line = lines[-1] if lines else ""

    receivers: list[str] = []
    focal_vars: set[str] = set()
    focal_calls: set[str] = set()
    target_types: set[str] = set()

    if language == "python":
        m_rec = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?:[A-Za-z_][A-Za-z0-9_]*)?$", cur_line)
        if m_rec: receivers.extend(m_rec)
        for l in recent_5:
            for m in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\.", l):
                if m not in STOP_SYMS: receivers.append(m)

        for l in recent_5:
            for m in re.findall(r"\b(?:if|elif|while|for)\s+([A-Za-z_]\w*)", l):
                if m not in STOP_SYMS: focal_vars.add(m)
            for m in re.findall(r"\b([A-Za-z_]\w*)\s*=", l):
                if m not in STOP_SYMS: focal_vars.add(m)
            for m in re.findall(r"\b([A-Za-z_]\w*)\s*\(", l):
                if m not in STOP_SYMS: focal_calls.add(m)

        assigned_types: dict[str, str] = {}
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", left_context[-2500:]):
            assigned_types[m.group(1)] = m.group(2)
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", left_context[-2500:]):
            assigned_types[m.group(1)] = m.group(2)

        imported_symbols: dict[str, str] = {}
        imports: list[str] = []
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

        for r in set(receivers) | focal_vars:
            if r in assigned_types: target_types.add(assigned_types[r])
            elif r in imported_symbols: target_types.add(imported_symbols[r])
            elif r and r[0].isupper(): target_types.add(r)
        for s in imports:
            if s and s[0].isupper(): target_types.add(s)

        cursor_syms = [s for s in WORD_RE.findall(cur_line) if s not in STOP_SYMS and len(s) > 2]
        if not cursor_syms and recent_5:
            cursor_syms = [s for s in WORD_RE.findall(" ".join(recent_5)) if s not in STOP_SYMS and len(s) > 2]

        return {
            "receivers": list(set(receivers)),
            "focal_vars": list(focal_vars),
            "focal_calls": list(focal_calls),
            "target_types": target_types,
            "cursor_symbols": list(set(cursor_syms)),
            "assigned_types": assigned_types,
            "imported_symbols": imported_symbols,
        }
    else:  # java
        target_types = set()
        recent = "\n".join(recent_5)
        for m in JAVA_CAST_RE.finditer(recent): target_types.add(m.group(1))
        for m in JAVA_TYPE_DECL_RE.finditer(recent): target_types.add(m.group(1))
        for m in JAVA_IMPORT_RE.finditer(left_context): target_types.add(m.group(2))

        receivers = []
        for l in recent_5:
            m_rec = JAVA_RECEIVER_RE.findall(l)
            if m_rec: receivers.extend([r[0] for r in m_rec])

        cursor_syms = [s for s in WORD_RE.findall(cur_line) if s not in STOP_SYMS and len(s) > 2]
        if not cursor_syms and recent_5:
            cursor_syms = [s for s in WORD_RE.findall(" ".join(recent_5)) if s not in STOP_SYMS and len(s) > 2]

        return {
            "receivers": list(set(receivers)),
            "focal_vars": list(target_types),
            "focal_calls": [],
            "target_types": target_types,
            "cursor_symbols": list(set(cursor_syms)),
            "assigned_types": {},
            "imported_symbols": {},
        }


# =========================================================================
# CSSI CONTRACT SYNTHESIS (v1 vs MES v2)
# =========================================================================

def synthesize_cssi_v1(pool_files: list[dict[str, Any]], scope_info: dict[str, Any], language: str) -> str:
    target_types = scope_info.get("target_types", set())
    cursor_syms = set(scope_info.get("cursor_symbols", []))
    contract_snippets: list[str] = []
    contract_tokens_est = 0

    if language == "python":
        for f in pool_files:
            content = f.get("content") or f.get("text", "")
            if not content: continue
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
                            if len(stripped) < 100: retained_lines.append(f"    {stripped}")
                    if retained_lines:
                        snippet = f"# Contract: {tname} (from {fpath})\nclass {tname}:\n" + "\n".join(retained_lines[:15]) + "\n"
                        contract_snippets.append(snippet)
                        contract_tokens_est += len(snippet.split())
                        break

            for sname in cursor_syms:
                if sname in STOP_SYMS: continue
                f_pat = re.compile(rf"(?:def|async def)\s+{re.escape(sname)}\s*\([^)]*\)[^\n:]*:\s*(?:[^\n]*)?", re.MULTILINE)
                for m in f_pat.finditer(content):
                    sig = m.group(0).strip()
                    snippet = f"# Contract: {sname} (from {fpath})\n{sig}: ...\n"
                    contract_snippets.append(snippet)
                    contract_tokens_est += len(snippet.split())
                    if contract_tokens_est > 350: break

            if contract_tokens_est > 350: break
    else:  # java
        for f in pool_files:
            content = f.get("content") or f.get("text", "")
            if not content: continue
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

            if contract_tokens_est > 350: break

    return "\n".join(contract_snippets).strip()


def synthesize_cssi_mes(pool_files: list[dict[str, Any]], candidates: list[dict[str, Any]], scope_info: dict[str, Any], language: str) -> str:
    target_types = scope_info.get("target_types", set())
    cursor_syms = set(scope_info.get("cursor_symbols", []))
    focal_calls = set(scope_info.get("focal_calls", []))
    contract_snippets: list[str] = []
    contract_tokens_est = 0
    seen_headers: set[str] = set()

    search_sources = pool_files + candidates

    if language == "python":
        for src in search_sources:
            content = src.get("content") or src.get("text") or src.get("code", "")
            if not content: continue
            fpath = src.get("path") or src.get("file_path", "")

            for tname in target_types:
                if tname in seen_headers: continue
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
                            if len(stripped) < 100: retained_lines.append(f"    {stripped}")
                    if retained_lines:
                        snippet = f"# Contract: {tname} (from {fpath})\nclass {tname}:\n" + "\n".join(retained_lines[:15]) + "\n"
                        contract_snippets.append(snippet)
                        seen_headers.add(tname)
                        contract_tokens_est += len(snippet.split())
                        break

            for cname in focal_calls | cursor_syms:
                if cname in STOP_SYMS or cname in seen_headers: continue
                f_pat = re.compile(rf"(?:def|async def)\s+{re.escape(cname)}\s*\([^)]*\)[^\n:]*:\s*(?:[^\n]*)?", re.MULTILINE)
                for m in f_pat.finditer(content):
                    sig = m.group(0).strip()
                    snippet = f"# Contract: {cname} (from {fpath})\n{sig}: ...\n"
                    contract_snippets.append(snippet)
                    seen_headers.add(cname)
                    contract_tokens_est += len(snippet.split())
                    if contract_tokens_est > 350: break

            if contract_tokens_est > 350: break
    else:  # java
        for src in search_sources:
            content = src.get("content") or src.get("text") or src.get("code", "")
            if not content: continue
            fpath = src.get("path") or src.get("file_path", "")

            for tname in target_types:
                if tname in seen_headers: continue
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
                        seen_headers.add(tname)
                        contract_tokens_est += len(snippet.split())
                        break

            if contract_tokens_est > 350: break

    return "\n".join(contract_snippets).strip()


# =========================================================================
# CHANNEL A RERANKING MODULES (AIRCoder RRF vs CACR)
# =========================================================================

def rank_aircoder_rrf(candidates: list[dict[str, Any]], query_file: str, left_context: str, scope_info: dict[str, Any]) -> list[dict[str, Any]]:
    if not candidates: return []
    q_path = Path(query_file)
    q_parts = q_path.parts
    q_name = q_path.name.lower()
    q_tokens = set(re.findall(r"\w+", left_context[-600:]))
    targets = set([s.lower() for s in scope_info.get("cursor_symbols", []) if len(s) > 1])

    scored = []
    for c in candidates:
        fp = c.get("file_path") or c.get("path", "")
        code = c.get("text") or c.get("code", "")
        if not code.strip(): continue

        c_path = Path(fp)
        c_parts = c_path.parts
        c_name = c_path.name.lower()
        max_nl = max(len(q_name), len(c_name))
        f_es = (1.0 - (levenshtein_distance(q_name, c_name) / max_nl)) if max_nl > 0 else 0.0

        common = 0
        for p1, p2 in zip(q_parts, c_parts):
            if p1 == p2: common += 1
            else: break
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
            "dense": dense, "sparse": sparse, "file_es": f_es, "file_dist": inv_dist,
            "dataflow": df_score, "class_match": cls_match, "def_match": def_match,
        }
        scored.append({"cand": c, "scores": s_dict, "rrf": 0.0})

    k = 60
    for metric in ["dense", "sparse", "file_es", "file_dist", "dataflow", "class_match", "def_match"]:
        scored.sort(key=lambda x: x["scores"].get(metric, 0.0), reverse=True)
        for rank, item in enumerate(scored):
            item["rrf"] += 1.0 / (k + (rank + 1))

    scored.sort(key=lambda x: x["rrf"], reverse=True)
    return [item["cand"] for item in scored]


def rank_cursor_affinity_cacr(candidates: list[dict[str, Any]], query_file: str, left_context: str, scope_info: dict[str, Any]) -> list[tuple[str, list[str], float]]:
    """Cursor-Affinity Coalesced Reranking (CACR).
    
    Scores each file by composite cursor-affinity and returns coalesced blocks
    in ASCENDING order of score (highest affinity file comes LAST, closest to cursor).
    """
    if not candidates: return []
    q_path = Path(query_file)
    q_parts = q_path.parts
    q_name = q_path.name.lower()
    recent_tokens = set(re.findall(r"\w+", left_context[-800:]))
    focal_vars = set([s.lower() for s in scope_info.get("focal_vars", [])])
    focal_calls = set([s.lower() for s in scope_info.get("focal_calls", [])])
    receivers = set([s.lower() for s in scope_info.get("receivers", [])])
    target_types = set([s.lower() for s in scope_info.get("target_types", [])])

    files_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        fp = c.get("file_path") or c.get("path", "")
        code = c.get("text") or c.get("code", "")
        if fp and code.strip():
            files_chunks[fp].append(c)

    scored_files: list[tuple[str, list[str], float]] = []
    for fp, c_list in files_chunks.items():
        c_path = Path(fp)
        c_parts = c_path.parts
        c_name = c_path.name.lower()
        max_nl = max(len(q_name), len(c_name))
        path_sim = (1.0 - (levenshtein_distance(q_name, c_name) / max_nl)) if max_nl > 0 else 0.0

        common = 0
        for p1, p2 in zip(q_parts, c_parts):
            if p1 == p2: common += 1
            else: break
        dist = (len(q_parts) - common) + (len(c_parts) - common)
        inv_dist = 1.0 if dist == 0 else (1.0 / dist)

        combined_code = "\n".join([c.get("text") or c.get("code", "") for c in c_list])
        code_lower = combined_code.lower()

        # Check semantic definition matches
        def_match = 1.0 if any(fn in code_lower for fn in focal_calls) else 0.0
        class_match = 1.0 if any(t in code_lower for t in target_types) else 0.0
        rec_match = 1.0 if any(r in code_lower for r in receivers) else 0.0
        var_match = 1.0 if any(v in code_lower for v in focal_vars) else 0.0

        # Lexical overlap
        c_tokens = set(re.findall(r"\w+", combined_code[:2000]))
        jaccard = len(recent_tokens & c_tokens) / max(1, len(recent_tokens | c_tokens))

        # Dense score if available
        dense = max([float(c.get("cosine") or c.get("cosine_score") or 0.0) for c in c_list])

        # Composite affinity score
        affinity = (
            3.0 * def_match +
            2.5 * class_match +
            2.0 * rec_match +
            1.5 * var_match +
            1.5 * path_sim +
            1.0 * inv_dist +
            2.0 * jaccard +
            1.0 * dense
        )

        code_snippets = [c.get("text") or c.get("code", "") for c in c_list]
        scored_files.append((fp, code_snippets, affinity))

    # Sort in ASCENDING order of affinity so highest affinity file is LAST (adjacent to cursor!)
    scored_files.sort(key=lambda x: x[2], reverse=False)
    return scored_files


# =========================================================================
# PROMPT ASSEMBLY FOR ALL 6 ARMS
# =========================================================================

def build_pilot_prompt(arm: str, task: dict[str, Any], tr: dict[str, Any], pool_files: list[dict[str, Any]], tok: Any) -> tuple[list[int], int, int]:
    lang = task["language"]
    left_context = task["left_context"]
    cur_path = task.get("path", "current_file")
    left_all_ids = tok.encode(left_context)
    comment_label = "#" if lang == "python" else "//"
    path_ids = tok.encode(f"{comment_label} file path: {cur_path}\n")

    raw_candidates = tr.get("aligncoder_top10") or tr.get("spc_chunks") or tr.get("retrieved_chunks") or []

    # ARM 1: aircoder_r (Baseline ACL 2026 Replication)
    if arm == "aircoder_r":
        scope_v1 = extract_scope_v1(left_context, lang)
        ranked = rank_aircoder_rrf(raw_candidates, cur_path, left_context, scope_v1)
        selected_chunks = []
        cur_tokens = 0
        for ch in ranked:
            fp = ch.get("file_path") or ch.get("path", "")
            code = ch.get("text") or ch.get("code", "")
            if not code.strip(): continue
            c_block = f"{comment_label} {fp}\n{code}\n\n"
            c_ids = tok.encode(c_block)
            if cur_tokens + len(c_ids) > 800: break
            cur_tokens += len(c_ids)
            selected_chunks.append((fp, code, c_ids))

        rag_open = "'''\n" if lang == "python" else "/*\n"
        rag_close = "\n'''\n\n" if lang == "python" else "\n*/\n\n"
        cross_body = ""
        for fp, code, _ in selected_chunks[::-1]:
            cross_body += f"{comment_label} {fp}\n{code}\n\n"
        cross_ids = tok.encode(rag_open + cross_body.strip() + rag_close) if cross_body.strip() else []

        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    # ARM 2: cadre_v1 (Previous Full CADRE: 3584 tokens, First-encounter grouping, v1 CSSI)
    elif arm == "cadre_v1":
        chunks_by_file: dict[str, list[str]] = defaultdict(list)
        for ch in raw_candidates:
            fp = ch.get("file_path") or ch.get("path", "")
            c = ch.get("text") or ch.get("code", "")
            if fp and c.strip(): chunks_by_file[fp].append(c)

        exemplar_ids: list[int] = []
        for fp, code_list in chunks_by_file.items():
            coalesced = "\n...\n".join(code_list)
            block = f"{comment_label} file path: {fp}\n{coalesced}\n\n"
            b_ids = tok.encode(block)
            if len(exemplar_ids) + len(b_ids) > 3200: break
            exemplar_ids.extend(b_ids)

        scope_v1 = extract_scope_v1(left_context, lang)
        cssi_text = synthesize_cssi_v1(pool_files, scope_v1, lang)
        cssi_ids: list[int] = []
        if cssi_text:
            cssi_block = f"{comment_label} === CURSOR STRUCTURAL CONTRACTS ===\n{cssi_text}\n{comment_label} ====================================\n\n"
            cssi_ids = tok.encode(cssi_block)[:384]

        cross_ids = exemplar_ids + cssi_ids
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    # ARM 3: cadre_v2_aro (+Ascending Relevance Ordering: reverse file order so best file is adjacent to cursor)
    elif arm == "cadre_v2_aro":
        chunks_by_file = defaultdict(list)
        for ch in raw_candidates:
            fp = ch.get("file_path") or ch.get("path", "")
            c = ch.get("text") or ch.get("code", "")
            if fp and c.strip(): chunks_by_file[fp].append(c)

        # Reverse file order: highest ranked file was first in raw_candidates, so in reverse it becomes LAST (closest to cursor)
        file_items = list(chunks_by_file.items())[::-1]
        exemplar_ids = []
        for fp, code_list in file_items:
            coalesced = "\n...\n".join(code_list)
            block = f"{comment_label} file path: {fp}\n{coalesced}\n\n"
            b_ids = tok.encode(block)
            if len(exemplar_ids) + len(b_ids) > 3200: break
            exemplar_ids.extend(b_ids)

        scope_v1 = extract_scope_v1(left_context, lang)
        cssi_text = synthesize_cssi_v1(pool_files, scope_v1, lang)
        cssi_ids = []
        if cssi_text:
            cssi_block = f"{comment_label} === CURSOR STRUCTURAL CONTRACTS ===\n{cssi_text}\n{comment_label} ====================================\n\n"
            cssi_ids = tok.encode(cssi_block)[:384]

        cross_ids = exemplar_ids + cssi_ids
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    # ARM 4: cadre_v2_mes (+Multi-line Enclosing Scope Analysis for CSSI)
    elif arm == "cadre_v2_mes":
        chunks_by_file = defaultdict(list)
        for ch in raw_candidates:
            fp = ch.get("file_path") or ch.get("path", "")
            c = ch.get("text") or ch.get("code", "")
            if fp and c.strip(): chunks_by_file[fp].append(c)

        file_items = list(chunks_by_file.items())[::-1]
        exemplar_ids = []
        for fp, code_list in file_items:
            coalesced = "\n...\n".join(code_list)
            block = f"{comment_label} file path: {fp}\n{coalesced}\n\n"
            b_ids = tok.encode(block)
            if len(exemplar_ids) + len(b_ids) > 3200: break
            exemplar_ids.extend(b_ids)

        scope_mes = extract_scope_mes(left_context, lang)
        cssi_text = synthesize_cssi_mes(pool_files, raw_candidates, scope_mes, lang)
        cssi_ids = []
        if cssi_text:
            cssi_block = f"{comment_label} === CURSOR STRUCTURAL CONTRACTS ===\n{cssi_text}\n{comment_label} ====================================\n\n"
            cssi_ids = tok.encode(cssi_block)[:384]

        cross_ids = exemplar_ids + cssi_ids
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    # ARM 5: cadre_v2_cacr (+Cursor-Affinity Coalesced Reranking for Channel A)
    elif arm == "cadre_v2_cacr":
        scope_mes = extract_scope_mes(left_context, lang)
        ranked_files = rank_cursor_affinity_cacr(raw_candidates, cur_path, left_context, scope_mes)

        exemplar_ids = []
        for fp, code_list, _ in ranked_files:
            coalesced = "\n...\n".join(code_list)
            block = f"{comment_label} file path: {fp}\n{coalesced}\n\n"
            b_ids = tok.encode(block)
            if len(exemplar_ids) + len(b_ids) > 3200: break
            exemplar_ids.extend(b_ids)

        cssi_text = synthesize_cssi_mes(pool_files, raw_candidates, scope_mes, lang)
        cssi_ids = []
        if cssi_text:
            cssi_block = f"{comment_label} === CURSOR STRUCTURAL CONTRACTS ===\n{cssi_text}\n{comment_label} ====================================\n\n"
            cssi_ids = tok.encode(cssi_block)[:384]

        cross_ids = exemplar_ids + cssi_ids
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    # ARM 6: cadre_x (Unified CADRE-X: MES + CACR + DAB Dynamic Adaptive Budgeting)
    elif arm == "cadre_x":
        scope_mes = extract_scope_mes(left_context, lang)
        ranked_files = rank_cursor_affinity_cacr(raw_candidates, cur_path, left_context, scope_mes)

        # Dynamic Adaptive Budgeting:
        # Preserve up to 5,500 tokens of in-file context, allocating remainder to cross-file
        target_in_file = min(len(left_all_ids), 5500)
        cross_budget = min(3800, max(1200, 8000 - target_in_file - len(path_ids) - 50))
        cssi_budget = min(384, cross_budget // 6)
        exemplar_budget = cross_budget - cssi_budget

        exemplar_ids = []
        for fp, code_list, _ in ranked_files:
            coalesced = "\n...\n".join(code_list)
            block = f"{comment_label} file path: {fp}\n{coalesced}\n\n"
            b_ids = tok.encode(block)
            if len(exemplar_ids) + len(b_ids) > exemplar_budget: break
            exemplar_ids.extend(b_ids)

        cssi_text = synthesize_cssi_mes(pool_files, raw_candidates, scope_mes, lang)
        cssi_ids = []
        if cssi_text:
            cssi_block = f"{comment_label} === CURSOR STRUCTURAL CONTRACTS ===\n{cssi_text}\n{comment_label} ====================================\n\n"
            cssi_ids = tok.encode(cssi_block)[:cssi_budget]

        cross_ids = exemplar_ids + cssi_ids
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    else:
        raise ValueError(f"Unknown arm: {arm}")


def run_evaluation(state_ns: dict[str, Any], tag: str = "cadre_v2_loop_256", batch_size: int = 128) -> dict[str, Any]:
    mb_data = state_ns.get("MB_DATA") or {}
    engine = state_ns.get("VLLM_ENGINE")
    if not engine or not mb_data:
        raise RuntimeError("VLLM_ENGINE or MB_DATA not found in state_ns!")

    tasks: list[dict[str, Any]] = mb_data["tasks"]
    traces: dict[Any, dict[str, Any]] = mb_data["traces"]
    files: dict[str, dict[str, Any]] = mb_data["files"]
    pools: dict[str, list[str]] = mb_data["pools"]
    parser_map: dict[str, Any] = mb_data["parser_map"]
    tok: Any = mb_data["tokenizer"]

    from vllm import SamplingParams
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        stop=["\n\n", "class ", "def "],
    )

    arms = [
        "aircoder_r",
        "cadre_v1",
        "cadre_v2_aro",
        "cadre_v2_mes",
        "cadre_v2_cacr",
        "cadre_x",
    ]

    print(f"Starting CADRE-X Exploration Loop: {len(tasks)} tasks across {len(arms)} arms...", flush=True)

    results_rows: list[dict[str, Any]] = []
    out_dir = ROOT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for batch_idx in range(0, len(tasks), batch_size):
        chunk_tasks = tasks[batch_idx : batch_idx + batch_size]
        print(f"\n[Batch {batch_idx//batch_size + 1}] Processing {len(chunk_tasks)} tasks...", flush=True)

        batch_prompts: list[list[int]] = []
        meta: list[tuple[str, dict[str, Any], int, int]] = []

        for task in chunk_tasks:
            key = (task["benchmark"], task["task_id"])
            tr = traces.get(key) or traces.get(task["task_id"])
            if not tr: continue

            pool_files = []
            if "file_ids_by_path" in task:
                pool_files = [files[fid] for fid in task["file_ids_by_path"].values() if fid in files]
            elif task.get("pool_id"):
                f_ids = pools.get(task["pool_id"], [])
                pool_files = [files[fid] for fid in f_ids if fid in files]

            for arm in arms:
                p_ids, p_len, c_len = build_pilot_prompt(arm, task, tr, pool_files, tok)
                batch_prompts.append(p_ids)
                meta.append((arm, task, p_len, c_len))

        t_gen = time.time()
        prompts_payload = [{"prompt_token_ids": p_ids} for p_ids in batch_prompts]
        outputs = engine.generate(prompts_payload, sampling_params=sp)
        gen_dur = time.time() - t_gen
        print(f"  Generated {len(outputs)} completions in {gen_dur:.1f}s ({len(outputs)/max(0.01, gen_dur):.1f} req/s)", flush=True)

        for (arm, task, p_len, c_len), out in zip(meta, outputs):
            token_ids = list(out.outputs[0].token_ids)
            raw_pred = tok.decode(token_ids, skip_special_tokens=True)

            bm = task["benchmark"]
            lang = task["language"]
            gt = task.get("target") or task.get("groundtruth", "")
            parser = parser_map.get(lang)

            pred = postprocess_code(task["left_context"], raw_pred, parser, lang, bm, gt)

            em = int(pred.strip() == gt.strip())
            es = cal_edit_sim_single(pred, gt)
            pred_ids = extract_identifiers(pred, lang)
            gt_ids = extract_identifiers(gt, lang)
            id_em = int(pred_ids == gt_ids)
            p_set = set(pred_ids)
            g_set = set(gt_ids)
            tp = len(p_set & g_set)
            fp = len(p_set - g_set)
            fn = len(g_set - p_set)
            id_f1 = (2 * tp / (2 * tp + fp + fn)) * 100.0 if (2 * tp + fp + fn) else 0.0

            results_rows.append({
                "task_id": task["task_id"],
                "benchmark": bm,
                "arm": arm,
                "em": em,
                "es": es,
                "id_em": id_em,
                "id_f1": id_f1,
                "prompt_tokens": p_len,
                "crossfile_tokens": c_len,
                "prediction": pred,
                "raw_prediction": raw_pred,
                "groundtruth": gt,
            })

    total_dur = time.time() - t_start
    print(f"\nEvaluation loop completed in {total_dur:.1f}s!", flush=True)

    # Compute summary metrics by arm and benchmark
    summary: dict[str, Any] = {
        "tag": tag,
        "total_tasks": len(tasks),
        "duration_seconds": total_dur,
        "arms": {},
    }

    for arm in arms:
        arm_rows = [r for r in results_rows if r["arm"] == arm]
        bm_metrics = {}
        for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
            bm_r = [r for r in arm_rows if r["benchmark"] == bm]
            if bm_r:
                n = len(bm_r)
                bm_metrics[bm] = {
                    "count": n,
                    "em": sum(r["em"] for r in bm_r) / n * 100.0,
                    "es": sum(r["es"] for r in bm_r) / n,
                    "id_em": sum(r["id_em"] for r in bm_r) / n * 100.0,
                    "id_f1": sum(r["id_f1"] for r in bm_r) / n,
                }

        total_n = len(arm_rows)
        micro_em = sum(r["em"] for r in arm_rows) / total_n * 100.0
        micro_es = sum(r["es"] for r in arm_rows) / total_n
        micro_id_em = sum(r["id_em"] for r in arm_rows) / total_n * 100.0
        micro_id_f1 = sum(r["id_f1"] for r in arm_rows) / total_n
        macro_em = sum(m["em"] for m in bm_metrics.values()) / max(1, len(bm_metrics))

        summary["arms"][arm] = {
            "micro_em": micro_em,
            "macro_em": macro_em,
            "micro_es": micro_es,
            "micro_id_em": micro_id_em,
            "micro_id_f1": micro_id_f1,
            "by_benchmark": bm_metrics,
        }

    # Save metrics and predictions
    metrics_path = out_dir / f"{tag}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}", flush=True)

    preds_path = out_dir / f"{tag}_predictions.jsonl.gz"
    with gzip.open(preds_path, "wt", encoding="utf-8") as f:
        for r in results_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved predictions to {preds_path}", flush=True)

    # Print summary table
    print("\n" + "="*80)
    print(f"{'Arm':<20} | {'Micro EM':<10} | {'Python':<10} | {'Java':<10} | {'API':<10} | {'Line':<10}")
    print("-" * 80)
    for arm in arms:
        m = summary["arms"][arm]
        bm = m["by_benchmark"]
        py = bm.get("cceval_python", {}).get("em", 0.0)
        jv = bm.get("cceval_java", {}).get("em", 0.0)
        api = bm.get("repoeval_api", {}).get("em", 0.0)
        line = bm.get("repoeval_line", {}).get("em", 0.0)
        print(f"{arm:<20} | {m['micro_em']:>7.2f}%   | {py:>7.2f}%   | {jv:>7.2f}%   | {api:>7.2f}%   | {line:>7.2f}%")
    print("="*80 + "\n")

    return summary


if __name__ == "__main__":
    import runpy
    # Standalone execution
    print("Initializing environment via colab_init_engine.py...", flush=True)
    init_mod = runpy.run_path("/content/colab_init_engine.py", run_name="__main__")
    state = {
        "VLLM_ENGINE": init_mod.get("VLLM_ENGINE") or init_mod.get("global_ns", {}).get("VLLM_ENGINE"),
        "MB_DATA": init_mod.get("MB_DATA") or init_mod.get("global_ns", {}).get("MB_DATA"),
    }
    if not state["VLLM_ENGINE"] or not state["MB_DATA"]:
        # Fallback: check if stored in a shared global namespace
        import colab_init_engine
        state["VLLM_ENGINE"] = getattr(colab_init_engine, "VLLM_ENGINE", None) or state["VLLM_ENGINE"]
        state["MB_DATA"] = getattr(colab_init_engine, "MB_DATA", None) or state["MB_DATA"]
    run_evaluation(state)
