import re
import keyword
from pathlib import Path
from typing import Any

ID_RE = re.compile(r"\b[_A-Za-z][_A-Za-z0-9]*\b")
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
        # 1. Receiver detection
        m_rec = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?:[A-Za-z_][A-Za-z0-9_]*)?$", cur_line)
        if m_rec:
            receivers.extend(m_rec)
        for l in recent_5:
            for m in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\.", l):
                if m not in STOP_SYMS:
                    receivers.append(m)

        # 2. Focal variables and call targets in the enclosing 5 lines
        for l in recent_5:
            for m in re.findall(r"\b(?:if|elif|while|for)\s+([A-Za-z_]\w*)", l):
                if m not in STOP_SYMS: focal_vars.add(m)
            for m in re.findall(r"\b([A-Za-z_]\w*)\s*=", l):
                if m not in STOP_SYMS: focal_vars.add(m)
            for m in re.findall(r"\b([A-Za-z_]\w*)\s*\(", l):
                if m not in STOP_SYMS: focal_calls.add(m)

        # 3. Type inference
        assigned_types: dict[str, str] = {}
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", left_context[-2500:]):
            assigned_types[m.group(1)] = m.group(2)
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", left_context[-2500:]):
            assigned_types[m.group(1)] = m.group(2)

        # 4. Imports
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
            "imports": imports,
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
            "imports": list(target_types),
        }


def synthesize_cssi_contract_v2(pool_files: list[dict[str, Any]], candidates: list[dict[str, Any]], scope_info: dict[str, Any], language: str) -> str:
    target_types = scope_info.get("target_types", set())
    cursor_syms = set(scope_info.get("cursor_symbols", []))
    focal_calls = set(scope_info.get("focal_calls", []))
    contract_snippets: list[str] = []
    contract_tokens_est = 0
    seen_headers: set[str] = set()

    # Search in both pool_files and candidate chunks!
    search_sources = pool_files + candidates

    if language == "python":
        for src in search_sources:
            content = src.get("content") or src.get("text") or src.get("code", "")
            if not content: continue
            fpath = src.get("path") or src.get("file_path", "")

            # 1. Target class contracts
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

            # 2. Focal call contracts (functions invoked right at cursor)
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

print("CSSI v2 loaded successfully!")
