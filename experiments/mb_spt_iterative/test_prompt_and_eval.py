#!/usr/bin/env python3
"""Test prompt construction and CSSI extraction across all 4 benchmarks."""

from __future__ import annotations

import gzip
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

BUNDLE_PATH = Path("experiments/mb_spt_iterative/full_benchmark_8004_bundle.tar.gz")

JAVA_TYPE_DECL_RE = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\s+([a-z_][A-Za-z0-9_]*)\s*[=;,)]')
JAVA_CAST_RE = re.compile(r'\(\(([A-Z][A-Za-z0-9_]*)\)\s*([A-Za-z0-9_]+)\)')
JAVA_IMPORT_RE = re.compile(r'import\s+(?:static\s+)?([A-Za-z0-9_.]+)\.([A-Z][A-Za-z0-9_]*);')
JAVA_RECEIVER_RE = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)?\s*$')

def extract_cursor_scope(left_context: str, language: str) -> dict:
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
        for m in re.finditer(r"(?:from\s+[A-Za-z0-9_.]+\s+import\s+([^#\n]+)|import\s+([^#\n]+))", left_context):
            raw = m.group(1) or m.group(2) or ""
            for s in raw.split(","):
                s = s.strip().split(" as ")[0].strip()
                if s: imports.append(s)
        return {
            "receivers": receivers,
            "assigned_types": assigned_types,
            "target_types": set(assigned_types.values()) | set(imports),
            "cursor_symbols": [w for w in re.findall(r"\w+", cur_line) if len(w) > 2],
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
        return {
            "receivers": receivers,
            "target_types": target_types,
            "cursor_symbols": [w for w in re.findall(r"\w+", cur_line) if len(w) > 2],
        }

def synthesize_cssi_multilang(pool_files: list[dict], scope_info: dict, language: str) -> str:
    target_types = scope_info["target_types"]
    cursor_symbols = set(scope_info["cursor_symbols"])
    contract_snippets = []

    if language == "python":
        for f in pool_files:
            content = f.get("text") or f.get("content", "")
            fpath = f.get("path", "")
            for tname in target_types:
                pattern = re.compile(rf"class\s+{re.escape(tname)}\b[^\n:]*:\s*\n((?:[ \t]+[^\n]*\n|\s*\n)*)", re.MULTILINE)
                for m in pattern.finditer(content):
                    lines = m.group(1).splitlines()
                    sigs = []
                    for line in lines:
                        st = line.strip()
                        if st.startswith("def ") or st.startswith("async def "):
                            sigs.append(f"    {st.split(':', 1)[0]}: ...")
                    if sigs:
                        snippet = f"# Contract: {tname} (from {fpath})\nclass {tname}:\n" + "\n".join(sigs[:12]) + "\n"
                        contract_snippets.append(snippet)
                        break
            if len("\n".join(contract_snippets).split()) > 350:
                break
    else:  # java
        for f in pool_files:
            content = f.get("text") or f.get("content", "")
            fpath = f.get("path", "")
            for tname in target_types:
                if f"class {tname}" in content or f"interface {tname}" in content:
                    lines = content.splitlines()
                    sigs = []
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
                        break
            if len("\n".join(contract_snippets).split()) > 350:
                break

    return "\n".join(contract_snippets).strip()

def test():
    with tarfile.open(BUNDLE_PATH) as tar:
        f_tasks = tar.extractfile("tasks.jsonl.gz")
        f_traces = tar.extractfile("retrieval_trace.jsonl.gz")
        f_files = tar.extractfile("files.jsonl.gz")

        files_map = {}
        with gzip.open(f_files, "rt") as gz_f:
            for line in gz_f:
                d = json.loads(line)
                files_map[d["file_id"]] = d

        bench_samples = defaultdict(list)
        with gzip.open(f_tasks, "rt") as gz_t, gzip.open(f_traces, "rt") as gz_tr:
            for t_line, tr_line in zip(gz_t, gz_tr):
                t = json.loads(t_line)
                tr = json.loads(tr_line)
                bm = t["benchmark"]
                if len(bench_samples[bm]) < 2:
                    bench_samples[bm].append((t, tr))

        for bm, pairs in bench_samples.items():
            print(f"\n==================== BENCHMARK: {bm} ====================")
            for task, tr in pairs:
                tid = task["task_id"]
                lang = task["language"]
                cands = tr.get("aligncoder_top10") or tr.get("spc_chunks") or []
                pool_files = [files_map[fid] for fid in task.get("file_ids_by_path", {}).values() if fid in files_map]

                scope = extract_cursor_scope(task["left_context"], lang)
                cssi = synthesize_cssi_multilang(pool_files, scope, lang)

                print(f"Task: {tid} ({lang})")
                print(f"  Candidates count: {len(cands)}")
                print(f"  Target types: {scope.get('target_types')}")
                print(f"  Pool files: {len(pool_files)}")
                print(f"  CSSI Contract Length: {len(cssi)} chars")
                if cssi:
                    print("  --- CSSI Snippet Preview ---")
                    for cl in cssi.splitlines()[:6]:
                        print("    ", cl)

test()
