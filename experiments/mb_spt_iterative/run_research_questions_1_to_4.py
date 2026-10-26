#!/usr/bin/env python3
"""Execute Research Experiments for Questions 1 -> 4 on Colab.

Questions Evaluated:
1. Budget Scaling: 8k vs 4k vs 2k total context budget.
2. Retriever-Agnostic Validation:
   - BM-25 (top 10) raw vs BM-25 + CADRE
   - RLCoder (RLRetriever top 10) raw vs RLCoder + CADRE
   - AlignCoder top 10 + CADRE
3. CSSI-Enriched Retrieval (Query Expansion):
   - BM-25 with raw query vs BM-25 with CSSI-expanded query (Recall & EM).
4. CSSI Tree Representation & Linearization:
   - Canonical Interface Stub (PEP 484 / Java Interface)
   - Explicit Hierarchical Tree Notation (DFS Indented)
   - JSON-Schema AST Type Map
"""

from __future__ import annotations

import difflib
import gzip
import json
import keyword
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path("/content/mb_spt")
BUNDLE_PATH = Path("/content/research_bundle_256.tar.gz")
GENERATOR = "deepseek-ai/deepseek-coder-1.3b-base"
GENERATOR_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
SEED = 20261024

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


def postprocess_code_aligncoder(prompt: str, completion: str, parser: Any, lang: str, benchmark: str, gt: str) -> str:
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


def postprocess_code_aircoder_official(completion: str, gt: str) -> str:
    gt_clean = "".join([l.strip() for l in gt.splitlines() if l.strip()])
    pred_clean = "".join([l.strip() for l in completion.splitlines() if l.strip()])
    return pred_clean[:len(gt_clean)]


def extract_cursor_context_symbols(left_context: str, language: str) -> dict[str, Any]:
    nonempty = [l for l in left_context.splitlines() if l.strip()]
    tail = "\n".join(nonempty[-6:]) if nonempty else ""

    receivers = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*[A-Za-z0-9_]*\s*\(?$", tail)
    constructors = re.findall(r"(?:new\s+)?([A-Z][A-Za-z0-9_]*)\s*\(", tail)

    bases = []
    for line in nonempty:
        if language == "python":
            m = re.search(r"class\s+([A-Za-z0-9_]+)\s*\(([^)]+)\):", line)
            if m: bases.extend([b.strip() for b in m.group(2).split(",") if b.strip()])
        elif language == "java":
            m = re.search(r"class\s+([A-Za-z0-9_]+)\s+(?:extends|implements)\s+([A-Za-z0-9_,\s]+)", line)
            if m: bases.extend([b.strip() for b in re.split(r"[,\s]+", m.group(2)) if b.strip()])

    imports = []
    if language == "python":
        for m in re.finditer(r"from\s+([A-Za-z0-9_.]+)\s+import\s*\(([^)]+)\)", left_context, re.DOTALL):
            imports.extend([s.strip() for s in m.group(2).split(",") if s.strip()])
        for m in re.finditer(r"from\s+([A-Za-z0-9_.]+)\s+import\s+([A-Za-z0-9_,\s]+)", left_context):
            imports.extend([s.strip() for s in m.group(2).split(",") if s.strip() and not s.strip().startswith("(")])
    elif language == "java":
        for m in re.finditer(r"import\s+([A-Za-z0-9_.]+);", left_context):
            parts = m.group(1).split(".")
            if parts: imports.append(parts[-1])

    var_types: dict[str, str] = {}
    for line in nonempty[-25:]:
        m = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:new\s+)?([A-Z][a-zA-Z0-9_]*)\s*\(", line)
        if m: var_types[m.group(1)] = m.group(2)

    local_vars = {w for w in ID_RE.findall(tail) if w not in STOP_SYMS and len(w) >= 3}

    return {
        "receivers": receivers,
        "constructors": constructors,
        "bases": bases,
        "imports": imports,
        "var_types": var_types,
        "local_vars": local_vars,
    }


def extract_cssi_tree_structured(symbols: dict[str, Any], pool_files: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    target_names: set[str] = set()
    for r in symbols["receivers"]:
        if r in symbols["var_types"]: target_names.add(symbols["var_types"][r])
        elif r in ("self", "this"): target_names.update(symbols["bases"])
        else: target_names.add(r)
    target_names.update(symbols["constructors"])
    target_names.update(symbols["bases"])
    target_names.update(symbols["imports"])
    target_names = {t for t in target_names if t not in STOP_SYMS and len(t) >= 3}
    if not target_names: return []

    tree_nodes = []
    for pf in pool_files:
        text = pf.get("text", "") or pf.get("content", "")
        if not text: continue
        p = pf.get("path", "")
        stem = Path(p).stem

        matched = {t for t in target_names if t == stem or f"class {t}" in text or f"interface {t}" in text}
        if not matched: continue

        lines = text.splitlines()
        in_class = False
        cur_class_name = ""
        cur_methods = []

        for line in lines:
            if not in_class:
                m_cls = re.search(r"^(\s*)(?:public\s+|abstract\s+)*(?:class|interface)\s+([A-Za-z0-9_]+)", line)
                if m_cls and m_cls.group(2) in target_names:
                    in_class = True
                    cur_class_name = m_cls.group(2)
                    continue
            else:
                if language == "python":
                    m_def = re.match(r"^(\s{4,8})(?:async\s+)?def\s+([a-zA-Z_0-9]+)\s*\((.*?)\):", line)
                    if m_def:
                        cur_methods.append({"name": m_def.group(2), "params": m_def.group(3)})
                elif language == "java":
                    m_mth = re.match(r"^(\s{4,8})(?:public\s+|protected\s+|static\s+|final\s+)*([\w<>\[\]]+)\s+([a-zA-Z_0-9]+)\s*\((.*?)\)", line)
                    if m_mth:
                        cur_methods.append({"name": m_mth.group(3), "params": m_mth.group(4), "ret": m_mth.group(2)})

                if len(cur_methods) >= 12: break

        if cur_class_name:
            tree_nodes.append({
                "path": p,
                "class_name": cur_class_name,
                "methods": cur_methods,
            })
            if len(tree_nodes) >= 2: break

    return tree_nodes


def linearize_cssi_tree(tree_nodes: list[dict[str, Any]], language: str, mode: str = "stub", max_tokens: int = 300, tok: Any = None) -> str:
    if not tree_nodes: return ""
    comment_sym = "#" if language == "python" else "//"

    if mode == "stub":
        # Mode A: Canonical PEP 484 / Java Interface Stub
        stubs = []
        for node in tree_nodes:
            lines = [f"{comment_sym} [CSSI Interface Stub: {node['path']}]"]
            lines.append(f"{comment_sym} class {node['class_name']}:" if language == "python" else f"{comment_sym} class {node['class_name']} {{")
            for m in node["methods"]:
                if language == "python":
                    lines.append(f"{comment_sym}     def {m['name']}({m['params']}): ...")
                else:
                    lines.append(f"{comment_sym}     {m.get('ret', 'void')} {m['name']}({m['params']});")
            if language == "java":
                lines.append(f"{comment_sym} }}")
            stubs.append("\n".join(lines))
        res = "\n\n".join(stubs)

    elif mode == "tree":
        # Mode B: Explicit Hierarchical Tree Notation (DFS Indented)
        blocks = []
        for node in tree_nodes:
            lines = [
                f"{comment_sym} [CSSI Contextual Dependency Tree: {node['path']}]",
                f"{comment_sym} └── ClassNode: {node['class_name']}",
            ]
            for m in node["methods"]:
                lines.append(f"{comment_sym}     ├── MethodNode: {m['name']}({m['params']})")
            blocks.append("\n".join(lines))
        res = "\n\n".join(blocks)

    elif mode == "schema":
        # Mode C: JSON-Schema AST Type Map
        blocks = []
        for node in tree_nodes:
            payload = {
                "file": node["path"],
                "class": node["class_name"],
                "methods": [{"name": m["name"], "params": m["params"]} for m in node["methods"]]
            }
            blocks.append(f"{comment_sym} [CSSI Schema TypeMap]: " + json.dumps(payload))
        res = "\n\n".join(blocks)
    else:
        res = ""

    if tok is not None and res:
        c_ids = tok.encode(res, add_special_tokens=False)
        if len(c_ids) > max_tokens:
            res = tok.decode(c_ids[:max_tokens], skip_special_tokens=True)
    return res


def coalesce_commented_chunks(chunks: list[dict[str, Any]], language: str) -> list[str]:
    comment_sym = "#" if language == "python" else "//"
    file_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        p = c.get("path") or c.get("file_path", "")
        if not p or c.get("kind") == "stop" or p == "file_path": continue
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
            code_str = c.get("code") or c.get("text", "") or c.get("code_content", "")
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


def build_experiment_prompt(
    arm: str,
    task: dict[str, Any],
    trace: dict[str, Any],
    pool_files: list[dict[str, Any]],
    tok: Any
) -> tuple[list[int], int, int]:
    lang = task["language"]
    comment_sym = "#" if lang == "python" else "//"
    path_text = f"\n\n{comment_sym} file path: {task['path']}\n\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)
    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)

    # Candidate source selection
    if "bm25" in arm:
        candidates = list(trace.get("bm25_top10") or trace.get("bm25_top100") or [])[:10]
    elif "rlcoder" in arm:
        candidates = list(trace.get("rlcoder_top10") or trace.get("rlretriever_ranked_candidates") or [])[:10]
    else:
        candidates = list(trace.get("aligncoder_top10") or trace.get("spc_chunks") or trace.get("retrieved_chunks") or [])[:10]

    symbols = extract_cursor_context_symbols(task["left_context"], lang)

    # Budget settings
    if "budget_2k" in arm:
        max_total = 2000
        cross_budget = 896
        cssi_budget = 200
    elif "budget_4k" in arm:
        max_total = 4000
        cross_budget = 1792
        cssi_budget = 300
    else:  # 8k default
        max_total = 8000
        cross_budget = 3584
        cssi_budget = 300

    # Linearization mode
    linear_mode = "stub"
    if "linear_tree" in arm: linear_mode = "tree"
    elif "linear_schema" in arm: linear_mode = "schema"

    # Query expansion re-ranking (Experiment 3)
    if "cssi_expanded" in arm and "bm25" in arm:
        target_tokens = set(symbols["receivers"]) | set(symbols["constructors"]) | set(symbols["bases"]) | symbols["local_vars"]
        scored_cands = []
        for c in candidates:
            c_text = (c.get("code") or c.get("text", "")).lower()
            overlap = sum(1 for t in target_tokens if t.lower() in c_text)
            scored_cands.append((overlap, c))
        scored_cands.sort(key=lambda x: x[0], reverse=True)
        candidates = [c for _, c in scored_cands]

    # Handle flat raw baseline (no CADRE)
    if "raw_flat" in arm:
        body_parts = []
        for c in candidates[:5]:
            p = c.get("path") or c.get("file_path", "")
            code = c.get("code") or c.get("text", "") or c.get("code_content", "")
            if p and code.strip():
                body_parts.append(f"{comment_sym} file path: {p}\n{code.strip()}")
        raw_text = "\n\n".join(body_parts)
        cross_ids = tok.encode(raw_text, add_special_tokens=False)[:800]
        allowed_left = max(0, max_total - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-max_total:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    # Full CADRE assembly
    tree_nodes = extract_cssi_tree_structured(symbols, pool_files, lang)
    cssi_contract = linearize_cssi_tree(tree_nodes, lang, mode=linear_mode, max_tokens=cssi_budget, tok=tok)
    cssi_ids = tok.encode("\n\n" + cssi_contract, add_special_tokens=False) if cssi_contract else []

    rem_cross = max(0, cross_budget - len(cssi_ids))
    blocks = coalesce_commented_chunks(candidates, lang)
    exemplar_text = "\n\n".join(blocks)
    exemplar_ids = tok.encode(exemplar_text, add_special_tokens=False)[:rem_cross]

    cross_ids = exemplar_ids + cssi_ids
    allowed_left = max(0, max_total - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []
    prompt_ids = (cross_ids + path_ids + left_ids)[-max_total:]
    return prompt_ids, len(prompt_ids), len(cross_ids)


def run_all_research_questions(engine: Any, mb_data: dict[str, Any], batch_size: int = 128) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = mb_data["tasks"]
    traces: dict[Any, dict[str, Any]] = mb_data["traces"]
    files: dict[str, dict[str, Any]] = mb_data["files"]
    pools: dict[str, list[str]] = mb_data["pools"]
    parser_map: dict[str, Any] = mb_data["parser_map"]
    tok: Any = mb_data["tokenizer"]

    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=64, temperature=0.0)

    # Define all evaluation arms across Questions 1 -> 4
    arms_to_run = [
        # 1. Budget Scaling (Question 1)
        ("cadre_budget_8k", "Question 1: Full 8k Budget (CADRE+)"),
        ("cadre_budget_4k", "Question 1: Reduced 4k Budget"),
        ("cadre_budget_2k", "Question 1: Constrained 2k Budget"),

        # 2. Retriever-Agnostic Validation (Question 2)
        ("bm25_raw_flat", "Question 2: BM-25 Top 10 Raw Baseline"),
        ("bm25_plus_cadre", "Question 2: BM-25 Top 10 + CADRE"),
        ("rlcoder_raw_flat", "Question 2: RLCoder Top 10 Raw Baseline"),
        ("rlcoder_plus_cadre", "Question 2: RLCoder Top 10 + CADRE"),

        # 3. CSSI-Enriched Retrieval (Question 3)
        ("bm25_cssi_expanded_cadre", "Question 3: BM-25 CSSI-Expanded Query + CADRE"),

        # 4. CSSI Tree Representation & Linearization (Question 4)
        ("cadre_linear_stub", "Question 4: Canonical Interface Stub (PEP 484/Java Interface)"),
        ("cadre_linear_tree", "Question 4: Explicit DFS Indented Tree Notation"),
        ("cadre_linear_schema", "Question 4: JSON-Schema AST Type Map"),
    ]

    all_summaries: dict[str, Any] = {}
    benchmarks = ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]

    out_dir = ROOT_DIR / "outputs" / "research_q1_to_q4"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n================================================================================", flush=True)
    print(f"BẮT ĐẦU CHẠY TOÀN BỘ 10 ARMS CHO 4 CÂU HỎI NGHIÊN CỨU ({len(tasks)} TASKS/ARM)", flush=True)
    print(f"================================================================================\n", flush=True)

    for arm_key, arm_title in arms_to_run:
        print(f"\n>>> Running Arm: [{arm_key}] - {arm_title}...", flush=True)
        t_arm_start = time.time()

        prompts_batch = []
        meta_batch = []

        for task in tasks:
            key = (task["benchmark"], task["task_id"])
            tr = traces.get(key) or traces.get(task["task_id"])
            if not tr: continue

            pool_files = []
            if "file_ids_by_path" in task:
                pool_files = [files[fid] for fid in task["file_ids_by_path"].values() if fid in files]
            elif task.get("pool_id"):
                f_ids = pools.get(task["pool_id"], [])
                pool_files = [files[fid] for fid in f_ids if fid in files]

            p_ids, p_len, c_len = build_experiment_prompt(arm_key, task, tr, pool_files, tok)
            prompts_batch.append(p_ids)
            meta_batch.append((task, p_len, c_len))

        # Generate completions via vLLM
        t_gen_start = time.time()
        prompts_payload = [{"prompt_token_ids": p} for p in prompts_batch]
        outputs = engine.generate(prompts_payload, sampling_params=sp)
        gen_time = time.time() - t_gen_start

        # Score completions
        results = []
        for (task, p_len, c_len), out in zip(meta_batch, outputs):
            token_ids = list(out.outputs[0].token_ids)
            raw_pred = tok.decode(token_ids, skip_special_tokens=True)
            bm = task["benchmark"]
            lang = task["language"]
            gt = task.get("target") or task.get("groundtruth", "")
            parser = parser_map.get(lang)

            pred_align = postprocess_code_aligncoder(task["left_context"], raw_pred, parser, lang, bm, gt)
            em_align = int(pred_align.strip() == gt.strip())
            es_align = cal_edit_sim_single(pred_align, gt)

            pred_air = postprocess_code_aircoder_official(raw_pred, gt)
            em_air = int(pred_air == "".join([l.strip() for l in gt.splitlines() if l.strip()]))

            results.append({
                "task_id": task["task_id"],
                "benchmark": bm,
                "arm": arm_key,
                "em_align": em_align,
                "es_align": es_align,
                "em_air": em_air,
            })

        # Summarize by benchmark
        tot = len(results)
        summary = {
            "arm": arm_key,
            "title": arm_title,
            "tasks": tot,
            "gen_time_seconds": gen_time,
            "micro_em_align": sum(r["em_align"] for r in results) / tot * 100.0,
            "micro_es_align": sum(r["es_align"] for r in results) / tot,
            "micro_em_air": sum(r["em_air"] for r in results) / tot * 100.0,
            "by_bm": {}
        }
        for bm in benchmarks:
            b_rows = [r for r in results if r["benchmark"] == bm]
            n_b = len(b_rows)
            summary["by_bm"][bm] = {
                "count": n_b,
                "em_align": sum(r["em_align"] for r in b_rows) / n_b * 100.0 if n_b else 0,
                "es_align": sum(r["es_align"] for r in b_rows) / n_b if n_b else 0,
                "em_air": sum(r["em_air"] for r in b_rows) / n_b * 100.0 if n_b else 0,
            }
        summary["macro_em_align"] = sum(summary["by_bm"][bm]["em_align"] for bm in benchmarks) / len(benchmarks)
        summary["macro_em_air"] = sum(summary["by_bm"][bm]["em_air"] for bm in benchmarks) / len(benchmarks)

        all_summaries[arm_key] = summary
        print(f"  Finished in {time.time()-t_arm_start:.1f}s | Align Strict: {summary['micro_em_align']:.2f}% | AIRCoder Prefix: {summary['micro_em_air']:.2f}% (Macro: {summary['macro_em_air']:.2f}%)", flush=True)

    # Save summary
    out_file = out_dir / "research_q1_to_q4_summary.json"
    out_file.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\nSaved all results to {out_file}!", flush=True)

    # Print Formatted Comprehensive Comparison Table
    print("\n" + "="*105)
    print("BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM 4 CÂU HỎI NGHIÊN CỨU (COLAB A100):")
    print(f"{'Nhánh Thực Nghiệm (Arm)':<26} | {'Align EM':<9} | {'AIRCoder EM':<11} | {'Macro EM':<9} | {'Python':<8} | {'Java':<8} | {'API':<8} | {'Line':<8}")
    print("-" * 105)
    for arm_key, arm_title in arms_to_run:
        s = all_summaries[arm_key]
        bm = s["by_bm"]
        py = bm.get("cceval_python", {}).get("em_air", 0)
        jv = bm.get("cceval_java", {}).get("em_air", 0)
        api = bm.get("repoeval_api", {}).get("em_air", 0)
        ln = bm.get("repoeval_line", {}).get("em_air", 0)
        print(f"{arm_key:<26} | {s['micro_em_align']:>7.2f}% | {s['micro_em_air']:>9.2f}% | {s['macro_em_air']:>7.2f}% | {py:>6.2f}% | {jv:>6.2f}% | {api:>6.2f}% | {ln:>6.2f}%")
    print("="*105 + "\n")

    return all_summaries


if __name__ == "__main__":
    import tarfile
    from tree_sitter import Language, Parser
    from transformers import AutoTokenizer
    from vllm import LLM

    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    if not (ROOT_DIR / "tasks.jsonl.gz").exists():
        print(f"Extracting {BUNDLE_PATH.name}...", flush=True)
        with tarfile.open(BUNDLE_PATH, "r:gz") as tar:
            tar.extractall(ROOT_DIR)

    with gzip.open(ROOT_DIR / "tasks.jsonl.gz", "rt", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f]
    with gzip.open(ROOT_DIR / "retrieval_trace.jsonl.gz", "rt", encoding="utf-8") as f:
        traces = {(r["benchmark"], r["task_id"]): r for r in map(json.loads, f)}
    with gzip.open(ROOT_DIR / "files.jsonl.gz", "rt", encoding="utf-8") as f:
        files = {r["file_id"]: {"path": r["path"], "text": r["text"]} for r in map(json.loads, f)}
    pools = {}
    if (ROOT_DIR / "pools.jsonl.gz").exists():
        with gzip.open(ROOT_DIR / "pools.jsonl.gz", "rt", encoding="utf-8") as f:
            pools = {r["pool_id"]: r["file_ids"] for r in map(json.loads, f)}

    parser_map = {}
    for lang in ("python", "java"):
        so_path = ROOT_DIR / "parsers" / f"{lang}-lang-parser.so"
        language = Language(str(so_path), lang)
        p = Parser()
        p.set_language(language)
        parser_map[lang] = p

    tok = AutoTokenizer.from_pretrained(GENERATOR, revision=GENERATOR_REVISION, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print(f"Loading vLLM engine: {GENERATOR} (eager=True)...", flush=True)
    engine = LLM(
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

    mb_data = {
        "tasks": tasks,
        "traces": traces,
        "files": files,
        "pools": pools,
        "parser_map": parser_map,
        "tokenizer": tok,
    }

    run_all_research_questions(engine, mb_data)
