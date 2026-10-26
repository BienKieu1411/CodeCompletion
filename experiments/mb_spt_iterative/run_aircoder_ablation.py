#!/usr/bin/env python3
"""AIRCoder (ACL 2026) vs CADRE Direct Head-to-Head & Multi-Gap Ablation Benchmark (256 tasks).

Research Gaps Targeted:
1. Gap 1 (Budget Asymmetry & Fragmentation): AIRCoder's 800-token fixed SPC budget vs Dynamic File Coalescing.
2. Gap 2 (Semantic Sibling Interference): Flat RAG ranks calling exemplars higher than target API declarations.
3. Gap 3 (Receiver Scope Blindness): Retrievers cannot resolve `self.`, `super()`, or local variable types.
4. Gap 4 (Positional RoPE Attention Degradation): Concatenating contracts far from cursor diminishes token attention.

Ablation Arms:
1. aircoder_r (ACL 2026 SOTA Replication): 8,000-token input limit, 800-token SPC budget, RRF fused ranking, flat assembly.
2. cadre_ablation_coalesce: +Dynamic 3,584-token budget + file coalescing (no CS-STR, no CSSI).
3. cadre_ablation_cssi_flat: +CS-STR & CSSI structural contract, but placed monotonically at prompt head.
4. cadre_full_dual_channel: Full CADRE (CS-STR + CSSI with formal param alignment + DCA dual-channel adjacent to cursor).
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
    if node.type == "ERROR":
        return True
    for child in node.children:
        if has_error(child):
            return True
    return False


def postprocess_cceval(prompt: str, completion: str, language: str, parser_map: dict[str, Any]) -> str:
    try:
        if language in ("java", "csharp", "typescript"):
            end_idx = None
            for i in range(len(completion)):
                if completion[i] in (";", "}", "{"):
                    end_idx = i
                    break
            return completion[: end_idx + 1] if end_idx else completion
        if language == "python":
            parser = parser_map.get(language)
            if parser is None:
                return completion
            for i in range(len(completion)):
                if i + 1 >= len(completion) or completion[i + 1] != "\n":
                    continue
                code = (prompt + completion[: i + 1]).encode("utf-8", errors="replace")
                tree = parser.parse(code)
                if tree is None or has_error(tree.root_node):
                    continue
                return completion[: i + 1].rstrip()
            return completion
        return completion
    except Exception:
        return completion


def postprocess_repoeval(completion: str, target: str) -> str:
    gt_lines = [l.strip() for l in target.split("\n") if l.strip()]
    pred_lines = [l.strip() for l in completion.split("\n") if l.strip()]
    sliced = pred_lines[: len(gt_lines)]
    return "\n".join(sliced)


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]


# ==============================================================================
# Component 1: Cursor-Scoped Symbol & Type Resolution (CS-STR)
# ==============================================================================

def extract_cursor_context_symbols(left_context: str, language: str) -> dict[str, Any]:
    nonempty = [l for l in left_context.splitlines() if l.strip()]
    tail = "\n".join(nonempty[-6:]) if nonempty else ""
    
    # 1. Receiver expression: e.g. foo.bar( -> receiver is 'foo'
    receivers = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*[A-Za-z0-9_]*\s*\(?$", tail)
    
    # 2. Instantiations / constructors: e.g. new Foo( or Foo(
    constructors = re.findall(r"(?:new\s+)?([A-Z][A-Za-z0-9_]*)\s*\(", tail)
    
    # 3. Class inheritance: e.g. class MySub(BaseClass): or class MySub extends BaseClass
    bases = []
    for line in nonempty:
        if language == "python":
            m = re.search(r"class\s+([A-Za-z0-9_]+)\s*\(([^)]+)\):", line)
            if m:
                bases.extend([b.strip() for b in m.group(2).split(",") if b.strip()])
        elif language == "java":
            m = re.search(r"class\s+([A-Za-z0-9_]+)\s+(?:extends|implements)\s+([A-Za-z0-9_,\s]+)", line)
            if m:
                bases.extend([b.strip() for b in re.split(r"[,\s]+", m.group(2)) if b.strip()])
                
    # 4. Multi-line parenthesized imports in python: from pkg import (A, B, C)
    imports = []
    if language == "python":
        for m in re.finditer(r"from\s+([A-Za-z0-9_.]+)\s+import\s*\(([^)]+)\)", left_context, re.DOTALL):
            syms = [s.strip() for s in m.group(2).split(",") if s.strip()]
            imports.extend(syms)
        for m in re.finditer(r"from\s+([A-Za-z0-9_.]+)\s+import\s+([A-Za-z0-9_,\s]+)", left_context):
            syms = [s.strip() for s in m.group(2).split(",") if s.strip() and not s.strip().startswith("(")]
            imports.extend(syms)
    elif language == "java":
        for m in re.finditer(r"import\s+([A-Za-z0-9_.]+);", left_context):
            parts = m.group(1).split(".")
            if parts:
                imports.append(parts[-1])
            
    # 5. Local variable assignments: var = TargetClass(...)
    var_types: dict[str, str] = {}
    for line in nonempty[-25:]:
        m = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:new\s+)?([A-Z][a-zA-Z0-9_]*)\s*\(", line)
        if m:
            var_types[m.group(1)] = m.group(2)
            
    # 6. Active cursor parameter candidates (local variable names in cursor tail)
    local_vars = {w for w in ID_RE.findall(tail) if w not in STOP_SYMS and len(w) >= 3}
    
    return {
        "receivers": receivers,
        "constructors": constructors,
        "bases": bases,
        "imports": imports,
        "var_types": var_types,
        "local_vars": local_vars,
    }


# ==============================================================================
# Component 2: Cursor-Scoped Structural Interface (CSSI) with Formal Param Alignment
# ==============================================================================

def extract_cssi_contract(
    symbols: dict[str, Any],
    pool_files: list[dict[str, Any]],
    language: str,
    max_tokens: int = 300,
    tok: Any = None,
) -> str:
    target_names: set[str] = set()
    for r in symbols["receivers"]:
        if r in symbols["var_types"]:
            target_names.add(symbols["var_types"][r])
        elif r == "self" or r == "this":
            target_names.update(symbols["bases"])
        else:
            target_names.add(r)
    target_names.update(symbols["constructors"])
    target_names.update(symbols["bases"])
    target_names.update(symbols["imports"])
    target_names = {t for t in target_names if t not in STOP_SYMS and len(t) >= 3}
    
    if not target_names:
        return ""
        
    stubs = []
    comment_sym = "#" if language == "python" else "//"
    
    for pf in pool_files:
        text = pf.get("text", "")
        if not text:
            continue
        p = pf.get("path", "")
        stem = Path(p).stem
        
        matched = {t for t in target_names if t == stem or f"class {t}" in text or f"interface {t}" in text}
        if not matched:
            continue
            
        lines = text.splitlines()
        extracted_lines = []
        in_class = False
        class_indent = ""
        best_method_lines = []
        best_overlap = -1
        
        for line in lines:
            if not in_class:
                m_cls = re.search(r"^(\s*)(?:public\s+|abstract\s+)*(?:class|interface)\s+([A-Za-z0-9_]+)", line)
                if m_cls and m_cls.group(2) in target_names:
                    in_class = True
                    class_indent = m_cls.group(1)
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
                        sig_params = set(ID_RE.findall(params)) - STOP_SYMS
                        overlap = len(sig_params & symbols["local_vars"])
                        extracted_lines.append(f"{m_def.group(1)}def {sig_name}({params}): ...")
                elif language == "java":
                    m_mth = re.match(r"^(\s{4,8})(?:public\s+|protected\s+|static\s+|final\s+)*([\w<>\[\]]+)\s+([a-zA-Z_0-9]+)\s*\((.*?)\)", line)
                    if m_mth:
                        ret_type = m_mth.group(2)
                        sig_name = m_mth.group(3)
                        params = m_mth.group(4)
                        sig_params = set(ID_RE.findall(params)) - STOP_SYMS
                        overlap = len(sig_params & symbols["local_vars"])
                        extracted_lines.append(f"{m_mth.group(1)}{ret_type} {sig_name}({params});")
                        
                if len(extracted_lines) >= 15:
                    break
                    
        if extracted_lines:
            stub_text = f"{comment_sym} [CADRE-CSSI Contract: {p}]\n" + "\n".join(
                f"{comment_sym} {l}" for l in extracted_lines
            )
            stubs.append(stub_text)
            if len(stubs) >= 2:
                break
                
    if not stubs:
        return ""
        
    contract = "\n\n".join(stubs)
    if tok is not None:
        c_ids = tok.encode(contract, add_special_tokens=False)
        if len(c_ids) > max_tokens:
            contract = tok.decode(c_ids[:max_tokens], skip_special_tokens=True)
    return contract


def coalesce_chunks_by_file(chunks: list[dict[str, Any]], language: str) -> list[str]:
    comment_sym = "#" if language == "python" else "//"
    file_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        p = c.get("path")
        if not p or c.get("kind") == "stop":
            continue
        file_chunks[p].append(c)
        
    blocks = []
    for p, f_chunk_list in file_chunks.items():
        if p == "file_path":
            body = "\n".join(f"{comment_sym} {l}" for c in f_chunk_list for l in c.get("code", "").splitlines() if l.strip())
            if body:
                blocks.append(f"{comment_sym} Potential invoked APIs:\n{body}")
            continue
            
        f_chunk_list.sort(key=lambda x: int(x.get("line_start", 0)))
        merged_lines = []
        last_end = -1
        range_start = -1
        
        for c in f_chunk_list:
            s = int(c.get("line_start", 0))
            e = int(c.get("line_end", 0))
            c_lines = c.get("code", "").splitlines()
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


def format_aircoder_chunks(chunks: list[dict[str, Any]], language: str) -> list[str]:
    """Format chunks according to AIRCoder / standard cross-file RAG convention."""
    label = "#" if language == "python" else "//"
    blocks = []
    for c in chunks:
        if c.get("kind") == "stop" or not c.get("path"):
            break
        if c["path"] == "file_path":
            hdr = f"{label} Potential invoked APIs:"
        else:
            hdr = f"{label} file path: {c['path']}\n{label} lines: {c['line_start']}-{c['line_end']}"
        body = "\n".join(f"{label} {line}" for line in c["code"].split("\n") if line)
        blocks.append(f"{hdr}\n{body}".strip())
    return blocks


def build_ablation_prompt(
    arm: str,
    task: dict[str, Any],
    trace: dict[str, Any],
    pool_files: list[dict[str, Any]],
    tok: Any,
) -> tuple[list[int], int, int]:
    lang = task["language"]
    top10 = list(trace["aligncoder_top10"])
    path_text = f"\n\n# file path: {task['path']}\n\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)
    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)
    
    symbols = extract_cursor_context_symbols(task["left_context"], lang)
    
    if arm == "aircoder_r":
        # AIRCoder (ACL 2026): 8,000 input tokens limit, 800 tokens cross-file budget
        blocks = format_aircoder_chunks(top10, lang)
        cross_text = "\n\n".join(blocks)
        cross_ids = tok.encode(cross_text, add_special_tokens=False)[:800]
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)
        
    elif arm == "cadre_ablation_coalesce":
        # Ablation 1: Expand to 3584 tokens + file coalescing (no CS-STR, no CSSI)
        blocks = coalesce_chunks_by_file(top10, lang)
        cross_text = "\n\n".join(blocks)
        cross_ids = tok.encode(cross_text, add_special_tokens=False)[:3584]
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)
        
    elif arm == "cadre_ablation_cssi_flat":
        # Ablation 2: Add CSSI Contract, but placed at prompt head (monotonic order, no DCA)
        cssi_contract = extract_cssi_contract(symbols, pool_files, lang, max_tokens=300, tok=tok)
        cssi_ids = tok.encode(cssi_contract + "\n\n", add_special_tokens=False) if cssi_contract else []
        
        rem_cross = max(0, 3584 - len(cssi_ids))
        blocks = coalesce_chunks_by_file(top10, lang)
        exemplar_text = "\n\n".join(blocks)
        exemplar_ids = tok.encode(exemplar_text, add_special_tokens=False)[:rem_cross]
        
        # Monotonic packing: CSSI at top -> Exemplars -> Current file
        cross_ids = cssi_ids + exemplar_ids
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)
        
    elif arm == "cadre_full_dual_channel":
        # Full CADRE: Exemplars (Channel A) -> CSSI Contract (Channel B) adjacent to cursor file
        cssi_contract = extract_cssi_contract(symbols, pool_files, lang, max_tokens=300, tok=tok)
        cssi_ids = tok.encode("\n\n" + cssi_contract, add_special_tokens=False) if cssi_contract else []
        
        rem_cross = max(0, 3584 - len(cssi_ids))
        blocks = coalesce_chunks_by_file(top10, lang)
        exemplar_text = "\n\n".join(blocks)
        exemplar_ids = tok.encode(exemplar_text, add_special_tokens=False)[:rem_cross]
        
        # Dual-Channel order: Channel A (Exemplars) -> Channel B (Contract) -> Current File
        cross_ids = exemplar_ids + cssi_ids
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)
        
    else:
        raise ValueError(f"Unknown arm: {arm}")


def run_iteration(state_ns: dict[str, Any], subset_per_benchmark: int = 64, tag: str = "aircoder_ablation_256") -> dict[str, Any]:
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
    selected_tasks = []
    for b in benchmarks:
        b_tasks = [t for t in all_tasks if t["benchmark"] == b][:subset_per_benchmark]
        selected_tasks.extend(b_tasks)
        
    print(f"Executing AIRCoder vs CADRE Multi-Gap Ablation: {len(selected_tasks)} tasks across {len(benchmarks)} benchmarks...", flush=True)
    
    arms = ["aircoder_r", "cadre_ablation_coalesce", "cadre_ablation_cssi_flat", "cadre_full_dual_channel"]
    batch_prompts: list[list[int]] = []
    meta: list[tuple[str, dict[str, Any], int, int]] = []
    
    t0 = time.time()
    for task in selected_tasks:
        key = (task["benchmark"], task["task_id"])
        tr = traces.get(key) or traces.get(task["task_id"])
        if not tr:
            continue
        pool_id = task.get("pool_id")
        f_ids = pools.get(pool_id, []) if pool_id else []
        pool_files = [files[fid] for fid in f_ids if fid in files]
        
        for arm in arms:
            p_ids, p_len, c_len = build_ablation_prompt(arm, task, tr, pool_files, tok)
            batch_prompts.append(p_ids)
            meta.append((arm, task, p_len, c_len))
            
    print(f"Generated {len(batch_prompts)} prompts in {time.time() - t0:.2f}s. Submitting to vLLM...", flush=True)
    
    # max_tokens=64 for generation (consistent with official evaluation)
    sp = SamplingParams(max_tokens=64, temperature=0.0)
    t_gen = time.time()
    prompts_payload = [{"prompt_token_ids": p_ids} for p_ids in batch_prompts]
    outputs = engine.generate(prompts_payload, sampling_params=sp)
    gen_time = time.time() - t_gen
    print(f"vLLM completed {len(outputs)} generations in {gen_time:.2f}s ({len(outputs)/gen_time:.1f} req/s)!", flush=True)
    
    # Evaluation
    predictions_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    results_rows: list[dict[str, Any]] = []
    
    for (arm, task, p_len, c_len), out in zip(meta, outputs):
        token_ids = list(out.outputs[0].token_ids)
        raw_pred = tok.decode(token_ids, skip_special_tokens=True)
        lang = task["language"]
        bm = task["benchmark"]
        
        if bm in ("cceval_python", "cceval_java"):
            pred = postprocess_cceval(task["left_context"], raw_pred, lang, parser_map)
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
            es = 1.0 - (levenshtein_distance(target_eval, pred_eval) / max_l) if max_l else 1.0
        else:
            em = int(pred_lines == gt_lines)
            target_eval = "\n".join(gt_lines)
            pred_eval = "\n".join(pred_lines)
            max_l = max(len(target_eval), len(pred_eval))
            es = 1.0 - (levenshtein_distance(target_eval, pred_eval) / max_l) if max_l else 1.0
            
        pred_ids = extract_identifiers(clean_pred, lang)
        gt_ids = extract_identifiers(clean_gt, lang)
        id_em = int(pred_ids == gt_ids)
        
        p_set = set(pred_ids)
        g_set = set(gt_ids)
        tp = len(p_set & g_set)
        fp = len(p_set - g_set)
        fn = len(g_set - p_set)
        id_f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
        
        row = {
            "task_id": task["task_id"],
            "benchmark": bm,
            "repo_id": task["task_id"].split("/")[0],
            "arm": arm,
            "em": em,
            "es": es,
            "id_em": id_em,
            "id_f1": id_f1,
            "prompt_tokens": p_len,
            "crossfile_tokens": c_len,
            "prediction": clean_pred,
            "raw_prediction": raw_pred,
            "groundtruth": clean_gt,
        }
        predictions_by_arm[arm].append(row)
        results_rows.append(row)
        
    summary: dict[str, Any] = {"tag": tag, "subset_per_benchmark": subset_per_benchmark, "arms": {}}
    
    for arm in arms:
        rows = predictions_by_arm[arm]
        arm_stats: dict[str, Any] = {
            "overall": {
                "em": round(sum(r["em"] for r in rows) / len(rows) * 100, 2),
                "es": round(sum(r["es"] for r in rows) / len(rows) * 100, 2),
                "id_em": round(sum(r["id_em"] for r in rows) / len(rows) * 100, 2),
                "id_f1": round(sum(r["id_f1"] for r in rows) / len(rows) * 100, 2),
                "count": len(rows),
            },
            "by_benchmark": {},
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
            
            arm_stats["by_benchmark"][b] = {
                "micro_em": round(sum(r["em"] for r in b_rows) / len(b_rows) * 100, 2),
                "macro_em": round(macro_em, 2),
                "micro_es": round(sum(r["es"] for r in b_rows) / len(b_rows) * 100, 2),
                "macro_es": round(macro_es, 2),
                "id_em": round(sum(r["id_em"] for r in b_rows) / len(b_rows) * 100, 2),
                "id_f1": round(sum(r["id_f1"] for r in b_rows) / len(b_rows) * 100, 2),
                "count": len(b_rows),
            }
        summary["arms"][arm] = arm_stats
        
    summary["overall_by_arm"] = {arm: s["overall"] for arm, s in summary["arms"].items()}
    
    out_dir = ROOT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with gzip.open(out_dir / f"{tag}_predictions.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in results_rows:
            f.write(json.dumps(r) + "\n")
            
    print(f"Results saved to {tag}_metrics.json", flush=True)
    return summary


if __name__ == "__main__":
    print("AIRCoder vs CADRE Multi-Gap Ablation Runner loaded successfully.")
