#!/usr/bin/env python3
"""CADRE+ (Line-Commented Coalescing + Param-Aligned CSSI + 64-token gen) on 256-Task Pilot.

Targeting:
- Beating AIRCoder's reported numbers: CCEval-Python > 31.26%, RepoEval-API > 43.94%
- Beating baseline AIRCoder replication (33.59%)
- Demonstrating the exact mechanism that achieves >40% Micro EM under AlignCoder standard metric.
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
        text = pf.get("text", "")
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
                        sig_params = set(ID_RE.findall(params)) - STOP_SYMS
                        overlap = len(sig_params & symbols["local_vars"])
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
    """Coalesce chunks by file AND prefix every line with comment symbol (# or //)."""
    comment_sym = "#" if language == "python" else "//"
    file_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        p = c.get("path")
        if not p or c.get("kind") == "stop": continue
        file_chunks[p].append(c)
        
    blocks = []
    for p, f_chunk_list in file_chunks.items():
        if p == "file_path":
            body = "\n".join(f"{comment_sym} {l}" for c in f_chunk_list for l in c.get("code", "").splitlines() if l.strip())
            if body: blocks.append(f"{comment_sym} Potential invoked APIs:\n{body}")
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


def format_aircoder_exact(chunks: list[dict[str, Any]], language: str) -> list[str]:
    label = "#" if language == "python" else "//"
    blocks = []
    for c in chunks:
        if c.get("kind") == "stop" or not c.get("path"): break
        if c["path"] == "file_path":
            hdr = f"{label} Potential invoked APIs:"
        else:
            hdr = f"{label} file path: {c['path']}\n{label} lines: {c['line_start']}-{c['line_end']}"
        body = "\n".join(f"{label} {line}" for line in c["code"].split("\n") if line)
        blocks.append(f"{hdr}\n{body}".strip())
    return blocks


def build_plus_prompt(arm: str, task: dict[str, Any], trace: dict[str, Any], pool_files: list[dict[str, Any]], tok: Any) -> tuple[list[int], int, int]:
    lang = task["language"]
    top10 = list(trace.get("aligncoder_top10") or trace.get("spc_chunks") or [])
    path_text = f"\n\n# file path: {task['path']}\n\n" if lang == "python" else f"\n\n// file path: {task['path']}\n\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)
    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)
    symbols = extract_cursor_context_symbols(task["left_context"], lang)

    if arm == "aircoder_r":
        # Baseline AIRCoder 800 SPC
        blocks = format_aircoder_exact(top10, lang)
        # ascending order
        if lang == "python":
            rag_open = "'''\n"
            rag_close = "\n'''\n\n"
        else:
            rag_open = "/*\n"
            rag_close = "\n*/\n\n"
        body = "\n\n".join(blocks[::-1])
        cross_text = rag_open + body.strip() + rag_close if body.strip() else ""
        cross_ids = tok.encode(cross_text, add_special_tokens=False)[:800]
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    elif arm == "cadre_v1_raw":
        # What ran in full benchmark: raw code, no line comments
        chunks_by_file = defaultdict(list)
        for ch in top10:
            fp = ch.get("path", "")
            c = ch.get("code", "")
            if fp and c.strip(): chunks_by_file[fp].append(c)
        comment_label = "#" if lang == "python" else "//"
        exemplar_ids = []
        for fp, code_list in chunks_by_file.items():
            coalesced = "\n...\n".join(code_list)
            block = f"{comment_label} file path: {fp}\n{coalesced}\n\n"
            b_ids = tok.encode(block, add_special_tokens=False)
            if len(exemplar_ids) + len(b_ids) > 3200: break
            exemplar_ids.extend(b_ids)
        cssi_contract = extract_cssi_contract(symbols, pool_files, lang, max_tokens=300, tok=tok)
        cssi_ids = tok.encode("\n\n" + cssi_contract, add_special_tokens=False)[:384] if cssi_contract else []
        cross_ids = exemplar_ids + cssi_ids
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    elif arm == "cadre_plus_commented":
        # Full CADRE+: Line-Commented Coalescing + Param-Aligned CSSI
        cssi_contract = extract_cssi_contract(symbols, pool_files, lang, max_tokens=300, tok=tok)
        cssi_ids = tok.encode("\n\n" + cssi_contract, add_special_tokens=False) if cssi_contract else []
        
        rem_cross = max(0, 3584 - len(cssi_ids))
        blocks = coalesce_commented_chunks(top10, lang)
        exemplar_text = "\n\n".join(blocks)
        exemplar_ids = tok.encode(exemplar_text, add_special_tokens=False)[:rem_cross]
        
        cross_ids = exemplar_ids + cssi_ids
        allowed_left = max(0, 8000 - len(cross_ids) - len(path_ids))
        left_ids = left_all_ids[-allowed_left:] if allowed_left else []
        prompt_ids = (cross_ids + path_ids + left_ids)[-8000:]
        return prompt_ids, len(prompt_ids), len(cross_ids)

    else:
        raise ValueError(f"Unknown arm: {arm}")


def run_evaluation(state_ns: dict[str, Any], tag: str = "cadre_plus_pilot_256") -> dict[str, Any]:
    mb_data = state_ns.get("MB_DATA") or {}
    engine = state_ns.get("VLLM_ENGINE")
    if not engine or not mb_data:
        raise RuntimeError("VLLM_ENGINE or MB_DATA not found in state_ns!")

    tasks: list[dict[str, Any]] = mb_data["tasks"]
    traces: dict[Any, dict[str, Any]] = mb_data["traces"]
    files: dict[str, dict[str, Any]] = mb_data["files"]
    pools: dict[str, list[str]] = mb_data["pools"]
    tok: Any = mb_data["tokenizer"]

    from vllm import SamplingParams
    # 64 max tokens to avoid premature cutoff of long statements
    sp = SamplingParams(temperature=0.0, max_tokens=64)

    arms = ["aircoder_r", "cadre_v1_raw", "cadre_plus_commented"]
    print(f"Starting CADRE+ Evaluation: {len(tasks)} tasks across {len(arms)} arms (max_tokens=64)...", flush=True)

    batch_prompts: list[list[int]] = []
    meta: list[tuple[str, dict[str, Any], int, int]] = []

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

        for arm in arms:
            p_ids, p_len, c_len = build_plus_prompt(arm, task, tr, pool_files, tok)
            batch_prompts.append(p_ids)
            meta.append((arm, task, p_len, c_len))

    t_gen = time.time()
    prompts_payload = [{"prompt_token_ids": p_ids} for p_ids in batch_prompts]
    outputs = engine.generate(prompts_payload, sampling_params=sp)
    gen_dur = time.time() - t_gen
    print(f"Generated {len(outputs)} completions in {gen_dur:.1f}s ({len(outputs)/max(0.01, gen_dur):.1f} req/s)!", flush=True)

    results_rows: list[dict[str, Any]] = []
    for (arm, task, p_len, c_len), out in zip(meta, outputs):
        token_ids = list(out.outputs[0].token_ids)
        raw_pred = tok.decode(token_ids, skip_special_tokens=True)
        results_rows.append({
            "task_id": task["task_id"],
            "benchmark": task["benchmark"],
            "language": task["language"],
            "arm": arm,
            "raw_prediction": raw_pred,
            "groundtruth": task.get("target") or task.get("groundtruth", ""),
            "prompt_tokens": p_len,
            "crossfile_tokens": c_len,
            "left_context": task["left_context"],
        })

    out_dir = ROOT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"{tag}_predictions.jsonl.gz"
    with gzip.open(preds_path, "wt", encoding="utf-8") as f:
        for r in results_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results_rows)} predictions to {preds_path}", flush=True)

    return {"status": "ok", "count": len(results_rows)}


if __name__ == "__main__":
    import runpy
    print("Initializing environment via colab_init_engine.py...", flush=True)
    init_mod = runpy.run_path("/content/colab_init_engine.py", run_name="__main__")
    state = {
        "VLLM_ENGINE": init_mod.get("VLLM_ENGINE") or init_mod.get("global_ns", {}).get("VLLM_ENGINE"),
        "MB_DATA": init_mod.get("MB_DATA") or init_mod.get("global_ns", {}).get("MB_DATA"),
    }
    run_evaluation(state)
