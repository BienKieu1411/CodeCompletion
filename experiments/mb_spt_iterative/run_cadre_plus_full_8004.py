#!/usr/bin/env python3
"""CADRE+ Full Benchmark Evaluation (8,004 Tasks) Across All 4 Benchmarks.

Setting:
- Architecture: CADRE+ (Line-Commented Coalescing + Param-Aligned CSSI + DCA Dual-Channel)
- Generator: deepseek-ai/deepseek-coder-1.3b-base (8k Context, Greedy Decoding T=0.0)
- Generation budget: max_tokens=64 (prevents truncation of long API calls)
- Metrics evaluated:
  1. AlignCoder Official Strict Metric (Tree-sitter AST Statement Match)
  2. AIRCoder Official Metric (Prefix Truncation [:len(target)])
  3. Edit Similarity (ES), Identifier EM, Identifier F1
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
            code_str = c.get("code") or c.get("text", "")
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


def build_cadre_plus_prompt(task: dict[str, Any], trace: dict[str, Any], pool_files: list[dict[str, Any]], tok: Any) -> tuple[list[int], int, int]:
    lang = task["language"]
    top10 = list(trace.get("aligncoder_top10") or trace.get("spc_chunks") or trace.get("retrieved_chunks") or [])
    comment_sym = "#" if lang == "python" else "//"
    path_text = f"\n\n{comment_sym} file path: {task['path']}\n\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)
    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)

    symbols = extract_cursor_context_symbols(task["left_context"], lang)
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


def run_full_evaluation(state_ns: dict[str, Any], tag: str = "cadre_plus_full_8004", batch_size: int = 500) -> dict[str, Any]:
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
    # max_tokens=64 prevents truncation on long statements
    sp = SamplingParams(max_tokens=64, temperature=0.0)

    total_tasks = len(tasks)
    print(f"=== STARTING CADRE+ FULL BENCHMARK EVALUATION: {total_tasks} TASKS ===", flush=True)

    results_rows: list[dict[str, Any]] = []
    out_dir = ROOT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = out_dir / f"{tag}_partial_results.jsonl"

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
        chunk_tasks = tasks[batch_idx : batch_idx + batch_size]
        remaining_chunk = [t for t in chunk_tasks if f"{t['benchmark']}::{t['task_id']}" not in processed_task_keys]
        if not remaining_chunk: continue

        print(f"\n[Batch {batch_idx//batch_size + 1}] Processing {len(remaining_chunk)} tasks (tasks {batch_idx} to {batch_idx + len(chunk_tasks)} / {total_tasks})...", flush=True)

        batch_prompts: list[list[int]] = []
        meta: list[tuple[dict[str, Any], int, int]] = []

        for task in remaining_chunk:
            key = (task["benchmark"], task["task_id"])
            tr = traces.get(key) or traces.get(task["task_id"])
            if not tr: continue

            pool_files = []
            if "file_ids_by_path" in task:
                pool_files = [files[fid] for fid in task["file_ids_by_path"].values() if fid in files]
            elif task.get("pool_id"):
                f_ids = pools.get(task["pool_id"], [])
                pool_files = [files[fid] for fid in f_ids if fid in files]

            p_ids, p_len, c_len = build_cadre_plus_prompt(task, tr, pool_files, tok)
            batch_prompts.append(p_ids)
            meta.append((task, p_len, c_len))

        t_gen = time.time()
        prompts_payload = [{"prompt_token_ids": p_ids} for p_ids in batch_prompts]
        outputs = engine.generate(prompts_payload, sampling_params=sp)
        gen_dur = time.time() - t_gen
        print(f"  Generated {len(outputs)} completions in {gen_dur:.1f}s ({len(outputs)/max(0.01, gen_dur):.1f} req/s)!", flush=True)

        with open(partial_path, "a", encoding="utf-8") as f_out:
            for (task, p_len, c_len), out in zip(meta, outputs):
                token_ids = list(out.outputs[0].token_ids)
                raw_pred = tok.decode(token_ids, skip_special_tokens=True)

                bm = task["benchmark"]
                lang = task["language"]
                gt = task.get("target") or task.get("groundtruth", "")
                parser = parser_map.get(lang)

                # Metric 1: AlignCoder Strict AST Match
                pred_align = postprocess_code_aligncoder(task["left_context"], raw_pred, parser, lang, bm, gt)
                em_align = int(pred_align.strip() == gt.strip())
                es_align = cal_edit_sim_single(pred_align, gt)

                # Metric 2: AIRCoder Prefix Match
                pred_air = postprocess_code_aircoder_official(raw_pred, gt)
                em_air = int(pred_air == "".join([l.strip() for l in gt.splitlines() if l.strip()]))

                pred_ids = extract_identifiers(pred_align, lang)
                gt_ids = extract_identifiers(gt, lang)
                id_em = int(pred_ids == gt_ids)
                p_set = set(pred_ids)
                g_set = set(gt_ids)
                tp = len(p_set & g_set)
                fp = len(p_set - g_set)
                fn = len(g_set - p_set)
                id_f1 = (2 * tp / (2 * tp + fp + fn)) * 100.0 if (2 * tp + fp + fn) else 0.0

                record = {
                    "task_id": task["task_id"],
                    "benchmark": bm,
                    "language": lang,
                    "arm": "cadre_plus",
                    "em_align": em_align,
                    "es_align": es_align,
                    "em_air": em_air,
                    "id_em": id_em,
                    "id_f1": id_f1,
                    "prompt_tokens": p_len,
                    "crossfile_tokens": c_len,
                    "raw_prediction": raw_pred,
                    "prediction": pred_align,
                    "groundtruth": gt,
                }
                results_rows.append(record)
                f_out.write(json.dumps(record) + "\n")

    total_dur = time.time() - t_start
    print(f"\n=== CADRE+ FULL EVALUATION COMPLETED IN {total_dur/60:.2f} MINUTES ===", flush=True)

    # Compute final metrics summary
    benchmarks = ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]
    summary: dict[str, Any] = {
        "tag": tag,
        "total_tasks": len(results_rows),
        "duration_minutes": total_dur / 60.0,
        "by_benchmark": {},
        "overall": {},
    }

    for bm in benchmarks:
        bm_rows = [r for r in results_rows if r["benchmark"] == bm]
        n = len(bm_rows)
        summary["by_benchmark"][bm] = {
            "count": n,
            "em_align": sum(r["em_align"] for r in bm_rows) / n * 100.0 if n else 0,
            "es_align": sum(r["es_align"] for r in bm_rows) / n if n else 0,
            "em_air": sum(r["em_air"] for r in bm_rows) / n * 100.0 if n else 0,
            "id_em": sum(r["id_em"] for r in bm_rows) / n * 100.0 if n else 0,
            "id_f1": sum(r["id_f1"] for r in bm_rows) / n if n else 0,
        }

    tot_n = len(results_rows)
    summary["overall"] = {
        "micro_em_align": sum(r["em_align"] for r in results_rows) / tot_n * 100.0 if tot_n else 0,
        "macro_em_align": sum(summary["by_benchmark"][bm]["em_align"] for bm in benchmarks) / len(benchmarks),
        "micro_es_align": sum(r["es_align"] for r in results_rows) / tot_n if tot_n else 0,
        "micro_em_air": sum(r["em_air"] for r in results_rows) / tot_n * 100.0 if tot_n else 0,
        "macro_em_air": sum(summary["by_benchmark"][bm]["em_air"] for bm in benchmarks) / len(benchmarks),
        "micro_id_em": sum(r["id_em"] for r in results_rows) / tot_n * 100.0 if tot_n else 0,
        "micro_id_f1": sum(r["id_f1"] for r in results_rows) / tot_n if tot_n else 0,
    }

    metrics_path = out_dir / f"{tag}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary metrics to {metrics_path}", flush=True)

    preds_path = out_dir / f"{tag}_predictions.jsonl.gz"
    with gzip.open(preds_path, "wt", encoding="utf-8") as f:
        for r in results_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved full predictions to {preds_path}", flush=True)

    # Print summary table
    print("\n" + "="*80)
    print("KẾT QUẢ CADRE+ FULL 8,004 TASKS (DEEPSEEK-CODER 1.3B):")
    print(f"{'Benchmark':<18} | {'Tasks':<7} | {'AlignCoder Strict EM':<22} | {'AIRCoder Prefix EM':<20}")
    print("-" * 80)
    for bm in benchmarks:
        b_m = summary["by_benchmark"][bm]
        print(f"{bm:<18} | {b_m['count']:>5}   | {b_m['em_align']:>6.2f}% ({b_m['es_align']:>5.2f} ES)      | {b_m['em_air']:>6.2f}%")
    ov = summary["overall"]
    print("-" * 80)
    print(f"{'Micro Overall':<18} | {tot_n:>5}   | {ov['micro_em_align']:>6.2f}% ({ov['micro_es_align']:>5.2f} ES)      | {ov['micro_em_air']:>6.2f}%")
    print(f"{'Macro Overall':<18} | {tot_n:>5}   | {ov['macro_em_align']:>6.2f}%                        | {ov['macro_em_air']:>6.2f}%")
    print("="*80 + "\n")

    return summary


if __name__ == "__main__":
    import sys
    g = globals()
    if g.get("VLLM_ENGINE") is not None and g.get("MB_DATA") is not None:
        print("Using existing in-memory VLLM_ENGINE and MB_DATA from Colab kernel globals.", flush=True)
        state = g
    else:
        print("Initializing environment via colab_init_engine.py...", flush=True)
        import colab_init_engine
        colab_init_engine.init_globals(g)
        state = g
    run_full_evaluation(state)
