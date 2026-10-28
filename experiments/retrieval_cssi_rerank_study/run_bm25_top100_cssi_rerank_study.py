#!/usr/bin/env python3
"""
run_bm25_top100_cssi_rerank_study.py
Evaluates Reranking directly on BM25 Top 100 candidates with CSSI on Google Colab A100.

Architecture:
1. Stage 1 (BM25):
   - Standard 20-line tail query against all chunked blocks in the repo.
   - Retrieves Top 100 candidate blocks per task.
2. Stage 2 (Reranking on Top 100):
   - Baseline: RLRetriever (UniXcoder dense cosine) picks Top 10.
   - Proposed (CSSI Rerank): RLRetriever dense score + CSSI Structural Alignment Score
     (Class match, callee method match, receiver match from TypedContextTree)
     reranks all 100 blocks and picks Top 10.
3. Stage 3 (Generation & Official Metrics):
   - Generates completions via vLLM (DeepSeek-Coder-1.3B).
   - Computes official EM and ES across benchmarks (cceval_python, cceval_java, repoeval_api, repoeval_line).
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import json
import os
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import subprocess
for pkg in ["rank_bm25", "fuzzywuzzy", "python-Levenshtein", "editdistance"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams
import editdistance
from fuzzywuzzy import fuzz

RETRIEVER_MODEL = "nov3630/RLCoder"
GENERATOR_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"

STOP_WORDS = {
    "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
    "del", "elif", "else", "except", "finally", "for", "from", "global", "if",
    "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
    "return", "try", "while", "with", "yield", "self", "cls", "True", "False",
    "None", "int", "str", "float", "bool", "list", "dict", "set", "tuple",
    "public", "private", "protected", "static", "final", "void", "new", "this",
    "super", "extends", "implements", "interface", "package", "throws", "throw",
    "catch", "instanceof", "synchronized", "volatile", "transient", "native",
}
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# -------------------------------------------------------------
# Metric Helpers
# -------------------------------------------------------------
def cal_edit_sim(pred: str, gt: str) -> float:
    return float(fuzz.ratio(pred.strip(), gt.strip()))


def repoeval_es(target: str, pred: str) -> float:
    t_lines = [l.strip() for l in target.splitlines() if l.strip()]
    p_lines = [l.strip() for l in pred.splitlines() if l.strip()][:len(t_lines)]
    t_str = "\n".join(t_lines)
    p_str = "\n".join(p_lines)
    if not t_str and not p_str: return 100.0
    max_l = max(len(t_str), len(p_str))
    if max_l == 0: return 100.0
    return (1.0 - (editdistance.eval(t_str, p_str) / max_l)) * 100.0


def get_py_statement_line(pred: str, gt: str) -> Tuple[str, str]:
    pred = re.sub(r"#.*", "", pred)
    gt = re.sub(r"#.*", "", gt)
    gt_lines = [l.strip() for l in gt.splitlines() if l.strip()]
    n = max(1, len(gt_lines))
    pred_lines = [l.strip() for l in pred.splitlines() if l.strip()][:n]
    return "\n".join(pred_lines), "\n".join(gt_lines)


def get_bracket_stmt(completion: str) -> str:
    end_idx = None
    for i in range(len(completion)):
        if completion[i] in [";", "}", "{"]:
            end_idx = i
            break
    return completion[:end_idx + 1] if end_idx is not None else completion


def extract_identifiers(code: str) -> Set[str]:
    clean = re.sub(r"#.*", "", code)
    clean = re.sub(r"//.*", "", clean)
    clean = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'', "", clean)
    words = WORD_RE.findall(clean)
    return set(w for w in words if w not in STOP_WORDS and len(w) > 1)


def evaluate_records_official(records: List[Dict[str, Any]], pred_key: str) -> Dict[str, Any]:
    by_bench = defaultdict(list)
    for r in records:
        bm = r["benchmark"]
        pred = r.get(pred_key, "")
        gt = r["groundtruth"]
        repo = r["task_id"].split("/")[0]

        if bm.startswith("repoeval"):
            t_lines = [l.strip() for l in gt.splitlines() if l.strip()]
            p_lines = [l.strip() for l in pred.splitlines() if l.strip()][:len(t_lines)]
            em = int(len(t_lines) == len(p_lines) and t_lines == p_lines)
            es = repoeval_es(gt, pred)
        elif bm == "cceval_java":
            gt_s = get_bracket_stmt(re.sub(r"//.*", "", gt)).strip()
            pred_s = get_bracket_stmt(re.sub(r"//.*", "", pred)).strip()
            em = int("".join(pred_s.split()) == "".join(gt_s.split()) and len(gt_s) > 0)
            es = cal_edit_sim(pred_s, gt_s)
        else:
            pred_s, gt_s = get_py_statement_line(pred, gt)
            em = int("".join(pred_s.split()) == "".join(gt_s.split()) and len(gt_s) > 0)
            es = cal_edit_sim(pred_s, gt_s)

        gt_ids = extract_identifiers(gt)
        pred_ids = extract_identifiers(pred)
        if not gt_ids and not pred_ids:
            id_em, id_f1 = 1, 100.0
        elif not gt_ids or not pred_ids:
            id_em, id_f1 = 0, 0.0
        else:
            id_em = int(gt_ids == pred_ids)
            p = len(gt_ids & pred_ids) / len(pred_ids)
            r = len(gt_ids & pred_ids) / len(gt_ids)
            id_f1 = (2 * p * r / (p + r)) * 100.0 if (p + r) > 0 else 0.0

        by_bench[bm].append({"repo": repo, "em": em, "es": es, "id_em": id_em, "id_f1": id_f1})

    summary = {}
    macro_em, macro_es, macro_id_em, macro_id_f1 = [], [], [], []
    for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
        items = by_bench[bm]
        if not items: continue
        n = len(items)
        if bm.startswith("repoeval"):
            repo_map = defaultdict(list)
            repo_es = defaultdict(list)
            repo_id_em = defaultdict(list)
            repo_id_f1 = defaultdict(list)
            for it in items:
                repo_map[it["repo"]].append(it["em"])
                repo_es[it["repo"]].append(it["es"])
                repo_id_em[it["repo"]].append(it["id_em"])
                repo_id_f1[it["repo"]].append(it["id_f1"])
            b_em = sum(sum(v)/len(v) for v in repo_map.values()) / len(repo_map) * 100.0
            b_es = sum(sum(v)/len(v) for v in repo_es.values()) / len(repo_es)
            b_id_em = sum(sum(v)/len(v) for v in repo_id_em.values()) / len(repo_id_em) * 100.0
            b_id_f1 = sum(sum(v)/len(v) for v in repo_id_f1.values()) / len(repo_id_f1)
        else:
            b_em = sum(it["em"] for it in items) / n * 100.0
            b_es = sum(it["es"] for it in items) / n
            b_id_em = sum(it["id_em"] for it in items) / n * 100.0
            b_id_f1 = sum(it["id_f1"] for it in items) / n

        summary[bm] = {
            "em": round(b_em, 2),
            "es": round(b_es, 2),
            "id_em": round(b_id_em, 2),
            "id_f1": round(b_id_f1, 2),
            "count": n,
        }
        macro_em.append(b_em)
        macro_es.append(b_es)
        macro_id_em.append(b_id_em)
        macro_id_f1.append(b_id_f1)

    if macro_em:
        summary["macro_mean"] = {
            "em": round(sum(macro_em) / len(macro_em), 2),
            "es": round(sum(macro_es) / len(macro_es), 2),
            "id_em": round(sum(macro_id_em) / len(macro_id_em), 2),
            "id_f1": round(sum(macro_id_f1) / len(macro_id_f1), 2),
        }
    return summary


# -------------------------------------------------------------
# Block Chunking & CSSI Signals
# -------------------------------------------------------------
def split_into_smaller_blocks(file_path: str, code_content: str, language: str) -> List[Dict[str, Any]]:
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
    temp = []
    for mb in mini_blocks:
        if len(mb) > max_len:
            for idx in range(0, len(mb), max_len):
                temp.append(mb[idx:idx + max_len])
        else:
            temp.append(mb)

    processed = []
    line_cur = 1
    for mb in temp:
        start_line = line_cur
        end_line = line_cur + len(mb) - 1
        processed.append({
            "file_path": file_path,
            "line_start": start_line,
            "line_end": end_line,
            "code_content": "\n".join(mb),
            "language": language,
            "is_stop": False,
        })
        line_cur = end_line + 1
    return processed


def extract_cssi_signals(left_context: str) -> Dict[str, Any]:
    lines = left_context.splitlines()
    recent = lines[-35:] if len(lines) >= 35 else lines
    recent_text = "\n".join(recent)
    cur_line = lines[-1] if lines else ""

    receivers = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", cur_line)
    callees = re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", cur_line)

    assigned_types = {}
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", recent_text):
        assigned_types[m.group(1)] = m.group(2)
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", recent_text):
        assigned_types[m.group(1)] = m.group(2)
    for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[;=]", recent_text):
        assigned_types[m.group(2)] = m.group(1)

    imported_symbols = {}
    for m in re.finditer(r"(?:from\s+([A-Za-z0-9_.]+)\s+import\s+([^#\n]+)|import\s+([^#\n]+))", left_context):
        if m.group(1) and m.group(2):
            raw_syms = m.group(2).replace("(", "").replace(")", "").split(",")
            for s in raw_syms:
                s = s.strip()
                if " as " in s:
                    orig, alias = s.split(" as ")
                    imported_symbols[alias.strip()] = orig.strip()
                elif s:
                    imported_symbols[s] = s
        elif m.group(3):
            for s in m.group(3).split(","):
                s = s.strip().rstrip(";")
                short = s.split(".")[-1]
                imported_symbols[short] = s

    target_types = set()
    for r in receivers:
        if r in assigned_types:
            target_types.add(assigned_types[r])
        elif r in imported_symbols:
            target_types.add(imported_symbols[r])
        elif r and r[0].isupper():
            target_types.add(r)

    return {
        "receivers": set(receivers),
        "callees": set(callees),
        "target_types": target_types,
        "cur_line_ids": set(WORD_RE.findall(cur_line)) - STOP_WORDS,
    }


def compute_cssi_bonus(code_content: str, signals: Dict[str, Any]) -> float:
    c_tokens = set(WORD_RE.findall(code_content))
    score = 0.0

    target_types = signals["target_types"]
    if target_types:
        for tt in target_types:
            if tt in c_tokens: score += 1.5
            if f"class {tt}" in code_content or f"interface {tt}" in code_content:
                score += 3.5

    callees = signals["callees"]
    if callees:
        for cal in callees:
            if cal in c_tokens: score += 1.2
            if f"def {cal}" in code_content or f"{cal}(" in code_content:
                score += 2.5

    receivers = signals["receivers"]
    if receivers:
        for r in receivers:
            if r in c_tokens: score += 0.8

    cur_line_ids = signals["cur_line_ids"]
    overlap = len(cur_line_ids & c_tokens)
    score += min(overlap * 0.3, 1.5)
    return score


# -------------------------------------------------------------
# Prompt Construction
# -------------------------------------------------------------
def coalesce_chunks_for_prompt(chunks: List[Dict[str, Any]], language: str, max_tokens: int = 1536) -> str:
    comment_sym = "#" if language == "python" else "//"
    chunks_by_file = defaultdict(list)
    for c in chunks:
        fp = c.get("file_path") or c.get("path") or ""
        if not fp or c.get("is_stop"):
            continue
        code = c.get("code_content") or c.get("code") or c.get("text") or ""
        if code.strip():
            chunks_by_file[fp].append(code.strip())

    blocks = []
    for fp, clist in chunks_by_file.items():
        header = f"{comment_sym} file path: {fp}\n"
        body = "\n...\n".join(clist)
        blocks.append(f"{header}{body}\n\n")

    full_text = "".join(blocks)
    words = full_text.split()
    if len(words) > max_tokens:
        full_text = " ".join(words[:max_tokens])
    return full_text


def build_completion_prompt(
    task: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    tok: Any,
    max_context: int = 3072,
    crossfile_budget: int = 1536,
    headroom: int = 64,
) -> str:
    lang = task["language"]
    cs = "#" if lang == "python" else "//"
    path = task.get("path", "file.py" if lang == "python" else "File.java")

    path_text = f"\n\n{cs} file path: {path}\n\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)

    cross_text = coalesce_chunks_for_prompt(candidates[:10], lang, max_tokens=crossfile_budget)
    cross_ids = tok.encode(cross_text, add_special_tokens=False)[:crossfile_budget]

    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)

    max_input_tokens = max_context - headroom
    allowed_left = max(0, max_input_tokens - len(cross_ids) - len(path_ids))
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []

    final_ids = (cross_ids + path_ids + left_ids)[-max_input_tokens:]
    return tok.decode(final_ids)


# -------------------------------------------------------------
# Main Execution: BM25 Top 100 -> CSSI Rerank -> Top 10
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/content")
    parser.add_argument("--output-dir", default="/content/cadre-results/bm25_top100_rerank")
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-per-bm", type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("EMPIRICAL STUDY: RERANKING BM25 TOP-100 WITH CSSI STRUCTURAL ALIGNMENT")
    print(f"Max Context: {args.max_model_len} | Generator: {GENERATOR_MODEL}")
    print("=" * 85, flush=True)

    # 1. Load Files & Tasks
    print("\n[Step 1/5] Loading benchmark tasks and repository files...", flush=True)
    bundle_path = data_root / "full_benchmark_8004_bundle.tar.gz"
    if not (data_root / "tasks.jsonl.gz").exists() and bundle_path.exists():
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(data_root)

    files_path = data_root / "files.jsonl.gz"
    tasks_path = data_root / "tasks.jsonl.gz"
    files = {r["file_id"]: {"path": r["path"], "text": r["text"]} for r in map(json.loads, gzip.open(files_path, "rt"))}
    all_tasks = [json.loads(line) for line in gzip.open(tasks_path, "rt")]

    if args.limit_per_bm:
        selected = []
        for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
            selected.extend([t for t in all_tasks if t["benchmark"] == bm][:args.limit_per_bm])
        tasks = selected
    elif args.smoke:
        selected = []
        for bm in ["cceval_python", "cceval_java", "repoeval_api", "repoeval_line"]:
            selected.extend([t for t in all_tasks if t["benchmark"] == bm][:50])
        tasks = selected
    else:
        tasks = all_tasks

    print(f"✓ Total tasks to evaluate: {len(tasks)}", flush=True)

    # 2. Stage 1: BM25 Extraction (Top 100 blocks per task)
    print("\n[Step 2/5] Running Stage 1: BM25 Candidate Extraction (Top 100 blocks)...", flush=True)
    t_bm25_start = time.time()
    bm25_top100_by_task = []

    for idx, t in enumerate(tasks):
        lang = t["language"]
        pool_fids = t.get("file_ids_by_path", {})
        blocks = []
        for rel_path, fid in pool_fids.items():
            if fid in files:
                blocks.extend(split_into_smaller_blocks(rel_path, files[fid]["text"], lang))

        if not blocks:
            top100 = []
        else:
            corpus = [b["code_content"].lower().split() for b in blocks]
            bm25 = BM25Okapi(corpus)
            qlines = [l for l in t["left_context"].splitlines() if l.strip()][-20:]
            qtokens = "\n".join(qlines).lower().split()
            scores = bm25.get_scores(qtokens)
            top100_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:100]
            top100 = [dict(blocks[i], bm25_score=float(scores[i])) for i in top100_idx]

        # Stop block
        top100.append({
            "file_path": "",
            "line_start": 0,
            "line_end": 0,
            "code_content": "Don't need cross file context for completion",
            "language": lang,
            "is_stop": True,
            "bm25_score": 0.0,
        })
        bm25_top100_by_task.append(top100)

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(tasks):
            print(f"  BM25 processed {idx + 1}/{len(tasks)} tasks ({time.time() - t_bm25_start:.1f}s)...", flush=True)

    print(f"✓ Stage 1 BM25 extraction completed in {time.time() - t_bm25_start:.1f}s.")

    # 3. Stage 2: Reranking on Top 100
    # Two arms:
    #   Arm A (Base Neural Rerank): RLRetriever dense cosine picks Top 10 from 100 blocks.
    #   Arm B (CSSI + Neural Rerank): RLRetriever dense score + CSSI structural score picks Top 10 from 100 blocks.
    print("\n[Step 3/5] Running Stage 2: Dense Neural & CSSI Structural Reranking on Top 100...", flush=True)
    t_ret_start = time.time()
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

    # Encode queries
    queries = ["\n".join([l for l in t["left_context"].splitlines() if l.strip()][-20:]) for t in tasks]
    q_embs = encode_texts(queries, is_query=True, max_len=256, batch_size=256)

    cands_base_top10_by_task = []
    cands_cssi_top10_by_task = []

    CHUNK_SIZE = 200
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

        chunk_c_embs = encode_texts(chunk_cands_text, is_query=False, max_len=512, batch_size=512)

        for t_offset, (s_idx, e_idx) in enumerate(chunk_slices):
            t_idx = chunk_start + t_offset
            q_emb = q_embs[t_idx:t_idx + 1]
            c_emb = chunk_c_embs[s_idx:e_idx]
            sims = torch.mm(q_emb, c_emb.t())[0].cpu().numpy()

            cands = bm25_top100_by_task[t_idx]
            t = tasks[t_idx]
            signals = extract_cssi_signals(t["left_context"])

            # 1. Base Rerank: Pick top 10 solely by neural dense cosine
            top10_base_idx = sims.argsort()[-10:][::-1]
            top10_base = [dict(cands[i], dense_score=float(sims[i])) for i in top10_base_idx]
            cands_base_top10_by_task.append(top10_base)

            # 2. Proposed CSSI Rerank on Top 100: Combine dense cosine with CSSI structural bonus
            scored_cssi = []
            for i, b in enumerate(cands):
                if b.get("is_stop"):
                    scored_cssi.append((-999.0, i))
                    continue
                d_score = float(sims[i])
                s_bonus = compute_cssi_bonus(b["code_content"], signals)
                # Dense score normalized ~ [0, 1] + Structural bonus
                combined = (d_score * 2.0) + s_bonus
                scored_cssi.append((combined, i))

            scored_cssi.sort(key=lambda x: x[0], reverse=True)
            top10_cssi_idx = [i for _, i in scored_cssi[:10]]
            top10_cssi = [dict(cands[i], combined_score=float(scored_cssi[rank][0])) for rank, i in enumerate(top10_cssi_idx)]
            cands_cssi_top10_by_task.append(top10_cssi)

    print(f"✓ Neural & CSSI Reranking on Top 100 completed in {time.time() - t_ret_start:.1f}s.")

    # 4. Stage 3: Prompt Building & vLLM Generation
    print("\n[Step 4/5] Building prompts and generating completions via vLLM on A100...", flush=True)
    gen_tok = AutoTokenizer.from_pretrained(GENERATOR_MODEL, trust_remote_code=True)
    if gen_tok.pad_token_id is None: gen_tok.pad_token = gen_tok.eos_token

    prompts_base = [build_completion_prompt(t, cands, tok=gen_tok, max_context=args.max_model_len, headroom=64) for t, cands in zip(tasks, cands_base_top10_by_task)]
    prompts_cssi = [build_completion_prompt(t, cands, tok=gen_tok, max_context=args.max_model_len, headroom=64) for t, cands in zip(tasks, cands_cssi_top10_by_task)]

    del rl_model
    torch.cuda.empty_cache()

    llm = LLM(
        model=GENERATOR_MODEL,
        tokenizer=GENERATOR_MODEL,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        stop=["\n\n\n", "# file path:", "// file path:"],
    )

    print(f"  Generating Arm 1: Base RLCoder (BM25 Top100 -> Dense -> Top10)...", flush=True)
    t_g1 = time.time()
    outs_base = llm.generate(prompts_base, sampling_params)
    print(f"  ✓ Arm 1 completed in {time.time() - t_g1:.2f}s ({len(prompts_base)/(time.time()-t_g1):.1f} req/s).", flush=True)

    print(f"  Generating Arm 2: Proposed CADRE (BM25 Top100 -> Dense+CSSI Rerank -> Top10)...", flush=True)
    t_g2 = time.time()
    outs_cssi = llm.generate(prompts_cssi, sampling_params)
    print(f"  ✓ Arm 2 completed in {time.time() - t_g2:.2f}s ({len(prompts_cssi)/(time.time()-t_g2):.1f} req/s).", flush=True)

    # 5. Evaluate Official EM/ES
    print("\n[Step 5/5] Evaluating Official EM & ES across both pipelines...", flush=True)
    records = []
    for idx, t in enumerate(tasks):
        records.append({
            "task_id": t["task_id"],
            "benchmark": t["benchmark"],
            "language": t["language"],
            "groundtruth": t["groundtruth"],
            "pred_base": outs_base[idx].outputs[0].text,
            "pred_cssi": outs_cssi[idx].outputs[0].text,
        })

    m_base = evaluate_records_official(records, "pred_base")
    m_cssi = evaluate_records_official(records, "pred_cssi")

    results = {
        "bm25_top100_dense_base": m_base,
        "bm25_top100_cssi_rerank": m_cssi,
    }

    # Save
    with open(out_dir / "bm25_top100_rerank_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    with gzip.open(out_dir / "bm25_top100_rerank_predictions.jsonl.gz", "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print("\n" + "=" * 95)
    print("FINAL RESULTS: BM25 TOP-100 RERANKING EXPERIMENT (A100 vLLM)")
    print("=" * 95)
    header = f"{'Configuration':<40} | {'Macro EM':<10} | {'Macro ES':<10} | {'Macro ID-EM':<12} | {'Macro ID-F1':<12}"
    print(header)
    print("-" * len(header))
    mb = m_base["macro_mean"]
    mc = m_cssi["macro_mean"]
    print(f"{'Arm 1: BM25 Top100 -> Dense Base':<40} | {mb['em']:>8.2f}% | {mb['es']:>8.2f}% | {mb['id_em']:>10.2f}% | {mb['id_f1']:>10.2f}%")
    print(f"{'Arm 2: BM25 Top100 -> CSSI Rerank':<40} | {mc['em']:>8.2f}% | {mc['es']:>8.2f}% | {mc['id_em']:>10.2f}% | {mc['id_f1']:>10.2f}%")
    print("-" * len(header))
    delta_em = mc['em'] - mb['em']
    delta_es = mc['es'] - mb['es']
    print(f"{'Improvement (Δ CSSI vs Base)':<40} | {delta_em:>+8.2f} pp | {delta_es:>+8.2f} pp | {mc['id_em'] - mb['id_em']:>+10.2f} pp | {mc['id_f1'] - mb['id_f1']:>+10.2f} pp")
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
