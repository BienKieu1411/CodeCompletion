#!/usr/bin/env python3
"""Iteration 4: Test Expression-Gated Anchor Recency Transport + Non-API Submodular Tail Promotion on Colab A100."""

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

from colab_init_engine import (
    CROSSFILE_TOKENS,
    GENERATOR_CONTEXT,
    GENERATOR_MAX_NEW,
    ROOT_DIR,
    SEED,
    init_globals,
)

ID_RE = re.compile(r"\b[_A-Za-z][_A-Za-z0-9]*\b")


def extract_cursor_markov_blanket(
    left_context: str,
    qh_outputs: list[str],
    window_lines: int = 5,
) -> tuple[set[str], set[str], set[str]]:
    nonempty = [line for line in left_context.splitlines() if line.strip()]
    cursor_tail = "\n".join(nonempty[-window_lines:])
    cursor_syms = set(ID_RE.findall(cursor_tail))
    draft_syms = set(ID_RE.findall("\n".join(qh_outputs)))

    min_count = 2 if len(qh_outputs) >= 2 else 1
    counts: Counter[str] = Counter()
    for out in qh_outputs:
        for s in set(ID_RE.findall(out)):
            if len(s) >= 4:
                counts[s] += 1
    consensus_syms = {s for s, c in counts.items() if c >= min_count}
    return cursor_syms, draft_syms, consensus_syms


def active_markov_edges(
    edges: list[dict[str, str]],
    mb_syms: set[str],
) -> list[dict[str, str]]:
    active = []
    for edge in edges:
        sym = edge.get("symbol", "")
        imp = edge.get("import", "")
        rel = edge.get("relation", "")
        if rel == "imports_module":
            continue
        if (sym and sym != "*" and sym in mb_syms) or (imp and imp.split(".")[-1] in mb_syms):
            active.append(edge)
    return active


def is_open_expression_cursor(left_context: str, top1_cos: float, threshold: float = 0.85) -> bool:
    """Theorem 3 Gate: True when top-1 chunk has high confidence (>=0.85) and cursor is mid-expression."""
    if top1_cos < threshold:
        return False
    nonempty = [line for line in left_context.splitlines() if line.strip()]
    if not nonempty:
        return False
    last = nonempty[-1].rstrip()
    if last.endswith(('"""', "'''", ";", "}", "{", ":")):
        return False
    # Open member access, call parenthesis, assignment, or 'new ' constructor trigger
    if last.endswith((".", "(", "=")) or re.search(r"\b(?:new|await|return|assertRaises)\s+[A-Za-z0-9_.]*$", last):
        return True
    return False


def submodular_tail_rerank(
    top10: list[dict[str, Any]],
    rerank_candidates: list[dict[str, Any]],
    cursor_syms: set[str],
    consensus_syms: set[str],
    act_edges: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Theorem 1 & 2: Lock Top-4, never boost synthetic api_chunks, promote uncovered MB real code chunks."""
    locked = list(top10[:4])
    locked_keys = {(c["path"], c["line_start"], c["line_end"], c["code"]) for c in locked}
    covered_syms: set[str] = set()
    for c in locked:
        covered_syms.update(ID_RE.findall(c.get("code", "")))

    act_dep_syms = {e.get("symbol", "") for e in act_edges if e.get("symbol")}
    uncovered_cons = consensus_syms - covered_syms
    uncovered_deps = act_dep_syms - covered_syms

    # If top-4 already covers all active dependencies and consensus symbols, keep top10 untouched
    if not uncovered_cons and not uncovered_deps:
        return top10

    pool = [
        c for c in rerank_candidates[:16]
        if (c["path"], c["line_start"], c["line_end"], c["code"]) not in locked_keys
    ]

    def score_cand(c: dict[str, Any]) -> float:
        # Never artificially boost synthetic api_chunks (path == 'file_path')
        if c.get("api") or c.get("path") == "file_path" or c.get("kind") == "stop":
            return float(c["cosine"])
        c_syms = set(ID_RE.findall(c.get("code", "")))
        if not (c_syms & cursor_syms):
            return float(c["cosine"])
        u_cons_hits = len(c_syms & uncovered_cons)
        u_dep_hits = len(c_syms & uncovered_deps)
        if u_cons_hits == 0 and u_dep_hits == 0:
            return float(c["cosine"])
        bonus = 0.012 * min(3, u_cons_hits) + 0.025 * min(2, u_dep_hits)
        return float(c["cosine"]) + bonus

    # Only apply if a rank 6..16 candidate strictly beats rank 5's score with uncovered symbols
    rank5_base = score_cand(pool[0]) if pool else 0.0
    best_later = max((score_cand(c) for c in pool[1:]), default=0.0)
    if best_later <= rank5_base + 1e-5:
        return top10

    scored_pool = sorted(pool, key=lambda c: -score_cand(c))
    return locked + scored_pool[:6]


def format_chunks_exact(chunks: list[dict[str, Any]], language: str) -> list[str]:
    label = "#" if language == "python" else "//"
    api_indices = [
        i for i, c in enumerate(chunks)
        if c.get("kind") != "stop" and c.get("path") == "file_path"
    ]
    last_api_idx = api_indices[-1] if api_indices else -1
    blocks = []
    for i, c in enumerate(chunks):
        if c.get("kind") == "stop" or not c.get("path"):
            break
        if c["path"] == "file_path":
            hdr = f"{label} Potential invoked APIs:" if i == last_api_idx else f"{label} Potentially invoked APIs:"
        else:
            hdr = f"{label} file path: {c['path']}\n{label} lines: {c['line_start']}-{c['line_end']}"
        body = "\n".join(f"{label} {line}" for line in c["code"].split("\n") if line)
        blocks.append(f"{hdr}\n{body}".strip())
    return blocks


def reverse_surviving_blocks(blocks: list[str], tok: Any) -> list[int]:
    surviving_blocks = []
    used_tokens = 0
    sep_tokens = len(tok.encode("\n\n", add_special_tokens=False))
    for b in blocks:
        b_ids = tok.encode(b, add_special_tokens=False)
        if used_tokens + len(b_ids) <= CROSSFILE_TOKENS:
            surviving_blocks.append(b)
            used_tokens += len(b_ids) + sep_tokens
        else:
            rem = CROSSFILE_TOKENS - used_tokens
            if rem > 20:
                surviving_blocks.append(tok.decode(b_ids[:rem], skip_special_tokens=True))
            break
    rev_joined = "\n\n".join(reversed(surviving_blocks))
    return tok.encode(rev_joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]


def move_top1_adjacent_ids(blocks: list[str], tok: Any) -> list[int]:
    """Keep [Rank 2, ..., Rank m] in order, and place Rank 1 immediately before # file path: <target>."""
    if len(blocks) <= 1:
        joined = "\n\n".join(blocks)
        return tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]
    b0 = blocks[0]
    b0_ids = tok.encode(b0, add_special_tokens=False)
    sep_tokens = len(tok.encode("\n\n", add_special_tokens=False))
    budget_for_rest = max(0, CROSSFILE_TOKENS - len(b0_ids) - sep_tokens)
    rest_blocks = []
    used = 0
    for b in blocks[1:]:
        b_ids = tok.encode(b, add_special_tokens=False)
        if used + len(b_ids) <= budget_for_rest:
            rest_blocks.append(b)
            used += len(b_ids) + sep_tokens
        else:
            rem = budget_for_rest - used
            if rem > 20:
                rest_blocks.append(tok.decode(b_ids[:rem], skip_special_tokens=True))
            break
    ordered = rest_blocks + [b0]
    joined = "\n\n".join(ordered)
    return tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]


def build_arm_prompt_ids(
    arm: str,
    task: dict[str, Any],
    trace: dict[str, Any],
    tok: Any,
) -> tuple[list[int], int, int]:
    lang = task["language"]
    path_text = f"\n\n# file path: {task['path']}\n\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)
    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)

    cursor_syms, draft_syms, consensus_syms = extract_cursor_markov_blanket(
        task["left_context"], trace["query_enhancement_outputs"], window_lines=5
    )
    mb_syms = cursor_syms | draft_syms
    act_edges = active_markov_edges(trace["dependency_edges"], mb_syms)
    top10 = list(trace["aligncoder_top10"])
    top1_cos = float(top10[0]["cosine"]) if top10 else 0.0
    recency_gate = is_open_expression_cursor(task["left_context"], top1_cos, threshold=0.85)

    if arm == "aligncoder_code":
        blocks = format_chunks_exact(top10, lang)
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]

    elif arm == "mb_recency_gated":
        blocks = format_chunks_exact(top10, lang)
        if recency_gate:
            cross_ids = reverse_surviving_blocks(blocks, tok)
        else:
            joined = "\n\n".join(blocks)
            cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]

    elif arm == "mb_top1_adjacent":
        blocks = format_chunks_exact(top10, lang)
        if recency_gate:
            cross_ids = move_top1_adjacent_ids(blocks, tok)
        else:
            joined = "\n\n".join(blocks)
            cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]

    elif arm == "mb_submodular_tail":
        selected = submodular_tail_rerank(
            top10, trace["rerank_candidates"], cursor_syms, consensus_syms, act_edges
        )
        blocks = format_chunks_exact(selected, lang)
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]

    elif arm == "mb_spt_v4":
        selected = submodular_tail_rerank(
            top10, trace["rerank_candidates"], cursor_syms, consensus_syms, act_edges
        )
        blocks = format_chunks_exact(selected, lang)
        if recency_gate:
            cross_ids = reverse_surviving_blocks(blocks, tok)
        else:
            joined = "\n\n".join(blocks)
            cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]

    else:
        raise ValueError(f"Unknown arm: {arm}")

    allowed_left = max(0, GENERATOR_CONTEXT - len(cross_ids) - len(path_ids) - 10)
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []
    prompt_ids = (cross_ids + path_ids + left_ids)[-GENERATOR_CONTEXT:]
    return prompt_ids, len(prompt_ids), len(cross_ids)


from run_iter3 import (
    aggregate_metrics,
    has_error,
    identifiers,
    levenshtein_distance,
    metric_row,
    paired_bootstrap,
    postprocess,
    remove_comments,
)


def run_iteration(global_ns: dict[str, Any], subset_per_benchmark: int = 16, tag: str = "iter4_64") -> dict[str, Any]:
    if "VLLM_ENGINE" not in global_ns or "MB_DATA" not in global_ns:
        init_globals(global_ns)
    from vllm import SamplingParams

    engine = global_ns["VLLM_ENGINE"]
    data = global_ns["MB_DATA"]
    all_tasks = data["tasks"]
    traces = data["traces"]
    parser_map = data["parser_map"]
    tok = data["tokenizer"]

    by_bench = defaultdict(list)
    for t in all_tasks:
        if len(by_bench[t["benchmark"]]) < subset_per_benchmark:
            by_bench[t["benchmark"]].append(t)
    selected_tasks = [t for b in sorted(by_bench) for t in by_bench[b]]
    print(f"[{tag}] Evaluating {len(selected_tasks)} tasks ({subset_per_benchmark}/benchmark)...", flush=True)

    arms = ["aligncoder_code", "mb_recency_gated", "mb_top1_adjacent", "mb_submodular_tail", "mb_spt_v4"]
    all_token_prompts = []
    prompt_meta = []
    for t in selected_tasks:
        trace = traces[(t["benchmark"], t["task_id"])]
        for arm in arms:
            p_ids, p_toks, c_toks = build_arm_prompt_ids(arm, t, trace, tok)
            all_token_prompts.append({"prompt_token_ids": p_ids})
            prompt_meta.append((arm, t, p_toks, c_toks))

    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=GENERATOR_MAX_NEW, n=1, seed=123)
    t0 = time.time()
    outputs = engine.generate(all_token_prompts, params, use_tqdm=True)
    elapsed = time.time() - t0
    print(f"[{tag}] Generated {len(all_token_prompts)} completions in {elapsed:.2f}s", flush=True)

    rows_by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    all_rows = []
    for (arm, t, p_toks, c_toks), req_out in zip(prompt_meta, outputs):
        token_ids = list(req_out.outputs[0].token_ids)
        raw_pred = tok.decode(token_ids, skip_special_tokens=True)
        row = metric_row(t, raw_pred, parser_map)
        row.update({"arm": arm, "raw_prediction": raw_pred, "prompt_tokens": p_toks, "crossfile_tokens": c_toks})
        rows_by_arm[arm].append(row)
        all_rows.append(row)

    summaries = {arm: aggregate_metrics(rows) for arm, rows in rows_by_arm.items()}
    bootstrap = paired_bootstrap(rows_by_arm, baseline="aligncoder_code")
    overall_by_arm = {}
    for arm, rows in rows_by_arm.items():
        overall_by_arm[arm] = {
            "em": round(100 * sum(r["em"] for r in rows) / len(rows), 4),
            "es": round(100 * sum(r["es"] for r in rows) / len(rows), 4),
            "id_f1": round(100 * sum(r["id_f1"] for r in rows) / len(rows), 4),
            "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / len(rows), 1),
            "mean_crossfile_tokens": round(sum(r["crossfile_tokens"] for r in rows) / len(rows), 1),
        }

    result = {
        "tag": tag,
        "sample_count": len(selected_tasks),
        "subset_per_benchmark": subset_per_benchmark,
        "generation_seconds": round(elapsed, 2),
        "overall_by_arm": overall_by_arm,
        "metrics_by_arm": summaries,
        "paired_bootstrap_vs_aligncoder_code": bootstrap,
    }
    out_dir = ROOT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with gzip.open(out_dir / f"{tag}_predictions.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    sub = int(os.environ.get("SUBSET_PER_BENCHMARK", "16"))
    tag = os.environ.get("ITER_TAG", f"iter4_{sub * 4}")
    run_iteration(globals(), subset_per_benchmark=sub, tag=tag)
