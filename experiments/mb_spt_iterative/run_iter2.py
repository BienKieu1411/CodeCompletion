#!/usr/bin/env python3
"""Iteration 2: Fix vLLM token_ids detokenization + test Induction-Preserving MB-SPT on Colab A100."""

from __future__ import annotations

import difflib
import gzip
import json
import keyword
import os
import random
import re
import time
from collections import defaultdict
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
) -> tuple[set[str], set[str]]:
    nonempty = [line for line in left_context.splitlines() if line.strip()]
    cursor_tail = "\n".join(nonempty[-window_lines:])
    cursor_syms = set(ID_RE.findall(cursor_tail))
    draft_syms = set(ID_RE.findall("\n".join(qh_outputs)))
    return cursor_syms, draft_syms


def active_markov_edges(
    edges: list[dict[str, str]],
    mb_syms: set[str],
) -> list[dict[str, str]]:
    """Theorem 1: Keep only dependency edges whose symbol is active in the Cursor Markov Blanket."""
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


def matching_edges_for_chunk(chunk: dict[str, Any], edges: list[dict[str, str]]) -> list[dict[str, str]]:
    matches = []
    is_api = chunk.get("api", chunk.get("path") == "file_path")
    c_paths = chunk.get("source_paths", []) if is_api else [chunk.get("path", "")]
    code = chunk.get("code", "")
    for edge in edges:
        if edge["path"] not in c_paths:
            continue
        sym = edge.get("symbol", "")
        if sym and sym != "*" and re.search(rf"\b{re.escape(sym)}\b", code):
            matches.append(edge)
    return matches


def active_graph_score(chunk: dict[str, Any], act_edges: list[dict[str, str]]) -> float:
    if not act_edges:
        return 0.0
    hits = {e.get("import", e.get("symbol", "")) for e in matching_edges_for_chunk(chunk, act_edges)}
    total = max(1, len({e.get("import", e.get("symbol", "")) for e in act_edges}))
    return min(1.0, len(hits) / total)


def format_chunks_exact(
    chunks: list[dict[str, Any]],
    language: str,
    act_edges: list[dict[str, str]] | None = None,
) -> list[str]:
    """Exact AlignCoder chunk formatting (preserving all lines for [A][B] induction heads)."""
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
        block = f"{hdr}\n{body}".strip()
        if act_edges:
            m_edges = matching_edges_for_chunk(c, act_edges)
            if m_edges:
                notes = list(dict.fromkeys(
                    f"{label} DEPENDENCY {e['relation']}: {e['import']} -> {e['path']}::{e['symbol']}"
                    for e in m_edges if e.get("symbol")
                ))[:1]
                if notes:
                    block = "\n".join(notes) + "\n" + block
        blocks.append(block)
    return blocks


def merge_overlapping_same_file_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge chunks from the exact same file whose line ranges overlap or touch, preserving contiguous lines."""
    out: list[dict[str, Any]] = []
    for c in chunks:
        if c.get("kind") == "stop" or not c.get("path"):
            break
        is_api = c.get("api", c["path"] == "file_path")
        if (
            out
            and not is_api
            and not out[-1].get("api", out[-1]["path"] == "file_path")
            and out[-1]["path"] == c["path"]
            and out[-1].get("line_start", 0) > 0
            and c.get("line_start", 0) > 0
        ):
            s1, e1 = out[-1]["line_start"], out[-1]["line_end"]
            s2, e2 = c["line_start"], c["line_end"]
            if s1 <= s2 <= e1 + 1:
                lines1 = out[-1]["code"].split("\n")
                lines2 = c["code"].split("\n")
                overlap = max(0, e1 - s2 + 1)
                merged_lines = lines1 + lines2[overlap:]
                out[-1] = dict(out[-1])
                out[-1]["line_end"] = max(e1, e2)
                out[-1]["code"] = "\n".join(merged_lines)
                continue
        out.append(c)
    return out


def build_arm_prompt(
    arm: str,
    task: dict[str, Any],
    trace: dict[str, Any],
    tok: Any,
) -> tuple[str, int, int]:
    lang = task["language"]
    path_text = f"\n\n# file path: {task['path']}\n\n"
    path_ids = tok.encode(path_text, add_special_tokens=False)
    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)

    cursor_syms_5, draft_syms = extract_cursor_markov_blanket(
        task["left_context"], trace["query_enhancement_outputs"], window_lines=5
    )
    cursor_syms_3, _ = extract_cursor_markov_blanket(
        task["left_context"], trace["query_enhancement_outputs"], window_lines=3
    )
    mb_syms_5 = cursor_syms_5 | draft_syms
    mb_syms_3 = cursor_syms_3 | draft_syms
    act_edges_5 = active_markov_edges(trace["dependency_edges"], mb_syms_5)
    act_edges_3 = active_markov_edges(trace["dependency_edges"], mb_syms_3)

    top10 = list(trace["aligncoder_top10"])

    if arm == "aligncoder_code":
        blocks = format_chunks_exact(top10, lang, act_edges=None)
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]

    elif arm == "span_merge_clean":
        merged = merge_overlapping_same_file_chunks(top10)
        blocks = format_chunks_exact(merged, lang, act_edges=None)
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]

    elif arm == "mb_gated_rerank_v2":
        # Lock Top-4 anchors; if active Markov Blanket edges exist, rerank Ranks 5..10 from top-25 candidates
        if act_edges_5:
            locked = top10[:4]
            locked_keys = {(c["path"], c["line_start"], c["line_end"], c["code"]) for c in locked}
            pool = [
                c for c in trace["rerank_candidates"][:25]
                if (c["path"], c["line_start"], c["line_end"], c["code"]) not in locked_keys
            ]
            pool_scored = sorted(
                pool,
                key=lambda c: -(c["cosine"] + 0.08 * active_graph_score(c, act_edges_5)),
            )
            selected = locked + pool_scored[:6]
        else:
            selected = top10
        blocks = format_chunks_exact(selected, lang, act_edges=None)
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]

    elif arm == "mb_active_dep_header":
        # Only attach dependency notes for edges active in the tight 3-line Cursor Markov Blanket
        blocks = format_chunks_exact(top10, lang, act_edges=act_edges_3 if act_edges_3 else None)
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]

    elif arm == "mb_spt_v2":
        # Combine Markov-Blanket Gated Tail Reranking + Same-File Span Merge + Active Dependency Header
        if act_edges_5:
            locked = top10[:4]
            locked_keys = {(c["path"], c["line_start"], c["line_end"], c["code"]) for c in locked}
            pool = [
                c for c in trace["rerank_candidates"][:25]
                if (c["path"], c["line_start"], c["line_end"], c["code"]) not in locked_keys
            ]
            pool_scored = sorted(
                pool,
                key=lambda c: -(c["cosine"] + 0.08 * active_graph_score(c, act_edges_5)),
            )
            selected = locked + pool_scored[:6]
        else:
            selected = top10
        merged = merge_overlapping_same_file_chunks(selected)
        blocks = format_chunks_exact(merged, lang, act_edges=act_edges_3 if act_edges_3 else None)
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]

    else:
        raise ValueError(f"Unknown arm: {arm}")

    allowed_left = max(0, GENERATOR_CONTEXT - len(cross_ids) - len(path_ids) - 10)
    left_ids = left_all_ids[-allowed_left:] if allowed_left else []
    prompt_ids = (cross_ids + path_ids + left_ids)[-GENERATOR_CONTEXT:]
    prompt = tok.decode(prompt_ids, skip_special_tokens=True)
    return prompt, len(prompt_ids), len(cross_ids)


def has_error(node: Any) -> bool:
    if node.type == "ERROR":
        return True
    return any(has_error(child) for child in node.children)


def remove_comments(code: str) -> str:
    return re.sub(r"#.*|//.*", "", code)


def postprocess(prompt: str, completion: str, language: str, parser_map: dict[str, Any]) -> str:
    try:
        if language in ("java", "csharp", "typescript"):
            for index, char in enumerate(completion):
                if char in (";", "}", "{"):
                    return completion[: index + 1] if index else completion
            return completion
        if language == "python":
            parser = parser_map.get(language)
            if parser is None:
                return completion
            for index, char in enumerate(completion):
                if char != "\n":
                    continue
                parsed = parser.parse((prompt + completion[:index]).encode("utf-8", errors="replace"))
                if parsed is not None and not has_error(parsed.root_node):
                    return completion[:index].rstrip()
            return completion
        return completion
    except Exception:
        return completion


def identifiers(code: str, language: str) -> list[str]:
    clean = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\'', "", code)
    names = re.findall(r"\w+", clean)
    if language == "python":
        ignored = set(keyword.kwlist) - {"True", "False"}
    else:
        keyword_path = ROOT_DIR / "aligncoder" / "utils" / "keywords" / "java.txt"
        ignored = {line.strip() for line in keyword_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return [name for name in names if re.match(r"[_a-zA-Z][_a-zA-Z0-9]*", name) and name not in ignored]


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


def metric_row(task: dict[str, Any], prediction: str, parser_map: dict[str, Any]) -> dict[str, Any]:
    pred = remove_comments(postprocess(task["left_context"], prediction, task["language"], parser_map)).strip()
    target = remove_comments(task["groundtruth"]).strip()
    pred_lines = [line.strip() for line in pred.splitlines() if line.strip()]
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    em = int(pred_lines == target_lines)
    try:
        from fuzzywuzzy import fuzz
        es = fuzz.ratio(target, pred) / 100.0
    except Exception:
        es = difflib.SequenceMatcher(None, target, pred).ratio()
    pred_ids, target_ids = set(identifiers(pred, task["language"])), set(identifiers(target, task["language"]))
    tp = len(pred_ids & target_ids)
    fp, fn = len(pred_ids - target_ids), len(target_ids - pred_ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    repo_em, repo_es = None, None
    if task["benchmark"].startswith("repoeval"):
        pred_eval_lines = pred_lines[: len(target_lines)]
        target_eval = "\n".join(target_lines)
        pred_eval = "\n".join(pred_eval_lines)
        distance = levenshtein_distance(target_eval, pred_eval)
        repo_es = 1.0 - distance / max(len(target_eval), len(pred_eval)) if max(len(target_eval), len(pred_eval)) else 1.0
        repo_em = float(pred_eval_lines == target_lines)
    return {
        "task_id": task["task_id"],
        "benchmark": task["benchmark"],
        "repo_id": task["task_id"].split("/")[0],
        "prediction": pred,
        "target": target,
        "em": em,
        "es": es,
        "repo_em": repo_em,
        "repo_es": repo_es,
        "id_em": float(pred_ids == target_ids),
        "id_precision": precision,
        "id_recall": recall,
        "id_f1": f1,
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for benchmark in sorted({row["benchmark"] for row in rows}):
        group = [row for row in rows if row["benchmark"] == benchmark]
        metric_keys = ("em", "es") if benchmark.startswith("repoeval") else (
            "em", "es", "id_em", "id_precision", "id_recall", "id_f1"
        )
        summary = {key: 100 * sum(row[key] for row in group) / len(group) for key in metric_keys}
        if benchmark.startswith("repoeval"):
            by_repo = defaultdict(list)
            for row in group:
                by_repo[row["repo_id"]].append(row)
            summary["repo_macro_em"] = 100 * sum(sum(r["repo_em"] for r in v) / len(v) for v in by_repo.values()) / len(by_repo)
            summary["repo_macro_es"] = 100 * sum(sum(r["repo_es"] for r in v) / len(v) for v in by_repo.values()) / len(by_repo)
        result[benchmark] = {k: round(v, 4) for k, v in summary.items()}
    return result


def paired_bootstrap(rows_by_arm: dict[str, list[dict[str, Any]]], baseline: str = "aligncoder_code", rounds: int = 2000) -> dict[str, Any]:
    rng = random.Random(SEED)
    baseline_rows = {(row["benchmark"], row["task_id"]): row for row in rows_by_arm[baseline]}
    out = {}
    for arm, arm_rows in rows_by_arm.items():
        if arm == baseline:
            continue
        current = {(row["benchmark"], row["task_id"]): row for row in arm_rows}
        paired_keys = sorted(baseline_rows.keys() & current.keys())
        paired = [(baseline_rows[key], current[key]) for key in paired_keys]
        diffs = {metric: [] for metric in ("em", "es", "id_f1")}
        for _ in range(rounds):
            sample = [paired[rng.randrange(len(paired))] for _ in paired]
            for metric in diffs:
                diffs[metric].append(100 * sum(b[metric] - a[metric] for a, b in sample) / len(sample))
        wins = sum(1 for a, b in paired if b["es"] > a["es"] + 1e-6)
        losses = sum(1 for a, b in paired if b["es"] < a["es"] - 1e-6)
        ties = len(paired) - wins - losses
        out[arm] = {
            "es_wins_losses_ties": [wins, losses, ties],
            **{
                metric: {
                    "mean_delta_pp": round(sum(values) / len(values), 4),
                    "ci95_pp": [
                        round(sorted(values)[int(0.025 * rounds)], 4),
                        round(sorted(values)[int(0.975 * rounds)], 4),
                    ],
                }
                for metric, values in diffs.items()
            },
        }
    return out


def run_iteration(global_ns: dict[str, Any], subset_per_benchmark: int = 16, tag: str = "iter2_64") -> dict[str, Any]:
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

    arms = ["aligncoder_code", "span_merge_clean", "mb_gated_rerank_v2", "mb_active_dep_header", "mb_spt_v2"]
    all_prompts = []
    prompt_meta = []
    for t in selected_tasks:
        trace = traces[(t["benchmark"], t["task_id"])]
        for arm in arms:
            p, p_toks, c_toks = build_arm_prompt(arm, t, trace, tok)
            all_prompts.append(p)
            prompt_meta.append((arm, t, p_toks, c_toks))

    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=GENERATOR_MAX_NEW, n=1, seed=123)
    t0 = time.time()
    outputs = engine.generate(all_prompts, params, use_tqdm=True)
    elapsed = time.time() - t0
    print(f"[{tag}] Generated {len(all_prompts)} completions in {elapsed:.2f}s", flush=True)

    rows_by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    all_rows = []
    for (arm, t, p_toks, c_toks), req_out in zip(prompt_meta, outputs):
        # CRITICAL FIX: Decode from token_ids using HuggingFace AutoTokenizer so spaces/newlines are preserved!
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
    tag = os.environ.get("ITER_TAG", f"iter2_{sub * 4}")
    run_iteration(globals(), subset_per_benchmark=sub, tag=tag)
