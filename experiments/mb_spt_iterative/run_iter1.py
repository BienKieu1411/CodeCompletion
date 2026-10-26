#!/usr/bin/env python3
"""Iteration 1: Test MB-SPT mathematical pillars on Colab GPU (64-task fast subset & 256-task full)."""

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
    """Return (cursor_syms, draft_syms) forming the Cursor Markov Blanket M_w."""
    nonempty = [line for line in left_context.splitlines() if line.strip()]
    cursor_tail = "\n".join(nonempty[-window_lines:])
    cursor_syms = set(ID_RE.findall(cursor_tail))
    draft_syms = set(ID_RE.findall("\n".join(qh_outputs)))
    return cursor_syms, draft_syms


def active_markov_edges(
    edges: list[dict[str, str]],
    mb_syms: set[str],
) -> list[dict[str, str]]:
    """Filter prefix import edges strictly to the Cursor Markov Blanket (Theorem 1)."""
    active = []
    for edge in edges:
        sym = edge.get("symbol", "")
        imp = edge.get("import", "")
        rel = edge.get("relation", "")
        if rel == "imports_module":
            if sym and sym in mb_syms:
                active.append(edge)
            continue
        if (sym and sym != "*" and sym in mb_syms) or (imp and imp.split(".")[-1] in mb_syms):
            active.append(edge)
    return active


def format_chunk_raw(c: dict[str, Any], language: str, is_last_api: bool = False) -> str:
    label = "#" if language == "python" else "//"
    if c.get("kind") == "stop" or not c.get("path"):
        return ""
    if c["path"] == "file_path":
        hdr = f"{label} Potential invoked APIs:" if is_last_api else f"{label} Potentially invoked APIs:"
    else:
        hdr = f"{label} file path: {c['path']}\n{label} lines: {c['line_start']}-{c['line_end']}"
    body = "\n".join(f"{label} {line}" for line in c["code"].split("\n") if line)
    return f"{hdr}\n{body}".strip()


def quotient_deduplicate_chunks(
    chunks: list[dict[str, Any]],
    language: str,
    guaranteed_left_text: str = "",
) -> list[str]:
    """Theorem 2a: Quotient Deduplication Q(S_0) & Same-File Adjacent Merging."""
    label = "#" if language == "python" else "//"
    seen_lines: set[str] = set()
    for line in guaranteed_left_text.splitlines():
        norm = line.strip()
        if len(norm) > 14 and not norm.startswith(("#", "//", "{", "}", "return", "else", "try")):
            seen_lines.add(norm)

    cleaned_items: list[dict[str, Any]] = []
    for c in chunks:
        if c.get("kind") == "stop" or not c.get("path"):
            break
        raw_lines = [line for line in c["code"].split("\n") if line.strip()]
        kept_lines = []
        for line in raw_lines:
            norm = line.strip()
            if len(norm) > 10 and norm in seen_lines:
                continue
            kept_lines.append(line)
            if len(norm) > 10:
                seen_lines.add(norm)
        if not kept_lines:
            continue
        cleaned_items.append({
            "path": c["path"],
            "line_start": c.get("line_start", 0),
            "line_end": c.get("line_end", 0),
            "lines": kept_lines,
            "api": c.get("api", c["path"] == "file_path"),
        })

    merged_items: list[dict[str, Any]] = []
    for item in cleaned_items:
        if (
            merged_items
            and not item["api"]
            and not merged_items[-1]["api"]
            and merged_items[-1]["path"] == item["path"]
            and merged_items[-1]["line_end"] > 0
            and item["line_start"] > 0
            and abs(item["line_start"] - merged_items[-1]["line_end"]) <= 3
        ):
            merged_items[-1]["line_end"] = max(merged_items[-1]["line_end"], item["line_end"])
            merged_items[-1]["lines"].extend(item["lines"])
        else:
            merged_items.append(item)

    blocks = []
    for item in merged_items:
        if item["api"] or item["path"] == "file_path":
            hdr = f"{label} Potentially invoked APIs:"
        else:
            hdr = f"{label} file path: {item['path']}\n{label} lines: {item['line_start']}-{item['line_end']}"
        body = "\n".join(f"{label} {line}" for line in item["lines"])
        blocks.append(f"{hdr}\n{body}".strip())
    return blocks


def extract_truncated_infile_skeleton(
    task: dict[str, Any],
    tok: Any,
    allowed_left_est: int,
    mb_syms: set[str],
    max_lines: int = 20,
) -> str:
    """Theorem 2b: Rescue active Markov-Blanket & declaration lines from truncated left_context prefix."""
    left_all_ids = tok.encode(task["left_context"], add_special_tokens=False)
    if len(left_all_ids) <= allowed_left_est:
        return ""
    trunc_ids = left_all_ids[:-allowed_left_est]
    trunc_text = tok.decode(trunc_ids, skip_special_tokens=True)
    surv_text = tok.decode(left_all_ids[-allowed_left_est:], skip_special_tokens=True)
    surv_lines = {line.strip() for line in surv_text.splitlines() if line.strip()}

    lang = task["language"]
    label = "#" if lang == "python" else "//"
    kept: list[str] = []
    seen: set[str] = set()
    for line in trunc_text.splitlines():
        ls = line.strip()
        if not ls or ls in surv_lines or ls in seen:
            continue
        if ls.startswith(("#", "//", "/*", "*")):
            continue
        line_syms = set(ID_RE.findall(ls))
        has_mb = bool(line_syms & mb_syms)
        is_decl = bool(
            re.search(
                r"^\s*(?:public |private |protected |static |final |async |def |class |interface |enum |import |from )"
                r"|\b(?:self|this)\.[A-Za-z_][A-Za-z0-9_]*\s*=",
                line,
            )
        )
        if has_mb or is_decl:
            kept.append(line)
            seen.add(ls)

    if not kept:
        return ""
    if len(kept) > max_lines:
        scored = [
            (2 if (set(ID_RE.findall(l)) & mb_syms) else 1, idx, l)
            for idx, l in enumerate(kept)
        ]
        top_indices = {idx for _, idx, _ in sorted(scored, key=lambda x: (-x[0], -x[1]))[:max_lines]}
        kept = [l for idx, l in enumerate(kept) if idx in top_indices]

    hdr = f"{label} file path: {task['path']}"
    body = "\n".join(f"{label} {l}" for l in kept)
    return f"{hdr}\n{body}".strip()


def trim_clean_cross_ids(cross_ids: list[int], tok: Any) -> list[int]:
    """Ensure cross-file context does not end mid-line or with a dangling header at the 1536 cap."""
    if len(cross_ids) < CROSSFILE_TOKENS:
        return cross_ids
    text = tok.decode(cross_ids, skip_special_tokens=True)
    if "\n" not in text:
        return cross_ids
    clean_text = text.rsplit("\n", 1)[0].rstrip()
    lines = clean_text.splitlines()
    while lines and (
        lines[-1].startswith(("# file path:", "// file path:", "# lines:", "// lines:", "# Potential", "// Potential"))
    ):
        lines.pop()
    clean_text = "\n".join(lines).rstrip()
    return tok.encode(clean_text, add_special_tokens=False)[:CROSSFILE_TOKENS]


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
    guaranteed_left_text = tok.decode(left_all_ids[-488:], skip_special_tokens=True)

    cursor_syms, draft_syms = extract_cursor_markov_blanket(
        task["left_context"], trace["query_enhancement_outputs"], window_lines=5
    )
    mb_syms = cursor_syms | draft_syms
    act_edges = active_markov_edges(trace["dependency_edges"], mb_syms)
    active_paths = {e["path"] for e in act_edges}
    active_edge_syms = {e["symbol"] for e in act_edges if e.get("symbol")}

    if arm == "aligncoder_code":
        blocks = [format_chunk_raw(c, lang) for c in trace["aligncoder_top10"]]
        blocks = [b for b in blocks if b]
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]
    elif arm == "q_dedup_only":
        blocks = quotient_deduplicate_chunks(trace["aligncoder_top10"], lang, guaranteed_left_text)
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]
        cross_ids = trim_clean_cross_ids(cross_ids, tok)
    elif arm == "mb_gated_rescue":
        rescued_candidates = []
        for c in trace["rerank_candidates"][10:25]:
            if c.get("kind") == "stop" or not c.get("path"):
                continue
            c_syms = set(ID_RE.findall(c["code"]))
            if (
                (c["path"] in active_paths and (c_syms & mb_syms))
                or (c_syms & active_edge_syms)
                or len(c_syms & cursor_syms & draft_syms) >= 2
            ):
                rescued_candidates.append(c)
        all_chunks = list(trace["aligncoder_top10"]) + rescued_candidates[:3]
        blocks = quotient_deduplicate_chunks(all_chunks, lang, guaranteed_left_text)
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]
        cross_ids = trim_clean_cross_ids(cross_ids, tok)
    elif arm == "mb_infile_skeleton":
        blocks = quotient_deduplicate_chunks(trace["aligncoder_top10"], lang, guaranteed_left_text)
        skel = extract_truncated_infile_skeleton(task, tok, 520, mb_syms, max_lines=20)
        if skel:
            insert_pos = min(len(blocks), 6)
            blocks.insert(insert_pos, skel)
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]
        cross_ids = trim_clean_cross_ids(cross_ids, tok)
    elif arm == "mb_spt_full":
        rescued_candidates = []
        for c in trace["rerank_candidates"][10:25]:
            if c.get("kind") == "stop" or not c.get("path"):
                continue
            c_syms = set(ID_RE.findall(c["code"]))
            if (
                (c["path"] in active_paths and (c_syms & mb_syms))
                or (c_syms & active_edge_syms)
                or len(c_syms & cursor_syms & draft_syms) >= 2
            ):
                rescued_candidates.append(c)
        all_chunks = list(trace["aligncoder_top10"]) + rescued_candidates[:3]
        blocks = quotient_deduplicate_chunks(all_chunks, lang, guaranteed_left_text)
        skel = extract_truncated_infile_skeleton(task, tok, 520, mb_syms, max_lines=18)
        if skel:
            insert_pos = min(len(blocks), 6)
            blocks.insert(insert_pos, skel)
        joined = "\n\n".join(blocks)
        cross_ids = tok.encode(joined[: CROSSFILE_TOKENS * 10], add_special_tokens=False)[:CROSSFILE_TOKENS]
        cross_ids = trim_clean_cross_ids(cross_ids, tok)
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


def run_iteration(global_ns: dict[str, Any], subset_per_benchmark: int = 16, tag: str = "iter1_64") -> dict[str, Any]:
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

    arms = ["aligncoder_code", "q_dedup_only", "mb_gated_rescue", "mb_infile_skeleton", "mb_spt_full"]
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
        raw_pred = req_out.outputs[0].text
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
    tag = os.environ.get("ITER_TAG", f"iter1_{sub * 4}")
    run_iteration(globals(), subset_per_benchmark=sub, tag=tag)
