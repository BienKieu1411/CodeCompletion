#!/usr/bin/env python3
"""Build compact full CrossCodeEval (2,665 Python + 2,139 Java = 4,804 tasks) evaluation bundle."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OBS_ROOT = ROOT / "results" / "aligncoder-rq1-deepseek13b-20261013-225803" / "observability"
OFF_ROOT = ROOT / "results" / "aligncoder-rq1-deepseek13b-20261013-225803" / "official_results"
DATA_ROOT = ROOT / "dataset" / "data4aligncoder" / "cceval"
OUT_BUNDLE = Path(__file__).resolve().parent / "full_cceval_bundle.tar.gz"

LINE_RE = re.compile(r"lines:\s*(\d+)-(\d+)")


def parse_line_range(desc: str) -> tuple[int, int]:
    m = LINE_RE.search(desc or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def normalize_candidate(c: dict, rank_fallback: int = 999) -> dict:
    fp = c.get("file_path", "")
    desc = c.get("description", "")
    s, e = parse_line_range(desc)
    is_stop = bool(c.get("stop_block") or fp == "")
    is_api = bool(c.get("api_block") or fp == "file_path")
    return {
        "path": fp,
        "line_start": s,
        "line_end": e,
        "code": c.get("code_content", ""),
        "cosine": round(float(c.get("cosine_score") or 0.0), 6),
        "rank": int(c.get("retriever_rank") or rank_fallback),
        "api": is_api,
        "kind": "stop" if is_stop else ("api" if is_api else "code"),
    }


def main() -> None:
    t0 = time.time()
    tasks_out = []
    traces_out = []
    files_map: dict[str, dict[str, str]] = {}

    for lang, bench, expected_n in [("python", "cceval_python", 2665), ("java", "cceval_java", 2139)]:
        print(f"Processing {bench} ({expected_n} tasks)...", flush=True)
        df = pd.read_parquet(DATA_ROOT / lang / "test.parquet")
        assert len(df) == expected_n, f"{len(df)} != {expected_n}"

        # Load query enhancement drafts from first 2*N lines of generation_events.jsonl.gz
        qh_by_task: dict[str, list[str]] = {}
        gen_path = OBS_ROOT / bench / "generation_events.jsonl.gz"
        with gzip.open(gen_path, "rt", encoding="utf-8") as f:
            for _ in range(expected_n * 2):
                line = f.readline()
                if not line:
                    break
                r = json.loads(line)
                if r.get("event") == "query_enhancement_postprocessed":
                    tid = r["task_id"]
                    if tid not in qh_by_task:
                        qh_by_task[tid] = r.get("deduplicated_postprocessed_completions", [])

        # Load unixcoder_rerank_pool from first 4*N lines of retrieval_events.jsonl.gz
        rerank_by_task: dict[str, tuple[list[dict], list[dict]]] = {}
        ret_path = OBS_ROOT / bench / "retrieval_events.jsonl.gz"
        with gzip.open(ret_path, "rt", encoding="utf-8") as f:
            for _ in range(expected_n * 4):
                line = f.readline()
                if not line:
                    break
                r = json.loads(line)
                if r.get("event") == "unixcoder_rerank_pool":
                    tid = r["task_id"]
                    if tid not in rerank_by_task:
                        pool = r.get("candidate_pool_after_bm25_and_api", [])
                        pool_sorted = sorted(
                            pool,
                            key=lambda x: (int(x.get("retriever_rank") or 9999), -float(x.get("cosine_score") or 0.0)),
                        )
                        top25 = [normalize_candidate(c, i + 1) for i, c in enumerate(pool_sorted[:25])]
                        # Map cosine_score onto selected_order
                        cos_lookup = {
                            (c.get("file_path", ""), c.get("description", ""), c.get("code_content", "")): float(
                                c.get("cosine_score") or 0.0
                            )
                            for c in pool
                        }
                        sel_raw = r.get("selected_order", [])
                        top10 = []
                        for i, c in enumerate(sel_raw):
                            nc = normalize_candidate(c, i + 1)
                            if nc["cosine"] == 0.0:
                                nc["cosine"] = round(
                                    cos_lookup.get(
                                        (c.get("file_path", ""), c.get("description", ""), c.get("code_content", "")),
                                        0.0,
                                    ),
                                    6,
                                )
                            top10.append(nc)
                        rerank_by_task[tid] = (top10, top25)

        assert len(qh_by_task) == expected_n, f"qh_by_task: {len(qh_by_task)}"
        assert len(rerank_by_task) == expected_n, f"rerank_by_task: {len(rerank_by_task)}"

        for _, row in df.iterrows():
            tid = row["task_id"]
            top10, top25 = rerank_by_task[tid]
            referenced_paths = {
                c["path"] for c in (top10 + top25) if c["path"] and c["path"] not in ("", "file_path")
            }
            task_file_ids: dict[str, str] = {}
            for cf in row["crossfile_context"]:
                p = cf["path"]
                if p in referenced_paths:
                    txt = cf["text"]
                    fid = hashlib.sha1((f"{bench}:{p}\0{txt}").encode("utf-8", errors="replace")).hexdigest()[:16]
                    if fid not in files_map:
                        files_map[fid] = {"file_id": fid, "path": p, "text": txt}
                    task_file_ids[p] = fid

            tasks_out.append(
                {
                    "benchmark": bench,
                    "task_id": tid,
                    "language": lang,
                    "path": row["path"],
                    "left_context": row["left_context"],
                    "groundtruth": row["groundtruth"],
                    "file_ids_by_path": task_file_ids,
                }
            )
            traces_out.append(
                {
                    "benchmark": bench,
                    "task_id": tid,
                    "query_enhancement_outputs": qh_by_task[tid],
                    "aligncoder_top10": top10,
                    "rerank_candidates": top25,
                }
            )

    print(
        f"Collected {len(tasks_out)} tasks, {len(traces_out)} traces, {len(files_map)} unique referenced source files in {time.time() - t0:.1f}s",
        flush=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with gzip.open(tmp_path / "tasks.jsonl.gz", "wt", encoding="utf-8") as f:
            for item in tasks_out:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with gzip.open(tmp_path / "retrieval_trace.jsonl.gz", "wt", encoding="utf-8") as f:
            for item in traces_out:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with gzip.open(tmp_path / "files.jsonl.gz", "wt", encoding="utf-8") as f:
            for item in files_map.values():
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Copy tree-sitter parsers and keyword lists
        parsers_dir = tmp_path / "parsers"
        parsers_dir.mkdir(parents=True, exist_ok=True)
        for lang in ("python", "java"):
            shutil.copy2(
                ROOT / "RepoClone" / "AlignCoder" / "utils" / "build" / f"{lang}-lang-parser.so",
                parsers_dir / f"{lang}-lang-parser.so",
            )
        kw_dir = tmp_path / "aligncoder" / "utils" / "keywords"
        kw_dir.mkdir(parents=True, exist_ok=True)
        for item in (ROOT / "RepoClone" / "AlignCoder" / "utils" / "keywords").iterdir():
            if item.is_file():
                shutil.copy2(item, kw_dir / item.name)

        # Copy official predictions for exact baseline comparison
        off_dir = tmp_path / "official_results"
        for bench in ("cceval_python", "cceval_java"):
            bdir = off_dir / bench
            bdir.mkdir(parents=True, exist_ok=True)
            for fn in ("prediction.jsonl", "results.json", "detailed_results.json"):
                shutil.copy2(OFF_ROOT / bench / fn, bdir / fn)

        with tarfile.open(OUT_BUNDLE, "w:gz") as tar:
            for item in sorted(tmp_path.iterdir()):
                tar.add(item, arcname=item.name)

    size_mb = OUT_BUNDLE.stat().st_size / (1024 * 1024)
    print(f"Built {OUT_BUNDLE} ({size_mb:.2f} MiB) in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
