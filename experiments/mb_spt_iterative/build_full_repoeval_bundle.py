#!/usr/bin/env python3
"""Build compact full RepoEval (1,600 API + 1,600 Line = 3,200 tasks) evaluation bundle."""

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

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "dataset" / "data4aligncoder" / "repoeval"
RESULTS_ROOT = ROOT / "RepoClone" / "AlignCoder" / "results"
OUT_BUNDLE = Path(__file__).resolve().parent / "full_repoeval_bundle.tar.gz"

LINE_RE = re.compile(r"lines:\s*(\d+)-(\d+)")


def parse_line_range(desc: str) -> tuple[int, int]:
    m = LINE_RE.search(desc or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def main() -> None:
    t0 = time.time()
    tasks_out = []
    traces_out = []
    files_map: dict[str, dict[str, str]] = {}

    for split in ("api_level", "line_level"):
        bench = "repoeval_api" if split == "api_level" else "repoeval_line"
        print(f"Processing {bench} (1600 tasks)...", flush=True)

        cand_by_task = {}
        cand_file = RESULTS_ROOT / bench / "prediction_with_candidates.jsonl"
        with cand_file.open("r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                cand_by_task[d["task_id"]] = d["candidates"]

        t0_data = pq.read_table(DATA_ROOT / split / "test_0.parquet").to_pylist()
        t1_data = pq.read_table(DATA_ROOT / split / "test_1.parquet").to_pylist()
        all_tasks = t0_data + t1_data
        assert len(all_tasks) == 1600, f"Expected 1600 tasks, got {len(all_tasks)}"

        for row in all_tasks:
            tid = row["task_id"]
            cands = cand_by_task.get(tid, [])
            norm_cands = []
            for i, c in enumerate(cands):
                fp = c.get("file_path", "")
                desc = c.get("description", "")
                s, e = parse_line_range(desc)
                norm_cands.append({
                    "path": fp,
                    "line_start": s,
                    "line_end": e,
                    "code": c.get("code_content", ""),
                    "cosine": round(1.0 - (i * 0.01), 6),
                    "rank": i + 1,
                    "api": fp == "file_path",
                    "kind": "api" if fp == "file_path" else "code",
                })

            referenced_paths = {
                c["path"] for c in norm_cands if c["path"] and c["path"] not in ("", "file_path")
            }
            task_file_ids: dict[str, str] = {}
            for cf in row.get("crossfile_context", []):
                p = cf.get("path", "")
                if p in referenced_paths:
                    txt = cf.get("text", "")
                    fid = hashlib.sha1(f"{bench}:{p}\0{txt}".encode("utf-8", errors="replace")).hexdigest()[:16]
                    if fid not in files_map:
                        files_map[fid] = {"file_id": fid, "path": p, "text": txt}
                    task_file_ids[p] = fid

            tasks_out.append({
                "benchmark": bench,
                "task_id": tid,
                "language": "python",
                "path": row["path"],
                "left_context": row["left_context"],
                "groundtruth": row["groundtruth"],
                "file_ids_by_path": task_file_ids,
            })

            traces_out.append({
                "benchmark": bench,
                "task_id": tid,
                "query_enhancement_outputs": [],
                "aligncoder_top10": norm_cands[:10],
                "rerank_candidates": norm_cands,
            })

    print(f"Packaged {len(tasks_out)} tasks, {len(traces_out)} traces, {len(files_map)} files in {time.time() - t0:.2f}s", flush=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        # Write tasks.jsonl.gz
        with gzip.open(tmp / "tasks.jsonl.gz", "wt", encoding="utf-8") as f:
            for t in tasks_out:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

        # Write retrieval_trace.jsonl.gz
        with gzip.open(tmp / "retrieval_trace.jsonl.gz", "wt", encoding="utf-8") as f:
            for tr in traces_out:
                f.write(json.dumps(tr, ensure_ascii=False) + "\n")

        # Write files.jsonl.gz
        with gzip.open(tmp / "files.jsonl.gz", "wt", encoding="utf-8") as f:
            for rec in files_map.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Copy official baseline predictions
        for bench in ("repoeval_api", "repoeval_line"):
            off_dir = tmp / "official_results" / bench
            off_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RESULTS_ROOT / bench / "prediction.jsonl", off_dir / "prediction.jsonl")
            shutil.copy2(RESULTS_ROOT / bench / "results.json", off_dir / "results.json")
            if (RESULTS_ROOT / bench / "detailed_results.json").exists():
                shutil.copy2(RESULTS_ROOT / bench / "detailed_results.json", off_dir / "detailed_results.json")

        # Copy parsers (both python and java)
        parser_dir = tmp / "parsers"
        parser_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "RepoClone" / "AlignCoder" / "utils" / "build" / "python-lang-parser.so", parser_dir / "python-lang-parser.so")
        shutil.copy2(ROOT / "RepoClone" / "AlignCoder" / "utils" / "build" / "java-lang-parser.so", parser_dir / "java-lang-parser.so")

        # Copy keyword lists
        kw_dir = tmp / "aligncoder" / "utils" / "keywords"
        kw_dir.mkdir(parents=True, exist_ok=True)
        src_kw = ROOT / "RepoClone" / "AlignCoder" / "utils" / "keywords"
        for kw_file in src_kw.glob("*"):
            if kw_file.is_file():
                shutil.copy2(kw_file, kw_dir / kw_file.name)

        # Create tarball
        print(f"Creating {OUT_BUNDLE}...", flush=True)
        with tarfile.open(OUT_BUNDLE, "w:gz") as tar:
            for item in sorted(tmp.iterdir()):
                tar.add(item, arcname=item.name)

    size_mb = OUT_BUNDLE.stat().st_size / (1024 * 1024)
    print(f"Created {OUT_BUNDLE} ({size_mb:.2f} MiB) in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
