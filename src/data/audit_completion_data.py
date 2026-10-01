"""Profile benchmark shape and old train labels; never evaluate/tune a model on test."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zlib

import pyarrow.parquet as pq

REVISION = "ce72fe3ef4987e15a9219fea4174bb80700e742c"
TOKENIZER_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
BENCHMARKS = {
    "cceval_python": ["cceval/python/test.parquet"],
    "cceval_java": ["cceval/java/test.parquet"],
    "repoeval_line": ["repoeval/line_level/test_0.parquet", "repoeval/line_level/test_1.parquet"],
    "repoeval_api": ["repoeval/api_level/test_0.parquet", "repoeval/api_level/test_1.parquet"],
}


def source_hash(text):
    return hashlib.sha256(text.replace("\r\n", "\n").replace("\r", "\n").strip().encode()).hexdigest()


def quantiles(values):
    values = sorted(values)
    return {f"p{p}": values[int((len(values)-1)*p/100)] if values else None
            for p in (10, 50, 90, 99)}


def profile(rows, tokenizer, cursor_contract="exact_prefix"):
    counts, kinds = Counter(), Counter()
    lengths, prefix_lengths, line_counts, examples = [], [], [], []
    for row in rows:
        if "payload" in row:
            row = json.loads(zlib.decompress(row["payload"]))
        left = row["left_context"] or ""
        target = row.get("groundtruth", row.get("target_code", "")) or ""
        tail = left.rsplit("\n", 1)[-1]
        counts["rows"] += 1
        counts["stored_prefix_nonblank_tail"] += bool(tail.strip())
        if cursor_contract == "exact_prefix":
            counts["cursor_midline"] += bool(tail.strip())
        counts["cursor_after_dot"] += tail.rstrip().endswith(".")
        counts["multiline_target"] += len(target.strip().splitlines()) > 1
        counts["target_call_syntax"] += bool(re.search(r"\b\w+\s*\(", target))
        counts["target_already_in_prefix"] += bool(target.strip()) and target.strip() in left
        counts["empty_target"] += not target.strip()
        kinds[row.get("target_kind", "benchmark")] += 1
        lengths.append(len(tokenizer.encode(target, add_special_tokens=False)))
        prefix_lengths.append(len(tokenizer.encode(left, add_special_tokens=False)))
        line_counts.append(len(target.strip().splitlines()))
        if len(examples) < 5:
            examples.append({"task_id": row.get("task_id"), "left_tail": left[-160:],
                             "target": target[:300]})
    return {"cursor_contract": cursor_contract, "counts": dict(counts), "target_tokens": quantiles(lengths),
            "prefix_tokens": quantiles(prefix_lengths), "target_lines": quantiles(line_counts),
            "target_kinds": dict(kinds), "examples_for_audit_only": examples}


def rows_from(paths, columns):
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=columns):
            yield from batch.to_pylist()


def exclusion_hashes(root):
    """File identities for exact-overlap exclusion only, never training examples."""
    hashes = set()
    for name, files in BENCHMARKS.items():
        for row in rows_from([root/p for p in files],
                             ["left_context", "groundtruth", "right_context", "crossfile_context"]):
            texts = [(row["left_context"] or "") + (row["groundtruth"] or "") +
                     (row["right_context"] or "")]
            if name.startswith("repoeval"):
                # RepoEval serializes line-joined segments without boundary newlines.
                # Include both possible reconstructions solely for overlap exclusion.
                texts.append("\n".join(row[k] or "" for k in
                                       ("left_context", "groundtruth", "right_context")))
            texts.extend(item["text"] for item in row["crossfile_context"] or [] if item["text"])
            for text in texts:
                if len(text) >= 200 and len(text.splitlines()) >= 8:
                    hashes.add(source_hash(text))
        print(f"exclusion {name}: {len(hashes)} unique nontrivial file hashes", flush=True)
    return sorted(hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Downloaded Data4AlignCoder/data")
    parser.add_argument("--old-train", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusion-output", type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base",
                                              revision=TOKENIZER_REVISION, local_files_only=True)
    tokenizer.model_max_length = 10**9  # Counting only; no long sequence is passed to a model.
    results = {"source_revision": REVISION, "tokenizer_revision": TOKENIZER_REVISION,
               "scope": "structural audit, not measured model difficulty", "datasets": {}}
    for name, paths in BENCHMARKS.items():
        result = profile(rows_from([args.root/p for p in paths],
                                    ["task_id", "left_context", "groundtruth"]), tokenizer,
                         "line_segments" if name.startswith("repoeval") else "exact_prefix")
        results["datasets"][name] = result
        print(name, json.dumps({k:v for k,v in result.items() if k != "examples_for_audit_only"}), flush=True)
    if args.old_train:
        result = profile(rows_from([args.old_train], ["payload"]), tokenizer)
        results["datasets"]["old_train_all_splits"] = result
        print("old_train_all_splits", json.dumps({k:v for k,v in result.items() if k != "examples_for_audit_only"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    if args.exclusion_output:
        hashes = exclusion_hashes(args.root)
        args.exclusion_output.write_text(json.dumps({"source_revision": REVISION,
                                                     "file_hashes": hashes}) + "\n")


if __name__ == "__main__":
    main()
