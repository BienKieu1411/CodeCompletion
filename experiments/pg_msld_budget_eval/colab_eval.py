#!/usr/bin/env python3
"""Paired cross-file-budget evaluation for the PG-MSLD UniXcoder checkpoint.

The input rows and per-row number of retrieved snippets are frozen from the
previous 400-row evaluation.  Both arms use the same 3,072-token total prompt
budget; only the cross-file allocation changes (512 vs. 2,344 tokens).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zlib

EXPECTED_CHECKPOINT_SHA256 = "30a6807a0b2727cf558d87308fe70a2bea6d40eb0893d002ae4a3105311686f7"
EXPECTED_VALID_SHA256 = "7658522ea93f31ce86ee50b90162cbeae9e18bb32bec5256c001e977364a8582"
CHECKPOINT_SCHEMA = "rlcoder_natural_random2k_slate15_weighted_positive_gain_listwise_kd_v7"
GENERATOR_ID = "deepseek-ai/deepseek-coder-1.3b-base"
GENERATOR_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"
RETRIEVER_ID = "microsoft/unixcoder-base"
RETRIEVER_REVISION = "5604afdc964f6c53782a6813140ade5216b99006"
TOTAL_CONTEXT_TOKENS = 3072
CROSS_FILE_BUDGETS = (512, 2344)
TOPK_CANDIDATES = 100
RETRIEVER_QUERY_TOKENS = 256
RETRIEVER_CANDIDATE_TOKENS = 512
GENERATOR_MAX_NEW_TOKENS = 64


def ensure_light_dependencies() -> None:
    requirements = {
        "pyarrow": "pyarrow",
        "rank_bm25": "rank-bm25",
        "fuzzywuzzy": "fuzzywuzzy",
        "tree_sitter": "tree-sitter>=0.25,<0.27",
        "tree_sitter_python": "tree-sitter-python",
        "tree_sitter_java": "tree-sitter-java",
        "transformers": "transformers>=4.45,<5",
    }
    missing = []
    for module, package in requirements.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_rlcoder_chunks(path: str, code: str, language: str) -> list[dict]:
    """Reproduce RLCoder's natural blank-line mini-block splitter (max 15 lines)."""
    mini_blocks: list[list[str]] = []
    current: list[str] = []
    for line in code.splitlines():
        if line.strip() == "":
            if current:
                mini_blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        mini_blocks.append(current)

    bounded: list[list[str]] = []
    for block in mini_blocks:
        if len(block) > 15:
            bounded.extend(block[i : i + 15] for i in range(0, len(block), 15))
        else:
            bounded.append(block)

    chunks: list[dict] = []
    current_content: list[str] = []
    total_lines = 0
    for block in bounded:
        if total_lines >= 5000:
            break
        if len(current_content) + len(block) <= 15:
            current_content.extend(block)
            total_lines += len(block)
        else:
            if current_content:
                chunks.append({
                    "path": path,
                    "start": total_lines - len(current_content) + 1,
                    "end": total_lines,
                    "text": "\n".join(current_content),
                    "language": language,
                })
            current_content = list(block)
            total_lines += len(block)
    if current_content:
        chunks.append({
            "path": path,
            "start": total_lines - len(current_content) + 1,
            "end": total_lines,
            "text": "\n".join(current_content),
            "language": language,
        })
    return chunks


def render_retriever_text(chunk: dict) -> str:
    marker = "#" if chunk["language"] == "python" else "//"
    description = f"file path: {chunk['path']}\nlines: {chunk['start']}-{chunk['end']}"
    lines = [f"{marker} {line}" for line in description.strip().split("\n") if line]
    lines.append("\n".join(f"{marker} {line}" for line in chunk["text"].split("\n")))
    return "\n".join(lines).strip()


def query_text(left_context: str) -> str:
    return "\n".join([line for line in left_context.split("\n") if line.strip()][-20:])


def bm25_pool(row: dict) -> tuple[str, list[dict]]:
    from rank_bm25 import BM25Okapi

    target_path = row["path"]
    language = row["language"]
    all_chunks: list[dict] = []
    for source_file in row["crossfile_context"] or []:
        path = str(source_file.get("path", ""))
        if not path or path == target_path:
            continue
        text = source_file.get("text") or ""
        all_chunks.extend(split_rlcoder_chunks(path, text, language))
    if not all_chunks:
        raise ValueError(f"No cross-file chunks for {row['task_id']}")
    index = BM25Okapi([chunk["text"].lower().split() for chunk in all_chunks])
    query = query_text(row["left_context"])
    scores = index.get_scores(query.split())
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOPK_CANDIDATES]
    candidates = []
    for rank, index_in_all in enumerate(order, start=1):
        candidates.append({
            **all_chunks[index_in_all],
            "bm25_rank": rank,
            "bm25_score": float(scores[index_in_all]),
            "retriever_text": render_retriever_text(all_chunks[index_in_all]),
        })
    return query, candidates


def legacy_retriever_ids(text: str, tokenizer, *, is_query: bool, max_length: int) -> list[int]:
    tokens = tokenizer.tokenize(text)
    if is_query:
        tokens = tokens[-(max_length - 4) :]
    else:
        tokens = tokens[: max_length - 4]
    token_ids = tokenizer.convert_tokens_to_ids(
        [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + tokens + [tokenizer.sep_token]
    )
    if len(token_ids) > max_length:
        raise AssertionError("UniXcoder input exceeded its fixed sequence length")
    return token_ids + [tokenizer.pad_token_id] * (max_length - len(token_ids))


def encode_unixcoder(model, rows: list[list[int]], tokenizer, device, batch_size: int) -> "object":
    import torch
    import torch.nn.functional as F

    outputs = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        width = max(len(row) for row in batch_rows)
        input_ids = torch.full(
            (len(batch_rows), width), tokenizer.pad_token_id,
            dtype=torch.long, device=device,
        )
        for offset, row in enumerate(batch_rows):
            input_ids[offset, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        with torch.inference_mode():
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
            outputs.append(F.normalize(pooled.float(), p=2, dim=1).cpu())
        if start == 0 or (start // batch_size + 1) % 100 == 0:
            print(f"UniXcoder embeddings: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return torch.cat(outputs, dim=0)


def build_crossfile_context(selected: list[dict], generator_tokenizer, budget: int, language: str) -> tuple[list[int], str]:
    """Match RLCoder's selected-block rendering and hard cross-file token clip."""
    rendered = "\n\n".join(render_retriever_text(chunk) for chunk in selected)
    # RLCoder first bounds the raw text to avoid huge tokenizer work, then clips token IDs.
    token_ids = generator_tokenizer.encode(
        rendered[: budget * 10], add_special_tokens=False
    )[:budget]
    text = generator_tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    return token_ids, text


def make_prompt(row: dict, selected: list[dict], tokenizer, total_budget: int, cross_budget: int) -> tuple[str, list[int], int, int]:
    cross_ids, cross_text = build_crossfile_context(selected, tokenizer, cross_budget, row["language"])
    # This follows RLCoder's generator prompt contract (including its literal '#' path label).
    path_context = f"\n\n# file path: {row['path']}\n\n"
    path_ids = tokenizer.encode(path_context, add_special_tokens=False)
    left_ids = tokenizer.encode(row["left_context"], add_special_tokens=False)
    allowed_left = max(0, total_budget - len(cross_ids) - len(path_ids) - 10)
    left_ids = left_ids[-allowed_left:] if allowed_left else []
    prompt_text = tokenizer.decode(cross_ids + path_ids + left_ids, clean_up_tokenization_spaces=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)[-total_budget:]
    return prompt_text, prompt_ids, len(cross_ids), len(left_ids)


def ensure_eval_scorers() -> None:
    ensure_light_dependencies()


def parse_has_error(source: str, language: str) -> bool:
    import tree_sitter as ts
    import tree_sitter_java
    import tree_sitter_python

    grammar = tree_sitter_python if language == "python" else tree_sitter_java
    parser = ts.Parser(ts.Language(grammar.language()))
    return parser.parse(source.encode("utf-8")).root_node.has_error


def python_one_statement(prompt: str, completion: str) -> str:
    for index in range(max(0, len(completion) - 1)):
        if completion[index + 1] != "\n":
            continue
        try:
            if not parse_has_error(prompt + completion[: index + 1], "python"):
                return completion[: index + 1].rstrip()
        except Exception:
            return completion
    return completion


def postprocess_completion(prompt: str, completion: str, language: str) -> str:
    if language == "python":
        completion = python_one_statement(prompt, completion)
    elif language == "java":
        end = next((i for i, char in enumerate(completion) if char in ";}{"), None)
        completion = completion[: end + 1] if end else completion
    return re.sub(r"#.*", "", re.sub(r"//.*", "", completion))


def levenshtein(left: str, right: str) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for i, char_right in enumerate(right, 1):
        current = [i]
        for j, char_left in enumerate(left, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]


def score_prediction(prompt: str, prediction: str, target: str, language: str, benchmark: str) -> dict:
    from fuzzywuzzy import fuzz

    processed = postprocess_completion(prompt, prediction, language)
    target = re.sub(r"#.*", "", re.sub(r"//.*", "", target))
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    prediction_lines = [line.strip() for line in processed.splitlines() if line.strip()]
    if benchmark.startswith("repoeval"):
        clipped = prediction_lines[: len(target_lines)]
        left, right = "\n".join(target_lines), "\n".join(clipped)
        em = float(len(target_lines) == len(clipped) and left == right)
        es = 1.0 - levenshtein(left, right) / max(len(left), len(right), 1)
    else:
        em = float(prediction_lines == target_lines)
        es = fuzz.ratio(processed.strip(), target.strip()) / 100.0
    return {"em": em, "es": es, "prediction": processed}


def summarize(rows: list[dict]) -> dict:
    by_benchmark: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row)
    metrics = {}
    for benchmark, values in sorted(by_benchmark.items()):
        if benchmark.startswith("repoeval"):
            by_repo: dict[str, list[dict]] = defaultdict(list)
            for value in values:
                by_repo[value["task_id"].split("/", 1)[0]].append(value)
            repo_groups = list(by_repo.values())
            em = sum(sum(x["em"] for x in group) / len(group) for group in repo_groups) / len(repo_groups)
            es = sum(sum(x["es"] for x in group) / len(group) for group in repo_groups) / len(repo_groups)
            repo_count = len(repo_groups)
        else:
            em = sum(x["em"] for x in values) / len(values)
            es = sum(x["es"] for x in values) / len(values)
            repo_count = None
        metrics[benchmark] = {"n": len(values), "em": em, "es": es, "repo_count": repo_count}
    metrics["macro_mean_4_benchmarks"] = {
        "em": sum(v["em"] for k, v in metrics.items() if k != "macro_mean_4_benchmarks") / 4,
        "es": sum(v["es"] for k, v in metrics.items() if k != "macro_mean_4_benchmarks") / 4,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--previous-details", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retriever-batch-size", type=int, default=64)
    parser.add_argument("--generator-batch-size", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=400)
    args = parser.parse_args()

    ensure_eval_scorers()
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("Colab GPU is required for this run")
    checkpoint_hash = sha256(args.checkpoint)
    valid_hash = sha256(args.valid)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected checkpoint SHA256: {checkpoint_hash}")
    if valid_hash != EXPECTED_VALID_SHA256:
        raise ValueError(f"Unexpected valid SHA256: {valid_hash}")

    details = [json.loads(line) for line in args.previous_details.read_text().splitlines() if line.strip()]
    # RepoEval line/API can share a task_id; benchmark is part of the key.
    detail_key = lambda item: (str(item["benchmark"]), str(item["task_id"]))
    count_by_id = {detail_key(item): int(item["selected_count"]) for item in details}
    candidate_count_by_id = {detail_key(item): int(item["candidate_count"]) for item in details}
    if len(count_by_id) != len(details):
        raise ValueError("Previous eval details contain duplicate task IDs")

    rows = []
    columns = ["task_id", "path", "left_context", "crossfile_context", "groundtruth", "benchmark", "language"]
    for batch in pq.ParquetFile(args.valid).iter_batches(batch_size=64, columns=columns):
        rows.extend(batch.to_pylist())
    rows = rows[: args.max_rows]
    ids = [(str(row["benchmark"]), str(row["task_id"])) for row in rows]
    if len(rows) != 400 or len(details) != 400 or set(ids) != set(count_by_id):
        raise ValueError(f"Expected the exact 400 prior eval tasks; parquet={len(rows)}, details={len(details)}")
    if sorted((row["benchmark"], row["language"]) for row in rows) != sorted(
        [("cceval_java", "java")] * 100 + [("cceval_python", "python")] * 100
        + [("repoeval_api", "python")] * 100 + [("repoeval_line", "python")] * 100
    ):
        raise ValueError("The 400 benchmark/language composition differs from the previous evaluation")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    signature = checkpoint.get("signature", {})
    if signature.get("schema") != CHECKPOINT_SCHEMA or int(checkpoint.get("updates", -1)) != 2480:
        raise ValueError(f"Unexpected checkpoint signature/updates: {signature.get('schema')}, {checkpoint.get('updates')}")

    device = torch.device("cuda:0")
    retriever_tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_ID, revision=RETRIEVER_REVISION)
    retriever_config = AutoConfig.from_pretrained(RETRIEVER_ID, revision=RETRIEVER_REVISION)
    retriever = AutoModel.from_config(retriever_config).eval().to(device)
    state = {key.removeprefix("model.module."): value for key, value in checkpoint["retriever"].items()}
    missing, unexpected = retriever.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"Checkpoint tensor mismatch; missing={missing}, unexpected={unexpected}")
    del checkpoint, state
    torch.cuda.empty_cache()

    generator_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_ID, revision=GENERATOR_REVISION)
    if generator_tokenizer.pad_token_id is None:
        generator_tokenizer.pad_token_id = generator_tokenizer.eos_token_id
    generator_tokenizer.padding_side = "left"

    prepared = []
    all_retriever_ids: list[list[int]] = []
    for index, row in enumerate(rows, start=1):
        query, candidates = bm25_pool(row)
        key = (str(row["benchmark"]), str(row["task_id"]))
        expected_candidates = min(TOPK_CANDIDATES, candidate_count_by_id[key])
        if len(candidates) != expected_candidates:
            raise ValueError(f"{row['task_id']}: expected {expected_candidates} BM25 candidates, got {len(candidates)}")
        selected_count = count_by_id[key]
        if not 0 <= selected_count <= 10:
            raise ValueError(f"Invalid frozen selected_count={selected_count} for {row['task_id']}")
        query_idx = len(all_retriever_ids)
        all_retriever_ids.append(legacy_retriever_ids(query, retriever_tokenizer, is_query=True,
                                                       max_length=RETRIEVER_QUERY_TOKENS))
        candidate_start = len(all_retriever_ids)
        for candidate in candidates:
            all_retriever_ids.append(legacy_retriever_ids(candidate["retriever_text"], retriever_tokenizer,
                                                           is_query=False, max_length=RETRIEVER_CANDIDATE_TOKENS))
        prepared.append({"row": row, "query": query, "candidates": candidates,
                         "query_vector_index": query_idx, "candidate_vector_start": candidate_start,
                         "selected_count": selected_count})
        if index % 50 == 0:
            print(f"Prepared BM25 top-{TOPK_CANDIDATES}: {index}/400 tasks", flush=True)

    vectors = encode_unixcoder(retriever, all_retriever_ids, retriever_tokenizer, device,
                               args.retriever_batch_size)
    del retriever, all_retriever_ids
    torch.cuda.empty_cache()
    for item in prepared:
        q = vectors[item["query_vector_index"]]
        start = item["candidate_vector_start"]
        candidate_count = len(item["candidates"])
        scores = vectors[start : start + candidate_count] @ q
        order = np.argsort(scores.numpy())[::-1]
        item["ranked_candidates"] = [
            {**item["candidates"][int(i)], "unixcoder_score": float(scores[int(i)])}
            for i in order[: item["selected_count"]]
        ]
    del vectors

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_ID, revision=GENERATOR_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    prompts_by_budget: dict[int, list[dict]] = {budget: [] for budget in CROSS_FILE_BUDGETS}
    for item in prepared:
        row = item["row"]
        for budget in CROSS_FILE_BUDGETS:
            prompt_text, prompt_ids, cross_tokens, left_tokens = make_prompt(
                row, item["ranked_candidates"], tokenizer, TOTAL_CONTEXT_TOKENS, budget
            )
            prompts_by_budget[budget].append({
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids,
                "cross_file_tokens": cross_tokens,
                "left_context_tokens": left_tokens,
            })
    del retriever_tokenizer
    torch.cuda.empty_cache()

    generator = AutoModelForCausalLM.from_pretrained(
        GENERATOR_ID, revision=GENERATOR_REVISION,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    ).to(device).eval()
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    all_prompt_rows = []
    for budget in CROSS_FILE_BUDGETS:
        all_prompt_rows.extend((budget, i) for i in range(len(rows)))
    generation_output: dict[tuple[int, int], str] = {}
    for start in range(0, len(all_prompt_rows), args.generator_batch_size):
        batch_keys = all_prompt_rows[start : start + args.generator_batch_size]
        input_ids = torch.full((len(batch_keys), TOTAL_CONTEXT_TOKENS), tokenizer.pad_token_id,
                               dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for batch_index, (budget, row_index) in enumerate(batch_keys):
            token_ids = prompts_by_budget[budget][row_index]["prompt_ids"]
            input_ids[batch_index, -len(token_ids):] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[batch_index, -len(token_ids):] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            generated = generator.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=TOTAL_CONTEXT_TOKENS + GENERATOR_MAX_NEW_TOKENS,
                do_sample=False, num_beams=1, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id, use_cache=True,
            )
        for output_index, key in enumerate(batch_keys):
            text = tokenizer.decode(generated[output_index, TOTAL_CONTEXT_TOKENS:], skip_special_tokens=True)
            generation_output[key] = text
        if (start // args.generator_batch_size + 1) % 10 == 0 or start + len(batch_keys) == len(all_prompt_rows):
            print(f"DeepSeek generation: {min(start + len(batch_keys), len(all_prompt_rows))}/800 prompts",
                  flush=True)
    del generator
    torch.cuda.empty_cache()

    arm_rows: dict[int, list[dict]] = {budget: [] for budget in CROSS_FILE_BUDGETS}
    for budget in CROSS_FILE_BUDGETS:
        for i, item in enumerate(prepared):
            row = item["row"]
            prompt_meta = prompts_by_budget[budget][i]
            raw_prediction = generation_output[(budget, i)]
            score = score_prediction(prompt_meta["prompt_text"], raw_prediction, row["groundtruth"],
                                     row["language"], row["benchmark"])
            arm_rows[budget].append({
                "task_id": str(row["task_id"]), "benchmark": row["benchmark"],
                "language": row["language"], "selected_count": item["selected_count"],
                "cross_file_tokens": prompt_meta["cross_file_tokens"],
                "left_context_tokens": prompt_meta["left_context_tokens"],
                "em": score["em"], "es": score["es"],
                "raw_prediction": raw_prediction, "prediction": score["prediction"],
                "prompt": prompt_meta["prompt_text"],
                "selected_candidates": item["ranked_candidates"],
            })

    result = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "checkpoint_sha256": checkpoint_hash, "checkpoint_schema": CHECKPOINT_SCHEMA,
            "checkpoint_updates": 2480, "valid_sha256": valid_hash, "n": len(rows),
            "generator": GENERATOR_ID, "generator_revision": GENERATOR_REVISION,
            "generator_backend": "Hugging Face Transformers, greedy decoding, bf16",
            "retriever": RETRIEVER_ID, "retriever_revision": RETRIEVER_REVISION,
            "retriever_input": "RLCoder encoder-only format; query=last 20 non-empty left-context lines; candidate=natural block with file/line header",
            "chunking": "RLCoder natural blank-line mini-blocks, max 15 non-empty lines",
            "bm25_pool": TOPK_CANDIDATES, "max_real_snippets": 10,
            "selection_count_control": "per-task selected_count frozen from the previous 400-row PG-MSLD eval; candidate identity reranked by the same checkpoint",
            "total_context_tokens_fixed": TOTAL_CONTEXT_TOKENS,
            "cross_file_budgets": list(CROSS_FILE_BUDGETS),
            "max_new_tokens": GENERATOR_MAX_NEW_TOKENS,
            "scoring": "same score_prediction convention as src/train/benchmark_validation.py; RepoEval macro-averaged by repo",
        },
        "arms": {},
        "paired_delta_high_minus_low": {},
        "elapsed_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(0),
    }
    for budget in CROSS_FILE_BUDGETS:
        result["arms"][str(budget)] = {
            "total_context_tokens": TOTAL_CONTEXT_TOKENS,
            "cross_file_budget": budget,
            "metrics": summarize(arm_rows[budget]),
            "mean_selected_snippets": sum(x["selected_count"] for x in arm_rows[budget]) / len(rows),
            "mean_cross_file_tokens": sum(x["cross_file_tokens"] for x in arm_rows[budget]) / len(rows),
            "mean_left_context_tokens": sum(x["left_context_tokens"] for x in arm_rows[budget]) / len(rows),
        }
    low, high = CROSS_FILE_BUDGETS
    low_by_id = {(x["benchmark"], x["task_id"]): x for x in arm_rows[low]}
    high_by_id = {(x["benchmark"], x["task_id"]): x for x in arm_rows[high]}
    delta_rows = []
    for task_key in low_by_id:
        lo, hi = low_by_id[task_key], high_by_id[task_key]
        delta_rows.append({"task_id": lo["task_id"], "benchmark": lo["benchmark"],
                           "delta_em": hi["em"] - lo["em"], "delta_es": hi["es"] - lo["es"]})
    delta_summary = summarize([{"task_id": x["task_id"], "benchmark": x["benchmark"],
                                "em": x["delta_em"], "es": x["delta_es"]} for x in delta_rows])
    # summarize() computes averages of values; its per-benchmark fields are mean paired deltas here.
    result["paired_delta_high_minus_low"] = delta_summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for budget, values in arm_rows.items():
        path = args.output_dir / f"predictions_crossfile_{budget}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
