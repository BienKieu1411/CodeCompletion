"""Leakage-safe end-to-end validation on the prepared benchmark rows.

``valid.parquet`` is an evaluation artifact, not a training-label artifact.
This module uses only the prepared model-input payload while selecting context;
``groundtruth`` is read only after generation for scoring and ``right_context``
is never read.  The reported CCEval and RepoEval scores follow the scoring
conventions used by the checked-in RLCoder/AlignCoder evaluators.
"""

from collections import defaultdict
import json
import re
from pathlib import Path
import zlib

import pyarrow.parquet as pq
from fuzzywuzzy import fuzz

from src.train.context_contract import ContextConfig, ContextRenderer
from src.train.select_conditional_context import select_context


VALID_SCHEMA = "ast_completion_validation_v1"


def unpack_payload(blob):
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError("Validation ast_payload must be a compressed byte string")
    return json.loads(zlib.decompress(bytes(blob)))


def validate_file(path):
    """Validate metadata without reading benchmark labels into memory."""
    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata or {}
    required = {"task_id", "path", "groundtruth", "left_context", "right_context",
                "crossfile_context", "ast_payload", "benchmark", "language"}
    if (metadata.get(b"artifact_schema") != VALID_SCHEMA.encode()
            or not required.issubset(parquet.schema_arrow.names)
            or parquet.metadata.num_rows == 0):
        raise ValueError("Need prepared ast_completion_validation_v1 valid.parquet")
    return {"rows": parquet.metadata.num_rows}


def _tree_has_error(source, language):
    import tree_sitter as ts
    import tree_sitter_java
    import tree_sitter_python

    grammar = tree_sitter_python if language == "python" else tree_sitter_java
    tree = ts.Parser(ts.Language(grammar.language())).parse(source.encode("utf-8"))
    return tree.root_node.has_error


def _python_one_statement(prompt, completion):
    """Mirror RLCoder's parser-assisted first-statement truncation."""
    for index in range(max(0, len(completion) - 1)):
        if completion[index + 1] != "\n":
            continue
        try:
            if not _tree_has_error(prompt + completion[:index + 1], "python"):
                return completion[:index + 1].rstrip()
        except Exception:
            # RLCoder keeps the unprocessed completion on parser failures.
            return completion
    return completion


def _bracket_statement(completion):
    # Keep the original evaluator's first-bracket rule.  In particular, do
    # not try to infer a target length from groundtruth before scoring.
    end = next((i for i, char in enumerate(completion) if char in ";}{"), None)
    return completion[:end + 1] if end else completion


def postprocess_completion(prompt, completion, language):
    if language == "python":
        completion = _python_one_statement(prompt, completion)
    elif language == "java":
        completion = _bracket_statement(completion)
    # This deliberately matches the benchmark evaluator's lightweight
    # comment removal. It is applied only to scoring, never to model input.
    return re.sub(r"#.*", "", re.sub(r"//.*", "", completion))


def _nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def _levenshtein(left, right):
    if len(left) > len(right):
        left, right = right, left
    row = list(range(len(left) + 1))
    for i, char_right in enumerate(right, 1):
        current = [i]
        for j, char_left in enumerate(left, 1):
            current.append(min(current[-1] + 1, row[j] + 1,
                               row[j - 1] + (char_left != char_right)))
        row = current
    return row[-1]


def score_prediction(prompt, prediction, target, language, benchmark):
    """Return one row score using the RLCoder/RepoEval conventions."""
    processed = postprocess_completion(prompt, prediction, language)
    target = re.sub(r"#.*", "", re.sub(r"//.*", "", target))
    target_lines = _nonempty_lines(target)
    prediction_lines = _nonempty_lines(processed)
    if benchmark.startswith("repoeval"):
        # RepoEval evaluates only the first target-length lines after the
        # language postprocessor and computes normalized character edit sim.
        clipped = prediction_lines[:len(target_lines)]
        left, right = "\n".join(target_lines), "\n".join(clipped)
        em = float(len(target_lines) == len(clipped) and left == right)
        es = 1.0 - _levenshtein(left, right) / max(len(left), len(right), 1)
    else:
        em = float(prediction_lines == target_lines)
        es = fuzz.ratio(processed.strip(), target.strip()) / 100.0
    return {"em": em, "es": es, "prediction": processed}


def _task_from_row(row, payload):
    """Build the allowlisted model task; benchmark labels never enter it."""
    candidates = payload.get("candidate_pool")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Validation row {row['task_id']} has no candidate pool")
    task = {
        "task_id": row["task_id"],
        "repo_uid": str(row["task_id"]).split("/", 1)[0],
        "language": row["language"],
        "file_path": payload["target_path"],
        "left_context": payload["model_left_context"],
        "candidates": candidates,
    }
    forbidden = {"groundtruth", "right_context", "target_code", "target_kind"}
    if forbidden.intersection(task):
        raise AssertionError("Benchmark gold leaked into validation model task")
    return task


def evaluate(model, path, generator_tokenizer, retriever_tokenizer, oracle, *,
             config=ContextConfig(), encoder_microbatch=64, check_continue=lambda: None,
             selector=None):
    """Run retrieval + one generator call per validation row without backward."""
    validate_file(path)
    rows = []
    prompts = []
    selected_counts = []
    selected_costs = []
    for batch in pq.ParquetFile(path).iter_batches(
            batch_size=16,
            columns=["task_id", "groundtruth", "ast_payload", "benchmark", "language"]):
        for row in batch.to_pylist():
            check_continue()
            payload = unpack_payload(row["ast_payload"])
            task = _task_from_row(row, payload)
            if selector is None:
                decision = select_context(
                    model, task, generator_tokenizer, retriever_tokenizer, config,
                    token_penalty=.01, threshold=0., max_moves=30, pair_shortlist=8,
                    encoder_microbatch=encoder_microbatch)
            else:
                decision = selector(model, task, generator_tokenizer, retriever_tokenizer,
                                    config, encoder_microbatch=encoder_microbatch)
            renderer = ContextRenderer(task, generator_tokenizer, config)
            prompt, cost = renderer.render(decision["indices"])
            rows.append({"row": row, "task": task, "prompt": prompt,
                         "indices": decision["indices"]})
            prompts.append(prompt)
            selected_counts.append(len(decision["indices"]))
            selected_costs.append(cost)
            if len(rows) % 25 == 0:
                print(json.dumps({"phase": "validation_retrieval", "completed": len(rows)}), flush=True)
    predictions = oracle.generate(prompts)
    if len(predictions) != len(rows):
        raise ValueError("Validation generator response count mismatch")

    details = []
    for item, prediction in zip(rows, predictions):
        source = item["row"]
        score = score_prediction(item["prompt"], prediction, source["groundtruth"],
                                 source["language"], source["benchmark"])
        details.append({
            "task_id": source["task_id"], "benchmark": source["benchmark"],
            "language": source["language"], "candidate_count": len(item["task"]["candidates"]),
            "selected_count": len(item["indices"]),
            "selected_token_cost": selected_costs[len(details)],
            "em": score["em"], "es": score["es"],
        })

    by_benchmark = defaultdict(list)
    for detail in details:
        by_benchmark[detail["benchmark"]].append(detail)
    benchmark_metrics = {}
    for benchmark, values in sorted(by_benchmark.items()):
        if benchmark.startswith("repoeval"):
            by_repo = defaultdict(list)
            for value in values:
                by_repo[value["task_id"].split("/", 1)[0]].append(value)
            groups = list(by_repo.values())
            em = sum(sum(v["em"] for v in group) / len(group) for group in groups) / len(groups)
            es = sum(sum(v["es"] for v in group) / len(group) for group in groups) / len(groups)
            repo_count = len(groups)
        else:
            em = sum(v["em"] for v in values) / len(values)
            es = sum(v["es"] for v in values) / len(values)
            repo_count = None
        benchmark_metrics[benchmark] = {"n": len(values), "em": em, "es": es,
                                        "repo_count": repo_count}
    return {
        "n": len(details), "benchmarks": benchmark_metrics,
        "mean_selected_count": sum(selected_counts) / max(len(selected_counts), 1),
        "mean_selected_token_cost": sum(selected_costs) / max(len(selected_costs), 1),
        "details": details,
    }
