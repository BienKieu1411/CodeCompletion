"""Portable AlignCoder-style evaluation files and metrics.

AlignCoder's public evaluation path writes ``prediction.jsonl`` first, then a
post-processing pass creates ``prediction_truncated.jsonl``,
``exact_match_idx.jsonl``, ``detailed_results.json`` and ``results.json``.
This module mirrors that file contract without depending on AlignCoder's local
tree-sitter shared objects, so the standalone ``src`` bundle remains enough to
run on a server.
"""

from __future__ import annotations

import ast
import json
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Sequence


IDENTIFIER_REGEX = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")
STRING_PATTERN = re.compile(
    r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''
)

LANGUAGE_KEYWORDS = {
    "python": {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    },
    "java": {
        "abstract",
        "assert",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extends",
        "final",
        "finally",
        "float",
        "for",
        "goto",
        "if",
        "implements",
        "import",
        "instanceof",
        "int",
        "interface",
        "long",
        "native",
        "new",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "short",
        "static",
        "strictfp",
        "super",
        "switch",
        "synchronized",
        "this",
        "throw",
        "throws",
        "transient",
        "try",
        "void",
        "volatile",
        "while",
    },
    "javascript": {
        "await",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "debugger",
        "default",
        "delete",
        "do",
        "else",
        "export",
        "extends",
        "finally",
        "for",
        "function",
        "if",
        "import",
        "in",
        "instanceof",
        "let",
        "new",
        "return",
        "super",
        "switch",
        "this",
        "throw",
        "try",
        "typeof",
        "var",
        "void",
        "while",
        "with",
        "yield",
    },
    "typescript": {
        "abstract",
        "any",
        "as",
        "async",
        "await",
        "boolean",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "declare",
        "default",
        "delete",
        "do",
        "else",
        "enum",
        "export",
        "extends",
        "finally",
        "for",
        "from",
        "function",
        "if",
        "implements",
        "import",
        "in",
        "interface",
        "let",
        "module",
        "namespace",
        "new",
        "number",
        "private",
        "protected",
        "public",
        "readonly",
        "return",
        "static",
        "string",
        "super",
        "switch",
        "this",
        "throw",
        "try",
        "type",
        "typeof",
        "var",
        "void",
        "while",
        "with",
        "yield",
    },
    "csharp": {
        "abstract",
        "as",
        "base",
        "bool",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "checked",
        "class",
        "const",
        "continue",
        "decimal",
        "default",
        "delegate",
        "do",
        "double",
        "else",
        "enum",
        "event",
        "explicit",
        "extern",
        "false",
        "finally",
        "fixed",
        "float",
        "for",
        "foreach",
        "goto",
        "if",
        "implicit",
        "in",
        "int",
        "interface",
        "internal",
        "is",
        "lock",
        "long",
        "namespace",
        "new",
        "null",
        "object",
        "operator",
        "out",
        "override",
        "params",
        "private",
        "protected",
        "public",
        "readonly",
        "ref",
        "return",
        "sbyte",
        "sealed",
        "short",
        "sizeof",
        "stackalloc",
        "static",
        "string",
        "struct",
        "switch",
        "this",
        "throw",
        "true",
        "try",
        "typeof",
        "uint",
        "ulong",
        "unchecked",
        "unsafe",
        "ushort",
        "using",
        "virtual",
        "void",
        "volatile",
        "while",
    },
}


def infer_language(path: str | None = None, fallback: str = "python") -> str:
    """Infer AlignCoder language name from a file or dataset path."""
    lowered = (path or "").lower()
    if "cceval_java" in lowered or lowered.endswith(".java") or "/java/" in lowered:
        return "java"
    if "typescript" in lowered or lowered.endswith(".ts") or lowered.endswith(".tsx"):
        return "typescript"
    if "javascript" in lowered or lowered.endswith(".js") or lowered.endswith(".jsx"):
        return "javascript"
    if "csharp" in lowered or lowered.endswith(".cs"):
        return "csharp"
    if "python" in lowered or lowered.endswith(".py"):
        return "python"
    return fallback


def remove_comments(code: str) -> str:
    """Match AlignCoder's simple comment removal for generated snippets."""
    code = re.sub(r"#.*", "", code or "")
    code = re.sub(r"//.*", "", code)
    return code


def extract_identifiers(source_code: str, language: str = "python") -> List[str]:
    """Extract non-keyword identifiers with the same broad shape as AlignCoder."""
    without_strings = STRING_PATTERN.sub("", source_code or "")
    keywords = LANGUAGE_KEYWORDS.get(language, set())
    return [
        token
        for token in IDENTIFIER_REGEX.findall(without_strings)
        if token not in keywords
    ]


def _first_bracket_statement(completion: str) -> str:
    for idx, char in enumerate(completion or ""):
        if char in {";", "}", "{"}:
            return completion[: idx + 1]
    return completion or ""


def _python_one_statement(prompt: str, completion: str) -> str:
    """Portable approximation of AlignCoder's tree-sitter truncation.

    The original evaluator returns the shortest generated prefix that makes
    ``prompt + prefix`` parseable and ends before a newline. When parsing is not
    possible because the prompt is a partial file, we keep the full completion,
    matching AlignCoder's failure fallback.
    """
    completion = completion or ""
    if not completion:
        return ""
    for idx in range(len(completion) - 1):
        if completion[idx + 1] != "\n":
            continue
        candidate = completion[: idx + 1].rstrip()
        try:
            ast.parse((prompt or "") + candidate)
        except SyntaxError:
            continue
        return candidate
    return completion


def postprocess_code_lines(prompt: str, completion: str, language: str) -> str:
    """Truncate a generated completion before metric calculation."""
    if language in {"java", "csharp", "typescript", "javascript"}:
        return _first_bracket_statement(completion)
    if language == "python":
        return _python_one_statement(prompt, completion)
    return completion or ""


def _nonempty_stripped_lines(text: str) -> List[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _fuzz_ratio(prediction: str, target: str) -> float:
    """Return a 0-100 ratio compatible with fuzzywuzzy.fuzz.ratio scale."""
    prediction = (prediction or "").strip()
    target = (target or "").strip()
    if prediction == target:
        return 100.0
    if not prediction and not target:
        return 100.0
    return float(round(100.0 * SequenceMatcher(None, prediction, target).ratio()))


def _levenshtein_distance(a: str, b: str) -> int:
    """Small dependency-free edit distance used by RepoEval-style ES."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _id_counts(pred_ids: Iterable[str], target_ids: Iterable[str]) -> tuple[int, int, int]:
    pred_set = set(pred_ids)
    target_set = set(target_ids)
    tp = len(pred_set & target_set)
    fp = len(pred_set - target_set)
    fn = len(target_set - pred_set)
    return tp, fp, fn


def _repoeval_em(target: str, prediction: str) -> float:
    target_lines = _nonempty_stripped_lines(target)
    prediction_lines = _nonempty_stripped_lines(prediction)[: len(target_lines)]
    if len(target_lines) != len(prediction_lines):
        return 0.0
    return 1.0 if target_lines == prediction_lines else 0.0


def _repoeval_es(target: str, prediction: str) -> float:
    target_lines = _nonempty_stripped_lines(target)
    prediction_lines = _nonempty_stripped_lines(prediction)[: len(target_lines)]
    target_str = "\n".join(target_lines)
    prediction_str = "\n".join(prediction_lines)
    denom = max(len(target_str), len(prediction_str))
    if denom == 0:
        return 1.0
    return 1.0 - (_levenshtein_distance(target_str, prediction_str) / denom)


def _compute_repoeval_metrics(
    truncated: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    repo_scores: Dict[str, Dict[str, List[float]]] = {}
    for row in truncated:
        task_id = str(row.get("task_id", ""))
        repo_id = task_id.split("/")[0] if "/" in task_id else task_id
        repo_id = repo_id or "unknown_repo"
        repo_bucket = repo_scores.setdefault(repo_id, {"em": [], "es": []})
        target = str(row.get("target") or "")
        prediction = str(row.get("pred") or "")
        repo_bucket["em"].append(_repoeval_em(target, prediction))
        repo_bucket["es"].append(_repoeval_es(target, prediction))

    if not repo_scores:
        return {"repoeval_em": 0.0, "repoeval_es": 0.0, "repoeval_total": 0}

    per_repo_em = [
        round(sum(bucket["em"]) / max(1, len(bucket["em"])), 4)
        for bucket in repo_scores.values()
    ]
    per_repo_es = [
        round(sum(bucket["es"]) / max(1, len(bucket["es"])), 4)
        for bucket in repo_scores.values()
    ]
    total = sum(len(bucket["em"]) for bucket in repo_scores.values())
    return {
        "repoeval_em": round(sum(per_repo_em) / len(per_repo_em) * 100, 4),
        "repoeval_es": round(sum(per_repo_es) / len(per_repo_es) * 100, 4),
        "repoeval_total": total,
    }


def compute_aligncoder_metrics(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compute AlignCoder-style metrics from in-memory prediction records."""
    truncated: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []
    exact_ids: List[str] = []

    for idx, record in enumerate(records):
        task_id = str(record.get("task_id", idx))
        language = infer_language(
            str(record.get("file_path") or record.get("dataset_path") or ""),
            fallback=str(record.get("language") or "python"),
        )
        pred = postprocess_code_lines(
            str(record.get("prompt") or ""),
            str(record.get("pred") or ""),
            language,
        )
        pred = remove_comments(pred)
        target = remove_comments(str(record.get("target") or ""))
        pred_lines = _nonempty_stripped_lines(pred)
        target_lines = _nonempty_stripped_lines(target)
        em_label = int(pred_lines == target_lines)
        pred_ids = extract_identifiers(pred, language)
        target_ids = extract_identifiers(target, language)
        identifier_em = int(pred_ids == target_ids)
        id_tp, id_fp, id_fn = _id_counts(pred_ids, target_ids)
        id_precision = id_tp / (id_tp + id_fp) if (id_tp + id_fp) else 0.0
        id_recall = id_tp / (id_tp + id_fn) if (id_tp + id_fn) else 0.0
        id_f1 = (
            2 * id_tp / (2 * id_tp + id_fp + id_fn)
            if (2 * id_tp + id_fp + id_fn)
            else 0.0
        )
        es = _fuzz_ratio(pred, target)

        if em_label:
            exact_ids.append(task_id)
        truncated.append(
            {
                "task_id": task_id,
                "pred": pred,
                "target": target,
                "pred_ids": pred_ids,
                "target_ids": target_ids,
                "language": language,
            }
        )
        detailed.append(
            {
                "task_id": task_id,
                "em": em_label,
                "es": es,
                "id_em": identifier_em,
                "id_precision": id_precision,
                "id_recall": id_recall,
                "id_f1": id_f1,
                "language": language,
            }
        )

    total = len(truncated)
    denom = max(1, total)
    repoeval = _compute_repoeval_metrics(truncated)
    return {
        "em": round(sum(row["em"] for row in detailed) / denom * 100, 4),
        "es": round(sum(row["es"] for row in detailed) / denom, 4),
        "id_em": round(sum(row["id_em"] for row in detailed) / denom * 100, 4),
        "id_precision": round(
            sum(row["id_precision"] for row in detailed) / denom * 100, 4
        ),
        "id_recall": round(
            sum(row["id_recall"] for row in detailed) / denom * 100, 4
        ),
        "id_f1": round(sum(row["id_f1"] for row in detailed) / denom * 100, 4),
        "total": total,
        **repoeval,
        "truncated_samples": truncated,
        "detailed_results": detailed,
        "exact_match_task_ids": exact_ids,
    }


def write_aligncoder_metric_files(
    output_dir: str,
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Write AlignCoder-style postprocessed metric artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = compute_aligncoder_metrics(records)
    truncated_path = os.path.join(output_dir, "prediction_truncated.jsonl")
    exact_match_path = os.path.join(output_dir, "exact_match_idx.jsonl")
    detailed_path = os.path.join(output_dir, "detailed_results.json")
    results_path = os.path.join(output_dir, "results.json")

    with open(truncated_path, "w", encoding="utf-8") as f:
        for row in metrics["truncated_samples"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(exact_match_path, "w", encoding="utf-8") as f:
        for task_id in metrics["exact_match_task_ids"]:
            f.write(f"{task_id}\n")
    with open(detailed_path, "w", encoding="utf-8") as f:
        for row in metrics["detailed_results"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    public_metrics = {
        key: metrics[key]
        for key in (
            "em",
            "es",
            "id_em",
            "id_precision",
            "id_recall",
            "id_f1",
            "total",
            "repoeval_em",
            "repoeval_es",
            "repoeval_total",
        )
    }
    public_metrics["aligncoder_report_em"] = (
        f"{public_metrics['em']}({public_metrics['repoeval_em']})"
    )
    public_metrics["aligncoder_report_es"] = (
        f"{public_metrics['es']}({public_metrics['repoeval_es']})"
    )
    public_metrics["metric_scale"] = "aligncoder_percent"
    public_metrics["postprocess"] = (
        "portable_aligncoder_style_no_tree_sitter_binary"
    )
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(public_metrics, f, indent=2, ensure_ascii=False)

    return {
        **public_metrics,
        "prediction_truncated_path": truncated_path,
        "exact_match_idx_path": exact_match_path,
        "detailed_results_path": detailed_path,
        "results_path": results_path,
    }
