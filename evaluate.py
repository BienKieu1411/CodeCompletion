"""
GraphFRL Evaluation Pipeline

Computes standard code completion metrics:
- Exact Match (EM): line-by-line comparison after whitespace normalization
- Edit Similarity (ES): editdistance-based (same as RepoEval/CCEval)
- Identifier EM: exact match on extracted identifiers
- Identifier F1: precision/recall on identifier sets

Compatible with AlignCoder evaluation format for fair comparison.
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_exact_match(prediction: str, ground_truth: str) -> int:
    """
    Line-level Exact Match.
    Strips whitespace, filters empty lines, truncates pred to GT length.
    """
    pred_lines = [l.strip() for l in prediction.splitlines() if l.strip()]
    gt_lines = [l.strip() for l in ground_truth.splitlines() if l.strip()]
    pred_lines = pred_lines[:len(gt_lines)]

    if len(pred_lines) != len(gt_lines):
        return 0
    return 1 if pred_lines == gt_lines else 0


def compute_edit_similarity(prediction: str, ground_truth: str) -> float:
    """
    Edit Similarity = 1 - (edit_distance / max_length).
    Uses editdistance package (same as RepoEval).
    Falls back to difflib if editdistance not installed.
    """
    pred_lines = [l.strip() for l in prediction.splitlines() if l.strip()]
    gt_lines = [l.strip() for l in ground_truth.splitlines() if l.strip()]
    pred_lines = pred_lines[:len(gt_lines)]

    pred_str = "\n".join(pred_lines)
    gt_str = "\n".join(gt_lines)

    if not gt_str and not pred_str:
        return 1.0
    if not gt_str or not pred_str:
        return 0.0

    try:
        import editdistance
        dist = editdistance.eval(gt_str, pred_str)
        max_len = max(len(gt_str), len(pred_str))
        return 1.0 - (dist / max_len) if max_len > 0 else 1.0
    except ImportError:
        import difflib
        return difflib.SequenceMatcher(None, pred_str, gt_str).ratio()


# ── Identifier Extraction ────────────────────────────────────────────────────

_IDENTIFIER_RE = re.compile(r'[_a-zA-Z][_a-zA-Z0-9]*')

# Python keywords to exclude from identifier matching
_PYTHON_KEYWORDS = {
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
    'try', 'while', 'with', 'yield',
}

_JAVA_KEYWORDS = {
    'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch',
    'char', 'class', 'const', 'continue', 'default', 'do', 'double',
    'else', 'enum', 'extends', 'final', 'finally', 'float', 'for',
    'goto', 'if', 'implements', 'import', 'instanceof', 'int',
    'interface', 'long', 'native', 'new', 'null', 'package', 'private',
    'protected', 'public', 'return', 'short', 'static', 'strictfp',
    'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
    'transient', 'try', 'void', 'volatile', 'while', 'true', 'false',
}


def extract_identifiers(code: str, language: str = "python") -> List[str]:
    """Extract unique identifiers from code, excluding language keywords."""
    # Remove string literals to avoid false identifiers
    code_clean = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', '', code)
    code_clean = re.sub(r"'([^'\\]*(\\.[^'\\]*)*)'", '', code_clean)
    # Remove comments
    code_clean = re.sub(r'#.*', '', code_clean)
    code_clean = re.sub(r'//.*', '', code_clean)

    keywords = _PYTHON_KEYWORDS if language == "python" else _JAVA_KEYWORDS
    tokens = _IDENTIFIER_RE.findall(code_clean)
    return [t for t in tokens if t not in keywords]


def compute_identifier_match(pred_ids: List[str], gt_ids: List[str]) -> Dict[str, float]:
    """Compute identifier precision, recall, F1."""
    pred_set = set(pred_ids)
    gt_set = set(gt_ids)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    id_em = 1 if pred_set == gt_set else 0

    return {
        "id_em": id_em,
        "id_precision": precision,
        "id_recall": recall,
        "id_f1": f1,
    }


# ── Post-processing ──────────────────────────────────────────────────────────

def postprocess_prediction(prediction: str, ground_truth: str = "") -> str:
    """
    Post-process generated code:
    1. Remove trailing empty lines
    2. Remove comments (for fair comparison)
    3. Truncate to same number of non-empty lines as ground truth
    """
    # Remove comments
    prediction = re.sub(r'#.*', '', prediction)
    prediction = re.sub(r'//.*', '', prediction)

    pred_lines = [l for l in prediction.splitlines()]

    if ground_truth:
        gt_nonempty = [l for l in ground_truth.splitlines() if l.strip()]
        n_target = len(gt_nonempty)
        # Keep only the first n_target non-empty lines
        kept = []
        count = 0
        for line in pred_lines:
            kept.append(line)
            if line.strip():
                count += 1
            if count >= n_target:
                break
        pred_lines = kept

    return "\n".join(pred_lines)


# ── Batch Evaluation ──────────────────────────────────────────────────────────

def evaluate_predictions(
    predictions: List[Dict],
    language: str = "python",
) -> Dict:
    """
    Evaluate a list of predictions.

    Each prediction dict must have:
      - task_id: str
      - pred: str (generated code)
      - target: str (ground truth)

    Returns:
      Dict with aggregate metrics: em, es, id_em, id_f1
    """
    em_scores = []
    es_scores = []
    id_em_scores = []
    id_f1_scores = []
    id_precision_scores = []
    id_recall_scores = []
    detailed = []

    for item in predictions:
        pred = postprocess_prediction(item["pred"], item["target"])
        gt = item["target"]

        em = compute_exact_match(pred, gt)
        es = compute_edit_similarity(pred, gt)
        pred_ids = extract_identifiers(pred, language)
        gt_ids = extract_identifiers(gt, language)
        id_metrics = compute_identifier_match(pred_ids, gt_ids)

        em_scores.append(em)
        es_scores.append(es)
        id_em_scores.append(id_metrics["id_em"])
        id_f1_scores.append(id_metrics["id_f1"])
        id_precision_scores.append(id_metrics["id_precision"])
        id_recall_scores.append(id_metrics["id_recall"])

        detailed.append({
            "task_id": item["task_id"],
            "em": em,
            "es": round(es, 4),
            "id_em": id_metrics["id_em"],
            "id_f1": round(id_metrics["id_f1"], 4),
            "id_precision": round(id_metrics["id_precision"], 4),
            "id_recall": round(id_metrics["id_recall"], 4),
        })

    n = len(predictions)
    results = {
        "em": round(sum(em_scores) / n * 100, 2) if n > 0 else 0.0,
        "es": round(sum(es_scores) / n * 100, 2) if n > 0 else 0.0,
        "id_em": round(sum(id_em_scores) / n * 100, 2) if n > 0 else 0.0,
        "id_f1": round(sum(id_f1_scores) / n * 100, 2) if n > 0 else 0.0,
        "id_precision": round(sum(id_precision_scores) / n * 100, 2) if n > 0 else 0.0,
        "id_recall": round(sum(id_recall_scores) / n * 100, 2) if n > 0 else 0.0,
        "total": n,
        "detailed": detailed,
    }
    return results


def save_results(
    predictions: List[Dict],
    output_dir: str,
    language: str = "python",
):
    """
    Save predictions and evaluation results to output_dir.

    Creates:
      - prediction.jsonl
      - results.json
      - detailed_results.jsonl
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions
    pred_path = os.path.join(output_dir, "prediction.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps({
                "task_id": item["task_id"],
                "pred": item["pred"],
            }, ensure_ascii=False) + "\n")

    # Evaluate
    results = evaluate_predictions(predictions, language)

    # Save aggregate results
    res_path = os.path.join(output_dir, "results.json")
    aggregate = {k: v for k, v in results.items() if k != "detailed"}
    with open(res_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    # Save detailed results
    detail_path = os.path.join(output_dir, "detailed_results.jsonl")
    with open(detail_path, "w", encoding="utf-8") as f:
        for d in results["detailed"]:
            f.write(json.dumps(d) + "\n")

    logger.info(
        f"Results saved to {output_dir}: "
        f"EM={results['em']}%, ES={results['es']}%, "
        f"ID_EM={results['id_em']}%, ID_F1={results['id_f1']}%"
    )

    return results


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    """Evaluate predictions from a JSONL file."""
    import argparse

    parser = argparse.ArgumentParser(description="GraphFRL Evaluation")
    parser.add_argument("--predictions", required=True, help="Path to prediction.jsonl")
    parser.add_argument("--ground_truth", required=True, help="Path to ground_truth.jsonl or test.jsonl")
    parser.add_argument("--output_dir", default="results/eval", help="Output directory")
    parser.add_argument("--language", default="python", help="Programming language")
    args = parser.parse_args()

    # Load predictions
    preds = {}
    with open(args.predictions, "r") as f:
        for line in f:
            item = json.loads(line)
            preds[item["task_id"]] = item["pred"]

    # Load ground truth
    merged = []
    with open(args.ground_truth, "r") as f:
        for line in f:
            item = json.loads(line)
            task_id = item.get("task_id", item.get("metadata", {}).get("task_id", ""))
            gt = item.get("groundtruth", item.get("target_code", item.get("target", "")))
            if task_id in preds:
                merged.append({
                    "task_id": task_id,
                    "pred": preds[task_id],
                    "target": gt,
                })

    print(f"Evaluating {len(merged)} predictions...")
    results = save_results(merged, args.output_dir, args.language)

    print(f"\n{'='*50}")
    print(f"  Exact Match (EM):    {results['em']}%")
    print(f"  Edit Similarity (ES): {results['es']}%")
    print(f"  Identifier EM:       {results['id_em']}%")
    print(f"  Identifier F1:       {results['id_f1']}%")
    print(f"  Total samples:       {results['total']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
