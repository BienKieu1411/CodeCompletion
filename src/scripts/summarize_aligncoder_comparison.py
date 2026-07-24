#!/usr/bin/env python3
"""Summarize ICAR results against AlignCoder-style result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


MAIN_BENCHMARKS = ("cceval_python", "cceval_java", "repoeval_line", "repoeval_api")


def _metric_float(value: Any) -> Optional[float]:
    if value is None or value == "-":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    # AlignCoder sometimes reports "cceval(repoeval)".
    text = text.split("(", 1)[0]
    try:
        return float(text)
    except ValueError:
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_result(root: Path, benchmark: str) -> Dict[str, Any]:
    direct = root / benchmark / "results.json"
    if direct.exists():
        return _read_json(direct)
    nested = root / "eval" / benchmark / "results.json"
    if nested.exists():
        return _read_json(nested)
    matches = sorted(root.glob(f"**/{benchmark}/results.json"))
    return _read_json(matches[0]) if matches else {}


def _row(
    benchmark: str,
    icar: Dict[str, Any],
    baseline: Dict[str, Any],
    metrics: Iterable[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"benchmark": benchmark}
    for metric in metrics:
        icar_value = _metric_float(icar.get(metric))
        baseline_value = _metric_float(baseline.get(metric))
        out[f"icar_{metric}"] = icar_value
        out[f"aligncoder_{metric}"] = baseline_value
        out[f"delta_{metric}"] = (
            round(icar_value - baseline_value, 4)
            if icar_value is not None and baseline_value is not None
            else None
        )
    return out


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _print_markdown(rows: list[Dict[str, Any]], metrics: list[str]) -> None:
    headers = ["benchmark"]
    for metric in metrics:
        headers.extend([f"icar_{metric}", f"aligncoder_{metric}", f"delta_{metric}"])
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(_format_cell(row.get(h)) for h in headers) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare ICAR AlignCoder-style results against a baseline root."
    )
    parser.add_argument("--icar-root", required=True, help="ICAR result root")
    parser.add_argument(
        "--aligncoder-root",
        default="",
        help="Optional AlignCoder result root containing <benchmark>/results.json",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=list(MAIN_BENCHMARKS),
        help="Benchmark labels to summarize",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["em", "es", "id_f1"],
        help="Metric keys to compare from results.json",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead")
    args = parser.parse_args()

    icar_root = Path(args.icar_root)
    baseline_root = Path(args.aligncoder_root) if args.aligncoder_root else None
    rows = []
    for benchmark in args.benchmarks:
        icar = _find_result(icar_root, benchmark)
        baseline = _find_result(baseline_root, benchmark) if baseline_root else {}
        rows.append(_row(benchmark, icar, baseline, args.metrics))

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        _print_markdown(rows, list(args.metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
