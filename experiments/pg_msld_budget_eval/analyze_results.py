#!/usr/bin/env python3
"""Paired task/repository bootstrap for the cross-file-budget Colab run."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2] / "results" / "pg-msld-crossfile-budget-colab-20261024"
METRICS = ROOT / "metrics.json"
LOW = ROOT / "predictions_crossfile_512.jsonl"
HIGH = ROOT / "predictions_crossfile_2344.jsonl"
N_BOOTSTRAP = 20_000


def load_rows(path: Path) -> dict[tuple[str, str], dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {(str(row["benchmark"]), str(row["task_id"])): row for row in rows}
    if len(rows) != 400 or len(result) != 400:
        raise ValueError(f"Expected 400 unique benchmark/task pairs in {path}; got {len(rows)}/{len(result)}")
    return result


def percentile(sorted_values: list[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def bootstrap_interval(units: list[float], rng: random.Random) -> tuple[float, float]:
    n = len(units)
    means = sorted(sum(units[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOTSTRAP))
    return percentile(means, 0.025), percentile(means, 0.975)


def main() -> None:
    metrics = json.loads(METRICS.read_text())
    low, high = load_rows(LOW), load_rows(HIGH)
    if low.keys() != high.keys():
        raise ValueError("Budget arms do not contain exactly the same task IDs")

    deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"em": [], "es": []})
    by_repo: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"em": [], "es": []})
    )
    wins = defaultdict(lambda: {"win": 0, "tie": 0, "loss": 0})
    for key in low:
        a, b = low[key], high[key]
        if a["benchmark"] != b["benchmark"]:
            raise ValueError(f"Mismatched benchmark for {key}")
        benchmark = key[0]
        d_em, d_es = b["em"] - a["em"], b["es"] - a["es"]
        deltas[benchmark]["em"].append(d_em)
        deltas[benchmark]["es"].append(d_es)
        repo = key[1].split("/", 1)[0]
        by_repo[benchmark][repo]["em"].append(d_em)
        by_repo[benchmark][repo]["es"].append(d_es)
        wins[benchmark]["win" if d_es > 0 else "loss" if d_es < 0 else "tie"] += 1

    rng = random.Random(20261024)
    benchmark_units: dict[str, dict[str, list[float]]] = {}
    report = {"paired_task_count": len(low), "bootstrap_replicates": N_BOOTSTRAP, "benchmarks": {}}
    for benchmark in sorted(deltas):
        is_repo = benchmark.startswith("repoeval")
        benchmark_units[benchmark] = {"em": [], "es": []}
        if is_repo:
            for repo, values in sorted(by_repo[benchmark].items()):
                for metric in ("em", "es"):
                    benchmark_units[benchmark][metric].append(sum(values[metric]) / len(values[metric]))
        else:
            benchmark_units[benchmark] = deltas[benchmark]

        report["benchmarks"][benchmark] = {
            "n_tasks": len(deltas[benchmark]["es"]),
            "bootstrap_unit": "repo" if is_repo else "task",
            "repo_count": len(benchmark_units[benchmark]["es"]) if is_repo else None,
            "delta_em": sum(benchmark_units[benchmark]["em"]) / len(benchmark_units[benchmark]["em"]),
            "delta_em_95ci": bootstrap_interval(benchmark_units[benchmark]["em"], rng),
            "delta_es": sum(benchmark_units[benchmark]["es"]) / len(benchmark_units[benchmark]["es"]),
            "delta_es_95ci": bootstrap_interval(benchmark_units[benchmark]["es"], rng),
            "task_level_es_win_tie_loss": wins[benchmark],
        }

    macro_bootstrap: dict[str, list[float]] = {"em": [], "es": []}
    for metric in ("em", "es"):
        for _ in range(N_BOOTSTRAP):
            sampled_benchmark_means = []
            for benchmark in sorted(benchmark_units):
                units = benchmark_units[benchmark][metric]
                n = len(units)
                sampled_benchmark_means.append(sum(units[rng.randrange(n)] for _ in range(n)) / n)
            macro_bootstrap[metric].append(sum(sampled_benchmark_means) / len(sampled_benchmark_means))
        macro_bootstrap[metric].sort()

    macro_delta = metrics["paired_delta_high_minus_low"]["macro_mean_4_benchmarks"]
    report["macro_delta_high_minus_low"] = {
        **macro_delta,
        "em_95ci": [percentile(macro_bootstrap["em"], 0.025), percentile(macro_bootstrap["em"], 0.975)],
        "es_95ci": [percentile(macro_bootstrap["es"], 0.025), percentile(macro_bootstrap["es"], 0.975)],
        "bootstrap_unit": "task for CCEval; repository for RepoEval; four benchmarks macro-averaged",
    }
    report["arm_metrics"] = metrics["arms"]
    report["paired_selection_overlap"] = {
        "selected_count_identical": all(low[k]["selected_count"] == high[k]["selected_count"] for k in low),
        "selected_snippets_identical": all(
            [x["path"] + ":" + str(x["start"]) + "-" + str(x["end"]) for x in low[k]["selected_candidates"]]
            == [x["path"] + ":" + str(x["start"]) + "-" + str(x["end"]) for x in high[k]["selected_candidates"]]
            for k in low
        ),
    }
    out = ROOT / "paired_analysis.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
