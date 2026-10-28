#!/usr/bin/env python3
"""Compare saved predictions with one scorer; this is not a controlled ablation."""
import collections
import gzip
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "failure_analysis"
SCORER = ROOT / "experiments/rlcoder_official_repro_20261026/run_cadre_v29_adaptive_slack.py"
PRED = ROOT / "experiments/retrieval_cssi_rerank_study/colab_results/bm25_top100_rerank/bm25_top100_rerank_predictions.jsonl.gz"
V29 = ROOT / "results/cadre_v29_official_8004/cadre_v29_8004_predictions.jsonl.gz"
BASE = ROOT / "experiments/rlcoder_official_repro_20261026/results_fixed"
BENCHMARKS = ("cceval_python", "cceval_java", "repoeval_api", "repoeval_line")


def main():
    spec = importlib.util.spec_from_file_location("cadre_v29_scorer", SCORER)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    with gzip.open(PRED, "rt") as f:
        rerank = [json.loads(line) for line in f]
    with gzip.open(V29, "rt") as f:
        v29 = {(r["benchmark"], str(r["task_id"])): r for r in map(json.loads, f)}
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    with gzip.open(OUT / "paired_task_diagnostics.jsonl.gz", "wt", encoding="utf8") as out:
        for bm in BENCHMARKS:
            with (BASE / bm / "prediction.jsonl").open() as f:
                baseline = {str(r["task_id"]): r["pred"] for r in map(json.loads, f)}
            rows = [r for r in rerank if r["benchmark"] == bm]
            counts = collections.Counter()
            for row in rows:
                key = (bm, str(row["task_id"]))
                other = v29[key]
                if row["groundtruth"] != other["groundtruth"]:
                    raise ValueError(f"ground truth mismatch: {key}")
                row["baseline"] = baseline[key[1]]
                row["v29"] = other["cadre_v29_final"]
                metrics = {name: scorer.evaluate_records_official([row], name)[bm]
                           for name in ("baseline", "pred_cssi", "v29")}
                baseline_em = metrics["baseline"]["em"] == 100
                rerank_em = metrics["pred_cssi"]["em"] == 100
                counts[("correct" if baseline_em else "wrong") + "_to_" +
                       ("correct" if rerank_em else "wrong")] += 1
                out.write(json.dumps({"benchmark": bm, "task_id": row["task_id"],
                                      "groundtruth": row["groundtruth"],
                                      "baseline_prediction": row["baseline"],
                                      "rerank_prediction": row["pred_cssi"],
                                      "v29_prediction": row["v29"],
                                      "scores": metrics}, ensure_ascii=False) + "\n")
            summary[bm] = {"count": len(rows), "outcomes": dict(counts),
                           "metrics": {name: scorer.evaluate_records_official(rows, name)[bm]
                                       for name in ("baseline", "pred_cssi", "v29")}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT}; all arms rescored using {SCORER.name}")


if __name__ == "__main__":
    main()
