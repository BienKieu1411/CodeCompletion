from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class AblationVariant:
    name: str
    infer_args: List[str]
    description: str


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _cli_path() -> str:
    return os.path.join(_repo_root(), "graphcoder_cli.py")


def _python_bin() -> str:
    return os.environ.get("PYTHON_BIN", "python3")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _aggregate_retrieval_meta(meta_items: List[Dict[str, Any]]) -> Dict[str, float]:
    if not meta_items:
        return {"avg_token_cost": 0.0, "avg_path_relevance": 0.0, "avg_path_len": 0.0}
    token_cost = 0.0
    path_rel = 0.0
    path_len = 0.0
    for x in meta_items:
        m = x.get("metadata", {})
        token_cost += float(m.get("token_cost", 0.0))
        path_rel += float(m.get("path_relevance", 0.0))
        paths = m.get("retrieval_path", []) or []
        mean_len = sum(len(p or []) for p in paths) / max(1, len(paths)) if paths else 0.0
        path_len += mean_len
    n = float(len(meta_items))
    return {
        "avg_token_cost": round(token_cost / n, 4),
        "avg_path_relevance": round(path_rel / n, 4),
        "avg_path_len": round(path_len / n, 4),
    }


def default_variants() -> List[AblationVariant]:
    return [
        AblationVariant("full", [], "Full GraphCoderRL"),
        AblationVariant("w_o_quant", ["--no-quantization"], "Disable semantic quantization"),
        AblationVariant("w_o_multi_hop", ["--no-multi-hop"], "Disable multi-hop traversal"),
        AblationVariant("w_o_left_anchor", ["--no-left-context-anchors"], "Disable left-context anchors"),
        AblationVariant("w_o_structural_edges", ["--no-structural-edges"], "Disable structural graph edges"),
        AblationVariant("w_o_control_dep", ["--no-control-dependency"], "Disable control dependency edges"),
        AblationVariant("w_o_overrides", ["--no-overrides"], "Disable overrides edges"),
    ]


def run_ablation_study(
    checkpoint: str,
    dataset: str,
    language: str,
    output_dir: str,
    model_name: str,
    max_samples: int,
    top_k: int,
    max_gen_length: int,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    variants = default_variants()

    results: List[Dict[str, Any]] = []
    for variant in variants:
        variant_dir = os.path.join(output_dir, variant.name)
        os.makedirs(variant_dir, exist_ok=True)

        cmd = [
            _python_bin(),
            _cli_path(),
            "infer",
            "--checkpoint", checkpoint,
            "--dataset", dataset,
            "--language", language,
            "--output-dir", variant_dir,
            "--model-name", model_name,
            "--max-samples", str(max_samples),
            "--top-k", str(top_k),
            "--max-gen-length", str(max_gen_length),
            "--save-retrieval-meta",
        ] + variant.infer_args

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=_repo_root(), capture_output=True, text=True)
        duration = time.time() - t0

        if proc.returncode != 0:
            results.append({
                "variant": variant.name,
                "description": variant.description,
                "ok": False,
                "returncode": proc.returncode,
                "stderr": proc.stderr[-2000:],
                "duration_sec": round(duration, 2),
            })
            continue

        result_json = _read_json(os.path.join(variant_dir, "results.json"))
        meta_items = _read_jsonl(os.path.join(variant_dir, "retrieval_metadata.jsonl"))
        meta_aggr = _aggregate_retrieval_meta(meta_items)

        row = {
            "variant": variant.name,
            "description": variant.description,
            "ok": True,
            "duration_sec": round(duration, 2),
            "em": result_json.get("em", 0.0),
            "es": result_json.get("es", 0.0),
            "id_em": result_json.get("id_em", 0.0),
            "id_f1": result_json.get("id_f1", 0.0),
            "total": result_json.get("total", 0),
            **meta_aggr,
        }
        results.append(row)

    summary = {
        "dataset": dataset,
        "language": language,
        "checkpoint": checkpoint,
        "variants": results,
    }

    json_path = os.path.join(output_dir, "ablation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, "ablation_summary.csv")
    fieldnames = [
        "variant", "description", "ok", "duration_sec", "em", "es", "id_em", "id_f1", "total",
        "avg_token_cost", "avg_path_relevance", "avg_path_len",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GraphCoderRL ablation study")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="repoeval")
    parser.add_argument("--language", default="python")
    parser.add_argument("--output-dir", default="results/research/ablation")
    parser.add_argument("--model-name", default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-gen-length", type=int, default=64)
    args = parser.parse_args(argv)

    summary = run_ablation_study(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        language=args.language,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_samples=args.max_samples,
        top_k=args.top_k,
        max_gen_length=args.max_gen_length,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
