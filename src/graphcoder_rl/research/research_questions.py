from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Tuple

from graphcoder_rl.research.ablation_study import _aggregate_retrieval_meta, _cli_path, _python_bin, _read_json, _read_jsonl, run_ablation_study


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _dependency_score(meta: Dict[str, Any]) -> float:
    paths = meta.get("retrieval_path", []) or []
    avg_path_len = sum(len(p or []) for p in paths) / max(1, len(paths)) if paths else 0.0
    coarse_files = len(meta.get("coarse_candidate_files", []) or [])
    path_rel = float(meta.get("path_relevance", 0.0))
    return avg_path_len + 0.2 * coarse_files + path_rel


def _load_detailed_map(detail_path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in _read_jsonl(detail_path):
        tid = str(item.get("task_id", ""))
        if tid:
            out[tid] = item
    return out


def _run_infer_with_meta(
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
    cmd = [
        _python_bin(), _cli_path(), "infer",
        "--checkpoint", checkpoint,
        "--dataset", dataset,
        "--language", language,
        "--output-dir", output_dir,
        "--model-name", model_name,
        "--max-samples", str(max_samples),
        "--top-k", str(top_k),
        "--max-gen-length", str(max_gen_length),
        "--save-retrieval-meta",
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=_repo_root(), capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"infer failed: {proc.stderr[-2000:]}")
    res = _read_json(os.path.join(output_dir, "results.json"))
    res["duration_sec"] = round(dt, 2)
    return res


def rq1_component_contribution(
    checkpoint: str,
    dataset: str,
    language: str,
    output_dir: str,
    model_name: str,
    max_samples: int,
    top_k: int,
    max_gen_length: int,
) -> Dict[str, Any]:
    return run_ablation_study(
        checkpoint=checkpoint,
        dataset=dataset,
        language=language,
        output_dir=os.path.join(output_dir, "rq1_component_contribution"),
        model_name=model_name,
        max_samples=max_samples,
        top_k=top_k,
        max_gen_length=max_gen_length,
    )


def rq2_dependency_heavy_subset(
    checkpoint: str,
    dataset: str,
    language: str,
    output_dir: str,
    model_name: str,
    max_samples: int,
    top_k: int,
    max_gen_length: int,
) -> Dict[str, Any]:
    rq_dir = os.path.join(output_dir, "rq2_dependency_heavy")
    _run_infer_with_meta(
        checkpoint=checkpoint,
        dataset=dataset,
        language=language,
        output_dir=rq_dir,
        model_name=model_name,
        max_samples=max_samples,
        top_k=top_k,
        max_gen_length=max_gen_length,
    )

    meta_items = _read_jsonl(os.path.join(rq_dir, "retrieval_metadata.jsonl"))
    detailed = _load_detailed_map(os.path.join(rq_dir, "detailed_results.jsonl"))

    scored: List[Tuple[str, float]] = []
    for item in meta_items:
        tid = str(item.get("task_id", ""))
        score = _dependency_score(item.get("metadata", {}))
        if tid:
            scored.append((tid, score))
    scored.sort(key=lambda x: x[1], reverse=True)

    n = len(scored)
    heavy_n = max(1, int(0.25 * n)) if n > 0 else 0
    heavy_ids = {tid for tid, _ in scored[:heavy_n]}

    def _aggregate(task_ids: List[str]) -> Dict[str, float]:
        if not task_ids:
            return {"em": 0.0, "es": 0.0, "id_em": 0.0, "id_f1": 0.0, "total": 0}
        em = es = id_em = id_f1 = 0.0
        cnt = 0
        for tid in task_ids:
            row = detailed.get(tid)
            if not row:
                continue
            em += float(row.get("em", 0.0))
            es += float(row.get("es", 0.0))
            id_em += float(row.get("id_em", 0.0))
            id_f1 += float(row.get("id_f1", 0.0))
            cnt += 1
        if cnt == 0:
            return {"em": 0.0, "es": 0.0, "id_em": 0.0, "id_f1": 0.0, "total": 0}
        return {
            "em": round(100.0 * em / cnt, 2),
            "es": round(es / cnt, 2),
            "id_em": round(100.0 * id_em / cnt, 2),
            "id_f1": round(id_f1 / cnt, 2),
            "total": cnt,
        }

    heavy_metrics = _aggregate([tid for tid, _ in scored[:heavy_n]])
    non_heavy_metrics = _aggregate([tid for tid, _ in scored[heavy_n:]])

    out = {
        "subset_rule": "top_25_percent_by_dependency_score",
        "dependency_score": "avg_path_len + 0.2*coarse_candidates + path_relevance",
        "heavy": heavy_metrics,
        "non_heavy": non_heavy_metrics,
        "total_with_meta": n,
    }
    with open(os.path.join(rq_dir, "rq2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def rq3_efficiency_tradeoff(
    checkpoint: str,
    dataset: str,
    language: str,
    output_dir: str,
    model_name: str,
    max_samples: int,
    top_k: int,
    max_gen_length: int,
) -> Dict[str, Any]:
    rq_dir = os.path.join(output_dir, "rq3_efficiency_tradeoff")
    os.makedirs(rq_dir, exist_ok=True)

    settings = [
        ("full", []),
        ("no_multi_hop", ["--no-multi-hop"]),
        ("no_quant", ["--no-quantization"]),
    ]
    rows: List[Dict[str, Any]] = []
    for name, extra in settings:
        out_dir = os.path.join(rq_dir, name)
        cmd = [
            _python_bin(), _cli_path(), "infer",
            "--checkpoint", checkpoint,
            "--dataset", dataset,
            "--language", language,
            "--output-dir", out_dir,
            "--model-name", model_name,
            "--max-samples", str(max_samples),
            "--top-k", str(top_k),
            "--max-gen-length", str(max_gen_length),
            "--save-retrieval-meta",
        ] + extra

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=_repo_root(), capture_output=True, text=True)
        dt = time.time() - t0
        if proc.returncode != 0:
            rows.append({"setting": name, "ok": False, "error": proc.stderr[-1000:]})
            continue

        res = _read_json(os.path.join(out_dir, "results.json"))
        meta = _read_jsonl(os.path.join(out_dir, "retrieval_metadata.jsonl"))
        aggr = _aggregate_retrieval_meta(meta)
        rows.append({
            "setting": name,
            "ok": True,
            "duration_sec": round(dt, 2),
            "em": res.get("em", 0.0),
            "es": res.get("es", 0.0),
            **aggr,
        })

    out = {"rq3": rows}
    with open(os.path.join(rq_dir, "rq3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def run_research_questions(
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
    rq1 = rq1_component_contribution(checkpoint, dataset, language, output_dir, model_name, max_samples, top_k, max_gen_length)
    rq2 = rq2_dependency_heavy_subset(checkpoint, dataset, language, output_dir, model_name, max_samples, top_k, max_gen_length)
    rq3 = rq3_efficiency_tradeoff(checkpoint, dataset, language, output_dir, model_name, max_samples, top_k, max_gen_length)
    summary = {"rq1": rq1, "rq2": rq2, "rq3": rq3}
    with open(os.path.join(output_dir, "research_questions_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GraphCoderRL research questions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="repoeval")
    parser.add_argument("--language", default="python")
    parser.add_argument("--output-dir", default="results/research")
    parser.add_argument("--model-name", default="deepseek-ai/deepseek-coder-1.3b-base")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-gen-length", type=int, default=64)
    args = parser.parse_args(argv)

    summary = run_research_questions(
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
