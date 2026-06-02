from __future__ import annotations

import argparse
import json
import logging
from types import SimpleNamespace

from graphcoder_rl.cli.config import GraphCoderCLIConfig

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("graphcoder_cli")


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-path", default="data/github_repos/python/train.parquet")
    parser.add_argument("--language", default="python")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--model-name", default="deepseek-ai/deepseek-coder-1.3b-base")


def _as_config(args: argparse.Namespace) -> GraphCoderCLIConfig:
    return GraphCoderCLIConfig(
        dataset_path=args.dataset_path,
        language=args.language,
        output_dir=args.output_dir,
        checkpoint=getattr(args, "checkpoint", None),
        pretrain_checkpoint=getattr(args, "pretrain_checkpoint", None),
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        top_k=getattr(args, "top_k", 3),
        num_epochs=getattr(args, "num_epochs", 1),
        batch_size=getattr(args, "batch_size", 2),
        completion_level=getattr(args, "completion_level", "line"),
        model_name=args.model_name,
    )


def _cmd_train(args: argparse.Namespace) -> dict:
    from graphcoder_rl.training.graphcoder_rl_train import train

    cfg = _as_config(args).to_dict()
    cfg.update(
        {
            "checkpoint_dir": args.checkpoint_dir,
            "log_dir": args.log_dir,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "ppo_update_steps": args.ppo_update_steps,
            "top_k": args.top_k,
            "validate_every": args.validate_every,
            "completion_level": args.completion_level,
            "fixed_train_size": args.fixed_train_size,
            "warm_start_steps": args.warm_start_steps,
            "coarse_scoring_mode": args.coarse_scoring_mode,
            "coarse_quantum_alpha": args.coarse_quantum_alpha,
        }
    )
    train(cfg)
    return {"status": "ok", "checkpoint_dir": args.checkpoint_dir}


def _cmd_infer(args: argparse.Namespace) -> dict:
    from graphcoder_rl.pipelines.graphcoder_rl_infer import infer

    ns = SimpleNamespace(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        language=args.language,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        top_k=args.top_k,
        max_gen_length=args.max_gen_length,
        save_retrieval_meta=args.save_retrieval_meta,
        coarse_scoring_mode=args.coarse_scoring_mode,
        coarse_quantum_alpha=args.coarse_quantum_alpha,
        enable_left_context_anchors=not args.no_left_context_anchors,
        enable_quantization=not args.no_quantization,
        enable_multi_hop=not args.no_multi_hop,
        enable_structural_edges=not args.no_structural_edges,
        enable_control_dependency=not args.no_control_dependency,
        enable_override_edges=not args.no_overrides,
        use_graph_cache=not args.no_graph_cache,
        use_ppl_entropy_cache=not args.no_ppl_cache,
    )
    return infer(ns)


def _cmd_eval(args: argparse.Namespace) -> dict:
    from graphcoder_rl.evaluation.graphcoder_rl_eval import evaluate_from_files

    return evaluate_from_files(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        language=args.language,
    )


def _cmd_pretrain_contrastive(args: argparse.Namespace) -> dict:
    from graphcoder_rl.training.contrastive_pretrain import run_contrastive_pretrain

    cfg = _as_config(args).to_dict()
    cfg.update(
        {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "completion_level": args.completion_level,
        }
    )
    return run_contrastive_pretrain(cfg)


def _cmd_build_graph_cache(args: argparse.Namespace) -> dict:
    from graphcoder_rl.pipelines.cache_builders import build_graph_cache

    graph_dir = f"{args.cache_dir}/graph"
    ppl_dir = f"{args.cache_dir}/ppl"
    return build_graph_cache(
        dataset_path=args.dataset_path,
        graph_cache_dir=graph_dir,
        ppl_cache_dir=ppl_dir,
        max_repos=args.max_repos,
    )


def _cmd_build_ppl_cache(args: argparse.Namespace) -> dict:
    from graphcoder_rl.pipelines.cache_builders import build_ppl_cache

    ppl_dir = f"{args.cache_dir}/ppl"
    return build_ppl_cache(
        dataset_path=args.dataset_path,
        ppl_cache_dir=ppl_dir,
        max_repos=args.max_repos,
    )


def _cmd_ablation_study(args: argparse.Namespace) -> dict:
    from graphcoder_rl.research.ablation_study import run_ablation_study

    return run_ablation_study(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        language=args.language,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_samples=args.max_samples,
        top_k=args.top_k,
        max_gen_length=args.max_gen_length,
    )


def _cmd_research_questions(args: argparse.Namespace) -> dict:
    from graphcoder_rl.research.research_questions import run_research_questions

    return run_research_questions(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        language=args.language,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_samples=args.max_samples,
        top_k=args.top_k,
        max_gen_length=args.max_gen_length,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphCoderRL unified CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train GraphCoderRL with PPO")
    _add_shared_args(p_train)
    p_train.add_argument("--checkpoint-dir", default="checkpoints/rl")
    p_train.add_argument("--log-dir", default="logs/rl")
    p_train.add_argument("--pretrain-checkpoint", default=None)
    p_train.add_argument("--num-epochs", type=int, default=1)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--ppo-update-steps", type=int, default=2)
    p_train.add_argument("--top-k", type=int, default=2)
    p_train.add_argument("--validate-every", type=int, default=1)
    p_train.add_argument("--completion-level", choices=["line", "block", "mixed"], default="line")
    p_train.add_argument("--fixed-train-size", type=int, default=2000)
    p_train.add_argument("--warm-start-steps", type=int, default=100)
    p_train.add_argument("--coarse-scoring-mode", choices=["dense", "quantum", "hybrid"], default="dense")
    p_train.add_argument("--coarse-quantum-alpha", type=float, default=0.5)
    p_train.set_defaults(func=_cmd_train)

    p_infer = sub.add_parser("infer", help="Run inference")
    _add_shared_args(p_infer)
    p_infer.add_argument("--checkpoint", default=None)
    p_infer.add_argument("--dataset", default="repoeval")
    p_infer.add_argument("--top-k", type=int, default=3)
    p_infer.add_argument("--max-gen-length", type=int, default=64)
    p_infer.add_argument("--save-retrieval-meta", action="store_true")
    p_infer.add_argument("--coarse-scoring-mode", choices=["dense", "quantum", "hybrid"], default="dense")
    p_infer.add_argument("--coarse-quantum-alpha", type=float, default=0.5)
    p_infer.add_argument("--no-left-context-anchors", action="store_true")
    p_infer.add_argument("--no-quantization", action="store_true")
    p_infer.add_argument("--no-multi-hop", action="store_true")
    p_infer.add_argument("--no-structural-edges", action="store_true")
    p_infer.add_argument("--no-control-dependency", action="store_true")
    p_infer.add_argument("--no-overrides", action="store_true")
    p_infer.add_argument("--no-graph-cache", action="store_true")
    p_infer.add_argument("--no-ppl-cache", action="store_true")
    p_infer.set_defaults(func=_cmd_infer)

    p_eval = sub.add_parser("eval", help="Evaluate predictions JSONL against ground truth JSONL")
    _add_shared_args(p_eval)
    p_eval.add_argument("--predictions", required=True)
    p_eval.add_argument("--ground-truth", required=True)
    p_eval.set_defaults(func=_cmd_eval)

    p_pre = sub.add_parser("pretrain-contrastive", help="Run contrastive pretrain for coarse retriever")
    _add_shared_args(p_pre)
    p_pre.add_argument("--num-epochs", type=int, default=1)
    p_pre.add_argument("--batch-size", type=int, default=8)
    p_pre.add_argument("--temperature", type=float, default=0.07)
    p_pre.add_argument("--learning-rate", type=float, default=2e-5)
    p_pre.add_argument("--completion-level", choices=["line", "block", "mixed"], default="line")
    p_pre.set_defaults(func=_cmd_pretrain_contrastive)

    p_gc = sub.add_parser("build-graph-cache", help="Build repository graph cache (includes incremental metadata)")
    _add_shared_args(p_gc)
    p_gc.add_argument("--max-repos", type=int, default=0)
    p_gc.set_defaults(func=_cmd_build_graph_cache)

    p_pc = sub.add_parser("build-ppl-cache", help="Build offline ppl/entropy cache for long entities")
    _add_shared_args(p_pc)
    p_pc.add_argument("--max-repos", type=int, default=0)
    p_pc.set_defaults(func=_cmd_build_ppl_cache)

    p_ab = sub.add_parser("ablation-study", help="Run ablation variants and collect summary metrics")
    _add_shared_args(p_ab)
    p_ab.add_argument("--checkpoint", required=True)
    p_ab.add_argument("--dataset", default="repoeval")
    p_ab.add_argument("--top-k", type=int, default=3)
    p_ab.add_argument("--max-gen-length", type=int, default=64)
    p_ab.set_defaults(func=_cmd_ablation_study)

    p_rq = sub.add_parser("run-rq", help="Run research questions RQ1/RQ2/RQ3")
    _add_shared_args(p_rq)
    p_rq.add_argument("--checkpoint", required=True)
    p_rq.add_argument("--dataset", default="repoeval")
    p_rq.add_argument("--top-k", type=int, default=3)
    p_rq.add_argument("--max-gen-length", type=int, default=64)
    p_rq.set_defaults(func=_cmd_research_questions)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
