from __future__ import annotations

import argparse
import json
import logging

from co_retrieval.cli.config import CoRetrievalCLIConfig


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("co_retrieval_cli")


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-path",
        default="data/github_repos/python/train.parquet",
        help=(
            "Dataset parquet path. Train mode also accepts comma-separated "
            "paths, e.g. python/train.parquet,java/train.parquet."
        ),
    )
    parser.add_argument("--language", default="python")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-samples", type=int, default=50)


def _cmd_train(args: argparse.Namespace) -> dict:
    from co_retrieval.runner import train

    retriever_lr = args.retriever_lr
    gate_lr = args.gate_lr
    soft_prompt_lr = args.soft_prompt_lr
    if retriever_lr is None:
        retriever_lr = 2e-5 if args.use_neural else 0.2
    if gate_lr is None:
        gate_lr = 1e-4 if args.use_neural else 0.1
    if soft_prompt_lr is None:
        soft_prompt_lr = 5e-3 if args.use_neural else 0.05
    train_epochs = args.train_epochs if args.train_epochs is not None else args.num_epochs

    fixed_train_size = (
        args.fixed_train_size if args.fixed_train_size is not None else args.max_samples
    )
    max_train_samples = (
        args.max_train_samples if args.max_train_samples is not None else args.max_samples
    )

    cfg = CoRetrievalCLIConfig(
        dataset_path=args.dataset_path,
        language=args.language,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        top_k=args.top_k,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        completion_level=args.completion_level,
        use_neural=args.use_neural,
    ).to_dict()

    cfg.update(
        {
            "checkpoint_dir": args.checkpoint_dir,
            "log_dir": args.log_dir,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "top_k": args.top_k,
            "dpo_beta": args.dpo_beta,
            "retriever_lr": retriever_lr,
            "gate_lr": gate_lr,
            "soft_prompt_lr": soft_prompt_lr,
            "completion_level": args.completion_level,
            "fixed_train_size": fixed_train_size,
            "max_train_samples": max_train_samples,
            "min_file_lines": args.min_file_lines,
            "min_file_chars": args.min_file_chars,
            "min_left_context_lines": args.min_left_context_lines,
            "max_chunk_lines": args.max_chunk_lines,
            "fallback_lines": args.fallback_lines,
            "random_seed": args.random_seed,
        }
    )

    # Neural-specific overrides
    if args.use_neural:
        cfg.update(
            {
                "encoder_name": args.encoder_name,
                "generator_name": args.generator_name,
                "experiment_mode": args.experiment_mode,
                "intent_mode": args.intent_mode,
                "gate_mode": args.gate_mode,
                "adapter_type": args.adapter_type,
                "include_oracle_strategy": not args.disable_oracle_strategy,
                "build_train_index": args.build_train_index,
                "refresh_train_index": args.refresh_train_index,
                "skip_train_eval": args.skip_train_eval,
                "warmup_steps": args.warmup_steps,
                "train_epochs": train_epochs,
                "epoch_budget_mode": args.epoch_budget_mode,
                "num_rounds": args.num_rounds,
                "steps_per_round_prompt": args.steps_per_round_prompt,
                "steps_per_round_retriever": args.steps_per_round_retriever,
                "steps_per_round_dpo": args.steps_per_round_dpo,
                "retriever_loss": args.retriever_loss,
                "lipo_tau": args.lipo_tau,
                "preference_margin": args.preference_margin,
                "utility_margin": args.utility_margin,
                "num_hard_negatives": args.num_hard_negatives,
                "preference_pool_top_k": args.preference_pool_top_k,
                "max_pairs_per_sample": args.max_pairs_per_sample,
                "leave_one_out_analysis_samples": args.leave_one_out_analysis_samples,
                "gate_quality_tolerance": args.gate_quality_tolerance,
                "gate_retrieval_reduction_target": args.gate_retrieval_reduction_target,
                "gate_use_retrieval_features": not args.disable_gate_retrieval_features,
                "gate_context_cost_weight": args.gate_context_cost_weight,
                "gate_context_cost_token_unit": args.gate_context_cost_token_unit,
                "gate_decision_threshold": args.gate_decision_threshold,
                "gate_calibrate_threshold": not args.disable_gate_calibration,
                "gate_calibration_samples": args.gate_calibration_samples,
                "gate_calibration_retrieval_penalty": (
                    args.gate_calibration_retrieval_penalty
                ),
                "num_prompt_tokens": args.num_prompt_tokens,
                "max_context_tokens": args.max_context_tokens,
                "encoder_max_length": args.encoder_max_length,
                "gate_hidden_dim": args.gate_hidden_dim,
                "gate_entropy_weight": args.gate_entropy_weight,
                "grad_clip_norm": args.grad_clip_norm,
                "generator_dtype": args.generator_dtype,
                "batch_encode_size": args.batch_encode_size,
                "max_new_tokens": args.max_new_tokens,
                "eval_ratio": args.eval_ratio,
                "max_eval_samples": args.max_eval_samples,
                "device": args.device,
                "query_entropy_threshold": args.query_entropy_threshold,
                "query_num_drafts": args.query_num_drafts,
                "query_draft_max_tokens": args.query_draft_max_tokens,
                "query_draft_temperature": args.query_draft_temperature,
                "query_draft_top_p": args.query_draft_top_p,
            }
        )
    else:
        cfg.update(
            {
                "sampled_contexts": args.sampled_contexts,
                "gate_threshold": args.gate_threshold,
            }
        )

    return train(cfg)


def _cmd_evaluate(args: argparse.Namespace) -> dict:
    from co_retrieval.runner import evaluate

    cfg = CoRetrievalCLIConfig(
        dataset_path=args.dataset_path,
        language=args.language,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        top_k=args.top_k,
        batch_size=args.batch_size,
        completion_level=args.completion_level,
        use_neural=True,
    ).to_dict()
    cfg.update(
        {
            "checkpoint_dir": args.checkpoint_dir,
            "log_dir": args.log_dir,
            "fixed_train_size": args.max_samples,
            "max_train_samples": args.max_samples,
            "max_eval_samples": args.max_samples,
            "min_file_lines": args.min_file_lines,
            "min_file_chars": args.min_file_chars,
            "min_left_context_lines": args.min_left_context_lines,
            "max_chunk_lines": args.max_chunk_lines,
            "fallback_lines": args.fallback_lines,
            "random_seed": args.random_seed,
            "top_k": args.top_k,
            "gate_mode": args.gate_mode,
            "gate_decision_threshold": args.gate_decision_threshold,
            "batch_encode_size": args.batch_encode_size,
            "max_new_tokens": args.max_new_tokens,
            "leave_one_out_analysis_samples": args.leave_one_out_analysis_samples,
            "include_analysis": not args.no_analysis,
            "include_policy_variants": args.include_policy_variants,
            "device": args.device,
            "generator_dtype": args.generator_dtype,
        }
    )
    return evaluate(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Co-Retrieval code-completion CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train Co-Retrieval pipeline")

    # Shared args
    _add_shared_args(p_train)
    p_train.add_argument("--checkpoint-dir", default="checkpoints/co_retrieval")
    p_train.add_argument("--log-dir", default="logs/co_retrieval")
    p_train.add_argument("--num-epochs", type=int, default=1)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--top-k", type=int, default=3)
    p_train.add_argument("--dpo-beta", type=float, default=0.1)
    p_train.add_argument("--retriever-lr", type=float, default=None)
    p_train.add_argument("--gate-lr", type=float, default=None)
    p_train.add_argument("--soft-prompt-lr", type=float, default=None)
    p_train.add_argument(
        "--completion-level", choices=["line", "block", "mixed"], default="mixed"
    )
    p_train.add_argument("--fixed-train-size", type=int, default=None)
    p_train.add_argument("--max-train-samples", type=int, default=None)
    p_train.add_argument("--min-file-lines", type=int, default=200)
    p_train.add_argument("--min-file-chars", type=int, default=2000)
    p_train.add_argument("--min-left-context-lines", type=int, default=30)
    p_train.add_argument("--max-chunk-lines", type=int, default=120)
    p_train.add_argument("--fallback-lines", type=int, default=40)
    p_train.add_argument("--random-seed", type=int, default=13)

    # Mode selection
    p_train.add_argument(
        "--use-neural", action="store_true", default=False,
        help="Use neural pipeline (7-phase, requires GPU) instead of proxy mode",
    )

    # Proxy-only args
    p_train.add_argument("--sampled-contexts", type=int, default=3)
    p_train.add_argument("--gate-threshold", type=float, default=0.5)

    # Neural-only args
    p_train.add_argument("--encoder-name", default="jinaai/jina-code-embeddings-1.5b")
    p_train.add_argument(
        "--generator-name", default="deepseek-ai/deepseek-coder-6.7b-base"
    )
    p_train.add_argument(
        "--experiment-mode",
        choices=[
            "intent_main",
            "raw_query_main",
            "retriever_only",
            "always_retrieve",
            "always_skip",
            "bm25",
            "dense_frozen",
            "sequential_adapter_first",
            "sequential_retriever_first",
        ],
        default="intent_main",
    )
    p_train.add_argument(
        "--intent-mode", choices=["static", "raw", "cost_aware"], default="static"
    )
    p_train.add_argument(
        "--gate-mode",
        choices=["learned", "always_retrieve", "always_skip", "rule"],
        default="learned",
    )
    p_train.add_argument(
        "--adapter-type", choices=["soft_prompt", "none"], default="soft_prompt"
    )
    p_train.add_argument("--disable-oracle-strategy", action="store_true", default=False)
    p_train.add_argument(
        "--build-train-index",
        action="store_true",
        default=False,
        help=(
            "Pre-encode all training chunks into a global dense index. "
            "Disabled by default because learned retriever weights make the "
            "cache stale and full-data indexing is expensive."
        ),
    )
    p_train.add_argument(
        "--refresh-train-index",
        action="store_true",
        default=False,
        help=(
            "Re-encode all training chunks after each co-training epoch. "
            "Very expensive on full data; keep disabled for main LiPO runs."
        ),
    )
    p_train.add_argument("--skip-train-eval", action="store_true", default=False)
    p_train.add_argument("--warmup-steps", type=int, default=200)
    p_train.add_argument(
        "--train-epochs",
        type=int,
        default=None,
        help=(
            "Neural full-pass training epochs. Defaults to --num-epochs; "
            "use with --epoch-budget-mode to make each co-training epoch "
            "consume one full sample/preference pass."
        ),
    )
    p_train.add_argument(
        "--epoch-budget-mode",
        action="store_true",
        default=False,
        help=(
            "For neural training, interpret epochs as full passes: prompt "
            "steps=len(train samples), retriever steps=len(preference groups), "
            "and gate steps=len(gate labels) per epoch."
        ),
    )
    p_train.add_argument("--num-rounds", type=int, default=2)
    p_train.add_argument("--steps-per-round-prompt", type=int, default=100)
    p_train.add_argument("--steps-per-round-retriever", type=int, default=None)
    p_train.add_argument("--steps-per-round-dpo", type=int, default=100)
    p_train.add_argument(
        "--retriever-loss", choices=["lipo", "dpo"], default="lipo"
    )
    p_train.add_argument("--lipo-tau", type=float, default=1.0)
    p_train.add_argument("--preference-margin", type=float, default=0.1)
    p_train.add_argument("--utility-margin", type=float, default=0.05)
    p_train.add_argument("--num-hard-negatives", type=int, default=10)
    p_train.add_argument("--preference-pool-top-k", type=int, default=20)
    p_train.add_argument("--max-pairs-per-sample", type=int, default=4)
    p_train.add_argument("--leave-one-out-analysis-samples", type=int, default=25)
    p_train.add_argument("--gate-quality-tolerance", type=float, default=0.01)
    p_train.add_argument("--gate-retrieval-reduction-target", type=float, default=0.20)
    p_train.add_argument(
        "--disable-gate-retrieval-features", action="store_true", default=False
    )
    p_train.add_argument("--gate-context-cost-weight", type=float, default=0.01)
    p_train.add_argument("--gate-context-cost-token-unit", type=int, default=512)
    p_train.add_argument("--gate-decision-threshold", type=float, default=0.5)
    p_train.add_argument(
        "--disable-gate-calibration", action="store_true", default=False
    )
    p_train.add_argument("--gate-calibration-samples", type=int, default=128)
    p_train.add_argument(
        "--gate-calibration-retrieval-penalty", type=float, default=0.05
    )
    p_train.add_argument("--num-prompt-tokens", type=int, default=50)
    p_train.add_argument("--max-context-tokens", type=int, default=4096)
    p_train.add_argument("--encoder-max-length", type=int, default=512)
    p_train.add_argument("--gate-hidden-dim", type=int, default=256)
    p_train.add_argument("--gate-entropy-weight", type=float, default=0.01)
    p_train.add_argument("--grad-clip-norm", type=float, default=1.0)
    p_train.add_argument(
        "--generator-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    p_train.add_argument("--batch-encode-size", type=int, default=32)
    p_train.add_argument("--max-new-tokens", type=int, default=128)
    p_train.add_argument("--eval-ratio", type=float, default=0.1)
    p_train.add_argument("--max-eval-samples", type=int, default=100)
    p_train.add_argument("--query-entropy-threshold", type=float, default=0.8)
    p_train.add_argument("--query-num-drafts", type=int, default=2)
    p_train.add_argument("--query-draft-max-tokens", type=int, default=32)
    p_train.add_argument("--query-draft-temperature", type=float, default=0.8)
    p_train.add_argument("--query-draft-top-p", type=float, default=0.95)
    p_train.add_argument("--device", default="cuda")

    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate a neural checkpoint and write AlignCoder-style predictions",
    )
    _add_shared_args(p_eval)
    p_eval.add_argument("--checkpoint-dir", required=True)
    p_eval.add_argument("--log-dir", default="logs/co_retrieval_eval")
    p_eval.add_argument("--batch-size", type=int, default=2)
    p_eval.add_argument("--top-k", type=int, default=3)
    p_eval.add_argument(
        "--completion-level", choices=["line", "block", "mixed"], default="mixed"
    )
    p_eval.add_argument("--min-file-lines", type=int, default=200)
    p_eval.add_argument("--min-file-chars", type=int, default=2000)
    p_eval.add_argument("--min-left-context-lines", type=int, default=30)
    p_eval.add_argument("--max-chunk-lines", type=int, default=120)
    p_eval.add_argument("--fallback-lines", type=int, default=40)
    p_eval.add_argument("--random-seed", type=int, default=13)
    p_eval.add_argument(
        "--gate-mode",
        choices=["learned", "always_retrieve", "always_skip", "rule"],
        default=None,
    )
    p_eval.add_argument("--gate-decision-threshold", type=float, default=None)
    p_eval.add_argument("--batch-encode-size", type=int, default=32)
    p_eval.add_argument("--max-new-tokens", type=int, default=128)
    p_eval.add_argument("--leave-one-out-analysis-samples", type=int, default=25)
    p_eval.add_argument("--no-analysis", action="store_true", default=False)
    p_eval.add_argument("--include-policy-variants", action="store_true", default=False)
    p_eval.add_argument(
        "--generator-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    p_eval.add_argument("--device", default="cuda")
    p_eval.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
