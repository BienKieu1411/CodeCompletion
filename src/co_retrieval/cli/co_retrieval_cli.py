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
    parser.add_argument("--dataset-path", default="data/github_repos/python/train.parquet")
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
                "warmup_steps": args.warmup_steps,
                "num_rounds": args.num_rounds,
                "steps_per_round_prompt": args.steps_per_round_prompt,
                "steps_per_round_dpo": args.steps_per_round_dpo,
                "preference_margin": args.preference_margin,
                "utility_margin": args.utility_margin,
                "num_hard_negatives": args.num_hard_negatives,
                "preference_pool_top_k": args.preference_pool_top_k,
                "max_pairs_per_sample": args.max_pairs_per_sample,
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
        "--completion-level", choices=["line", "block", "mixed"], default="line"
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
    p_train.add_argument("--generator-name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
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
        ],
        default="intent_main",
    )
    p_train.add_argument("--intent-mode", choices=["static", "raw"], default="static")
    p_train.add_argument(
        "--gate-mode",
        choices=["learned", "always_retrieve", "always_skip", "rule"],
        default="learned",
    )
    p_train.add_argument(
        "--adapter-type", choices=["soft_prompt", "none"], default="soft_prompt"
    )
    p_train.add_argument("--disable-oracle-strategy", action="store_true", default=False)
    p_train.add_argument("--warmup-steps", type=int, default=200)
    p_train.add_argument("--num-rounds", type=int, default=2)
    p_train.add_argument("--steps-per-round-prompt", type=int, default=100)
    p_train.add_argument("--steps-per-round-dpo", type=int, default=100)
    p_train.add_argument("--preference-margin", type=float, default=0.1)
    p_train.add_argument("--utility-margin", type=float, default=0.05)
    p_train.add_argument("--num-hard-negatives", type=int, default=10)
    p_train.add_argument("--preference-pool-top-k", type=int, default=20)
    p_train.add_argument("--max-pairs-per-sample", type=int, default=4)
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
    p_train.add_argument("--device", default="cuda")

    p_train.set_defaults(func=_cmd_train)

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
