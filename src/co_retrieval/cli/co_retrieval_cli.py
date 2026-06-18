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


def _as_config(args: argparse.Namespace) -> CoRetrievalCLIConfig:
    return CoRetrievalCLIConfig(
        dataset_path=args.dataset_path,
        language=args.language,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        top_k=args.top_k,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        completion_level=args.completion_level,
    )


def _cmd_train(args: argparse.Namespace) -> dict:
    from co_retrieval.runner import train

    cfg = _as_config(args).to_dict()
    cfg.update(
        {
            "checkpoint_dir": args.checkpoint_dir,
            "log_dir": args.log_dir,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "top_k": args.top_k,
            "sampled_contexts": args.sampled_contexts,
            "dpo_beta": args.dpo_beta,
            "retriever_lr": args.retriever_lr,
            "gate_lr": args.gate_lr,
            "soft_prompt_lr": args.soft_prompt_lr,
            "gate_threshold": args.gate_threshold,
            "completion_level": args.completion_level,
            "fixed_train_size": args.fixed_train_size,
            "max_train_samples": args.max_train_samples,
            "min_file_lines": args.min_file_lines,
            "min_file_chars": args.min_file_chars,
            "min_left_context_lines": args.min_left_context_lines,
            "max_chunk_lines": args.max_chunk_lines,
            "fallback_lines": args.fallback_lines,
            "random_seed": args.random_seed,
        }
    )
    return train(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Co-Retrieval code-completion CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train Co-Retrieval with DPO, gate, and soft prompt")
    _add_shared_args(p_train)
    p_train.add_argument("--checkpoint-dir", default="checkpoints/co_retrieval")
    p_train.add_argument("--log-dir", default="logs/co_retrieval")
    p_train.add_argument("--num-epochs", type=int, default=1)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--top-k", type=int, default=3)
    p_train.add_argument("--sampled-contexts", type=int, default=3)
    p_train.add_argument("--dpo-beta", type=float, default=0.1)
    p_train.add_argument("--retriever-lr", type=float, default=0.2)
    p_train.add_argument("--gate-lr", type=float, default=0.1)
    p_train.add_argument("--soft-prompt-lr", type=float, default=0.05)
    p_train.add_argument("--gate-threshold", type=float, default=0.5)
    p_train.add_argument("--completion-level", choices=["line", "block", "mixed"], default="line")
    p_train.add_argument("--fixed-train-size", type=int, default=2000)
    p_train.add_argument("--max-train-samples", type=int, default=2000)
    p_train.add_argument("--min-file-lines", type=int, default=200)
    p_train.add_argument("--min-file-chars", type=int, default=2000)
    p_train.add_argument("--min-left-context-lines", type=int, default=30)
    p_train.add_argument("--max-chunk-lines", type=int, default=120)
    p_train.add_argument("--fallback-lines", type=int, default=40)
    p_train.add_argument("--random-seed", type=int, default=13)
    p_train.set_defaults(func=_cmd_train)

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
