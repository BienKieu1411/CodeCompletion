"""Co-Retrieval training entrypoint.

This keeps the existing DatasetLoader sample construction intact and swaps the
old PPO training loop for the novelty-aligned DPO/gate/soft-prompt loop.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

from co_retrieval.chunking import CodeChunk, RepositoryChunker
from co_retrieval.training import CoTrainingConfig, CoTrainingTrainer, TrainingSample
from co_retrieval.data.repository_dataset_loader import DatasetLoader


logger = logging.getLogger(__name__)


def _split_dataset_paths(dataset_path: Any) -> List[str]:
    if isinstance(dataset_path, (list, tuple)):
        return [str(path).strip() for path in dataset_path if str(path).strip()]
    text = str(dataset_path or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,\n]+", text) if part.strip()]


def _dataset_repo_prefix(dataset_path: str) -> str:
    path = os.path.normpath(dataset_path)
    parent = os.path.basename(os.path.dirname(path))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(path)))
    parts = [part for part in (grandparent, parent) if part]
    return "_".join(parts) or os.path.basename(path)


def _text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _sample_to_training_sample(
    sample: Dict[str, Any], chunker: RepositoryChunker
) -> TrainingSample:
    chunks: List[CodeChunk] = []
    repo_id = _text_or_empty(sample.get("repo_id", ""))
    for file_path, content in sorted(
        (sample.get("crossfile_context") or {}).items()
    ):
        clean_path = _text_or_empty(file_path)
        chunk_path = f"{repo_id}/{clean_path}" if repo_id else clean_path
        chunks.extend(
            chunker.chunk_source(
                chunk_path,
                _text_or_empty(content),
            )
        )

    return TrainingSample(
        left_context=_text_or_empty(sample.get("left_context", "")),
        target=_text_or_empty(sample.get("ground_truth", "")),
        file_path=_text_or_empty(sample.get("id", "current_file.py")),
        candidate_chunks=chunks,
        repo_id=repo_id,
        task_id=_text_or_empty(sample.get("task_id", "")),
    )


def _collect_training_samples(
    cfg: Dict[str, Any], chunker: RepositoryChunker
) -> List[TrainingSample]:
    random.seed(int(cfg.get("random_seed", 42)))
    requested_fixed_size = int(
        cfg.get("fixed_train_size", cfg.get("max_samples", 2000))
    )
    max_samples = int(cfg.get("max_train_samples", requested_fixed_size))
    fixed_train = bool(cfg.get("fixed_train", True))
    if requested_fixed_size <= 0 or max_samples <= 0:
        fixed_train = False

    dataset_paths = _split_dataset_paths(
        cfg.get("dataset_path", "data/github_repos/python/train.parquet")
    )
    if not dataset_paths:
        raise RuntimeError("No training dataset path configured.")

    out: List[TrainingSample] = []
    for path_index, dataset_path in enumerate(dataset_paths):
        if 0 < max_samples <= len(out):
            break

        remaining_paths = len(dataset_paths) - path_index
        if fixed_train and max_samples > 0:
            remaining_samples = max_samples - len(out)
            fixed_size_for_path = max(
                1, (remaining_samples + remaining_paths - 1) // remaining_paths
            )
        else:
            fixed_size_for_path = max(1, requested_fixed_size)

        loader = DatasetLoader(
            dataset_path=dataset_path,
            use_fim=bool(cfg.get("use_fim", False)),
            completion_level=cfg.get("completion_level", "mixed"),
            fixed_train=fixed_train,
            fixed_train_size=fixed_size_for_path,
            fixed_train_max_attempts=int(cfg.get("fixed_train_max_attempts", 20000)),
            min_file_lines=int(cfg.get("min_file_lines", 200)),
            min_file_chars=int(cfg.get("min_file_chars", 2000)),
            min_left_context_lines=int(cfg.get("min_left_context_lines", 30)),
        )
        loader.prepare_dataset()
        repo_prefix = _dataset_repo_prefix(dataset_path)

        for batch in loader.get_epoch_batches(batch_size=int(cfg.get("batch_size", 4))):
            for sample in batch:
                if len(dataset_paths) > 1 and sample.get("repo_id"):
                    sample = dict(sample)
                    sample["repo_id"] = f"{repo_prefix}:{sample['repo_id']}"
                item = _sample_to_training_sample(sample, chunker)
                if item.target.strip():
                    out.append(item)
                if 0 < max_samples <= len(out):
                    return out
    return out


def _collect_global_chunks(samples: Sequence[TrainingSample]) -> List[CodeChunk]:
    global_chunks: List[CodeChunk] = []
    seen: set[str] = set()
    for sample in samples:
        for chunk in sample.candidate_chunks or []:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                global_chunks.append(chunk)
    return global_chunks


def _collect_evaluation_samples(
    cfg: Dict[str, Any],
    chunker: RepositoryChunker,
) -> List[TrainingSample]:
    """Load AlignCoder-style eval parquet when present, else use train sampler."""
    random.seed(int(cfg.get("random_seed", 42)))
    dataset_path = cfg.get("dataset_path", "data/github_repos/python/train.parquet")
    dataset_paths = _split_dataset_paths(dataset_path)
    if len(dataset_paths) > 1:
        max_samples = int(
            cfg.get("max_eval_samples", cfg.get("max_train_samples", 100))
        )
        samples: List[TrainingSample] = []
        for path in dataset_paths:
            if 0 < max_samples <= len(samples):
                break
            sub_cfg = dict(cfg)
            sub_cfg["dataset_path"] = path
            if max_samples > 0:
                sub_cfg["max_eval_samples"] = max_samples - len(samples)
            samples.extend(_collect_evaluation_samples(sub_cfg, chunker))
        return samples

    if os.path.exists(dataset_path) and str(dataset_path).endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(dataset_path)
        columns = set(df.columns)
        target_column = (
            "groundtruth"
            if "groundtruth" in columns
            else "ground_truth"
            if "ground_truth" in columns
            else "target_code"
            if "target_code" in columns
            else None
        )
        if {"left_context", "crossfile_context"}.issubset(columns) and target_column:
            max_samples = int(
                cfg.get("max_eval_samples", cfg.get("max_train_samples", 100))
            )
            rows = df if max_samples <= 0 else df.head(max_samples)
            samples: List[TrainingSample] = []
            for idx, row in rows.iterrows():
                task_id = row.get("task_id", row.get("id", f"task_{idx}"))
                sample = {
                    "id": row.get("path", task_id),
                    "task_id": task_id,
                    "repo_id": row.get("repo_id", str(task_id)),
                    "left_context": row.get("left_context", ""),
                    "right_context": row.get("right_context", ""),
                    "ground_truth": row.get(target_column, ""),
                    "crossfile_context": DatasetLoader._parse_crossfile(
                        row.get("crossfile_context", {})
                    ),
                }
                item = _sample_to_training_sample(sample, chunker)
                if item.target.strip():
                    samples.append(item)
            return samples
    return _collect_training_samples(cfg, chunker)


def _split_train_eval(
    samples: Sequence[TrainingSample],
    eval_ratio: float,
    max_eval_samples: int,
    random_seed: int,
) -> Tuple[List[TrainingSample], List[TrainingSample]]:
    """Legacy random split — kept for proxy mode only."""
    sample_list = list(samples)
    if len(sample_list) < 2 or eval_ratio <= 0 or max_eval_samples <= 0:
        return sample_list, []

    rng = random.Random(random_seed)
    rng.shuffle(sample_list)
    eval_size = min(max_eval_samples, max(1, int(len(sample_list) * eval_ratio)))
    eval_size = min(eval_size, len(sample_list) - 1)
    return sample_list[eval_size:], sample_list[:eval_size]


def _split_by_repository(
    samples: Sequence[TrainingSample],
    eval_ratio: float,
    max_eval_samples: int,
    random_seed: int,
) -> Tuple[List[TrainingSample], List[TrainingSample]]:
    """Repository-disjoint split to prevent data leakage.

    Samples from the same repository will never appear in both train and eval.
    """
    sample_list = list(samples)
    if len(sample_list) < 2 or eval_ratio <= 0 or max_eval_samples <= 0:
        return sample_list, []

    # Group only by the repository identity preserved by DatasetLoader.  File
    # paths are repository-relative and cannot safely identify a repository.
    repo_to_samples: Dict[str, List[TrainingSample]] = {}
    for sample in sample_list:
        if not sample.repo_id:
            logger.warning(
                "Repository-disjoint split unavailable because sample %r has "
                "no repo_id; using all samples for training.",
                sample.file_path,
            )
            return sample_list, []
        repo_to_samples.setdefault(sample.repo_id, []).append(sample)

    if len(repo_to_samples) < 2:
        logger.warning(
            "Repository-disjoint evaluation requires at least two repositories; "
            "using all %d samples for training.",
            len(sample_list),
        )
        return sample_list, []

    repos = sorted(repo_to_samples.keys())  # deterministic ordering
    rng = random.Random(random_seed)
    rng.shuffle(repos)

    # Split repos
    eval_repo_count = max(1, int(len(repos) * eval_ratio))
    eval_repo_count = min(eval_repo_count, len(repos) - 1)  # keep ≥1 train repo
    eval_repos = set(repos[:eval_repo_count])

    train = [
        s
        for repo_id, ss in repo_to_samples.items()
        if repo_id not in eval_repos
        for s in ss
    ]
    eval_ = [
        s
        for repo_id, ss in repo_to_samples.items()
        if repo_id in eval_repos
        for s in ss
    ]

    # Cap eval size
    if len(eval_) > max_eval_samples:
        rng.shuffle(eval_)
        eval_ = eval_[:max_eval_samples]

    logger.info(
        "Repository-disjoint split: %d repos (%d train, %d eval), "
        "%d train samples, %d eval samples",
        len(repos),
        len(repos) - len(eval_repos),
        len(eval_repos),
        len(train),
        len(eval_),
    )
    return train, eval_


def _train_proxy(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Proxy mode training (original — no GPU needed)."""
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints/co_retrieval")
    log_dir = cfg.get("log_dir", "logs/co_retrieval")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    chunker = RepositoryChunker(
        max_chunk_lines=int(cfg.get("max_chunk_lines", 120)),
        fallback_lines=int(cfg.get("fallback_lines", 40)),
    )
    samples = _collect_training_samples(cfg, chunker)
    if not samples:
        raise RuntimeError(
            "No training samples were constructed. Check dataset path and strict filters."
        )

    global_chunks: List[CodeChunk] = []
    seen: set[str] = set()
    for sample in samples:
        for chunk in sample.candidate_chunks or []:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                global_chunks.append(chunk)

    training_cfg = CoTrainingConfig(
        epochs=int(cfg.get("num_epochs", cfg.get("epochs", 1))),
        top_k=int(cfg.get("top_k", 3)),
        sampled_contexts=int(cfg.get("sampled_contexts", 3)),
        dpo_beta=float(cfg.get("dpo_beta", 0.1)),
        retriever_lr=float(cfg.get("retriever_lr", 0.2)),
        gate_lr=float(cfg.get("gate_lr", 0.1)),
        soft_prompt_lr=float(cfg.get("soft_prompt_lr", 0.05)),
        gate_threshold=float(cfg.get("gate_threshold", 0.5)),
        random_seed=int(cfg.get("random_seed", 13)),
    )
    trainer = CoTrainingTrainer(global_chunks, training_cfg)
    history = trainer.train(samples)

    ckpt = {
        "framework": "Co-Retrieval",
        "created_at": datetime.now().isoformat(),
        "config": asdict(training_cfg),
        "num_samples": len(samples),
        "num_chunks": len(global_chunks),
        "retriever_weights": trainer.retriever.weights,
        "gate": {"bias": trainer.gate.bias, "weights": trainer.gate.weights},
        "soft_prompt": {
            "update_count": trainer.model.soft_prompt.update_count,
            "symbol_affinities": trainer.model.soft_prompt.symbol_affinities,
        },
        "history": [asdict(item) for item in history],
    }
    checkpoint_path = os.path.join(checkpoint_dir, "co_retrieval_checkpoint.json")
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2, ensure_ascii=False)

    log_path = os.path.join(log_dir, "training_history.jsonl")
    with open(log_path, "w", encoding="utf-8") as f:
        for row in ckpt["history"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "status": "ok",
        "framework": "Co-Retrieval (proxy)",
        "checkpoint": checkpoint_path,
        "log_path": log_path,
        "num_samples": len(samples),
        "num_chunks": len(global_chunks),
        "final": ckpt["history"][-1] if ckpt["history"] else {},
    }
    logger.info("Co-Retrieval proxy training complete: %s", summary)
    return summary


def _train_neural(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Neural mode training (7-phase pipeline, requires GPU)."""
    from co_retrieval.neural_training import NeuralCoTrainer, NeuralCoTrainingConfig

    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints/co_retrieval_neural")
    log_dir = cfg.get("log_dir", "logs/co_retrieval_neural")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    chunker = RepositoryChunker(
        max_chunk_lines=int(cfg.get("max_chunk_lines", 120)),
        fallback_lines=int(cfg.get("fallback_lines", 40)),
    )
    samples = _collect_training_samples(cfg, chunker)
    if not samples:
        raise RuntimeError("No training samples found.")

    eval_ratio = float(cfg.get("eval_ratio", 0.1))
    max_eval_samples = int(cfg.get("max_eval_samples", 100))
    if bool(cfg.get("skip_train_eval", False)):
        train_samples, eval_samples = samples, []
        logger.info("Skipping train-time eval; using all %d samples for training.", len(samples))
    else:
        train_samples, eval_samples = _split_by_repository(
            samples,
            eval_ratio=eval_ratio,
            max_eval_samples=max_eval_samples,
            random_seed=int(cfg.get("random_seed", 42)),
        )

    neural_cfg = NeuralCoTrainingConfig(
        encoder_name=cfg.get("encoder_name", "microsoft/unixcoder-base"),
        generator_name=cfg.get(
            "generator_name", "deepseek-ai/deepseek-coder-6.7b-base"
        ),
        encoder_max_length=int(cfg.get("encoder_max_length", 512)),
        num_prompt_tokens=int(cfg.get("num_prompt_tokens", 50)),
        max_context_tokens=int(cfg.get("max_context_tokens", 4096)),
        gate_hidden_dim=int(cfg.get("gate_hidden_dim", 256)),
        gate_entropy_weight=float(cfg.get("gate_entropy_weight", 0.01)),
        gate_use_retrieval_features=bool(
            cfg.get("gate_use_retrieval_features", True)
        ),
        gate_context_cost_weight=float(cfg.get("gate_context_cost_weight", 0.01)),
        gate_context_cost_token_unit=int(
            cfg.get("gate_context_cost_token_unit", 512)
        ),
        gate_decision_threshold=float(cfg.get("gate_decision_threshold", 0.5)),
        gate_calibrate_threshold=bool(cfg.get("gate_calibrate_threshold", True)),
        gate_calibration_samples=int(cfg.get("gate_calibration_samples", 128)),
        gate_calibration_retrieval_penalty=float(
            cfg.get("gate_calibration_retrieval_penalty", 0.05)
        ),
        top_k=int(cfg.get("top_k", 3)),
        experiment_mode=cfg.get("experiment_mode", "intent_main"),
        intent_mode=cfg.get("intent_mode", "static"),
        gate_mode=cfg.get("gate_mode", "learned"),
        adapter_type=cfg.get("adapter_type", "soft_prompt"),
        include_oracle_strategy=bool(cfg.get("include_oracle_strategy", True)),
        build_train_index=bool(cfg.get("build_train_index", False)),
        refresh_train_index=bool(cfg.get("refresh_train_index", False)),
        warmup_steps=int(cfg.get("warmup_steps", 200)),
        train_epochs=int(cfg.get("train_epochs", cfg.get("num_epochs", 1))),
        epoch_budget_mode=bool(cfg.get("epoch_budget_mode", False)),
        num_rounds=int(cfg.get("num_rounds", 2)),
        steps_per_round_prompt=int(cfg.get("steps_per_round_prompt", 100)),
        steps_per_round_retriever=(
            int(cfg["steps_per_round_retriever"])
            if cfg.get("steps_per_round_retriever") is not None
            else None
        ),
        steps_per_round_dpo=int(cfg.get("steps_per_round_dpo", 100)),
        retriever_loss=cfg.get("retriever_loss", "lipo"),
        lipo_tau=float(cfg.get("lipo_tau", 1.0)),
        dpo_beta=float(cfg.get("dpo_beta", 0.1)),
        preference_margin=float(cfg.get("preference_margin", 0.1)),
        utility_margin=float(cfg.get("utility_margin", 0.05)),
        num_hard_negatives=int(cfg.get("num_hard_negatives", 10)),
        preference_pool_top_k=int(cfg.get("preference_pool_top_k", 20)),
        max_pairs_per_sample=int(cfg.get("max_pairs_per_sample", 4)),
        leave_one_out_analysis_samples=int(
            cfg.get("leave_one_out_analysis_samples", 25)
        ),
        gate_quality_tolerance=float(cfg.get("gate_quality_tolerance", 0.01)),
        gate_retrieval_reduction_target=float(
            cfg.get("gate_retrieval_reduction_target", 0.20)
        ),
        retriever_lr=float(cfg.get("retriever_lr", 2e-5)),
        gate_lr=float(cfg.get("gate_lr", 1e-4)),
        soft_prompt_lr=float(cfg.get("soft_prompt_lr", 5e-3)),
        grad_clip_norm=float(cfg.get("grad_clip_norm", 1.0)),
        device=cfg.get("device", "cuda"),
        generator_dtype=cfg.get("generator_dtype", "float16"),
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        random_seed=int(cfg.get("random_seed", 42)),
        max_new_tokens=int(cfg.get("max_new_tokens", 128)),
        batch_encode_size=int(cfg.get("batch_encode_size", 32)),
        eval_ratio=eval_ratio,
        max_eval_samples=max_eval_samples,
        query_entropy_threshold=float(cfg.get("query_entropy_threshold", 0.8)),
        query_num_drafts=int(cfg.get("query_num_drafts", 2)),
        query_draft_max_tokens=int(cfg.get("query_draft_max_tokens", 32)),
        query_draft_temperature=float(cfg.get("query_draft_temperature", 0.8)),
        query_draft_top_p=float(cfg.get("query_draft_top_p", 0.95)),
    )

    trainer = NeuralCoTrainer(neural_cfg)
    result = trainer.train(train_samples, eval_samples=eval_samples)
    result["num_collected_samples"] = len(samples)

    logger.info("Co-Retrieval neural training complete: %s", result.get("status"))
    return result


def _evaluate_neural(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Neural checkpoint evaluation with AlignCoder-style prediction files."""
    from co_retrieval.neural_training import NeuralCoTrainer, NeuralCoTrainingConfig

    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints/co_retrieval_neural")
    output_dir = cfg.get("output_dir", os.path.join("results", "co_retrieval_eval"))
    log_dir = cfg.get("log_dir", "logs/co_retrieval_neural_eval")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_cfg: Dict[str, Any] = {}
    meta_path = os.path.join(checkpoint_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            checkpoint_cfg = json.load(f).get("config", {})

    runtime_cfg = dict(checkpoint_cfg)
    runtime_cfg.update(
        {
            "checkpoint_dir": checkpoint_dir,
            "log_dir": log_dir,
            "device": cfg.get("device", checkpoint_cfg.get("device", "cuda")),
            "generator_dtype": cfg.get(
                "generator_dtype", checkpoint_cfg.get("generator_dtype", "float16")
            ),
            "top_k": int(cfg.get("top_k", checkpoint_cfg.get("top_k", 3))),
            "max_new_tokens": int(
                cfg.get("max_new_tokens", checkpoint_cfg.get("max_new_tokens", 128))
            ),
            "batch_encode_size": int(
                cfg.get(
                    "batch_encode_size",
                    checkpoint_cfg.get("batch_encode_size", 32),
                )
            ),
            "leave_one_out_analysis_samples": int(
                cfg.get(
                    "leave_one_out_analysis_samples",
                    checkpoint_cfg.get("leave_one_out_analysis_samples", 25),
                )
            ),
        }
    )
    if cfg.get("gate_mode") is not None:
        runtime_cfg["gate_mode"] = cfg["gate_mode"]
    if cfg.get("gate_decision_threshold") is not None:
        runtime_cfg["gate_decision_threshold"] = float(
            cfg["gate_decision_threshold"]
        )

    valid_fields = NeuralCoTrainingConfig.__dataclass_fields__
    neural_cfg = NeuralCoTrainingConfig(
        **{key: value for key, value in runtime_cfg.items() if key in valid_fields}
    )

    chunker = RepositoryChunker(
        max_chunk_lines=int(cfg.get("max_chunk_lines", 120)),
        fallback_lines=int(cfg.get("fallback_lines", 40)),
    )
    samples = _collect_evaluation_samples(cfg, chunker)
    if not samples:
        raise RuntimeError("No evaluation samples found.")
    chunks = _collect_global_chunks(samples)
    if not chunks:
        raise RuntimeError("No candidate chunks found for evaluation.")

    trainer = NeuralCoTrainer(neural_cfg)
    trainer.load_checkpoint(checkpoint_dir)
    trainer.phase0_build_index(chunks)
    metrics = trainer.evaluate_to_files(
        samples,
        output_dir,
        include_analysis=bool(cfg.get("include_analysis", True)),
    )
    policy_variants = (
        trainer.evaluate_policy_variants(samples)
        if bool(cfg.get("include_policy_variants", False))
        else {}
    )
    result = {
        "status": "ok",
        "framework": "Co-Retrieval (neural)",
        "checkpoint_dir": checkpoint_dir,
        "output_dir": output_dir,
        "metrics": metrics,
        "eval_policy_variants": policy_variants,
        "num_eval_samples": len(samples),
        "num_chunks": len(chunks),
        "oracle_used_for_eval": False,
        "inference_safe_strategy_check": True,
        "generator_backbone_frozen": True,
        "adapter_type": neural_cfg.adapter_type,
    }
    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    result["result_path"] = result_path
    logger.info("Co-Retrieval neural evaluation complete: %s", result_path)
    return result


def train(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Run Co-Retrieval training — dispatches between proxy and neural mode."""
    cfg = config or {}
    use_neural = bool(cfg.get("use_neural", False))

    if use_neural:
        logger.info(
            "Starting NEURAL mode training (schedule=%s, adapter=%s, "
            "generator frozen)",
            cfg.get("experiment_mode", "intent_main"),
            cfg.get("adapter_type", "soft_prompt"),
        )
        return _train_neural(cfg)
    else:
        logger.info("Starting PROXY mode training (no GPU)")
        return _train_proxy(cfg)


def evaluate(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Run neural Co-Retrieval evaluation from a saved checkpoint."""
    cfg = config or {}
    if not bool(cfg.get("use_neural", True)):
        raise ValueError("Only neural evaluation is implemented.")
    logger.info("Starting NEURAL mode evaluation")
    return _evaluate_neural(cfg)
