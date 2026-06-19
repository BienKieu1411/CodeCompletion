"""Co-Retrieval training entrypoint.

This keeps the existing DatasetLoader sample construction intact and swaps the
old PPO training loop for the novelty-aligned DPO/gate/soft-prompt loop.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

from co_retrieval.chunking import CodeChunk, RepositoryChunker
from co_retrieval.training import CoTrainingConfig, CoTrainingTrainer, TrainingSample
from co_retrieval.data.repository_dataset_loader import DatasetLoader


logger = logging.getLogger(__name__)


def _sample_to_training_sample(
    sample: Dict[str, Any], chunker: RepositoryChunker
) -> TrainingSample:
    chunks: List[CodeChunk] = []
    for file_path, content in sorted(
        (sample.get("crossfile_context") or {}).items()
    ):
        chunks.extend(chunker.chunk_source(str(file_path), str(content or "")))

    return TrainingSample(
        left_context=sample.get("left_context", ""),
        target=sample.get("ground_truth", ""),
        file_path=sample.get("id", "current_file.py"),
        candidate_chunks=chunks,
    )


def _collect_training_samples(
    cfg: Dict[str, Any], chunker: RepositoryChunker
) -> List[TrainingSample]:
    loader = DatasetLoader(
        dataset_path=cfg.get("dataset_path", "data/github_repos/python/train.parquet"),
        use_fim=bool(cfg.get("use_fim", False)),
        completion_level=cfg.get("completion_level", "line"),
        fixed_train=bool(cfg.get("fixed_train", True)),
        fixed_train_size=int(
            cfg.get("fixed_train_size", cfg.get("max_samples", 2000))
        ),
        fixed_train_max_attempts=int(cfg.get("fixed_train_max_attempts", 20000)),
        min_file_lines=int(cfg.get("min_file_lines", 200)),
        min_file_chars=int(cfg.get("min_file_chars", 2000)),
        min_left_context_lines=int(cfg.get("min_left_context_lines", 30)),
    )
    loader.prepare_dataset()

    max_samples = int(cfg.get("max_train_samples", cfg.get("fixed_train_size", 2000)))
    out: List[TrainingSample] = []
    for batch in loader.get_epoch_batches(batch_size=int(cfg.get("batch_size", 4))):
        for sample in batch:
            item = _sample_to_training_sample(sample, chunker)
            if item.target.strip():
                out.append(item)
            if 0 < max_samples <= len(out):
                return out
    return out


def _split_train_eval(
    samples: Sequence[TrainingSample],
    eval_ratio: float,
    max_eval_samples: int,
    random_seed: int,
) -> Tuple[List[TrainingSample], List[TrainingSample]]:
    sample_list = list(samples)
    if len(sample_list) < 2 or eval_ratio <= 0 or max_eval_samples <= 0:
        return sample_list, []

    rng = random.Random(random_seed)
    rng.shuffle(sample_list)
    eval_size = min(max_eval_samples, max(1, int(len(sample_list) * eval_ratio)))
    eval_size = min(eval_size, len(sample_list) - 1)
    return sample_list[eval_size:], sample_list[:eval_size]


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
    train_samples, eval_samples = _split_train_eval(
        samples,
        eval_ratio=eval_ratio,
        max_eval_samples=max_eval_samples,
        random_seed=int(cfg.get("random_seed", 42)),
    )

    neural_cfg = NeuralCoTrainingConfig(
        encoder_name=cfg.get("encoder_name", "jinaai/jina-code-embeddings-1.5b"),
        generator_name=cfg.get("generator_name", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        encoder_max_length=int(cfg.get("encoder_max_length", 512)),
        num_prompt_tokens=int(cfg.get("num_prompt_tokens", 50)),
        max_context_tokens=int(cfg.get("max_context_tokens", 4096)),
        gate_hidden_dim=int(cfg.get("gate_hidden_dim", 256)),
        gate_entropy_weight=float(cfg.get("gate_entropy_weight", 0.01)),
        top_k=int(cfg.get("top_k", 3)),
        experiment_mode=cfg.get("experiment_mode", "intent_main"),
        intent_mode=cfg.get("intent_mode", "static"),
        gate_mode=cfg.get("gate_mode", "learned"),
        adapter_type=cfg.get("adapter_type", "soft_prompt"),
        include_oracle_strategy=bool(cfg.get("include_oracle_strategy", True)),
        warmup_steps=int(cfg.get("warmup_steps", 200)),
        num_rounds=int(cfg.get("num_rounds", 2)),
        steps_per_round_prompt=int(cfg.get("steps_per_round_prompt", 100)),
        steps_per_round_dpo=int(cfg.get("steps_per_round_dpo", 100)),
        dpo_beta=float(cfg.get("dpo_beta", 0.1)),
        preference_margin=float(cfg.get("preference_margin", 0.1)),
        utility_margin=float(cfg.get("utility_margin", 0.05)),
        num_hard_negatives=int(cfg.get("num_hard_negatives", 10)),
        preference_pool_top_k=int(cfg.get("preference_pool_top_k", 20)),
        max_pairs_per_sample=int(cfg.get("max_pairs_per_sample", 4)),
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
    )

    trainer = NeuralCoTrainer(neural_cfg)
    result = trainer.train(train_samples, eval_samples=eval_samples)
    result["num_collected_samples"] = len(samples)

    logger.info("Co-Retrieval neural training complete: %s", result.get("status"))
    return result


def train(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Run Co-Retrieval training — dispatches between proxy and neural mode."""
    cfg = config or {}
    use_neural = bool(cfg.get("use_neural", False))

    if use_neural:
        logger.info("Starting NEURAL mode training (7-phase pipeline)")
        return _train_neural(cfg)
    else:
        logger.info("Starting PROXY mode training (no GPU)")
        return _train_proxy(cfg)
