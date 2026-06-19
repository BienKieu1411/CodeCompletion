from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class CoRetrievalCLIConfig:
    """Configuration for Co-Retrieval CLI (both proxy and neural modes)."""

    # ── Shared ────────────────────────────────────────────────────────────
    dataset_path: str = "data/github_repos/python/train.parquet"
    language: str = "python"
    output_dir: str = "results"
    checkpoint: str | None = None
    pretrain_checkpoint: str | None = None
    cache_dir: str = "cache"
    max_samples: int = 50
    top_k: int = 3
    num_epochs: int = 1
    batch_size: int = 2
    completion_level: str = "line"

    # ── Mode ──────────────────────────────────────────────────────────────
    use_neural: bool = False

    # ── Neural: model names ───────────────────────────────────────────────
    encoder_name: str = "jinaai/jina-code-embeddings-1.5b"
    generator_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # ── Neural: architecture ──────────────────────────────────────────────
    num_prompt_tokens: int = 50
    max_context_tokens: int = 4096
    gate_hidden_dim: int = 256
    encoder_max_length: int = 512
    experiment_mode: str = "intent_main"
    intent_mode: str = "static"
    gate_mode: str = "learned"
    adapter_type: str = "soft_prompt"
    include_oracle_strategy: bool = True

    # ── Neural: training ──────────────────────────────────────────────────
    retriever_lr: float = 2e-5
    gate_lr: float = 1e-4
    soft_prompt_lr: float = 5e-3
    dpo_beta: float = 0.1
    warmup_steps: int = 200
    num_rounds: int = 2
    steps_per_round_prompt: int = 100
    steps_per_round_dpo: int = 100
    preference_margin: float = 0.1
    utility_margin: float = 0.05
    num_hard_negatives: int = 10
    preference_pool_top_k: int = 20
    max_pairs_per_sample: int = 4
    grad_clip_norm: float = 1.0
    gate_entropy_weight: float = 0.01
    batch_encode_size: int = 32
    max_new_tokens: int = 128
    eval_ratio: float = 0.1
    max_eval_samples: int = 100

    # ── Neural: device ────────────────────────────────────────────────────
    device: str = "cuda"
    generator_dtype: str = "float16"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
