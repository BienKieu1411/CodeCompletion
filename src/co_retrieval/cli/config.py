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
    batch_size: int = 8
    completion_level: str = "mixed"

    # ── Mode ──────────────────────────────────────────────────────────────
    use_neural: bool = False

    # ── Neural: model names ───────────────────────────────────────────────
    encoder_name: str = "microsoft/unixcoder-base"
    generator_name: str = "deepseek-ai/deepseek-coder-6.7b-base"

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
    build_train_index: bool = False
    refresh_train_index: bool = False
    skip_train_eval: bool = False
    resample_train_each_epoch: bool = False

    # ── Neural: training ──────────────────────────────────────────────────
    retriever_lr: float = 2e-5
    gate_lr: float = 1e-4
    soft_prompt_lr: float = 5e-3
    retriever_loss: str = "lipo"
    lipo_tau: float = 1.0
    dpo_beta: float = 0.1
    warmup_steps: int = 200
    train_epochs: int = 1
    epoch_budget_mode: bool = False
    num_rounds: int = 2
    steps_per_round_prompt: int = 100
    steps_per_round_retriever: int | None = None
    steps_per_round_dpo: int = 100
    preference_margin: float = 0.1
    utility_margin: float = 0.05
    num_hard_negatives: int = 10
    preference_pool_top_k: int = 20
    max_pairs_per_sample: int = 4
    utility_score_microbatch_size: int = 2
    leave_one_out_analysis_samples: int = 25
    gate_quality_tolerance: float = 0.01
    gate_retrieval_reduction_target: float = 0.20
    gate_use_retrieval_features: bool = True
    gate_context_cost_weight: float = 0.01
    gate_context_cost_token_unit: int = 512
    gate_decision_threshold: float = 0.5
    gate_calibrate_threshold: bool = True
    gate_calibration_samples: int = 128
    gate_calibration_retrieval_penalty: float = 0.05
    grad_clip_norm: float = 1.0
    gate_entropy_weight: float = 0.01
    batch_encode_size: int = 32
    max_new_tokens: int = 128
    eval_ratio: float = 0.1
    max_eval_samples: int = 100

    # ── Neural: device ────────────────────────────────────────────────────
    device: str = "cuda"
    retriever_device: str | None = None
    generator_dtype: str = "float16"
    eval_index_mode: str = "global"
    eval_index_dir: str | None = None
    eval_index_shard_size: int = 50_000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
