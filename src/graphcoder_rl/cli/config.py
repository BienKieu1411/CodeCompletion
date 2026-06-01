from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class GraphCoderCLIConfig:
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
    warm_start_steps: int = 100
    completion_level: str = "line"
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-base"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
