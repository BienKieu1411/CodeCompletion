"""Co-Retrieval components for repository-level code completion."""

from co_retrieval.chunking import CodeChunk, RepositoryChunker
from co_retrieval.training import CoTrainingConfig, CoTrainingTrainer, TrainingSample

__all__ = [
    "CodeChunk",
    "RepositoryChunker",
    "CoTrainingConfig",
    "CoTrainingTrainer",
    "TrainingSample",
    "train",
]


def __getattr__(name: str):
    if name == "train":
        from co_retrieval.runner import train

        return train
    raise AttributeError(name)
