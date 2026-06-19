"""Co-Retrieval components for repository-level code completion."""

from co_retrieval.chunking import CodeChunk, RepositoryChunker
from co_retrieval.training import CoTrainingConfig, CoTrainingTrainer, TrainingSample

__all__ = [
    # Core
    "CodeChunk",
    "RepositoryChunker",
    # Proxy mode
    "CoTrainingConfig",
    "CoTrainingTrainer",
    "TrainingSample",
    # Entrypoint
    "train",
]


def __getattr__(name: str):
    if name == "train":
        from co_retrieval.runner import train

        return train

    # Lazy imports for neural components (avoid GPU dependency at import time)
    _neural_exports = {
        "DenseRetriever": "co_retrieval.dense_retriever",
        "NeuralGate": "co_retrieval.neural_gate",
        "SoftPromptLLM": "co_retrieval.soft_prompt",
        "EmbeddingCache": "co_retrieval.embedding_cache",
        "NeuralCoTrainer": "co_retrieval.neural_training",
        "NeuralCoTrainingConfig": "co_retrieval.neural_training",
    }
    if name in _neural_exports:
        import importlib

        module = importlib.import_module(_neural_exports[name])
        return getattr(module, name)

    raise AttributeError(name)
