import sys
from types import ModuleType, SimpleNamespace


class _FakeAuto:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError("This unit test must not load HuggingFace models")


fake_transformers = ModuleType("transformers")
fake_transformers.AutoModel = _FakeAuto
fake_transformers.AutoModelForCausalLM = _FakeAuto
fake_transformers.AutoTokenizer = _FakeAuto
sys.modules.setdefault("transformers", fake_transformers)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from co_retrieval.chunking import CodeChunk
from co_retrieval.dense_retriever import DenseRetriever
from co_retrieval.embedding_cache import EmbeddingCache
from co_retrieval.neural_training import NeuralCoTrainer


def _chunk(symbol: str) -> CodeChunk:
    return CodeChunk(
        file_path=f"{symbol}.py",
        start_line=1,
        end_line=1,
        chunk_type="function",
        text=f"def {symbol}(): pass",
        defined_symbols=[symbol],
    )


class TinyDenseRetriever(DenseRetriever):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self._device = "cpu"
        self._reference_encoder = None
        self.query_param = nn.Parameter(torch.tensor([1.0, 0.2]))
        self.chunk_vectors = {
            "chosen": torch.tensor([1.0, 0.0]),
            "rejected": torch.tensor([0.0, 1.0]),
        }

    def encode_query(self, query: str) -> torch.Tensor:
        return F.normalize(self.query_param, p=2, dim=0)

    def encode_chunks(
        self,
        chunks,
        batch_size: int = 32,
        encoder=None,
    ) -> torch.Tensor:
        return torch.stack(
            [self.chunk_vectors[chunk.defined_symbols[0]] for chunk in chunks]
        )


class DummyGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.3))

    def log_probs(self, query_vec: torch.Tensor):
        return F.logsigmoid(self.logit), F.logsigmoid(-self.logit)


def test_retriever_dpo_does_not_backprop_into_gate_by_default():
    retriever = TinyDenseRetriever()
    gate = DummyGate()
    chosen = _chunk("chosen")
    rejected = _chunk("rejected")

    loss = retriever.dpo_loss(
        query_text="query",
        chosen_chunks=[chosen],
        rejected_chunks=[rejected],
        gate=gate,
        beta=0.1,
    )
    loss.backward()

    assert retriever.query_param.grad is not None
    assert gate.logit.grad is None


class FakeRetriever:
    def __init__(self) -> None:
        self.live_retrieve_called = False

    def encode_query(self, query: str) -> torch.Tensor:
        return torch.tensor([0.0, 1.0])

    def retrieve_chunks(self, query, chunks, top_k=3):
        self.live_retrieve_called = True
        return list(chunks)[:top_k]


def test_current_retrieval_uses_refreshed_embedding_cache_when_available():
    chunks = [_chunk("alpha"), _chunk("beta"), _chunk("gamma")]
    cache = EmbeddingCache(dim=2, use_faiss=False)

    def encode_fn(texts):
        vectors = []
        for text in texts:
            if "alpha" in text:
                vectors.append([1.0, 0.0])
            elif "beta" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return np.asarray(vectors, dtype=np.float32)

    cache.build_from_chunks(chunks, encode_fn)
    retriever = FakeRetriever()
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(top_k=2)
    trainer.embedding_cache = cache
    trainer.retriever = retriever

    selected = trainer._retrieve_current("query", chunks, top_k=2)

    assert [chunk.defined_symbols[0] for chunk in selected] == ["beta", "gamma"]
    assert retriever.live_retrieve_called is False
