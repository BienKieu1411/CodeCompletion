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
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from co_retrieval.chunking import CodeChunk
from co_retrieval.dense_retriever import DenseRetriever
from co_retrieval.embedding_cache import EmbeddingCache
from co_retrieval.neural_training import (
    INFERENCE_SAFE_STRATEGIES,
    GateTrainingExample,
    NeuralCoTrainer,
    PreferenceData,
    _gate_defense_status,
    _gate_label_metrics,
    _safe_corr,
)


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


def test_inference_strategy_whitelist_rejects_target_aware_and_unknown_modes():
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    for strategy in INFERENCE_SAFE_STRATEGIES:
        trainer.config = SimpleNamespace(experiment_mode=strategy)
        trainer._assert_inference_safe_strategy(strategy, mode="eval")

    for strategy in ("oracle", "hard_neg", "target_symbol", "gold_overlap"):
        trainer.config = SimpleNamespace(experiment_mode=strategy)
        with pytest.raises(RuntimeError):
            trainer._assert_inference_safe_strategy(mode="predict")


def _preference_data():
    return PreferenceData(
        pairs=[],
        gate_examples=[
            GateTrainingExample(
                query="q",
                retrieve_is_better=True,
                max_utility=1.0,
                best_strategy="current",
            )
        ],
        gate_positive_count=1,
        gate_negative_count=0,
        mean_max_utility=1.0,
    )


def _sequential_trainer():
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(
        warmup_steps=2,
        num_rounds=3,
        steps_per_round_prompt=5,
        steps_per_round_dpo=7,
    )
    trainer.use_adapter = True
    trainer.calls = []

    def phase1(samples, steps=None, context_mode="mixed"):
        trainer.calls.append(("prompt", steps, context_mode))
        return {"phase1_loss": 0.0}

    def phase2(samples):
        trainer.calls.append(("preference", trainer.use_adapter))
        return _preference_data()

    def phase3(preference_data, steps=None):
        trainer.calls.append(("dpo", steps))
        return {"phase3_dpo_loss": 0.0, "phase3_gate_loss": 0.0}

    def refresh():
        trainer.calls.append(("refresh",))

    trainer.phase1_warmup_soft_prompt = phase1
    trainer.phase2_build_preference_data = phase2
    trainer.phase3_dpo_training = phase3
    trainer.phase4_refresh_index = refresh
    return trainer


def test_sequential_adapter_first_uses_single_preference_build_and_same_budget():
    trainer = _sequential_trainer()

    result = trainer.phase5_sequential_adapter_first(samples=[object()])

    assert trainer.calls == [
        ("prompt", 17, "mixed"),
        ("preference", True),
        ("dpo", 21),
        ("refresh",),
    ]
    assert result[0]["preference_data_builds"] == 1
    assert result[0]["prompt_steps_budget"] == 17
    assert result[0]["dpo_steps_budget"] == 21


def test_sequential_retriever_first_does_not_refresh_preferences_after_adapter():
    trainer = _sequential_trainer()

    result = trainer.phase5_sequential_retriever_first(samples=[object()])

    assert trainer.calls == [
        ("preference", False),
        ("dpo", 21),
        ("refresh",),
        ("prompt", 17, "retriever"),
    ]
    assert trainer.use_adapter is True
    assert result[0]["preference_data_builds"] == 1


def test_gate_policy_ablation_restores_original_gate_mode():
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(gate_mode="rule")
    seen_modes = []

    def phase6(samples, include_analysis=True):
        seen_modes.append(trainer.config.gate_mode)
        return {
            "exact_match": 0.0,
            "edit_similarity": 0.0,
            "identifier_f1": 0.0,
            "retrieval_rate": 0.0,
        }

    trainer.phase6_evaluate = phase6
    variants = trainer.evaluate_policy_variants([object()])

    assert seen_modes == ["learned", "always_retrieve", "always_skip"]
    assert set(variants) == {"learned", "always_retrieve", "always_skip"}
    assert trainer.config.gate_mode == "rule"


def test_gate_defense_status_requires_all_quality_metrics_and_retrieval_drop():
    learned = {
        "exact_match": 0.50,
        "edit_similarity": 0.70,
        "identifier_f1": 0.79,
        "retrieval_rate": 0.40,
    }
    always = {
        "exact_match": 0.51,
        "edit_similarity": 0.71,
        "identifier_f1": 0.80,
        "retrieval_rate": 0.70,
    }

    status = _gate_defense_status(
        learned,
        always,
        quality_tolerance=0.01,
        retrieval_reduction_target=0.20,
    )

    assert status["status"] == "compute_reduction_without_quality_loss"
    bad = dict(learned, identifier_f1=0.78)
    assert (
        _gate_defense_status(
            bad,
            always,
            quality_tolerance=0.01,
            retrieval_reduction_target=0.20,
        )["status"]
        == "gate_claim_not_supported"
    )


def test_gate_label_metrics_and_safe_correlation_handle_edge_cases():
    metrics = _gate_label_metrics(
        [True, True, False, False],
        [0.9, 0.2, 0.8, 0.1],
    )
    assert metrics["confusion_matrix"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["auc"] == 0.75

    single_class = _gate_label_metrics([True, True], [0.3, 0.7])
    assert single_class["auc"] is None
    assert _safe_corr([1.0, 1.0], [0.2, 0.4]) == 0.0


class LeaveOneOutGenerator:
    def teacher_forcing_nll(
        self,
        left_context,
        target,
        retrieved_chunks=None,
        use_soft_prompt=True,
    ):
        symbols = {
            chunk.defined_symbols[0] for chunk in (retrieved_chunks or [])
        }
        if symbols == {"helpful", "noisy"}:
            return torch.tensor(4.0)
        if symbols == {"noisy"}:
            return torch.tensor(8.0)
        if symbols == {"helpful"}:
            return torch.tensor(3.0)
        return torch.tensor(10.0)


def test_leave_one_out_analysis_identifies_helpful_and_noisy_chunks():
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(
        leave_one_out_analysis_samples=1,
        experiment_mode="intent_main",
    )
    trainer.use_adapter = True
    trainer.generator = LeaveOneOutGenerator()
    trainer._assert_inference_safe_strategy = lambda *args, **kwargs: None
    trainer._retrieve_for_generation = lambda sample, candidates, mode="": [
        _chunk("helpful"),
        _chunk("noisy"),
    ]
    sample = SimpleNamespace(
        left_context="x",
        target="target",
        candidate_chunks=[_chunk("helpful"), _chunk("noisy")],
    )

    result = trainer.leave_one_out_analysis([sample])

    assert result["num_sets"] == 1
    assert result["positive_contribution_count"] == 1
    assert result["negative_contribution_count"] == 1
