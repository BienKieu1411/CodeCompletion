import random
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

from co_retrieval.aligncoder_metrics import (
    compute_aligncoder_metrics,
    write_aligncoder_metric_files,
)
from co_retrieval.chunking import CodeChunk
from co_retrieval.context_utility import ContextScore
from co_retrieval.dense_retriever import DenseRetriever
from co_retrieval.embedding_cache import EmbeddingCache
from co_retrieval.neural_training import (
    GATE_FEATURE_NAMES,
    INFERENCE_SAFE_STRATEGIES,
    GateTrainingExample,
    NeuralCoTrainer,
    PreferenceData,
    _gate_defense_status,
    _gate_label_metrics,
    _scheduled_training_items,
    _select_gate_threshold,
    _safe_corr,
)
from co_retrieval.intent import CostAwareQueryEnhancer, IntentSketcher, _score_entropy
from co_retrieval.soft_prompt import SoftPromptLLM, TokenBudgetManager


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


def test_lipo_includes_stop_score_and_backpropagates_only_to_retriever():
    retriever = TinyDenseRetriever()
    gate = DummyGate()
    loss = retriever.lipo_loss(
        query_text="query",
        candidate_chunks_list=[[], [_chunk("chosen")], [_chunk("rejected")]],
        utilities=[0.0, 2.0, -1.0],
        tau=1.0,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert retriever.query_param.grad is not None
    assert gate.logit.grad is None

    with pytest.raises(ValueError):
        retriever.lipo_loss("query", [[], [_chunk("chosen")]], [0.0, 1.0], tau=0)


def test_lipo_temperature_applies_to_predicted_and_target_distributions():
    retriever = TinyDenseRetriever()
    with torch.no_grad():
        retriever.query_param.copy_(torch.tensor([1.0, 0.0]))

    loss = retriever.lipo_loss(
        query_text="query",
        candidate_chunks_list=[[], [_chunk("chosen")], [_chunk("rejected")]],
        utilities=[0.0, 2.0, -1.0],
        tau=2.0,
    )

    expected = F.kl_div(
        F.log_softmax(torch.tensor([0.0, 1.0, 0.0]) / 2.0, dim=-1).unsqueeze(0),
        F.softmax(torch.tensor([0.0, 2.0, -1.0]) / 2.0, dim=-1).unsqueeze(0),
        reduction="batchmean",
    )
    assert loss.item() == pytest.approx(expected.item())


def test_aligncoder_metric_writer_emits_expected_artifacts(tmp_path):
    records = [
        {
            "task_id": "java-1",
            "prompt": "class A { void f() {",
            "pred": "foo();\nbar();",
            "target": "foo(); // generated comment\n",
            "file_path": "Example.java",
        }
    ]

    metrics = write_aligncoder_metric_files(str(tmp_path), records)

    assert metrics["em"] == 100.0
    assert metrics["total"] == 1
    assert metrics["metric_scale"] == "aligncoder_percent"
    assert (tmp_path / "prediction_truncated.jsonl").exists()
    assert (tmp_path / "exact_match_idx.jsonl").read_text().strip() == "java-1"
    assert (tmp_path / "detailed_results.json").exists()
    assert (tmp_path / "results.json").exists()


def test_aligncoder_metrics_include_repoeval_style_line_cropping():
    metrics = compute_aligncoder_metrics(
        [
                {
                    "task_id": "repo-a/1",
                    "prompt": "",
                    "pred": "target_line\nextra_line",
                    "target": "target_line",
                    "file_path": "x.go",
                    "language": "go",
                }
        ]
    )

    assert metrics["em"] == 0.0
    assert metrics["repoeval_em"] == 100.0
    assert metrics["repoeval_es"] == 100.0


def test_soft_prompt_generation_reserves_new_token_headroom():
    class EosModel:
        def __call__(self, inputs_embeds, attention_mask):
            logits = torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1], 2)
            logits[..., 0] = 10.0
            return SimpleNamespace(logits=logits)

    class DecodeOnlyTokenizer:
        eos_token_id = 0

        def decode(self, ids, skip_special_tokens=True):
            return "decoded" if ids else ""

    model = SoftPromptLLM.__new__(SoftPromptLLM)
    model.num_prompt_tokens = 4
    model.budget_manager = TokenBudgetManager(max_tokens=20, num_prompt_tokens=4)
    model._device = "cpu"
    model.model = EosModel()
    model.tokenizer = DecodeOnlyTokenizer()
    seen = {}

    def prepare_inputs(text, *, max_text_tokens=None):
        seen["max_text_tokens"] = max_text_tokens
        return torch.zeros(1, 6, 2), torch.ones(1, 6, dtype=torch.long)

    model._prepare_inputs = prepare_inputs

    assert model.generate("left", max_new_tokens=5, use_soft_prompt=True) == ""
    assert seen["max_text_tokens"] == 11


def test_evaluate_to_files_preserves_task_id_and_writes_aligncoder_files(tmp_path):
    class FileEvalGenerator:
        def generate(self, *args, **kwargs):
            return "foo();"

        def teacher_forcing_nll(self, *args, **kwargs):
            return torch.tensor(1.0)

    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(
        max_new_tokens=4,
        experiment_mode="intent_main",
        gate_decision_threshold=0.5,
        leave_one_out_analysis_samples=0,
    )
    trainer.use_adapter = False
    trainer.generator = FileEvalGenerator()
    trainer._chunks = []
    trainer._should_retrieve = lambda *args, **kwargs: False
    trainer._gate_probability = lambda *args, **kwargs: 0.1
    sample = SimpleNamespace(
        left_context="class A { void f() {",
        target="foo();",
        file_path="Example.java",
        repo_id="repo-a",
        task_id=123,
        candidate_chunks=[],
    )

    metrics = trainer.evaluate_to_files([sample], str(tmp_path), include_analysis=False)

    pred_row = (tmp_path / "prediction.jsonl").read_text().strip()
    assert '"task_id": "123"' in pred_row
    assert (tmp_path / "prediction_truncated.jsonl").exists()
    assert (tmp_path / "results.json").exists()
    assert metrics["aligncoder_exact_match"] == 1.0



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


def test_current_retrieval_respects_explicit_zero_top_k():
    chunks = [_chunk("alpha"), _chunk("beta")]
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(top_k=2)
    trainer.embedding_cache = EmbeddingCache(dim=2, use_faiss=False)
    trainer.retriever = FakeRetriever()

    assert trainer._retrieve_current("query", chunks, top_k=0) == []


def test_embedding_cache_rejects_zero_top_k_and_detects_content_changes():
    chunk = _chunk("alpha")
    cache = EmbeddingCache(dim=2, use_faiss=False)
    cache.build_from_chunks(
        [chunk], lambda texts: np.asarray([[1.0, 0.0]], dtype=np.float32)
    )

    assert cache.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=0) == []
    changed = CodeChunk(
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        chunk_type=chunk.chunk_type,
        text="def alpha(): return 1",
        defined_symbols=chunk.defined_symbols,
    )
    assert not cache.is_valid_for([changed])


def test_phase4_clears_stale_global_index_when_refresh_is_disabled():
    chunks = [_chunk("alpha")]
    cache = EmbeddingCache(dim=2, use_faiss=False)
    cache.build_from_chunks(
        chunks, lambda texts: np.asarray([[1.0, 0.0]], dtype=np.float32)
    )
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(refresh_train_index=False)
    trainer._chunks = chunks
    trainer.embedding_cache = cache
    trainer.phase4_refresh_index()

    assert trainer.embedding_cache.is_empty


def test_retrieve_with_encoder_returns_empty_for_nonpositive_top_k():
    retriever = DenseRetriever.__new__(DenseRetriever)
    chunks = [_chunk("alpha")]

    assert retriever.retrieve_with_encoder("query", chunks, top_k=0) == []
    assert retriever.retrieve_with_encoder("query", chunks, top_k=-1) == []


def test_inference_strategy_whitelist_rejects_target_aware_and_unknown_modes():
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    for strategy in INFERENCE_SAFE_STRATEGIES:
        trainer.config = SimpleNamespace(experiment_mode=strategy)
        trainer._assert_inference_safe_strategy(strategy, mode="eval")

    for strategy in ("oracle", "hard_neg", "target_symbol", "gold_overlap"):
        trainer.config = SimpleNamespace(experiment_mode=strategy)
        with pytest.raises(RuntimeError):
            trainer._assert_inference_safe_strategy(mode="predict")


def test_gate_supervision_uses_only_deployed_strategy():
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(experiment_mode="intent_main")
    scored = [
        SimpleNamespace(name="oracle", is_stop=False, utility=10.0),
        SimpleNamespace(name="bm25", is_stop=False, utility=2.0),
        SimpleNamespace(name="current", is_stop=False, utility=-1.0),
    ]

    selected = trainer._gate_supervision_score(scored)

    assert selected.name == "current"
    assert selected.utility == -1.0


def test_preference_data_lipo_group_includes_stop_and_deduplicates_contexts():
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(
        experiment_mode="intent_main",
        utility_margin=0.05,
        preference_margin=0.1,
        max_pairs_per_sample=4,
        gate_use_retrieval_features=False,
    )
    trainer.use_adapter = True
    context = [_chunk("chosen")]
    trainer._chunks = context
    trainer._prepare_retrieval_query = lambda *args: ("intent query", context)
    trainer._gate_query = lambda *args: "intent query"
    trainer._strategy_candidates = lambda *args, **kwargs: []
    trainer.utility_scorer = SimpleNamespace(
        score=lambda *args, **kwargs: [
            ContextScore("oracle", context, False, 2.0, 3.0, "intent query"),
            ContextScore("current", context, False, 2.0, 3.0, "intent query"),
            ContextScore("stop", [], True, 5.0, 0.0, "intent query"),
        ]
    )
    sample = SimpleNamespace(
        left_context="service.fetch_",
        target="fetch_user()",
        candidate_chunks=context,
    )

    data = trainer.phase2_build_preference_data([sample])

    assert len(data.lipo_groups) == 1
    group = data.lipo_groups[0]
    assert len(group.candidate_chunks) == 2
    assert "stop" in group.strategy_names
    assert any(not chunks for chunks in group.candidate_chunks)


def test_gate_training_features_use_inference_available_gate_query():
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(
        experiment_mode="intent_main",
        utility_margin=0.05,
        preference_margin=0.1,
        max_pairs_per_sample=4,
        top_k=1,
        include_oracle_strategy=False,
    )
    trainer.use_adapter = True
    context = [_chunk("chosen")]
    trainer._chunks = context
    trainer.retriever = SimpleNamespace(initial_encoder=None)
    trainer._prepare_retrieval_query = lambda *args: ("enhanced query", context)
    trainer._gate_query = lambda *args: "cheap gate query"
    trainer._mine_hard_negatives = lambda *args, **kwargs: []
    seen_queries = []

    def gate_features(sample, candidates, *, retrieval_query=None, **kwargs):
        seen_queries.append(retrieval_query)
        return (1.0,)

    trainer._gate_features = gate_features
    trainer.utility_scorer = SimpleNamespace(
        score=lambda *args, **kwargs: [
            ContextScore("current", context, False, 2.0, 3.0, "enhanced query"),
            ContextScore("stop", [], True, 5.0, 0.0, "enhanced query"),
        ]
    )
    sample = SimpleNamespace(
        left_context="service.fetch_",
        target="fetch_user()",
        candidate_chunks=context,
    )

    data = trainer.phase2_build_preference_data([sample])

    assert data.gate_examples[0].query == "cheap gate query"
    assert seen_queries == ["cheap gate query"]


def test_gate_adjusted_utility_penalizes_context_cost_for_stop_label():
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(
        max_context_tokens=4096,
        gate_context_cost_token_unit=10,
        gate_context_cost_weight=0.5,
    )
    long_context = [
        CodeChunk(
            file_path="chosen.py",
            start_line=1,
            end_line=1,
            chunk_type="function",
            text="x" * 80,
            defined_symbols=["chosen"],
        )
    ]
    score = SimpleNamespace(is_stop=False, chunks=long_context, utility=1.0)

    assert trainer._gate_adjusted_utility(score) <= 0.0


def test_gate_features_are_inference_safe_and_fixed_width():
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(
        gate_use_retrieval_features=True,
        top_k=2,
        preference_pool_top_k=10,
        max_context_tokens=512,
        gate_context_cost_token_unit=128,
    )
    chunks = [_chunk("alpha"), _chunk("beta")]
    trainer._gate_query = lambda sample: "alpha.call"
    trainer._retrieve_current_with_scores = lambda *args, **kwargs: (
        [chunks[0], chunks[1]],
        [0.9, 0.1],
    )
    sample = SimpleNamespace(left_context="alpha.call", target="")

    features = trainer._gate_features(sample, chunks)

    assert len(features) == len(GATE_FEATURE_NAMES)
    assert features[0] == pytest.approx(0.9)
    assert features[1] == pytest.approx(0.8)
    assert features[-1] > 0.0


def test_normalized_entropy_and_cost_aware_query_paths():
    assert _score_entropy([0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert _score_entropy([10.0, -10.0, -10.0]) < 0.01

    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(
        intent_mode="cost_aware",
        top_k=3,
        query_entropy_threshold=0.8,
        query_num_drafts=2,
        query_draft_max_tokens=32,
        query_draft_temperature=0.8,
        query_draft_top_p=0.95,
    )
    trainer.intent_sketcher = IntentSketcher()
    trainer.query_enhancer = CostAwareQueryEnhancer(
        trainer.intent_sketcher, entropy_threshold=0.8
    )
    trainer.use_adapter = True
    trainer._query_enhancement_stats = {
        "queries": 0.0,
        "entropy_sum": 0.0,
        "entropy_count": 0.0,
        "sampling_count": 0.0,
        "draft_count": 0.0,
        "changed_count": 0.0,
    }
    chunks = [_chunk("alpha"), _chunk("beta"), _chunk("gamma")]
    trainer._retrieve_current_with_scores = lambda *args, **kwargs: (
        chunks,
        [0.0, 0.0, 0.0],
    )
    reretrieved = []
    trainer._retrieve_current = lambda query, *args, **kwargs: (
        reretrieved.append(query) or chunks
    )
    trainer.generator = SimpleNamespace(
        generate_drafts=lambda *args, **kwargs: ["refund_payment()", "refund_user"]
    )
    sample = SimpleNamespace(left_context="result = service.refund_", target="")

    query, selected = trainer._prepare_retrieval_query(sample, chunks)

    assert "refund_payment" in query
    assert selected == chunks
    assert len(reretrieved) == 1
    assert trainer._query_enhancement_stats["sampling_count"] == 1
    assert trainer._query_enhancement_stats["changed_count"] == 1


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

    def phase3(preference_data, steps=None, gate_steps=None):
        trainer.calls.append(("retriever", steps, gate_steps))
        return {"phase3_retriever_loss": 0.0, "phase3_gate_loss": 0.0}

    def refresh():
        trainer.calls.append(("refresh",))

    trainer.phase1_warmup_soft_prompt = phase1
    trainer.phase2_build_preference_data = phase2
    trainer.phase3_retriever_training = phase3
    trainer.phase4_refresh_index = refresh
    return trainer


def test_sequential_adapter_first_uses_single_preference_build_and_same_budget():
    trainer = _sequential_trainer()

    result = trainer.phase5_sequential_adapter_first(samples=[object()])

    assert trainer.calls == [
        ("prompt", 17, "mixed"),
        ("preference", True),
        ("retriever", 21, 21),
        ("refresh",),
    ]
    assert result[0]["preference_data_builds"] == 1
    assert result[0]["prompt_steps_budget"] == 17
    assert result[0]["retriever_steps_budget"] == 21


def test_sequential_retriever_first_does_not_refresh_preferences_after_adapter():
    trainer = _sequential_trainer()

    result = trainer.phase5_sequential_retriever_first(samples=[object()])

    assert trainer.calls == [
        ("preference", False),
        ("retriever", 21, 21),
        ("refresh",),
        ("prompt", 17, "retriever"),
    ]
    assert trainer.use_adapter is True
    assert result[0]["preference_data_builds"] == 1


def test_epoch_budget_mode_uses_full_pass_steps_without_extra_refreshes():
    trainer = _sequential_trainer()
    trainer.config.epoch_budget_mode = True
    trainer.config.train_epochs = 10
    samples = [object(), object(), object()]

    result = trainer.phase5_sequential_adapter_first(samples=samples)

    assert trainer.calls == [
        ("prompt", 32, "mixed"),
        ("preference", True),
        ("retriever", 0, 10),
        ("refresh",),
    ]
    assert result[0]["epoch_budget_mode"] is True
    assert result[0]["train_epochs"] == 10
    assert result[0]["prompt_steps_budget"] == 32
    assert result[0]["retriever_steps_budget"] == 0
    assert result[0]["gate_steps_budget"] == 10


def test_train_registers_chunks_without_building_global_index_by_default():
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(
        experiment_mode="intent_main",
        build_train_index=False,
        refresh_train_index=False,
        gate_calibrate_threshold=False,
        gate_decision_threshold=0.5,
        train_epochs=1,
        epoch_budget_mode=False,
    )
    calls = []
    trainer.phase0_build_index = lambda chunks: calls.append(("build", len(chunks)))
    trainer.phase0_register_chunks = lambda chunks: calls.append(
        ("register", len(chunks))
    )
    trainer.phase1_warmup_soft_prompt = lambda samples: {"phase1_loss": 0.0}
    trainer.phase5_co_training = lambda samples: []
    trainer.gate_defense_status = lambda variants: {"status": "not_evaluated"}
    trainer._save_checkpoint = lambda payload: calls.append(("save",))

    result = trainer.train(
        [
            SimpleNamespace(
                left_context="x",
                target="y",
                candidate_chunks=[_chunk("ctx")],
            )
        ],
        eval_samples=[],
    )

    assert calls[0] == ("register", 1)
    assert ("build", 1) not in calls
    assert result["build_train_index"] is False
    assert result["refresh_train_index"] is False


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


def test_gate_defense_status_prefers_aligncoder_metrics_when_available():
    learned = {
        "exact_match": 0.80,
        "edit_similarity": 0.80,
        "identifier_f1": 0.80,
        "aligncoder_exact_match": 0.49,
        "aligncoder_edit_similarity": 0.69,
        "aligncoder_identifier_f1": 0.79,
        "retrieval_rate": 0.40,
    }
    always = {
        "exact_match": 0.10,
        "edit_similarity": 0.10,
        "identifier_f1": 0.10,
        "aligncoder_exact_match": 0.50,
        "aligncoder_edit_similarity": 0.70,
        "aligncoder_identifier_f1": 0.80,
        "retrieval_rate": 0.70,
    }

    status = _gate_defense_status(
        learned,
        always,
        quality_tolerance=0.01,
        retrieval_reduction_target=0.20,
    )

    assert status["status"] == "compute_reduction_without_quality_loss"
    assert status["metric_sources"] == {
        "exact_match": "aligncoder_exact_match",
        "edit_similarity": "aligncoder_edit_similarity",
        "identifier_f1": "aligncoder_identifier_f1",
    }


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


def test_gate_threshold_selection_penalizes_unnecessary_retrieval():
    selected = _select_gate_threshold(
        [True, False, False],
        [0.9, 0.8, 0.1],
        retrieval_penalty=0.5,
    )

    assert selected["threshold"] >= 0.8
    assert selected["predicted_retrieval_rate"] < 1.0


def test_scheduled_training_items_cycles_to_exact_budget():
    items = _scheduled_training_items(["a", "b"], 5, random.Random(0))

    assert len(items) == 5
    assert set(items) == {"a", "b"}


def test_phase1_soft_prompt_cycles_to_exact_step_budget():
    class CountingGenerator:
        def __init__(self):
            self.prompt_embeddings = nn.Parameter(torch.tensor([1.0]))
            self.calls = 0

        def generation_loss(self, *args, **kwargs):
            self.calls += 1
            return self.prompt_embeddings.sum() * 0.0 + 1.0

    generator = CountingGenerator()
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(
        warmup_steps=5,
        warmup_no_context_ratio=1.0,
        warmup_oracle_ratio=0.0,
        top_k=3,
        grad_clip_norm=1.0,
    )
    trainer.use_adapter = True
    trainer.rng = random.Random(0)
    trainer._chunks = []
    trainer.generator = generator
    trainer.prompt_opt = torch.optim.SGD([generator.prompt_embeddings], lr=0.0)
    samples = [
        SimpleNamespace(left_context="a", target="b", candidate_chunks=[]),
        SimpleNamespace(left_context="c", target="d", candidate_chunks=[]),
    ]

    result = trainer.phase1_warmup_soft_prompt(samples, steps=5)

    assert generator.calls == 5
    assert result["phase1_loss"] == 1.0


def test_phase1_soft_prompt_respects_explicit_zero_steps():
    generator = SimpleNamespace(prompt_embeddings=nn.Parameter(torch.tensor([1.0])))
    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(warmup_steps=5)
    trainer.use_adapter = True
    trainer.generator = generator

    result = trainer.phase1_warmup_soft_prompt([object()], steps=0)

    assert result["phase1_loss"] == 0.0


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


def test_phase6_keeps_adapter_enabled_for_retrieve_and_skip_branches():
    class RecordingGenerator:
        def __init__(self):
            self.flags = []

        def generate(self, *args, use_soft_prompt=True, **kwargs):
            self.flags.append(("generate", use_soft_prompt))
            return "prediction"

        def teacher_forcing_nll(self, *args, use_soft_prompt=True, **kwargs):
            self.flags.append(("nll", use_soft_prompt))
            return torch.tensor(1.0)

    trainer = NeuralCoTrainer.__new__(NeuralCoTrainer)
    trainer.config = SimpleNamespace(max_new_tokens=4, experiment_mode="intent_main")
    trainer.use_adapter = True
    trainer.generator = RecordingGenerator()
    trainer._chunks = [_chunk("ctx")]
    trainer._query_enhancement_stats = {
        "queries": 0.0,
        "entropy_sum": 0.0,
        "entropy_count": 0.0,
        "sampling_count": 0.0,
        "draft_count": 0.0,
        "changed_count": 0.0,
    }
    trainer._assert_inference_safe_strategy = lambda *args, **kwargs: None
    trainer._should_retrieve = (
        lambda sample, candidates: sample.left_context == "retrieve"
    )
    trainer._retrieve_for_generation = lambda *args, **kwargs: [_chunk("ctx")]
    samples = [
        SimpleNamespace(
            left_context="retrieve", target="target", candidate_chunks=[_chunk("ctx")]
        ),
        SimpleNamespace(
            left_context="skip", target="target", candidate_chunks=[_chunk("ctx")]
        ),
    ]

    trainer.phase6_evaluate(samples, include_analysis=False)

    assert trainer.generator.flags
    assert all(flag is True for _, flag in trainer.generator.flags)
