"""Neural Co-Training pipeline — 7-phase training specification.

Phase 0 — Build repo index (AST chunking → UniXcoder embed → FAISS)
Phase 1 — Warm-up soft prompt (20% no-ctx, 50% oracle, 30% noisy; CE loss)
Phase 2 — Build preference data (6 strategies, teacher-forcing NLL)
Phase 3 — DPO-style train retriever; train gate from utility labels
Phase 4 — Refresh FAISS index (re-embed after retriever update)
Phase 5 — Alternating co-training epochs/rounds (P1→P2→P3→P4 repeated)
Phase 6 — Final evaluation (retrieval + generation metrics)
"""

from __future__ import annotations

import logging
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import AdamW

from co_retrieval.chunking import CodeChunk
from co_retrieval.aligncoder_metrics import (
    compute_aligncoder_metrics,
    write_aligncoder_metric_files,
)
from co_retrieval.context_utility import ContextCandidate, ContextUtilityScorer
from co_retrieval.dense_retriever import DenseRetriever
from co_retrieval.embedding_cache import EmbeddingCache, ShardedEmbeddingCache
from co_retrieval.intent import (
    CostAwareQueryEnhancer,
    IntentSketcher,
    _score_entropy,
)
from co_retrieval.neural_gate import NeuralGate
from co_retrieval.quality_metrics import (
    exact_match,
    edit_similarity,
    identifier_f1,
    identifier_set,
)
from co_retrieval.soft_prompt import SoftPromptLLM
from co_retrieval.training import TrainingSample

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")
INFERENCE_SAFE_STRATEGIES = {
    "current",
    "bm25",
    "dense_frozen",
    "learned_retriever",
}
TRAIN_ONLY_STRATEGIES = {
    "oracle",
    "hard_neg",
    "target_symbol",
    "gold_overlap",
    "future_context",
}
GATE_FEATURE_NAMES = (
    "top1_score",
    "top1_margin",
    "score_entropy",
    "score_std",
    "selected_count_ratio",
    "candidate_pool_ratio",
    "query_identifier_ratio",
    "context_token_ratio",
    "identifier_overlap_ratio",
)
# Gate labels are selected from the single deployed strategy, while these sets
# enforce the broader train/inference safety boundary.


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class NeuralCoTrainingConfig:
    """Full configuration for the 7-phase training pipeline."""

    # Model names
    encoder_name: str = "microsoft/unixcoder-base"
    generator_name: str = "deepseek-ai/deepseek-coder-6.7b-base"

    # Encoder
    encoder_max_length: int = 512
    # Evaluation may keep retrieval on CPU while generation remains on GPU.
    retriever_device: Optional[str] = None

    # Generator
    num_prompt_tokens: int = 50
    max_context_tokens: int = 4096

    # Gate
    gate_hidden_dim: int = 256
    gate_entropy_weight: float = 0.01
    gate_use_retrieval_features: bool = True
    gate_context_cost_weight: float = 0.01
    gate_context_cost_token_unit: int = 512
    gate_decision_threshold: float = 0.5
    gate_calibrate_threshold: bool = True
    gate_calibration_samples: int = 128
    gate_calibration_retrieval_penalty: float = 0.05

    # Retrieval
    top_k: int = 3
    experiment_mode: str = "intent_main"
    intent_mode: str = "static"
    gate_mode: str = "learned"
    adapter_type: str = "soft_prompt"
    include_oracle_strategy: bool = True
    build_train_index: bool = False
    refresh_train_index: bool = False

    # Phase 1 — Soft Prompt Warm-up
    warmup_steps: int = 200
    warmup_no_context_ratio: float = 0.20
    warmup_oracle_ratio: float = 0.50
    # warmup_noisy_ratio = 1.0 - no_context - oracle = 0.30

    # Phase 2 — Preference Data
    preference_margin: float = 0.1
    utility_margin: float = 0.05
    num_hard_negatives: int = 10
    preference_pool_top_k: int = 20
    max_pairs_per_sample: int = 4
    utility_score_microbatch_size: int = 2
    leave_one_out_analysis_samples: int = 25
    gate_quality_tolerance: float = 0.01
    gate_retrieval_reduction_target: float = 0.20

    # Phase 3 — Retriever training
    retriever_loss: str = "lipo"  # "lipo" (default) or "dpo" (ablation)
    lipo_tau: float = 1.0  # temperature for LiPO soft label distribution
    dpo_beta: float = 0.1  # beta for DPO loss (only used when retriever_loss="dpo")

    # Phase 5 — Co-training epochs / legacy fixed-step rounds
    train_epochs: int = 1
    epoch_budget_mode: bool = False
    num_rounds: int = 2
    steps_per_round_prompt: int = 100
    steps_per_round_retriever: Optional[int] = None
    # Deprecated compatibility alias, used when the new field is unset.
    steps_per_round_dpo: int = 100

    # Optimiser LRs
    retriever_lr: float = 2e-5
    gate_lr: float = 1e-4
    soft_prompt_lr: float = 5e-3

    # Regularisation
    grad_clip_norm: float = 1.0

    # Hardware
    device: str = "cuda"
    generator_dtype: str = "float16"

    # Checkpointing
    checkpoint_dir: str = "checkpoints/co_retrieval_neural"
    log_dir: str = "logs/co_retrieval_neural"

    # Misc
    random_seed: int = 42
    max_new_tokens: int = 128
    eval_skip_nll: bool = False
    eval_batch_size: int = 1
    batch_encode_size: int = 32
    eval_ratio: float = 0.1
    max_eval_samples: int = 100

    # Cost-aware query enhancement (active for intent_mode="cost_aware").
    query_entropy_threshold: float = 0.8
    query_num_drafts: int = 2
    query_draft_max_tokens: int = 32
    query_draft_temperature: float = 0.8
    query_draft_top_p: float = 0.95
    # Evaluation retrieval index. ``sharded`` is built separately and loaded
    # as CPU mmap files during generation.
    eval_index_mode: str = "global"
    eval_index_dir: Optional[str] = None
    eval_index_shard_size: int = 50_000


# ── BM25-like simple scorer ──────────────────────────────────────────────────


def _bm25_retrieve(
    query: str,
    chunks: Sequence[CodeChunk],
    top_k: int = 3,
) -> List[CodeChunk]:
    """Simple lexical BM25-like retrieval for C_bm25 strategy."""
    query_tokens = set(_IDENTIFIER_RE.findall(query))
    if not query_tokens:
        return list(chunks[:top_k])

    scored: List[Tuple[float, CodeChunk]] = []
    for chunk in chunks:
        chunk_tokens = set(
            chunk.defined_symbols + chunk.used_symbols + chunk.call_names
        )
        overlap = query_tokens & chunk_tokens
        score = len(overlap) * 1.5
        if chunk.chunk_type in {"method", "function", "class_header"}:
            score += 0.15
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def _approx_token_count(text: str) -> int:
    """Cheap token estimate for gate cost features without invoking LLM tokenizers."""
    return max(1, int(np.ceil(len(text) / 4.0))) if text else 0


def _scheduled_training_items(
    items: Sequence[Any],
    steps: int,
    rng: random.Random,
) -> List[Any]:
    """Return exactly ``steps`` items by reshuffling and cycling through data."""
    if steps <= 0 or not items:
        return []
    pool = list(items)
    scheduled: List[Any] = []
    while len(scheduled) < steps:
        rng.shuffle(pool)
        scheduled.extend(pool)
    return scheduled[:steps]


def _oracle_chunks(
    target: str,
    chunks: Sequence[CodeChunk],
    top_k: int = 3,
) -> List[CodeChunk]:
    """Find oracle chunks — those containing symbols from the target."""
    target_symbols = identifier_set(target)
    if not target_symbols:
        return []

    scored: List[Tuple[int, CodeChunk]] = []
    for chunk in chunks:
        chunk_symbols = set(
            chunk.defined_symbols + chunk.call_names + chunk.method_names
        )
        overlap = len(target_symbols & chunk_symbols)
        if overlap > 0:
            scored.append((overlap, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def _safe_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation with a safe 0.0 fallback for degenerate inputs."""
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> Optional[float]:
    """Return ROC AUC, or None when only one class is present."""
    if len(labels) != len(scores) or not labels:
        return None
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0.0
    for pos in positives:
        for neg in negatives:
            total += 1.0
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else None


def _gate_label_metrics(
    labels: Sequence[bool],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Gate calibration metrics against utility-derived retrieve/skip labels."""
    tp = fp = tn = fn = 0
    for label, prob in zip(labels, probabilities):
        pred = prob >= threshold
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and not label:
            tn += 1
        else:
            fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "auc": _binary_auc(labels, probabilities),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "num_samples": len(labels),
    }


def _select_gate_threshold(
    labels: Sequence[bool],
    probabilities: Sequence[float],
    *,
    retrieval_penalty: float = 0.05,
) -> Dict[str, Any]:
    """Choose a gate threshold from utility labels with a retrieval-rate penalty."""
    if len(labels) != len(probabilities) or not labels:
        return {
            "threshold": 0.5,
            "objective": 0.0,
            "predicted_retrieval_rate": 0.0,
            "metrics": _gate_label_metrics([], []),
        }
    candidates = sorted(set(float(p) for p in probabilities))
    candidates = [0.0] + candidates + [1.0]
    best: Dict[str, Any] | None = None
    for threshold in candidates:
        metrics = _gate_label_metrics(labels, probabilities, threshold=threshold)
        predicted_rate = sum(p >= threshold for p in probabilities) / max(
            1, len(probabilities)
        )
        objective = metrics["f1"] - retrieval_penalty * predicted_rate
        candidate = {
            "threshold": float(threshold),
            "objective": float(objective),
            "predicted_retrieval_rate": float(predicted_rate),
            "metrics": metrics,
        }
        if best is None:
            best = candidate
            continue
        if objective > best["objective"] + 1e-12:
            best = candidate
        elif abs(objective - best["objective"]) <= 1e-12 and predicted_rate < best[
            "predicted_retrieval_rate"
        ]:
            best = candidate
    return best or {
        "threshold": 0.5,
        "objective": 0.0,
        "predicted_retrieval_rate": 0.0,
        "metrics": _gate_label_metrics([], []),
    }


def _gate_defense_status(
    learned: Dict[str, float],
    always_retrieve: Dict[str, float],
    *,
    quality_tolerance: float = 0.01,
    retrieval_reduction_target: float = 0.20,
) -> Dict[str, Any]:
    """Classify whether learned gate supports quality or efficiency claims."""
    metric_sources = {
        "exact_match": (
            "aligncoder_exact_match"
            if "aligncoder_exact_match" in learned
            and "aligncoder_exact_match" in always_retrieve
            else "exact_match"
        ),
        "edit_similarity": (
            "aligncoder_edit_similarity"
            if "aligncoder_edit_similarity" in learned
            and "aligncoder_edit_similarity" in always_retrieve
            else "edit_similarity"
        ),
        "identifier_f1": (
            "aligncoder_identifier_f1"
            if "aligncoder_identifier_f1" in learned
            and "aligncoder_identifier_f1" in always_retrieve
            else "identifier_f1"
        ),
    }
    deltas = {
        metric: learned.get(source, 0.0) - always_retrieve.get(source, 0.0)
        for metric, source in metric_sources.items()
    }
    retrieval_reduction = always_retrieve.get("retrieval_rate", 0.0) - learned.get(
        "retrieval_rate", 0.0
    )
    quality_win = all(delta >= 0.0 for delta in deltas.values()) and any(
        delta > 0.0 for delta in deltas.values()
    )
    eps = 1e-12
    no_quality_loss = all(
        delta >= -quality_tolerance - eps for delta in deltas.values()
    )
    compute_win = (
        no_quality_loss
        and retrieval_reduction >= retrieval_reduction_target - eps
    )
    if quality_win:
        status = "adaptive_quality_improvement"
    elif compute_win:
        status = "compute_reduction_without_quality_loss"
    else:
        status = "gate_claim_not_supported"
    return {
        "status": status,
        "quality_deltas": deltas,
        "metric_sources": metric_sources,
        "retrieval_rate_reduction": retrieval_reduction,
        "quality_tolerance": quality_tolerance,
        "retrieval_reduction_target": retrieval_reduction_target,
    }


# ── Preference Pair ───────────────────────────────────────────────────────────


@dataclass
class PreferencePair:
    """A single DPO training pair."""

    query: str  # retrieval query
    chosen_chunks: List[CodeChunk]
    rejected_chunks: List[CodeChunk]
    chosen_is_stop: bool = False
    rejected_is_stop: bool = False
    chosen_nll: float = 0.0
    rejected_nll: float = 0.0
    chosen_utility: float = 0.0
    rejected_utility: float = 0.0
    chosen_strategy: str = ""
    rejected_strategy: str = ""
    gate_query: str = ""


@dataclass
class GateTrainingExample:
    """Retrieve/skip supervision from maximum context utility."""

    query: str
    retrieve_is_better: bool
    max_utility: float
    best_strategy: str = ""
    adjusted_utility: float = 0.0
    context_cost: float = 0.0
    features: Tuple[float, ...] = field(default_factory=tuple)


@dataclass
class PreferenceData:
    """Preference pairs plus gate supervision and analysis counters."""

    pairs: List[PreferencePair] = field(default_factory=list)
    lipo_groups: List["LipoGroup"] = field(default_factory=list)
    gate_examples: List[GateTrainingExample] = field(default_factory=list)
    pair_type_counts: Dict[str, int] = field(default_factory=dict)
    strategy_counts: Dict[str, int] = field(default_factory=dict)
    gate_positive_count: int = 0
    gate_negative_count: int = 0
    mean_max_utility: float = 0.0
    mean_gate_adjusted_utility: float = 0.0
    mean_gate_context_cost: float = 0.0


@dataclass
class LipoGroup:
    """A listwise training example for LiPO.

    Contains all scored candidates for one query, preserving
    the full utility spectrum for soft-label distribution.
    """

    query: str
    candidate_chunks: List[List[CodeChunk]]
    utilities: List[float]
    strategy_names: List[str]


# ── NeuralCoTrainer ───────────────────────────────────────────────────────────


class NeuralCoTrainer:
    """7-phase co-training pipeline for Co-Retrieval."""

    def __init__(self, config: NeuralCoTrainingConfig) -> None:
        self.config = config
        self._apply_experiment_defaults()
        self._validate_config()
        self.rng = random.Random(config.random_seed)
        torch.manual_seed(config.random_seed)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        gen_dtype = dtype_map.get(config.generator_dtype, torch.float16)

        # Components.  Evaluation can move retrieval off the generation GPU;
        # training keeps the historical single-device behavior by default.
        retriever_device = config.retriever_device or config.device
        logger.info(
            "Init DenseRetriever (%s) on %s…",
            config.encoder_name,
            retriever_device,
        )
        self.retriever = DenseRetriever(
            model_name=config.encoder_name,
            max_length=config.encoder_max_length,
            device=retriever_device,
            encode_batch_size=config.batch_encode_size,
        )
        self.retriever.save_initial_copy()  # frozen pretrained encoder baseline

        logger.info("Init NeuralGate…")
        self.gate = NeuralGate(
            input_dim=self.retriever.hidden_size,
            hidden_dim=config.gate_hidden_dim,
            entropy_weight=config.gate_entropy_weight,
            feature_dim=(
                len(GATE_FEATURE_NAMES)
                if config.gate_use_retrieval_features
                else 0
            ),
        ).to(retriever_device)

        logger.info("Init SoftPromptLLM (%s)…", config.generator_name)
        self.generator = SoftPromptLLM(
            model_name=config.generator_name,
            num_prompt_tokens=config.num_prompt_tokens,
            max_context_tokens=config.max_context_tokens,
            device=config.device,
            dtype=gen_dtype,
        )
        assert_frozen = getattr(getattr(self, "generator", None), "assert_backbone_frozen", None)
        if callable(assert_frozen):
            assert_frozen()
        self.use_adapter = config.adapter_type == "soft_prompt"
        if config.adapter_type not in {"soft_prompt", "none"}:
            raise ValueError(
                f"Unsupported adapter_type={config.adapter_type!r}; "
                "supported values are 'soft_prompt' and 'none'."
            )
        self.intent_sketcher = IntentSketcher()
        self.query_enhancer = CostAwareQueryEnhancer(
            self.intent_sketcher,
            entropy_threshold=config.query_entropy_threshold,
        )
        self._query_enhancement_stats: Dict[str, float] = {
            "queries": 0.0,
            "entropy_sum": 0.0,
            "entropy_count": 0.0,
            "sampling_count": 0.0,
            "draft_count": 0.0,
            "changed_count": 0.0,
        }
        self.utility_scorer = ContextUtilityScorer(
            self.generator,
            micro_batch_size=config.utility_score_microbatch_size,
        )

        # FAISS cache
        self.embedding_cache = EmbeddingCache(dim=self.retriever.hidden_size)

        # Optimisers
        self.retriever_opt = AdamW(
            self.retriever.parameters(), lr=config.retriever_lr
        )
        self.gate_opt = AdamW(self.gate.parameters(), lr=config.gate_lr)
        self.prompt_opt = AdamW(
            [self.generator.prompt_embeddings], lr=config.soft_prompt_lr
        )

        if torch.cuda.is_available() and str(config.device).startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Global chunks
        self._chunks: List[CodeChunk] = []
        self._chunk_map: Dict[str, CodeChunk] = {}

    def _validate_config(self) -> None:
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.config.preference_pool_top_k <= 0:
            raise ValueError("preference_pool_top_k must be positive")
        if self.config.batch_encode_size <= 0:
            raise ValueError("batch_encode_size must be positive")
        if self.config.utility_score_microbatch_size <= 0:
            raise ValueError("utility_score_microbatch_size must be positive")
        if self.config.retriever_loss not in {"lipo", "dpo"}:
            raise ValueError("retriever_loss must be either 'lipo' or 'dpo'")
        if self.config.lipo_tau <= 0:
            raise ValueError("lipo_tau must be greater than zero")
        if self.config.gate_context_cost_weight < 0:
            raise ValueError("gate_context_cost_weight must be non-negative")
        if self.config.gate_context_cost_token_unit <= 0:
            raise ValueError("gate_context_cost_token_unit must be positive")
        if not 0.0 <= self.config.gate_decision_threshold <= 1.0:
            raise ValueError("gate_decision_threshold must be in [0, 1]")
        if self.config.gate_calibration_samples < 0:
            raise ValueError("gate_calibration_samples must be non-negative")
        if self.config.gate_calibration_retrieval_penalty < 0:
            raise ValueError("gate_calibration_retrieval_penalty must be non-negative")
        if self.config.intent_mode not in {"raw", "static", "cost_aware"}:
            raise ValueError(
                "intent_mode must be one of 'raw', 'static', or 'cost_aware'"
            )
        if not 0.0 <= self.config.query_entropy_threshold <= 1.0:
            raise ValueError("query_entropy_threshold must be in [0, 1]")
        if self.config.query_num_drafts < 1:
            raise ValueError("query_num_drafts must be at least one")
        if self.config.query_draft_max_tokens < 1:
            raise ValueError("query_draft_max_tokens must be at least one")
        if self.config.query_draft_temperature <= 0:
            raise ValueError("query_draft_temperature must be greater than zero")
        if not 0.0 < self.config.query_draft_top_p <= 1.0:
            raise ValueError("query_draft_top_p must be in (0, 1]")

    def _retriever_steps_per_round(self) -> int:
        value = getattr(self.config, "steps_per_round_retriever", None)
        return self.config.steps_per_round_dpo if value is None else value

    def _prompt_steps_per_round(self, samples: Sequence[TrainingSample]) -> int:
        if getattr(self.config, "epoch_budget_mode", False):
            return len(samples)
        return self.config.steps_per_round_prompt

    def _retriever_budget(
        self,
        preference_data: PreferenceData,
        epochs: int = 1,
    ) -> tuple[Optional[int], Optional[int]]:
        if not getattr(self.config, "epoch_budget_mode", False):
            steps = epochs * self._retriever_steps_per_round()
            return steps, steps

        epochs = max(0, epochs)
        if getattr(self.config, "retriever_loss", "lipo") == "lipo":
            retriever_unit = len(preference_data.lipo_groups)
        else:
            retriever_unit = len(preference_data.pairs)
        gate_unit = len(preference_data.gate_examples)
        return epochs * retriever_unit, epochs * gate_unit

    def _apply_experiment_defaults(self) -> None:
        mode = self.config.experiment_mode
        if mode == "raw_query_main":
            self.config.intent_mode = "raw"
        elif mode == "retriever_only":
            self.config.adapter_type = "none"
        elif mode == "always_retrieve":
            self.config.gate_mode = "always_retrieve"
        elif mode == "always_skip":
            self.config.gate_mode = "always_skip"
            self.config.adapter_type = "none"
        elif mode == "bm25":
            self.config.gate_mode = "always_retrieve"
            self.config.adapter_type = "none"
        elif mode == "dense_frozen":
            self.config.gate_mode = "always_retrieve"
            self.config.adapter_type = "none"

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 0 — Build Repo Index
    # ══════════════════════════════════════════════════════════════════════

    def phase0_build_index(self, chunks: Sequence[CodeChunk]) -> None:
        """AST chunks → UniXcoder embeddings → FAISS index."""
        self._chunks = list(chunks)
        self._chunk_map = {c.chunk_id: c for c in chunks}

        self.embedding_cache.build_from_chunks(
            chunks,
            encode_fn=self.retriever.encode_texts_numpy,
            batch_size=self.config.batch_encode_size,
            show_progress=True,
        )
        logger.info("Phase 0: built index with %d chunks", len(chunks))

    def phase0_register_chunks(self, chunks: Sequence[CodeChunk]) -> None:
        """Register chunks for training without pre-encoding a global index."""
        self._chunks = list(chunks)
        self._chunk_map = {c.chunk_id: c for c in chunks}
        logger.info(
            "Phase 0: registered %d chunks without global index "
            "(sample-local retrieval during train)",
            len(chunks),
        )

    def build_eval_sharded_index(
        self,
        chunks: Sequence[CodeChunk],
        directory: str,
    ) -> None:
        """Build a disk-backed CPU index for evaluation or analysis."""
        self._chunks = list(chunks)
        self._chunk_map = {c.chunk_id: c for c in chunks}
        cache = ShardedEmbeddingCache(
            dim=self.retriever.hidden_size,
            shard_size=self.config.eval_index_shard_size,
        )
        cache.build_from_chunks(
            chunks,
            encode_fn=self.retriever.encode_texts_numpy,
            batch_size=self.config.batch_encode_size,
            directory=directory,
            show_progress=True,
        )
        self.embedding_cache = cache

    def load_eval_sharded_index(self, directory: str) -> bool:
        """Load a previously built CPU mmap evaluation index."""
        cache = ShardedEmbeddingCache(
            dim=self.retriever.hidden_size,
            shard_size=self.config.eval_index_shard_size,
        )
        if not cache.load(directory):
            return False
        self.embedding_cache = cache
        return True

    def offload_retriever_for_eval(self) -> None:
        """Move retrieval and gate to CPU, leaving GPU for generation only."""
        self.retriever.to("cpu")
        self.retriever._device = "cpu"
        self.gate.to("cpu")
        self.config.retriever_device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "Evaluation memory layout: retriever/gate on CPU; generator remains on %s",
            self.config.device,
        )

    def _register_additional_chunks(
        self, samples: Sequence[TrainingSample]
    ) -> None:
        """Add chunks from a newly sampled epoch without building an index."""
        added = 0
        for sample in samples:
            for chunk in sample.candidate_chunks or []:
                if chunk.chunk_id in self._chunk_map:
                    continue
                self._chunk_map[chunk.chunk_id] = chunk
                self._chunks.append(chunk)
                added += 1
        if added:
            logger.info(
                "Registered %d additional chunks from resampled epoch "
                "(total=%d)",
                added,
                len(self._chunks),
            )

    # ── Query / strategy helpers ─────────────────────────────────────────

    def _retrieval_query(self, sample: TrainingSample) -> str:
        if self.config.intent_mode == "raw":
            return sample.left_context
        return self.intent_sketcher.build_query(sample.left_context)

    def _gate_query(self, sample: TrainingSample) -> str:
        """Use the same cheap query representation at gate train/inference."""
        return self._retrieval_query(sample)

    def _gate_supervision_score(self, scored: Sequence[Any]) -> Optional[Any]:
        """Select only the retrieval strategy deployed by this experiment."""
        strategy = self._inference_strategy_name()
        self._assert_inference_safe_strategy(strategy, mode="gate_supervision")
        return next(
            (score for score in scored if score.name == strategy and not score.is_stop),
            None,
        )

    def _inference_strategy_name(self) -> str:
        mode = self.config.experiment_mode
        if mode == "bm25":
            return "bm25"
        if mode == "dense_frozen":
            return "dense_frozen"
        if mode in {
            "intent_main",
            "raw_query_main",
            "retriever_only",
            "always_retrieve",
            "always_skip",
            "sequential_adapter_first",
            "sequential_retriever_first",
        }:
            return "current"
        return mode

    def _assert_inference_safe_strategy(
        self,
        strategy: Optional[str] = None,
        *,
        mode: str = "inference",
    ) -> None:
        selected = strategy or self._inference_strategy_name()
        if selected not in INFERENCE_SAFE_STRATEGIES:
            raise RuntimeError(
                f"Unsafe retrieval strategy {selected!r} in {mode}; "
                f"allowed strategies are {sorted(INFERENCE_SAFE_STRATEGIES)}"
            )

    def _encode_query_numpy(self, query: str) -> np.ndarray:
        with torch.no_grad():
            vec = self.retriever.encode_query(query).detach().cpu().numpy()
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def _retrieve_current(
        self,
        query: str,
        candidates: Sequence[CodeChunk],
        top_k: Optional[int] = None,
    ) -> List[CodeChunk]:
        """Retrieve with the refreshed Phase 0/4 embedding cache when possible."""
        chunk_list = list(candidates)
        k = self.config.top_k if top_k is None else top_k
        if not chunk_list:
            return []
        if self.embedding_cache.is_empty or any(
            chunk.chunk_id not in self.embedding_cache.chunk_map
            for chunk in chunk_list
        ):
            return self.retriever.retrieve_chunks(query, chunk_list, top_k=k)

        query_vec = self._encode_query_numpy(query)
        chunk_ids = [chunk.chunk_id for chunk in chunk_list]
        vectors = self.embedding_cache.get_vectors_by_ids(chunk_ids)
        if vectors.size == 0:
            return []
        scores = vectors @ query_vec
        order = np.argsort(scores)[::-1][: min(k, len(chunk_list))]
        return [chunk_list[int(i)] for i in order]

    def _retrieve_current_with_scores(
        self,
        query: str,
        candidates: Sequence[CodeChunk],
        top_k: Optional[int] = None,
    ) -> tuple[List[CodeChunk], List[float]]:
        """Retrieve once and expose selected scores for entropy estimation."""
        chunk_list = list(candidates)
        k = self.config.top_k if top_k is None else top_k
        if not chunk_list:
            return [], []
        if not self.embedding_cache.is_empty and all(
            chunk.chunk_id in self.embedding_cache.chunk_map for chunk in chunk_list
        ):
            query_vec = self._encode_query_numpy(query)
            vectors = self.embedding_cache.get_vectors_by_ids(
                [chunk.chunk_id for chunk in chunk_list]
            )
            scores = vectors @ query_vec
            order = np.argsort(scores)[::-1][: min(k, len(chunk_list))]
            return (
                [chunk_list[int(i)] for i in order],
                [float(scores[int(i)]) for i in order],
            )

        selected = self.retriever.retrieve_chunks(query, chunk_list, top_k=k)
        try:
            with torch.no_grad():
                query_vec = self.retriever.encode_query(query)
                chunk_vecs = self.retriever.encode_chunks(selected)
                scores = (query_vec.unsqueeze(0) @ chunk_vecs.T).squeeze(0)
            return selected, [float(value) for value in scores.detach().cpu().tolist()]
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return selected, []

    def _context_cost_units(self, chunks: Sequence[CodeChunk]) -> float:
        """Normalised context cost used by the stop gate label."""
        if not chunks:
            return 0.0
        approx_tokens = sum(_approx_token_count(chunk.text) for chunk in chunks)
        max_context_tokens = getattr(self.config, "max_context_tokens", 4096)
        cost_token_unit = getattr(self.config, "gate_context_cost_token_unit", 512)
        capped_tokens = min(approx_tokens, max(1, max_context_tokens))
        return capped_tokens / max(1, cost_token_unit)

    def _gate_adjusted_utility(self, score: Any) -> float:
        """Utility after charging a small retrieval/context cost."""
        if score is None or getattr(score, "is_stop", False):
            return 0.0
        context_cost = self._context_cost_units(getattr(score, "chunks", []))
        cost_weight = getattr(self.config, "gate_context_cost_weight", 0.0)
        return float(score.utility) - (
            cost_weight * context_cost
        )

    def _identifier_overlap_ratio(
        self,
        query: str,
        chunks: Sequence[CodeChunk],
    ) -> float:
        query_ids = set(_IDENTIFIER_RE.findall(query))
        if not query_ids or not chunks:
            return 0.0
        chunk_ids: set[str] = set()
        for chunk in chunks:
            chunk_ids.update(chunk.defined_symbols)
            chunk_ids.update(chunk.used_symbols)
            chunk_ids.update(chunk.call_names)
            chunk_ids.update(chunk.method_names)
        return len(query_ids & chunk_ids) / max(1, len(query_ids))

    def _gate_features(
        self,
        sample: TrainingSample,
        candidates: Sequence[CodeChunk],
        *,
        retrieval_query: Optional[str] = None,
        current_context: Optional[List[CodeChunk]] = None,
        scores: Optional[Sequence[float]] = None,
    ) -> Tuple[float, ...]:
        """Inference-safe uncertainty and cost features for the stop gate."""
        if not getattr(self.config, "gate_use_retrieval_features", True):
            return ()
        query = retrieval_query or self._gate_query(sample)
        selected = current_context
        score_values = list(scores or [])
        if selected is None:
            selected, score_values = self._retrieve_current_with_scores(
                query, candidates, top_k=getattr(self.config, "top_k", 3)
            )
        selected = list(selected or [])
        if not score_values and selected:
            score_values = [0.0 for _ in selected]

        top1 = float(score_values[0]) if score_values else 0.0
        top1_margin = (
            float(score_values[0]) - float(score_values[1])
            if len(score_values) >= 2
            else 0.0
        )
        entropy = _score_entropy(score_values) if score_values else 1.0
        score_std = float(np.std(np.asarray(score_values, dtype=np.float32))) if score_values else 0.0
        top_k = getattr(self.config, "top_k", 3)
        preference_pool_top_k = getattr(self.config, "preference_pool_top_k", 20)
        max_context_tokens = getattr(self.config, "max_context_tokens", 4096)
        cost_token_unit = getattr(self.config, "gate_context_cost_token_unit", 512)
        selected_ratio = len(selected) / max(1, top_k)
        candidate_pool_ratio = min(
            1.0,
            len(candidates) / max(1, preference_pool_top_k),
        )
        query_identifier_ratio = min(
            1.0,
            len(set(_IDENTIFIER_RE.findall(query))) / 32.0,
        )
        context_token_ratio = min(
            1.0,
            self._context_cost_units(selected)
            * cost_token_unit
            / max(1, max_context_tokens),
        )
        identifier_overlap = self._identifier_overlap_ratio(query, selected)
        return (
            top1,
            top1_margin,
            entropy,
            score_std,
            selected_ratio,
            candidate_pool_ratio,
            query_identifier_ratio,
            context_token_ratio,
            identifier_overlap,
        )

    def _gate_feature_tensor(self, features: Sequence[float]) -> Optional[torch.Tensor]:
        if not getattr(self.config, "gate_use_retrieval_features", True):
            return None
        values = list(features)
        if not values:
            values = [0.0] * len(GATE_FEATURE_NAMES)
        return torch.tensor(
            values,
            dtype=torch.float32,
            device=getattr(
                self.retriever, "_device", getattr(self.config, "device", "cpu")
            ),
        )

    def _prepare_retrieval_query(
        self,
        sample: TrainingSample,
        candidates: Sequence[CodeChunk],
    ) -> tuple[str, Optional[List[CodeChunk]]]:
        """Return query and an optional already-computed current retrieval."""
        base_query = self._retrieval_query(sample)
        if self.config.intent_mode != "cost_aware":
            return base_query, None

        cheap_context, scores = self._retrieve_current_with_scores(
            base_query, candidates, top_k=self.config.top_k
        )
        stats = self._query_enhancement_stats
        stats["queries"] += 1
        entropy = _score_entropy(scores)
        if len(scores) >= 2:
            stats["entropy_sum"] += entropy
            stats["entropy_count"] += 1
        if len(scores) < 2 or entropy < self.config.query_entropy_threshold:
            return base_query, cheap_context

        stats["sampling_count"] += 1
        try:
            drafts = self.generator.generate_drafts(
                sample.left_context,
                num_drafts=self.config.query_num_drafts,
                max_new_tokens=self.config.query_draft_max_tokens,
                temperature=self.config.query_draft_temperature,
                top_p=self.config.query_draft_top_p,
                use_soft_prompt=self.use_adapter,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Draft generation failed; using static query: %s", exc)
            return base_query, cheap_context

        stats["draft_count"] += len(drafts)
        result = self.query_enhancer.enhance(
            sample.left_context,
            retriever_scores=scores,
            draft_completions=drafts,
        )
        if not result.query_changed:
            return base_query, cheap_context
        stats["changed_count"] += 1
        return result.query, self._retrieve_current(
            result.query, candidates, top_k=self.config.top_k
        )

    def _strategy_candidates(
        self,
        sample: TrainingSample,
        candidates: Sequence[CodeChunk],
        retrieval_query: str,
        *,
        include_train_only: bool = True,
        current_context: Optional[List[CodeChunk]] = None,
    ) -> List[ContextCandidate]:
        strategies = [
            ContextCandidate("stop", [], is_stop=True, retrieval_query=retrieval_query)
        ]

        bm25_ctx = _bm25_retrieve(retrieval_query, candidates, self.config.top_k)
        strategies.append(
            ContextCandidate("bm25", bm25_ctx, retrieval_query=retrieval_query)
        )

        if self.retriever.initial_encoder is not None:
            frozen_ctx = self.retriever.retrieve_with_encoder(
                retrieval_query,
                candidates,
                top_k=self.config.top_k,
                encoder=self.retriever.initial_encoder,
            )
            strategies.append(
                ContextCandidate(
                    "dense_frozen", frozen_ctx, retrieval_query=retrieval_query
                )
            )

        current_ctx = current_context
        ranked_current: Optional[List[CodeChunk]] = None
        if current_ctx is None:
            pool_k = max(self.config.top_k, self.config.preference_pool_top_k)
            ranked_current = self._retrieve_current(
                retrieval_query, candidates, top_k=pool_k
            )
            current_ctx = ranked_current[: self.config.top_k]
        strategies.append(
            ContextCandidate("current", current_ctx, retrieval_query=retrieval_query)
        )

        if include_train_only:
            hard_neg_ctx = self._mine_hard_negatives(
                sample,
                candidates,
                retrieval_query,
                ranked_chunks=ranked_current,
            )
            if hard_neg_ctx:
                strategies.append(
                    ContextCandidate(
                        "hard_neg", hard_neg_ctx, retrieval_query=retrieval_query
                    )
                )

        if include_train_only and self.config.include_oracle_strategy:
            oracle_ctx = _oracle_chunks(sample.target, candidates, self.config.top_k)
            if oracle_ctx:
                strategies.append(
                    ContextCandidate(
                        "oracle", oracle_ctx, retrieval_query=retrieval_query
                    )
                )

        return strategies

    def _retrieve_for_generation(
        self,
        sample: TrainingSample,
        candidates: Sequence[CodeChunk],
        *,
        mode: str = "inference",
    ) -> List[CodeChunk]:
        self._assert_inference_safe_strategy(mode=mode)
        query, current_context = self._prepare_retrieval_query(sample, candidates)
        if self.config.experiment_mode == "bm25":
            return _bm25_retrieve(query, candidates, self.config.top_k)
        if (
            self.config.experiment_mode == "dense_frozen"
            and self.retriever.initial_encoder is not None
        ):
            return self.retriever.retrieve_with_encoder(
                query,
                candidates,
                top_k=self.config.top_k,
                encoder=self.retriever.initial_encoder,
            )
        if current_context is not None:
            return current_context
        return self._retrieve_current(query, candidates, top_k=self.config.top_k)

    def _should_retrieve(
        self,
        sample: TrainingSample,
        candidates: Sequence[CodeChunk],
    ) -> bool:
        if self.config.gate_mode == "always_retrieve":
            return True
        if self.config.gate_mode == "always_skip":
            return False
        query = self._gate_query(sample)
        if self.config.gate_mode == "rule":
            if not sample.left_context.strip():
                return False
            return "." in sample.left_context.rstrip().splitlines()[-1]
        with torch.no_grad():
            q_vec = self.retriever.encode_query(query)
            features = self._gate_feature_tensor(
                self._gate_features(sample, candidates, retrieval_query=query)
            )
            return self.gate.should_retrieve(
                q_vec,
                features,
                threshold=getattr(self.config, "gate_decision_threshold", 0.5),
            )

    def _gate_probability(
        self,
        sample: TrainingSample,
        candidates: Sequence[CodeChunk],
    ) -> float:
        with torch.no_grad():
            query = self._gate_query(sample)
            q_vec = self.retriever.encode_query(query)
            features = self._gate_feature_tensor(
                self._gate_features(sample, candidates, retrieval_query=query)
            )
            value = self.gate(q_vec.detach(), features)
        return float(value.view(-1)[0].item())

    def _utility_gate_label(
        self,
        sample: TrainingSample,
        candidates: Sequence[CodeChunk],
    ) -> tuple[bool, float]:
        """Build an analysis label without target-aware retrieval strategies."""
        if not candidates:
            return False, 0.0
        retrieval_query, current_context = self._prepare_retrieval_query(
            sample, candidates
        )
        strategies = self._strategy_candidates(
            sample,
            candidates,
            retrieval_query,
            include_train_only=False,
            current_context=current_context,
        )
        with torch.no_grad():
            scored = self.utility_scorer.score(
                sample.left_context,
                sample.target,
                strategies,
                use_adapter=self.use_adapter,
            )
        best_retrieve = self._gate_supervision_score(scored)
        max_utility = best_retrieve.utility if best_retrieve else 0.0
        adjusted_utility = self._gate_adjusted_utility(best_retrieve)
        return adjusted_utility > self.config.utility_margin, max_utility

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 1 — Warm-up Soft Prompt
    # ══════════════════════════════════════════════════════════════════════

    def phase1_warmup_soft_prompt(
        self,
        samples: Sequence[TrainingSample],
        steps: Optional[int] = None,
        context_mode: str = "mixed",
    ) -> Dict[str, float]:
        """Train soft prompt with mixed context (CE loss).

        Context mixing: 20% no-ctx, 50% oracle, 30% noisy/retrieved.
        Freeze: LLM, Retriever, Gate.  Update: Soft Prompt only.
        """
        assert_frozen = getattr(getattr(self, "generator", None), "assert_backbone_frozen", None)
        if callable(assert_frozen):
            assert_frozen()
        if not self.use_adapter:
            return {"phase1_loss": 0.0}

        n_steps = self.config.warmup_steps if steps is None else steps
        if n_steps <= 0:
            return {"phase1_loss": 0.0}

        logger.info("Phase 1: Warm-up Soft Prompt (%d steps)…", n_steps)
        total_loss = 0.0
        sample_schedule = _scheduled_training_items(samples, n_steps, self.rng)
        if not sample_schedule:
            return {"phase1_loss": 0.0}

        no_ctx_ratio = self.config.warmup_no_context_ratio
        oracle_ratio = self.config.warmup_oracle_ratio

        for i, sample in enumerate(sample_schedule):
            candidates = list(sample.candidate_chunks or self._chunks)
            r = self.rng.random()

            if context_mode == "retriever":
                ctx_chunks = (
                    self._retrieve_for_generation(
                        sample, candidates, mode="adapter_training"
                    )
                    if candidates
                    else []
                )
            elif r < no_ctx_ratio:
                # 20% — No context
                ctx_chunks: List[CodeChunk] = []
            elif r < no_ctx_ratio + oracle_ratio:
                # 50% — Oracle context
                ctx_chunks = _oracle_chunks(
                    sample.target, candidates, self.config.top_k
                )
            else:
                # 30% — Noisy / retrieved context
                if candidates:
                    k = min(self.config.top_k, len(candidates))
                    ctx_chunks = self.rng.sample(candidates, k)
                else:
                    ctx_chunks = []

            self.prompt_opt.zero_grad()
            loss = self.generator.generation_loss(
                left_context=sample.left_context,
                target=sample.target,
                retrieved_chunks=ctx_chunks if ctx_chunks else None,
            )

            if loss is not None and not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [self.generator.prompt_embeddings],
                    self.config.grad_clip_norm,
                )
                self.prompt_opt.step()
                total_loss += loss.item()

            if (i + 1) % 50 == 0:
                logger.info(
                    "  Phase 1 step %d/%d  loss=%.4f",
                    i + 1, n_steps, total_loss / (i + 1),
                )

        avg = total_loss / max(1, n_steps)
        logger.info("Phase 1 done: avg_loss=%.4f", avg)
        return {"phase1_loss": avg}

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 2 — Build Preference Data
    # ══════════════════════════════════════════════════════════════════════

    def phase2_build_preference_data(
        self,
        samples: Sequence[TrainingSample],
    ) -> PreferenceData:
        """Create DPO pairs using teacher-forcing NLL (no decoding).

        Strategies are ranked by context utility: NLL(stop) - NLL(context).
        """
        assert_frozen = getattr(getattr(self, "generator", None), "assert_backbone_frozen", None)
        if callable(assert_frozen):
            assert_frozen()
        logger.info("Phase 2: Building preference data (%d samples)…", len(samples))
        pairs: List[PreferencePair] = []
        lipo_groups: List[LipoGroup] = []
        gate_examples: List[GateTrainingExample] = []
        pair_type_counter: Counter[str] = Counter()
        strategy_counter: Counter[str] = Counter()
        gate_positive_count = 0
        total_max_utility = 0.0
        total_gate_adjusted_utility = 0.0
        total_gate_context_cost = 0.0

        for i, sample in enumerate(samples):
            candidates = list(sample.candidate_chunks or self._chunks)
            if not candidates:
                continue

            retrieval_query, current_context = self._prepare_retrieval_query(
                sample, candidates
            )
            strategies = self._strategy_candidates(
                sample,
                candidates,
                retrieval_query,
                current_context=current_context,
            )
            with torch.inference_mode():
                scored = self.utility_scorer.score(
                    sample.left_context,
                    sample.target,
                    strategies,
                    use_adapter=self.use_adapter,
                )

            # ── Gate label: ONLY from inference-safe strategies ────────────
            # Gate must answer "will the DEPLOYED retriever help?",
            # not "does a good context EXIST somewhere in the repo?".
            # Oracle/hard_neg must NOT influence gate labels.
            best_gate_retrieve = self._gate_supervision_score(scored)
            gate_adjusted_utility = self._gate_adjusted_utility(best_gate_retrieve)
            gate_context_cost = (
                self._context_cost_units(best_gate_retrieve.chunks)
                if best_gate_retrieve
                else 0.0
            )
            retrieve_is_better = gate_adjusted_utility > self.config.utility_margin
            gate_max_utility = (
                best_gate_retrieve.utility if best_gate_retrieve else 0.0
            )
            gate_positive_count += int(retrieve_is_better)
            total_max_utility += gate_max_utility
            total_gate_adjusted_utility += gate_adjusted_utility
            total_gate_context_cost += gate_context_cost
            gate_examples.append(
                GateTrainingExample(
                    query=self._gate_query(sample),
                    retrieve_is_better=retrieve_is_better,
                    max_utility=gate_max_utility,
                    best_strategy=(
                        best_gate_retrieve.name if best_gate_retrieve else "stop"
                    ),
                    adjusted_utility=gate_adjusted_utility,
                    context_cost=gate_context_cost,
                    features=self._gate_features(
                        sample,
                        candidates,
                        retrieval_query=self._gate_query(sample),
                    ),
                )
            )
            for score in scored:
                strategy_counter[score.name] += 1

            # ── LiPO: collect listwise group with full utility spectrum ──
            unique_scored = []
            seen_contexts: set[tuple[bool, tuple[str, ...]]] = set()
            for score in scored:
                context_key = (
                    bool(score.is_stop),
                    tuple(sorted(chunk.chunk_id for chunk in score.chunks)),
                )
                if context_key in seen_contexts:
                    continue
                seen_contexts.add(context_key)
                unique_scored.append(score)
            has_stop = any(score.is_stop for score in unique_scored)
            has_context = any(not score.is_stop for score in unique_scored)
            if has_stop and has_context:
                lipo_groups.append(
                    LipoGroup(
                        query=retrieval_query,
                        candidate_chunks=[s.chunks for s in unique_scored],
                        utilities=[s.utility for s in unique_scored],
                        strategy_names=[s.name for s in unique_scored],
                    )
                )

            # ── DPO: form pairwise pairs (ablation / backward compat) ──
            pairs_added = 0
            for chosen_idx, chosen in enumerate(scored):
                for rejected in scored[chosen_idx + 1 :]:
                    if (
                        chosen.utility - rejected.utility
                        < self.config.preference_margin
                    ):
                        continue
                    pairs.append(
                        PreferencePair(
                            query=retrieval_query,
                            chosen_chunks=chosen.chunks,
                            rejected_chunks=rejected.chunks,
                            chosen_is_stop=chosen.is_stop,
                            rejected_is_stop=rejected.is_stop,
                            chosen_nll=chosen.nll,
                            rejected_nll=rejected.nll,
                            chosen_utility=chosen.utility,
                            rejected_utility=rejected.utility,
                            chosen_strategy=chosen.name,
                            rejected_strategy=rejected.name,
                            gate_query=sample.left_context,
                        )
                    )
                    pair_type_counter[f"{chosen.name}>{rejected.name}"] += 1
                    pairs_added += 1
                    if pairs_added >= self.config.max_pairs_per_sample:
                        break
                if pairs_added >= self.config.max_pairs_per_sample:
                    break

            if (i + 1) % 100 == 0:
                logger.info(
                    "  Phase 2: %d/%d samples, %d pairs so far",
                    i + 1, len(samples), len(pairs),
                )

        logger.info(
            "Phase 2 done: %d LiPO groups, %d DPO pairs, %d gate labels "
            "from %d samples (%.1f%%)",
            len(lipo_groups), len(pairs), len(gate_examples), len(samples),
            100 * len(pairs) / max(1, len(samples)),
        )
        return PreferenceData(
            pairs=pairs,
            lipo_groups=lipo_groups,
            gate_examples=gate_examples,
            pair_type_counts=dict(pair_type_counter),
            strategy_counts=dict(strategy_counter),
            gate_positive_count=gate_positive_count,
            gate_negative_count=len(gate_examples) - gate_positive_count,
            mean_max_utility=total_max_utility / max(1, len(gate_examples)),
            mean_gate_adjusted_utility=total_gate_adjusted_utility
            / max(1, len(gate_examples)),
            mean_gate_context_cost=total_gate_context_cost
            / max(1, len(gate_examples)),
        )

    def _mine_hard_negatives(
        self,
        sample: TrainingSample,
        candidates: Sequence[CodeChunk],
        retrieval_query: str,
        ranked_chunks: Optional[Sequence[CodeChunk]] = None,
    ) -> List[CodeChunk]:
        """Find chunks that the retriever ranks high but are not relevant."""
        target_symbols = identifier_set(sample.target)
        if not target_symbols:
            return []

        top_ranked = list(ranked_chunks) if ranked_chunks is not None else self._retrieve_current(
            retrieval_query,
            candidates,
            top_k=self.config.preference_pool_top_k,
        )

        hard_negs: List[CodeChunk] = []
        for chunk in top_ranked:
            chunk_symbols = set(
                chunk.defined_symbols + chunk.call_names + chunk.method_names
            )
            if not (chunk_symbols & target_symbols):
                hard_negs.append(chunk)
            if len(hard_negs) >= self.config.num_hard_negatives:
                break

        return hard_negs[: self.config.top_k]

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 3 — Retriever Training (LiPO or DPO) + Gate
    # ══════════════════════════════════════════════════════════════════════

    def phase3_retriever_training(
        self,
        preference_data: PreferenceData,
        steps: Optional[int] = None,
        gate_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Train retriever (LiPO or DPO) and gate (BCE from utility labels).

        Freeze: LLM, Soft Prompt.  Update: Retriever, Gate.
        Default loss is LiPO (listwise); DPO (pairwise) is available as ablation.
        """
        assert_frozen = getattr(getattr(self, "generator", None), "assert_backbone_frozen", None)
        if callable(assert_frozen):
            assert_frozen()
        use_lipo = self.config.retriever_loss == "lipo"
        lipo_groups = preference_data.lipo_groups
        pairs = preference_data.pairs
        gate_examples = preference_data.gate_examples

        if use_lipo:
            retriever_steps = (len(lipo_groups) if steps is None else max(0, steps))
            loss_name = "LiPO"
        else:
            retriever_steps = (len(pairs) if steps is None else max(0, steps))
            loss_name = "DPO"

        if gate_steps is None:
            gate_step_budget = len(gate_examples) if steps is None else max(0, steps)
        else:
            gate_step_budget = max(0, gate_steps)
        if use_lipo and not lipo_groups:
            retriever_steps = 0
        if not use_lipo and not pairs:
            retriever_steps = 0
        if not gate_examples:
            gate_step_budget = 0
        if retriever_steps <= 0 and gate_step_budget <= 0:
            return {
                "phase3_retriever_loss": 0.0,
                "phase3_retriever_loss_type": loss_name,
                "phase3_gate_loss": 0.0,
                "phase3_retriever_steps": 0,
                "phase3_gate_steps": 0,
            }

        logger.info(
            "Phase 3: %s training (%d retriever steps, %d gate labels)…",
            loss_name,
            retriever_steps,
            gate_step_budget,
        )

        # Only DPO needs a frozen reference encoder. Copying a large encoder for
        # LiPO would waste substantial host/GPU memory.
        if not use_lipo:
            self.retriever.refresh_reference()

        total_retriever_loss = 0.0
        total_gate_loss = 0.0

        # ── Retriever training ─────────────────────────────────────────
        if use_lipo:
            group_schedule = _scheduled_training_items(
                lipo_groups, retriever_steps, self.rng
            )
            for i, group in enumerate(group_schedule):
                self.retriever_opt.zero_grad()
                loss = self.retriever.lipo_loss(
                    query_text=group.query,
                    candidate_chunks_list=group.candidate_chunks,
                    utilities=group.utilities,
                    tau=self.config.lipo_tau,
                )
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.retriever.parameters(),
                        self.config.grad_clip_norm * 5,
                    )
                    self.retriever_opt.step()
                    total_retriever_loss += loss.item()

                if (i + 1) % 50 == 0:
                    logger.info(
                        "  Phase 3 LiPO step %d/%d  loss=%.4f",
                        i + 1,
                        retriever_steps,
                        total_retriever_loss / (i + 1),
                    )
        else:
            pair_schedule = _scheduled_training_items(
                pairs, retriever_steps, self.rng
            )
            for i, pair in enumerate(pair_schedule):
                self.retriever_opt.zero_grad()
                loss = self.retriever.dpo_loss(
                    query_text=pair.query,
                    chosen_chunks=pair.chosen_chunks,
                    rejected_chunks=pair.rejected_chunks,
                    beta=self.config.dpo_beta,
                    chosen_is_stop=pair.chosen_is_stop,
                    rejected_is_stop=pair.rejected_is_stop,
                )
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.retriever.parameters(),
                        self.config.grad_clip_norm * 5,
                    )
                    self.retriever_opt.step()
                    total_retriever_loss += loss.item()

                if (i + 1) % 50 == 0:
                    logger.info(
                        "  Phase 3 DPO step %d/%d  loss=%.4f",
                        i + 1,
                        retriever_steps,
                        total_retriever_loss / (i + 1),
                    )

        # ── Gate training — BCE from cost-aware utility labels ──────────
        gate_schedule = _scheduled_training_items(
            gate_examples, gate_step_budget, self.rng
        )
        for i, example in enumerate(gate_schedule):
            with torch.no_grad():
                q_vec = self.retriever.encode_query(example.query)
            features = self._gate_feature_tensor(example.features)
            g = self.gate(q_vec.detach(), features)
            g_loss = self.gate.gate_loss(g, example.retrieve_is_better)
            if not torch.isnan(g_loss):
                self.gate_opt.zero_grad()
                g_loss.backward()
                self.gate_opt.step()
                total_gate_loss += g_loss.item()

            if (i + 1) % 50 == 0:
                logger.info(
                        "  Phase 3 gate step %d/%d  gate=%.4f",
                        i + 1,
                        gate_step_budget,
                        total_gate_loss / (i + 1),
                    )

        result = {
            "phase3_retriever_loss": total_retriever_loss / max(1, retriever_steps),
            "phase3_retriever_loss_type": loss_name,
            "phase3_gate_loss": total_gate_loss / max(1, gate_step_budget),
            "phase3_gate_labels": gate_step_budget,
            "phase3_retriever_steps": retriever_steps,
            "phase3_gate_steps": gate_step_budget,
        }
        logger.info("Phase 3 done: %s", result)
        return result

    def phase3_dpo_training(
        self,
        preference_data: PreferenceData,
        steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Deprecated alias for :meth:`phase3_retriever_training`."""
        return self.phase3_retriever_training(preference_data, steps=steps)

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 4 — Refresh Index
    # ══════════════════════════════════════════════════════════════════════

    def phase4_refresh_index(self) -> None:
        """Re-embed all chunks with updated retriever → rebuild FAISS."""
        if not self._chunks:
            return
        if not getattr(self.config, "refresh_train_index", False):
            # A global cache built before retriever updates is no longer valid.
            # Keep sample-local retrieval on the live encoder instead of
            # silently serving rankings from the pre-training weights.
            if not self.embedding_cache.is_empty:
                self.embedding_cache.clear()
            logger.info(
                "Phase 4: skipped global index refresh; "
                "sample-local retrieval will use current retriever weights"
            )
            return
        logger.info("Phase 4: Refreshing FAISS index (%d chunks)…", len(self._chunks))
        self.embedding_cache.build_from_chunks(
            self._chunks,
            encode_fn=self.retriever.encode_texts_numpy,
            batch_size=self.config.batch_encode_size,
            show_progress=True,
        )
        logger.info("Phase 4 done: index rebuilt")

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 5 — Alternating Co-training Rounds
    # ══════════════════════════════════════════════════════════════════════

    def phase5_co_training(
        self,
        samples: Sequence[TrainingSample],
        num_rounds: Optional[int] = None,
        sample_provider: Optional[
            Callable[[int], Sequence[TrainingSample]]
        ] = None,
    ) -> List[Dict[str, Any]]:
        """Alternating epochs/rounds: P1 → P2 → P3 → P4, repeated.

        ``sample_provider`` optionally supplies a fresh training subset for
        each epoch.  This is deliberately handled here, before Phase 1/2, so
        preference data and all per-epoch budgets are computed from the same
        sampled subset.
        """
        rounds = self.config.num_rounds if num_rounds is None else num_rounds
        epoch_budget = getattr(self.config, "epoch_budget_mode", False)
        unit_label = "epoch" if epoch_budget else "round"
        if epoch_budget and num_rounds is None:
            rounds = self.config.train_epochs
        round_history: List[Dict[str, Any]] = []

        for r in range(1, rounds + 1):
            logger.info("═══ Co-training %s %d/%d ═══", unit_label, r, rounds)
            epoch_samples = (
                list(sample_provider(r))
                if sample_provider is not None
                else list(samples)
            )
            if not epoch_samples:
                raise RuntimeError(
                    f"No training samples were produced for epoch {r}."
                )
            if sample_provider is not None:
                self._register_additional_chunks(epoch_samples)
                logger.info(
                    "Epoch %d: using freshly resampled %d training samples",
                    r,
                    len(epoch_samples),
                )

            prompt_steps = self._prompt_steps_per_round(epoch_samples)

            # P1: Train soft prompt
            p1 = self.phase1_warmup_soft_prompt(
                epoch_samples, steps=prompt_steps
            )

            # P2: Build preference data
            preference_data = self.phase2_build_preference_data(epoch_samples)

            # P3: DPO train retriever; BCE train gate from utility labels
            retriever_steps, gate_steps = self._retriever_budget(preference_data)
            p3 = self.phase3_retriever_training(
                preference_data,
                steps=retriever_steps,
                gate_steps=gate_steps,
            )

            # P4: Refresh index
            self.phase4_refresh_index()

            round_result = {
                "round": r,
                "epoch": r if epoch_budget else None,
                "epoch_budget_mode": epoch_budget,
                "resampled_train_each_epoch": sample_provider is not None,
                "num_train_samples": len(epoch_samples),
                "prompt_steps_budget": prompt_steps,
                "retriever_steps_budget": retriever_steps,
                "gate_steps_budget": gate_steps,
                **p1,
                "num_dpo_pairs": len(preference_data.pairs),
                "num_gate_examples": len(preference_data.gate_examples),
                "gate_positive_count": preference_data.gate_positive_count,
                "gate_positive_ratio": preference_data.gate_positive_count
                / max(1, len(preference_data.gate_examples)),
                "mean_max_utility": preference_data.mean_max_utility,
                "mean_gate_adjusted_utility": (
                    preference_data.mean_gate_adjusted_utility
                ),
                "mean_gate_context_cost": preference_data.mean_gate_context_cost,
                "pair_type_counts": preference_data.pair_type_counts,
                "strategy_counts": preference_data.strategy_counts,
                **p3,
            }
            round_history.append(round_result)
            logger.info("Co-training %s %d result: %s", unit_label, r, round_result)

        return round_history

    def phase5_sequential_adapter_first(
        self,
        samples: Sequence[TrainingSample],
    ) -> List[Dict[str, Any]]:
        """Sequential baseline: adapter first, then retriever/gate once."""
        epochs = (
            self.config.train_epochs
            if getattr(self.config, "epoch_budget_mode", False)
            else self.config.num_rounds
        )
        prompt_steps = self.config.warmup_steps + (
            epochs * self._prompt_steps_per_round(samples)
        )
        p1 = self.phase1_warmup_soft_prompt(samples, steps=prompt_steps)
        preference_data = self.phase2_build_preference_data(samples)
        retriever_steps, gate_steps = self._retriever_budget(preference_data, epochs)
        p3 = self.phase3_retriever_training(
            preference_data,
            steps=retriever_steps,
            gate_steps=gate_steps,
        )
        self.phase4_refresh_index()
        return [
            {
                "round": 1,
                "schedule": "sequential_adapter_first",
                "epoch_budget_mode": getattr(self.config, "epoch_budget_mode", False),
                "train_epochs": epochs,
                "prompt_steps_budget": prompt_steps,
                "retriever_steps_budget": retriever_steps,
                "gate_steps_budget": gate_steps,
                "preference_data_builds": 1,
                "index_refreshes": 1,
                **p1,
                "num_dpo_pairs": len(preference_data.pairs),
                "num_gate_examples": len(preference_data.gate_examples),
                "gate_positive_count": preference_data.gate_positive_count,
                "gate_positive_ratio": preference_data.gate_positive_count
                / max(1, len(preference_data.gate_examples)),
                "mean_max_utility": preference_data.mean_max_utility,
                "mean_gate_adjusted_utility": (
                    preference_data.mean_gate_adjusted_utility
                ),
                "mean_gate_context_cost": preference_data.mean_gate_context_cost,
                "pair_type_counts": preference_data.pair_type_counts,
                "strategy_counts": preference_data.strategy_counts,
                **p3,
            }
        ]

    def phase5_sequential_retriever_first(
        self,
        samples: Sequence[TrainingSample],
    ) -> List[Dict[str, Any]]:
        """Sequential baseline: retriever/gate first, then adapter once."""
        epochs = (
            self.config.train_epochs
            if getattr(self.config, "epoch_budget_mode", False)
            else self.config.num_rounds
        )
        prompt_steps = self.config.warmup_steps + (
            epochs * self._prompt_steps_per_round(samples)
        )

        original_use_adapter = self.use_adapter
        self.use_adapter = False
        try:
            preference_data = self.phase2_build_preference_data(samples)
            retriever_steps, gate_steps = self._retriever_budget(
                preference_data, epochs
            )
            p3 = self.phase3_retriever_training(
                preference_data,
                steps=retriever_steps,
                gate_steps=gate_steps,
            )
        finally:
            self.use_adapter = original_use_adapter

        self.phase4_refresh_index()
        p1 = self.phase1_warmup_soft_prompt(
            samples,
            steps=prompt_steps,
            context_mode="retriever",
        )
        return [
            {
                "round": 1,
                "schedule": "sequential_retriever_first",
                "epoch_budget_mode": getattr(self.config, "epoch_budget_mode", False),
                "train_epochs": epochs,
                "prompt_steps_budget": prompt_steps,
                "retriever_steps_budget": retriever_steps,
                "gate_steps_budget": gate_steps,
                "preference_data_builds": 1,
                "index_refreshes": 1,
                **p1,
                "num_dpo_pairs": len(preference_data.pairs),
                "num_gate_examples": len(preference_data.gate_examples),
                "gate_positive_count": preference_data.gate_positive_count,
                "gate_positive_ratio": preference_data.gate_positive_count
                / max(1, len(preference_data.gate_examples)),
                "mean_max_utility": preference_data.mean_max_utility,
                "mean_gate_adjusted_utility": (
                    preference_data.mean_gate_adjusted_utility
                ),
                "mean_gate_context_cost": preference_data.mean_gate_context_cost,
                "pair_type_counts": preference_data.pair_type_counts,
                "strategy_counts": preference_data.strategy_counts,
                **p3,
            }
        ]

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 6 — Final Evaluation
    # ══════════════════════════════════════════════════════════════════════

    def _reset_eval_memory_stats(self) -> None:
        if torch.cuda.is_available() and str(self.config.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

    def _log_eval_memory(self, sample_index: int) -> None:
        if not (
            torch.cuda.is_available()
            and str(self.config.device).startswith("cuda")
        ):
            return
        # Periodic cache release prevents allocator fragmentation after long
        # prefill/NLL passes without synchronizing on every sample.
        if sample_index % 8 == 0:
            torch.cuda.empty_cache()
        if sample_index % 25 == 0:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(
                "Eval memory sample=%d allocated=%.2fGiB reserved=%.2fGiB "
                "peak=%.2fGiB context=%d attention=%s",
                sample_index,
                allocated,
                reserved,
                peak,
                self.config.max_context_tokens,
                getattr(self.generator, "attention_implementation", "unknown"),
            )

    def phase6_evaluate(
        self,
        test_samples: Sequence[TrainingSample],
        *,
        include_analysis: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate retrieval + generation + end-to-end."""
        self._assert_inference_safe_strategy(mode="eval")
        logger.info("Phase 6: Evaluating on %d samples…", len(test_samples))
        self._reset_eval_memory_stats()

        total_em = 0.0
        total_edit_sim = 0.0
        total_id_f1 = 0.0
        total_retrieve_count = 0
        total_nll_with = 0.0
        total_nll_without = 0.0
        nll_improvements: List[float] = []
        edit_values: List[float] = []
        id_f1_values: List[float] = []
        gate_labels: List[bool] = []
        gate_probs: List[float] = []
        aligncoder_records: List[Dict[str, Any]] = []
        query_stats = {key: 0.0 for key in self._query_enhancement_stats}
        n = 0

        for idx, sample in enumerate(test_samples):
            candidates = list(sample.candidate_chunks or self._chunks)
            should_retrieve = self._should_retrieve(sample, candidates)
            query_stats_before = dict(self._query_enhancement_stats)

            if should_retrieve and candidates:
                ctx = self._retrieve_for_generation(sample, candidates, mode="eval")
                pred = self.generator.generate(
                    sample.left_context,
                    max_new_tokens=self.config.max_new_tokens,
                    retrieved_chunks=ctx,
                    use_soft_prompt=self.use_adapter,
                )
                total_retrieve_count += 1
            else:
                ctx = []
                # Keep adapter treatment consistent between retrieve/skip
                # to avoid confounding context effect with adapter effect.
                pred = self.generator.generate(
                    sample.left_context,
                    max_new_tokens=self.config.max_new_tokens,
                    use_soft_prompt=self.use_adapter,
                )

            for key, value in self._query_enhancement_stats.items():
                query_stats[key] += value - query_stats_before[key]

            em = exact_match(pred, sample.target)
            edit = edit_similarity(pred, sample.target)
            id_f1 = identifier_f1(pred, sample.target)
            total_em += em
            total_edit_sim += edit
            total_id_f1 += id_f1
            edit_values.append(edit)
            id_f1_values.append(id_f1)
            raw_task_id = getattr(sample, "task_id", "")
            task_id = str(raw_task_id) if raw_task_id not in ("", None) else str(
                getattr(sample, "repo_id", "")
            )
            file_path = str(getattr(sample, "file_path", "current_file.py"))
            if not task_id:
                task_id = f"{file_path}:{idx}"
            aligncoder_records.append(
                {
                    "task_id": task_id,
                    "prompt": sample.left_context,
                    "target": sample.target,
                    "pred": pred,
                    "file_path": file_path,
                }
            )

            # NLL comparison — consistent adapter treatment
            with torch.inference_mode():
                nll_with = self.generator.teacher_forcing_nll(
                    sample.left_context, sample.target,
                    retrieved_chunks=ctx if ctx else None,
                    use_soft_prompt=self.use_adapter,
                ).item()
                nll_without = self.generator.teacher_forcing_nll(
                    sample.left_context, sample.target,
                    use_soft_prompt=self.use_adapter,
                ).item()
            total_nll_with += nll_with
            total_nll_without += nll_without
            nll_improvements.append(nll_without - nll_with)
            if include_analysis:
                label, _ = self._utility_gate_label(sample, candidates)
                gate_labels.append(label)
                gate_probs.append(self._gate_probability(sample, candidates))
            n += 1
            self._log_eval_memory(idx + 1)

        denom = max(1, n)
        nll_output_correlation = {
            "nll_vs_edit_corr": _safe_corr(nll_improvements, edit_values),
            "nll_vs_identifier_f1_corr": _safe_corr(
                nll_improvements, id_f1_values
            ),
        }
        aligncoder_metric_values = compute_aligncoder_metrics(aligncoder_records)
        aligncoder_public = {
            key: aligncoder_metric_values[key]
            for key in (
                "em",
                "es",
                "id_em",
                "id_precision",
                "id_recall",
                "id_f1",
                "total",
                "repoeval_em",
                "repoeval_es",
                "repoeval_total",
            )
        }
        metrics = {
            "exact_match": total_em / denom,
            "edit_similarity": total_edit_sim / denom,
            "identifier_f1": total_id_f1 / denom,
            "aligncoder_exact_match": aligncoder_metric_values["em"] / 100.0,
            "aligncoder_edit_similarity": aligncoder_metric_values["es"] / 100.0,
            "aligncoder_identifier_f1": aligncoder_metric_values["id_f1"] / 100.0,
            "aligncoder_metrics": aligncoder_public,
            "retrieval_rate": total_retrieve_count / denom,
            "gate_decision_threshold": getattr(
                self.config, "gate_decision_threshold", 0.5
            ),
            "avg_nll_with_retrieval": total_nll_with / denom,
            "avg_nll_without_retrieval": total_nll_without / denom,
            "nll_improvement": (total_nll_without - total_nll_with) / denom,
            "num_samples": n,
            "nll_output_correlation": nll_output_correlation,
            "gate_label_metrics": _gate_label_metrics(gate_labels, gate_probs)
            if include_analysis
            else {},
            "oracle_used_for_eval": False,
            "inference_safe_strategy_check": True,
            "query_enhancement": {
                "mean_normalized_entropy": query_stats["entropy_sum"]
                / max(1.0, query_stats["entropy_count"]),
                "sampling_rate": query_stats["sampling_count"]
                / max(1.0, query_stats["queries"]),
                "num_drafts": int(query_stats["draft_count"]),
                "query_changed_rate": query_stats["changed_count"]
                / max(1.0, query_stats["queries"]),
                "num_queries": int(query_stats["queries"]),
            },
        }
        if include_analysis:
            metrics["leave_one_out_analysis"] = self.leave_one_out_analysis(
                test_samples
            )
        logger.info(
            "Phase 6 results: %s",
            {
                k: round(v, 4) if isinstance(v, (int, float)) else v
                for k, v in metrics.items()
            },
        )
        return metrics

    def evaluate_to_files(
        self,
        test_samples: Sequence[TrainingSample],
        output_dir: str,
        *,
        include_analysis: bool = True,
    ) -> Dict[str, Any]:
        """Generate AlignCoder-style prediction files and summary metrics."""
        import json

        self._assert_inference_safe_strategy(mode="evaluate_to_files")
        skip_eval_nll = bool(getattr(self.config, "eval_skip_nll", False))
        if skip_eval_nll and include_analysis:
            raise ValueError(
                "eval_skip_nll is incompatible with include_analysis because "
                "utility/gate analysis requires teacher-forcing NLL."
            )
        if skip_eval_nll:
            logger.info(
                "Fast eval enabled: skipping per-sample teacher-forcing NLL "
                "passes; output metrics remain enabled."
            )
        os.makedirs(output_dir, exist_ok=True)
        self._reset_eval_memory_stats()
        pred_path = os.path.join(output_dir, "prediction.jsonl")
        cand_path = os.path.join(output_dir, "prediction_with_candidates.jsonl")

        total_em = 0.0
        total_edit_sim = 0.0
        total_id_f1 = 0.0
        total_retrieve_count = 0
        total_nll_with = 0.0
        total_nll_without = 0.0
        nll_improvements: List[float] = []
        edit_values: List[float] = []
        id_f1_values: List[float] = []
        gate_labels: List[bool] = []
        gate_probs: List[float] = []
        aligncoder_records: List[Dict[str, Any]] = []
        n = 0
        eval_batch_size = max(1, int(getattr(self.config, "eval_batch_size", 1)))

        with open(pred_path, "w", encoding="utf-8") as f_pred, open(
            cand_path, "w", encoding="utf-8"
        ) as f_cand:
            for batch_start in range(0, len(test_samples), eval_batch_size):
                batch_samples = list(
                    test_samples[batch_start : batch_start + eval_batch_size]
                )
                prepared: List[
                    Tuple[TrainingSample, List[CodeChunk], List[CodeChunk]]
                ] = []
                for sample in batch_samples:
                    candidates = list(sample.candidate_chunks or self._chunks)
                    ctx: List[CodeChunk] = []
                    should_retrieve = self._should_retrieve(sample, candidates)
                    if should_retrieve and candidates:
                        ctx = self._retrieve_for_generation(
                            sample, candidates, mode="evaluate_to_files"
                        )
                        total_retrieve_count += 1
                    prepared.append((sample, candidates, ctx))

                if eval_batch_size > 1 and hasattr(self.generator, "generate_batch"):
                    preds = self.generator.generate_batch(
                        [sample.left_context for sample, _, _ in prepared],
                        max_new_tokens=self.config.max_new_tokens,
                        retrieved_chunks_list=[ctx or None for _, _, ctx in prepared],
                        use_soft_prompt=self.use_adapter,
                    )
                else:
                    preds = [
                        self.generator.generate(
                            sample.left_context,
                            max_new_tokens=self.config.max_new_tokens,
                            retrieved_chunks=ctx or None,
                            use_soft_prompt=self.use_adapter,
                        )
                        for sample, _, ctx in prepared
                    ]

                for batch_offset, ((sample, candidates, ctx), pred) in enumerate(
                    zip(prepared, preds)
                ):
                    idx = batch_start + batch_offset

                    raw_task_id = getattr(sample, "task_id", "")
                    task_id = str(raw_task_id) if raw_task_id not in ("", None) else str(
                        getattr(sample, "repo_id", "")
                    )
                    file_path = str(getattr(sample, "file_path", "current_file.py"))
                    if not task_id:
                        task_id = f"{file_path}:{idx}"
                    f_pred.write(
                        json.dumps({"task_id": task_id, "pred": pred}, ensure_ascii=False)
                        + "\n"
                    )
                    aligncoder_records.append(
                        {
                            "task_id": task_id,
                            "prompt": sample.left_context,
                            "target": sample.target,
                            "pred": pred,
                            "file_path": file_path,
                        }
                    )
                    f_cand.write(
                        json.dumps(
                            {
                                "task_id": task_id,
                                "input": sample.left_context[-512:],
                                "target": sample.target,
                                "pred": pred,
                                "retrieval_used": bool(ctx),
                                "gate_probability": self._gate_probability(
                                    sample, candidates
                                ),
                                "gate_decision_threshold": getattr(
                                    self.config, "gate_decision_threshold", 0.5
                                ),
                                "candidates": [
                                    {
                                        "chunk_id": chunk.chunk_id,
                                        "file_path": chunk.file_path,
                                        "start_line": chunk.start_line,
                                        "end_line": chunk.end_line,
                                        "chunk_type": chunk.chunk_type,
                                        "defined_symbols": chunk.defined_symbols,
                                        "text": chunk.text,
                                    }
                                    for chunk in ctx
                                ],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    em = exact_match(pred, sample.target)
                    edit = edit_similarity(pred, sample.target)
                    id_f1 = identifier_f1(pred, sample.target)
                    total_em += em
                    total_edit_sim += edit
                    total_id_f1 += id_f1
                    edit_values.append(edit)
                    id_f1_values.append(id_f1)

                    if not skip_eval_nll:
                        with torch.inference_mode():
                            nll_with = self.generator.teacher_forcing_nll(
                                sample.left_context,
                                sample.target,
                                retrieved_chunks=ctx if ctx else None,
                                use_soft_prompt=self.use_adapter,
                            ).item()
                            nll_without = self.generator.teacher_forcing_nll(
                                sample.left_context,
                                sample.target,
                                retrieved_chunks=None,
                                use_soft_prompt=self.use_adapter,
                            ).item()
                        total_nll_with += nll_with
                        total_nll_without += nll_without
                        nll_improvements.append(nll_without - nll_with)
                    if include_analysis:
                        label, _ = self._utility_gate_label(sample, candidates)
                        gate_labels.append(label)
                        gate_probs.append(self._gate_probability(sample, candidates))
                    n += 1
                    self._log_eval_memory(idx + 1)

        aligncoder_metrics = write_aligncoder_metric_files(
            output_dir, aligncoder_records
        )
        denom = max(1, n)
        metrics = {
            "exact_match": total_em / denom,
            "edit_similarity": total_edit_sim / denom,
            "identifier_f1": total_id_f1 / denom,
            "aligncoder_exact_match": aligncoder_metrics["em"] / 100.0,
            "aligncoder_edit_similarity": aligncoder_metrics["es"] / 100.0,
            "aligncoder_identifier_f1": aligncoder_metrics["id_f1"] / 100.0,
            "retrieval_rate": total_retrieve_count / denom,
            "gate_decision_threshold": getattr(
                self.config, "gate_decision_threshold", 0.5
            ),
            "avg_nll_with_retrieval": (
                None if skip_eval_nll else total_nll_with / denom
            ),
            "avg_nll_without_retrieval": (
                None if skip_eval_nll else total_nll_without / denom
            ),
            "nll_improvement": (
                None
                if skip_eval_nll
                else (total_nll_without - total_nll_with) / denom
            ),
            "num_samples": n,
            "nll_output_correlation": (
                {"status": "skipped"}
                if skip_eval_nll
                else {
                    "nll_vs_edit_corr": _safe_corr(nll_improvements, edit_values),
                    "nll_vs_identifier_f1_corr": _safe_corr(
                        nll_improvements, id_f1_values
                    ),
                }
            ),
            "gate_label_metrics": _gate_label_metrics(gate_labels, gate_probs)
            if include_analysis
            else {},
            "prediction_path": pred_path,
            "prediction_with_candidates_path": cand_path,
            "aligncoder_metrics": aligncoder_metrics,
            "oracle_used_for_eval": False,
            "inference_safe_strategy_check": True,
        }
        if include_analysis:
            metrics["leave_one_out_analysis"] = self.leave_one_out_analysis(
                test_samples
            )
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
        metrics["metrics_path"] = metrics_path
        return metrics

    def leave_one_out_analysis(
        self,
        samples: Sequence[TrainingSample],
    ) -> Dict[str, Any]:
        """Analysis-only chunk contribution for top-k context sets."""
        self._assert_inference_safe_strategy(mode="leave_one_out_analysis")
        limit = max(0, self.config.leave_one_out_analysis_samples)
        if limit <= 0:
            return {"num_sets": 0, "num_chunks": 0}

        contributions: List[float] = []
        noisy_chunks = 0
        positive_chunks = 0
        num_sets = 0

        for sample in list(samples)[:limit]:
            candidates = list(sample.candidate_chunks or self._chunks)
            if not candidates:
                continue
            ctx = self._retrieve_for_generation(
                sample, candidates, mode="leave_one_out_analysis"
            )
            if len(ctx) <= 1:
                continue
            with torch.inference_mode():
                full_nll = self.generator.teacher_forcing_nll(
                    sample.left_context,
                    sample.target,
                    retrieved_chunks=ctx,
                    use_soft_prompt=self.use_adapter,
                ).item()
                num_sets += 1
                for idx in range(len(ctx)):
                    reduced = ctx[:idx] + ctx[idx + 1 :]
                    reduced_nll = self.generator.teacher_forcing_nll(
                        sample.left_context,
                        sample.target,
                        retrieved_chunks=reduced if reduced else None,
                        use_soft_prompt=self.use_adapter,
                    ).item()
                    contribution = reduced_nll - full_nll
                    contributions.append(contribution)
                    positive_chunks += int(contribution > 0.0)
                    noisy_chunks += int(contribution < 0.0)

        num_chunks = len(contributions)
        return {
            "num_sets": num_sets,
            "num_chunks": num_chunks,
            "positive_contribution_count": positive_chunks,
            "negative_contribution_count": noisy_chunks,
            "mean_contribution": float(np.mean(contributions))
            if contributions
            else 0.0,
            "noisy_chunk_fraction": noisy_chunks / max(1, num_chunks),
        }

    def evaluate_policy_variants(
        self,
        test_samples: Sequence[TrainingSample],
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate learned gate against always-retrieve and always-skip policies."""
        if not test_samples:
            return {}
        original_gate_mode = self.config.gate_mode
        variants: Dict[str, Dict[str, Any]] = {}
        try:
            for mode in ("learned", "always_retrieve", "always_skip"):
                self.config.gate_mode = mode
                variants[mode] = self.phase6_evaluate(
                    test_samples,
                    include_analysis=False,
                )
        finally:
            self.config.gate_mode = original_gate_mode
        return variants

    def calibrate_gate_threshold(
        self,
        samples: Sequence[TrainingSample],
    ) -> Dict[str, Any]:
        """Tune the learned gate threshold from utility-derived labels."""
        if self.config.gate_mode != "learned":
            return {
                "status": "skipped",
                "reason": f"gate_mode={self.config.gate_mode}",
                "threshold": self.config.gate_decision_threshold,
            }
        limit = max(0, self.config.gate_calibration_samples)
        if limit <= 0:
            return {
                "status": "skipped",
                "reason": "gate_calibration_samples=0",
                "threshold": self.config.gate_decision_threshold,
            }

        labels: List[bool] = []
        probabilities: List[float] = []
        for sample in list(samples)[:limit]:
            candidates = list(sample.candidate_chunks or self._chunks)
            if not candidates:
                continue
            label, _ = self._utility_gate_label(sample, candidates)
            labels.append(label)
            probabilities.append(self._gate_probability(sample, candidates))

        selected = _select_gate_threshold(
            labels,
            probabilities,
            retrieval_penalty=self.config.gate_calibration_retrieval_penalty,
        )
        self.config.gate_decision_threshold = float(selected["threshold"])
        return {
            "status": "ok" if labels else "no_labels",
            "threshold": self.config.gate_decision_threshold,
            "num_samples": len(labels),
            "retrieval_penalty": self.config.gate_calibration_retrieval_penalty,
            **selected,
        }

    def gate_defense_status(
        self,
        variants: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        learned = variants.get("learned")
        always_retrieve = variants.get("always_retrieve")
        if not learned or not always_retrieve:
            return {"status": "not_evaluated"}
        return _gate_defense_status(
            learned,
            always_retrieve,
            quality_tolerance=self.config.gate_quality_tolerance,
            retrieval_reduction_target=self.config.gate_retrieval_reduction_target,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  Full pipeline
    # ══════════════════════════════════════════════════════════════════════

    def train(
        self,
        samples: Sequence[TrainingSample],
        chunks: Optional[Sequence[CodeChunk]] = None,
        eval_samples: Optional[Sequence[TrainingSample]] = None,
        sample_provider: Optional[
            Callable[[int], Sequence[TrainingSample]]
        ] = None,
    ) -> Dict[str, Any]:
        """Run the complete 7-phase pipeline.

        Parameters
        ----------
        samples : training samples with left_context, target, candidate_chunks.
        chunks : global candidate chunks (if None, collected from samples).
        """
        # Collect global chunks
        if chunks is not None:
            all_chunks = list(chunks)
        else:
            all_chunks = []
            seen: set[str] = set()
            for s in samples:
                for c in s.candidate_chunks or []:
                    if c.chunk_id not in seen:
                        seen.add(c.chunk_id)
                        all_chunks.append(c)

        if not all_chunks:
            logger.warning("No chunks — aborting training")
            return {"status": "no_chunks"}

        # Phase 0
        if getattr(self.config, "build_train_index", False):
            self.phase0_build_index(all_chunks)
        else:
            self.phase0_register_chunks(all_chunks)

        schedule = self.config.experiment_mode
        p1: Dict[str, float] = {}
        if self.config.experiment_mode == "sequential_adapter_first":
            round_history = self.phase5_sequential_adapter_first(samples)
        elif self.config.experiment_mode == "sequential_retriever_first":
            if sample_provider is not None:
                raise ValueError(
                    "resample_train_each_epoch requires experiment_mode="
                    "'intent_main' so preference data is rebuilt per epoch"
                )
            round_history = self.phase5_sequential_retriever_first(samples)
        else:
            schedule = "alternating"
            # Phase 1 — Initial soft prompt warm-up
            p1 = self.phase1_warmup_soft_prompt(samples)
            # Phase 5 — Alternating co-training epochs/rounds (contains P1-P4)
            if sample_provider is None:
                round_history = self.phase5_co_training(samples)
            else:
                round_history = self.phase5_co_training(
                    samples, sample_provider=sample_provider
                )

        gate_calibration = (
            self.calibrate_gate_threshold(samples)
            if self.config.gate_calibrate_threshold
            else {
                "status": "disabled",
                "threshold": self.config.gate_decision_threshold,
            }
        )

        # Phase 6 — Final held-out evaluation when provided.
        eval_metrics = self.phase6_evaluate(eval_samples) if eval_samples else {}
        eval_policy_variants = (
            self.evaluate_policy_variants(eval_samples) if eval_samples else {}
        )
        gate_status = self.gate_defense_status(eval_policy_variants)

        # Save final checkpoint
        self._save_checkpoint(
            {
                "rounds": round_history,
                "eval": eval_metrics,
                "eval_policy_variants": eval_policy_variants,
                "gate_defense_status": gate_status,
                "gate_calibration": gate_calibration,
            }
        )

        return {
            "status": "ok",
            "schedule": schedule,
            "train_epochs": self.config.train_epochs,
            "epoch_budget_mode": self.config.epoch_budget_mode,
            "resample_train_each_epoch": sample_provider is not None,
            "build_train_index": self.config.build_train_index,
            "refresh_train_index": self.config.refresh_train_index,
            "initial_warmup": p1,
            "rounds": round_history,
            "eval": eval_metrics,
            "eval_policy_variants": eval_policy_variants,
            "gate_defense_status": gate_status,
            "gate_calibration": gate_calibration,
            "gate_label_metrics": eval_metrics.get("gate_label_metrics", {}),
            "nll_output_correlation": eval_metrics.get(
                "nll_output_correlation", {}
            ),
            "leave_one_out_analysis": eval_metrics.get(
                "leave_one_out_analysis", {}
            ),
            "oracle_used_for_eval": False,
            "inference_safe_strategy_check": True,
            "generator_backbone_frozen": True,
            "adapter_type": getattr(self.config, "adapter_type", "soft_prompt"),
            "num_samples": len(samples),
            "num_eval_samples": len(eval_samples or []),
            "num_chunks": len(all_chunks),
        }

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, sample: TrainingSample) -> str:
        """Single-sample inference with trained pipeline."""
        self._assert_inference_safe_strategy(mode="predict")
        candidates = list(sample.candidate_chunks or self._chunks)
        if self._should_retrieve(sample, candidates) and candidates:
            ctx = self._retrieve_for_generation(sample, candidates, mode="predict")
            return self.generator.generate(
                sample.left_context,
                max_new_tokens=self.config.max_new_tokens,
                retrieved_chunks=ctx,
                use_soft_prompt=self.use_adapter,
            )
        # Skip branch — keep adapter treatment consistent
        return self.generator.generate(
            sample.left_context,
            max_new_tokens=self.config.max_new_tokens,
            use_soft_prompt=self.use_adapter,
        )

    # ── Checkpointing ─────────────────────────────────────────────────────

    def _save_checkpoint(self, history: Any) -> None:
        import json

        ckpt_dir = self.config.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        self.retriever.save_pretrained(os.path.join(ckpt_dir, "retriever"))
        torch.save(self.gate.state_dict(), os.path.join(ckpt_dir, "gate.pt"))
        self.generator.save_prompt(os.path.join(ckpt_dir, "soft_prompt.pt"))

        meta = {
            "created_at": datetime.now().isoformat(),
            "config": asdict(self.config),
            "history": history if isinstance(history, list) else [history],
        }
        with open(os.path.join(ckpt_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Checkpoint saved to %s", ckpt_dir)

    def load_checkpoint(self, ckpt_dir: str) -> None:
        # ``DenseRetriever.load_pretrained`` replaces the encoder module.  The
        # optimizer created in ``__init__`` would otherwise keep references to
        # the old GPU parameters, temporarily retaining a second full encoder
        # during evaluation/checkpoint reload.
        self.retriever_opt = None
        self.retriever.load_pretrained(os.path.join(ckpt_dir, "retriever"))
        self.retriever_opt = AdamW(
            self.retriever.parameters(),
            lr=getattr(self.config, "retriever_lr", 2e-5),
        )
        gate_path = os.path.join(ckpt_dir, "gate.pt")
        if os.path.exists(gate_path):
            self.gate.load_state_dict(
                torch.load(gate_path, map_location=self.config.device)
            )
        prompt_path = os.path.join(ckpt_dir, "soft_prompt.pt")
        if os.path.exists(prompt_path):
            self.generator.load_prompt(prompt_path)
        logger.info("Loaded checkpoint from %s", ckpt_dir)
