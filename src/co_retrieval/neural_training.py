"""Neural Co-Training pipeline — 7-phase training specification.

Phase 0 — Build repo index (AST chunking → Jina embed → FAISS)
Phase 1 — Warm-up soft prompt (20% no-ctx, 50% oracle, 30% noisy; CE loss)
Phase 2 — Build preference data (6 strategies, teacher-forcing NLL)
Phase 3 — DPO-style train retriever; train gate from utility labels
Phase 4 — Refresh FAISS index (re-embed after retriever update)
Phase 5 — Alternating co-training rounds (P1→P2→P3→P4 repeated)
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import AdamW

from co_retrieval.chunking import CodeChunk
from co_retrieval.context_utility import ContextCandidate, ContextUtilityScorer
from co_retrieval.dense_retriever import DenseRetriever
from co_retrieval.embedding_cache import EmbeddingCache
from co_retrieval.intent import IntentSketcher
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


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class NeuralCoTrainingConfig:
    """Full configuration for the 7-phase training pipeline."""

    # Model names
    encoder_name: str = "jinaai/jina-code-embeddings-1.5b"
    generator_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # Encoder
    encoder_max_length: int = 512

    # Generator
    num_prompt_tokens: int = 50
    max_context_tokens: int = 4096

    # Gate
    gate_hidden_dim: int = 256
    gate_entropy_weight: float = 0.01

    # Retrieval
    top_k: int = 3
    experiment_mode: str = "intent_main"
    intent_mode: str = "static"
    gate_mode: str = "learned"
    adapter_type: str = "soft_prompt"
    include_oracle_strategy: bool = True

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
    leave_one_out_analysis_samples: int = 25
    gate_quality_tolerance: float = 0.01
    gate_retrieval_reduction_target: float = 0.20

    # Phase 3 — DPO
    dpo_beta: float = 0.1

    # Phase 5 — Co-training rounds
    num_rounds: int = 2
    steps_per_round_prompt: int = 100
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
    batch_encode_size: int = 32
    eval_ratio: float = 0.1
    max_eval_samples: int = 100


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


def _gate_defense_status(
    learned: Dict[str, float],
    always_retrieve: Dict[str, float],
    *,
    quality_tolerance: float = 0.01,
    retrieval_reduction_target: float = 0.20,
) -> Dict[str, Any]:
    """Classify whether learned gate supports quality or efficiency claims."""
    metrics = ("exact_match", "edit_similarity", "identifier_f1")
    deltas = {
        metric: learned.get(metric, 0.0) - always_retrieve.get(metric, 0.0)
        for metric in metrics
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


@dataclass
class PreferenceData:
    """Preference pairs plus gate supervision and analysis counters."""

    pairs: List[PreferencePair] = field(default_factory=list)
    gate_examples: List[GateTrainingExample] = field(default_factory=list)
    pair_type_counts: Dict[str, int] = field(default_factory=dict)
    strategy_counts: Dict[str, int] = field(default_factory=dict)
    gate_positive_count: int = 0
    gate_negative_count: int = 0
    mean_max_utility: float = 0.0


# ── NeuralCoTrainer ───────────────────────────────────────────────────────────


class NeuralCoTrainer:
    """7-phase co-training pipeline for Co-Retrieval."""

    def __init__(self, config: NeuralCoTrainingConfig) -> None:
        self.config = config
        self._apply_experiment_defaults()
        self.rng = random.Random(config.random_seed)
        torch.manual_seed(config.random_seed)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        gen_dtype = dtype_map.get(config.generator_dtype, torch.float16)

        # Components
        logger.info("Init DenseRetriever (%s)…", config.encoder_name)
        self.retriever = DenseRetriever(
            model_name=config.encoder_name,
            max_length=config.encoder_max_length,
            device=config.device,
        )
        self.retriever.save_initial_copy()  # C_jina baseline

        logger.info("Init NeuralGate…")
        self.gate = NeuralGate(
            input_dim=self.retriever.hidden_size,
            hidden_dim=config.gate_hidden_dim,
            entropy_weight=config.gate_entropy_weight,
        ).to(config.device)

        logger.info("Init SoftPromptLLM (%s)…", config.generator_name)
        self.generator = SoftPromptLLM(
            model_name=config.generator_name,
            num_prompt_tokens=config.num_prompt_tokens,
            max_context_tokens=config.max_context_tokens,
            device=config.device,
            dtype=gen_dtype,
        )
        self.use_adapter = config.adapter_type == "soft_prompt"
        if config.adapter_type not in {"soft_prompt", "none"}:
            raise ValueError(
                f"Unsupported adapter_type={config.adapter_type!r}; "
                "supported values are 'soft_prompt' and 'none'."
            )
        self.intent_sketcher = IntentSketcher()
        self.utility_scorer = ContextUtilityScorer(self.generator)

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

        # Global chunks
        self._chunks: List[CodeChunk] = []
        self._chunk_map: Dict[str, CodeChunk] = {}

    def _apply_experiment_defaults(self) -> None:
        mode = self.config.experiment_mode
        if mode == "raw_query_main":
            self.config.intent_mode = "raw"
        elif mode == "intent_main":
            self.config.intent_mode = "static"
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
        elif mode in {"sequential_adapter_first", "sequential_retriever_first"}:
            self.config.intent_mode = "static"

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 0 — Build Repo Index
    # ══════════════════════════════════════════════════════════════════════

    def phase0_build_index(self, chunks: Sequence[CodeChunk]) -> None:
        """AST chunks → Jina embed → FAISS index."""
        self._chunks = list(chunks)
        self._chunk_map = {c.chunk_id: c for c in chunks}

        self.embedding_cache.build_from_chunks(
            chunks,
            encode_fn=self.retriever.encode_texts_numpy,
            batch_size=self.config.batch_encode_size,
            show_progress=True,
        )
        logger.info("Phase 0: built index with %d chunks", len(chunks))

    # ── Query / strategy helpers ─────────────────────────────────────────

    def _retrieval_query(self, sample: TrainingSample) -> str:
        if self.config.intent_mode == "raw":
            return sample.left_context
        return self.intent_sketcher.build_query(sample.left_context)

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
        k = top_k or self.config.top_k
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

    def _strategy_candidates(
        self,
        sample: TrainingSample,
        candidates: Sequence[CodeChunk],
        retrieval_query: str,
        *,
        include_train_only: bool = True,
    ) -> List[ContextCandidate]:
        strategies = [
            ContextCandidate("stop", [], is_stop=True, retrieval_query=retrieval_query)
        ]

        bm25_ctx = _bm25_retrieve(retrieval_query, candidates, self.config.top_k)
        strategies.append(
            ContextCandidate("bm25", bm25_ctx, retrieval_query=retrieval_query)
        )

        if self.retriever.initial_encoder is not None:
            jina_ctx = self.retriever.retrieve_with_encoder(
                retrieval_query,
                candidates,
                top_k=self.config.top_k,
                encoder=self.retriever.initial_encoder,
            )
            strategies.append(
                ContextCandidate(
                    "dense_frozen", jina_ctx, retrieval_query=retrieval_query
                )
            )

        current_ctx = self._retrieve_current(
            retrieval_query, candidates, top_k=self.config.top_k
        )
        strategies.append(
            ContextCandidate("current", current_ctx, retrieval_query=retrieval_query)
        )

        if include_train_only:
            hard_neg_ctx = self._mine_hard_negatives(sample, candidates, retrieval_query)
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
        query = self._retrieval_query(sample)
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
        return self._retrieve_current(query, candidates, top_k=self.config.top_k)

    def _should_retrieve(self, sample: TrainingSample) -> bool:
        if self.config.gate_mode == "always_retrieve":
            return True
        if self.config.gate_mode == "always_skip":
            return False
        query = (
            sample.left_context
            if self.config.gate_mode != "rule"
            else self._retrieval_query(sample)
        )
        if self.config.gate_mode == "rule":
            if not sample.left_context.strip():
                return False
            return "." in sample.left_context.rstrip().splitlines()[-1]
        with torch.no_grad():
            q_vec = self.retriever.encode_query(query)
            return self.gate.should_retrieve(q_vec)

    def _gate_probability(self, sample: TrainingSample) -> float:
        with torch.no_grad():
            q_vec = self.retriever.encode_query(sample.left_context)
            value = self.gate(q_vec.detach())
        return float(value.view(-1)[0].item())

    def _utility_gate_label(
        self,
        sample: TrainingSample,
        candidates: Sequence[CodeChunk],
    ) -> tuple[bool, float]:
        """Build an analysis label without target-aware retrieval strategies."""
        if not candidates:
            return False, 0.0
        retrieval_query = self._retrieval_query(sample)
        strategies = self._strategy_candidates(
            sample,
            candidates,
            retrieval_query,
            include_train_only=False,
        )
        with torch.no_grad():
            scored = self.utility_scorer.score(
                sample.left_context,
                sample.target,
                strategies,
                use_adapter=self.use_adapter,
            )
        best_retrieve = max(
            (score for score in scored if not score.is_stop),
            key=lambda score: score.utility,
            default=None,
        )
        max_utility = best_retrieve.utility if best_retrieve else 0.0
        return max_utility > self.config.utility_margin, max_utility

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
        if not self.use_adapter:
            return {"phase1_loss": 0.0}

        n_steps = steps or self.config.warmup_steps
        n_steps = min(n_steps, len(samples))
        if n_steps <= 0:
            return {"phase1_loss": 0.0}

        logger.info("Phase 1: Warm-up Soft Prompt (%d steps)…", n_steps)
        total_loss = 0.0
        sample_list = list(samples)
        self.rng.shuffle(sample_list)

        no_ctx_ratio = self.config.warmup_no_context_ratio
        oracle_ratio = self.config.warmup_oracle_ratio

        for i, sample in enumerate(sample_list[:n_steps]):
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
        logger.info("Phase 2: Building preference data (%d samples)…", len(samples))
        pairs: List[PreferencePair] = []
        gate_examples: List[GateTrainingExample] = []
        pair_type_counter: Counter[str] = Counter()
        strategy_counter: Counter[str] = Counter()
        gate_positive_count = 0
        total_max_utility = 0.0

        for i, sample in enumerate(samples):
            candidates = list(sample.candidate_chunks or self._chunks)
            if not candidates:
                continue

            retrieval_query = self._retrieval_query(sample)
            strategies = self._strategy_candidates(sample, candidates, retrieval_query)
            with torch.no_grad():
                scored = self.utility_scorer.score(
                    sample.left_context,
                    sample.target,
                    strategies,
                    use_adapter=self.use_adapter,
                )

            best_retrieve = max(
                (score for score in scored if not score.is_stop),
                key=lambda score: score.utility,
                default=None,
            )
            retrieve_is_better = (
                best_retrieve is not None
                and best_retrieve.utility > self.config.utility_margin
            )
            gate_positive_count += int(retrieve_is_better)
            total_max_utility += best_retrieve.utility if best_retrieve else 0.0
            gate_examples.append(
                GateTrainingExample(
                    query=sample.left_context,
                    retrieve_is_better=retrieve_is_better,
                    max_utility=best_retrieve.utility if best_retrieve else 0.0,
                    best_strategy=best_retrieve.name if best_retrieve else "stop",
                )
            )
            for score in scored:
                strategy_counter[score.name] += 1

            # Form multiple informative pairs, not just best-vs-worst.
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
            "Phase 2 done: %d pairs and %d gate labels from %d samples (%.1f%%)",
            len(pairs), len(gate_examples), len(samples),
            100 * len(pairs) / max(1, len(samples)),
        )
        return PreferenceData(
            pairs=pairs,
            gate_examples=gate_examples,
            pair_type_counts=dict(pair_type_counter),
            strategy_counts=dict(strategy_counter),
            gate_positive_count=gate_positive_count,
            gate_negative_count=len(gate_examples) - gate_positive_count,
            mean_max_utility=total_max_utility / max(1, len(gate_examples)),
        )

    def _mine_hard_negatives(
        self,
        sample: TrainingSample,
        candidates: Sequence[CodeChunk],
        retrieval_query: str,
    ) -> List[CodeChunk]:
        """Find chunks that the retriever ranks high but are not relevant."""
        target_symbols = identifier_set(sample.target)
        if not target_symbols:
            return []

        top_ranked = self._retrieve_current(
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
    #  Phase 3 — DPO-style Training Retriever + Gate
    # ══════════════════════════════════════════════════════════════════════

    def phase3_dpo_training(
        self,
        preference_data: PreferenceData,
        steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """DPO ranking loss → update Retriever; utility labels → update Gate.

        Freeze: LLM, Soft Prompt.  Update: Retriever (DPO), Gate (BCE only).
        """
        pairs = preference_data.pairs
        gate_examples = preference_data.gate_examples
        dpo_steps = min(steps or len(pairs), len(pairs))
        gate_steps = min(steps or len(gate_examples), len(gate_examples))
        if dpo_steps <= 0 and gate_steps <= 0:
            return {"phase3_dpo_loss": 0.0, "phase3_gate_loss": 0.0}

        logger.info(
            "Phase 3: DPO training (%d pairs, %d gate labels)…",
            dpo_steps,
            gate_steps,
        )

        # Snapshot reference at start
        self.retriever.refresh_reference()

        total_dpo_loss = 0.0
        total_gate_loss = 0.0
        pair_list = list(pairs)
        self.rng.shuffle(pair_list)

        for i, pair in enumerate(pair_list[:dpo_steps]):
            # DPO loss updates only retriever. Gate supervision is separate below.
            self.retriever_opt.zero_grad()

            dpo_loss = self.retriever.dpo_loss(
                query_text=pair.query,
                chosen_chunks=pair.chosen_chunks,
                rejected_chunks=pair.rejected_chunks,
                beta=self.config.dpo_beta,
                chosen_is_stop=pair.chosen_is_stop,
                rejected_is_stop=pair.rejected_is_stop,
            )

            if not torch.isnan(dpo_loss):
                dpo_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.retriever.parameters(),
                    self.config.grad_clip_norm * 5,
                )
                self.retriever_opt.step()
                total_dpo_loss += dpo_loss.item()

            if (i + 1) % 50 == 0:
                logger.info(
                    "  Phase 3 DPO step %d/%d  dpo=%.4f",
                    i + 1,
                    dpo_steps,
                    total_dpo_loss / (i + 1),
                )

        gate_list = list(gate_examples)
        self.rng.shuffle(gate_list)
        for i, example in enumerate(gate_list[:gate_steps]):
            with torch.no_grad():
                q_vec = self.retriever.encode_query(example.query)
            g = self.gate(q_vec.detach())
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
                    gate_steps,
                    total_gate_loss / (i + 1),
                )

        result = {
            "phase3_dpo_loss": total_dpo_loss / max(1, dpo_steps),
            "phase3_gate_loss": total_gate_loss / max(1, gate_steps),
            "phase3_gate_labels": gate_steps,
        }
        logger.info("Phase 3 done: %s", result)
        return result

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 4 — Refresh Index
    # ══════════════════════════════════════════════════════════════════════

    def phase4_refresh_index(self) -> None:
        """Re-embed all chunks with updated retriever → rebuild FAISS."""
        if not self._chunks:
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
    ) -> List[Dict[str, Any]]:
        """Alternating rounds: P1 → P2 → P3 → P4, repeated."""
        rounds = num_rounds or self.config.num_rounds
        round_history: List[Dict[str, Any]] = []

        for r in range(1, rounds + 1):
            logger.info("═══ Co-training Round %d/%d ═══", r, rounds)

            # P1: Train soft prompt
            p1 = self.phase1_warmup_soft_prompt(
                samples, steps=self.config.steps_per_round_prompt
            )

            # P2: Build preference data
            preference_data = self.phase2_build_preference_data(samples)

            # P3: DPO train retriever; BCE train gate from utility labels
            p3 = self.phase3_dpo_training(
                preference_data, steps=self.config.steps_per_round_dpo
            )

            # P4: Refresh index
            self.phase4_refresh_index()

            round_result = {
                "round": r,
                **p1,
                "num_dpo_pairs": len(preference_data.pairs),
                "num_gate_examples": len(preference_data.gate_examples),
                "gate_positive_count": preference_data.gate_positive_count,
                "gate_positive_ratio": preference_data.gate_positive_count
                / max(1, len(preference_data.gate_examples)),
                "mean_max_utility": preference_data.mean_max_utility,
                "pair_type_counts": preference_data.pair_type_counts,
                "strategy_counts": preference_data.strategy_counts,
                **p3,
            }
            round_history.append(round_result)
            logger.info("Round %d result: %s", r, round_result)

        return round_history

    def phase5_sequential_adapter_first(
        self,
        samples: Sequence[TrainingSample],
    ) -> List[Dict[str, Any]]:
        """Sequential baseline: adapter first, then retriever/gate once."""
        prompt_steps = self.config.warmup_steps + (
            self.config.num_rounds * self.config.steps_per_round_prompt
        )
        dpo_steps = self.config.num_rounds * self.config.steps_per_round_dpo
        p1 = self.phase1_warmup_soft_prompt(samples, steps=prompt_steps)
        preference_data = self.phase2_build_preference_data(samples)
        p3 = self.phase3_dpo_training(preference_data, steps=dpo_steps)
        self.phase4_refresh_index()
        return [
            {
                "round": 1,
                "schedule": "sequential_adapter_first",
                "prompt_steps_budget": prompt_steps,
                "dpo_steps_budget": dpo_steps,
                "preference_data_builds": 1,
                "index_refreshes": 1,
                **p1,
                "num_dpo_pairs": len(preference_data.pairs),
                "num_gate_examples": len(preference_data.gate_examples),
                "gate_positive_count": preference_data.gate_positive_count,
                "gate_positive_ratio": preference_data.gate_positive_count
                / max(1, len(preference_data.gate_examples)),
                "mean_max_utility": preference_data.mean_max_utility,
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
        prompt_steps = self.config.warmup_steps + (
            self.config.num_rounds * self.config.steps_per_round_prompt
        )
        dpo_steps = self.config.num_rounds * self.config.steps_per_round_dpo

        original_use_adapter = self.use_adapter
        self.use_adapter = False
        try:
            preference_data = self.phase2_build_preference_data(samples)
            p3 = self.phase3_dpo_training(preference_data, steps=dpo_steps)
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
                "prompt_steps_budget": prompt_steps,
                "dpo_steps_budget": dpo_steps,
                "preference_data_builds": 1,
                "index_refreshes": 1,
                **p1,
                "num_dpo_pairs": len(preference_data.pairs),
                "num_gate_examples": len(preference_data.gate_examples),
                "gate_positive_count": preference_data.gate_positive_count,
                "gate_positive_ratio": preference_data.gate_positive_count
                / max(1, len(preference_data.gate_examples)),
                "mean_max_utility": preference_data.mean_max_utility,
                "pair_type_counts": preference_data.pair_type_counts,
                "strategy_counts": preference_data.strategy_counts,
                **p3,
            }
        ]

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 6 — Final Evaluation
    # ══════════════════════════════════════════════════════════════════════

    def phase6_evaluate(
        self,
        test_samples: Sequence[TrainingSample],
        *,
        include_analysis: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate retrieval + generation + end-to-end."""
        self._assert_inference_safe_strategy(mode="eval")
        logger.info("Phase 6: Evaluating on %d samples…", len(test_samples))

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
        n = 0

        for sample in test_samples:
            candidates = list(sample.candidate_chunks or self._chunks)
            should_retrieve = self._should_retrieve(sample)

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
                pred = self.generator.generate(
                    sample.left_context,
                    max_new_tokens=self.config.max_new_tokens,
                    use_soft_prompt=False,
                )

            em = exact_match(pred, sample.target)
            edit = edit_similarity(pred, sample.target)
            id_f1 = identifier_f1(pred, sample.target)
            total_em += em
            total_edit_sim += edit
            total_id_f1 += id_f1
            edit_values.append(edit)
            id_f1_values.append(id_f1)

            # NLL comparison
            with torch.no_grad():
                nll_with = self.generator.teacher_forcing_nll(
                    sample.left_context, sample.target,
                    retrieved_chunks=ctx if ctx else None,
                    use_soft_prompt=bool(ctx) and self.use_adapter,
                ).item()
                nll_without = self.generator.teacher_forcing_nll(
                    sample.left_context, sample.target,
                    use_soft_prompt=False,
                ).item()
            total_nll_with += nll_with
            total_nll_without += nll_without
            nll_improvements.append(nll_without - nll_with)
            if include_analysis:
                label, _ = self._utility_gate_label(sample, candidates)
                gate_labels.append(label)
                gate_probs.append(self._gate_probability(sample))
            n += 1

        denom = max(1, n)
        nll_output_correlation = {
            "nll_vs_edit_corr": _safe_corr(nll_improvements, edit_values),
            "nll_vs_identifier_f1_corr": _safe_corr(
                nll_improvements, id_f1_values
            ),
        }
        metrics = {
            "exact_match": total_em / denom,
            "edit_similarity": total_edit_sim / denom,
            "identifier_f1": total_id_f1 / denom,
            "retrieval_rate": total_retrieve_count / denom,
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
            with torch.no_grad():
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
                        use_soft_prompt=bool(reduced) and self.use_adapter,
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
        self.phase0_build_index(all_chunks)

        schedule = self.config.experiment_mode
        p1: Dict[str, float] = {}
        if self.config.experiment_mode == "sequential_adapter_first":
            round_history = self.phase5_sequential_adapter_first(samples)
        elif self.config.experiment_mode == "sequential_retriever_first":
            round_history = self.phase5_sequential_retriever_first(samples)
        else:
            schedule = "alternating"
            # Phase 1 — Initial soft prompt warm-up
            p1 = self.phase1_warmup_soft_prompt(samples)
            # Phase 5 — Alternating co-training rounds (contains P1-P4)
            round_history = self.phase5_co_training(samples)

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
            }
        )

        return {
            "status": "ok",
            "schedule": schedule,
            "initial_warmup": p1,
            "rounds": round_history,
            "eval": eval_metrics,
            "eval_policy_variants": eval_policy_variants,
            "gate_defense_status": gate_status,
            "gate_label_metrics": eval_metrics.get("gate_label_metrics", {}),
            "nll_output_correlation": eval_metrics.get(
                "nll_output_correlation", {}
            ),
            "leave_one_out_analysis": eval_metrics.get(
                "leave_one_out_analysis", {}
            ),
            "oracle_used_for_eval": False,
            "inference_safe_strategy_check": True,
            "num_samples": len(samples),
            "num_eval_samples": len(eval_samples or []),
            "num_chunks": len(all_chunks),
        }

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, sample: TrainingSample) -> str:
        """Single-sample inference with trained pipeline."""
        self._assert_inference_safe_strategy(mode="predict")
        candidates = list(sample.candidate_chunks or self._chunks)
        if self._should_retrieve(sample) and candidates:
            ctx = self._retrieve_for_generation(sample, candidates, mode="predict")
            return self.generator.generate(
                sample.left_context,
                max_new_tokens=self.config.max_new_tokens,
                retrieved_chunks=ctx,
                use_soft_prompt=self.use_adapter,
            )
        return self.generator.generate(
            sample.left_context,
            max_new_tokens=self.config.max_new_tokens,
            use_soft_prompt=False,
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
        self.retriever.load_pretrained(os.path.join(ckpt_dir, "retriever"))
        gate_path = os.path.join(ckpt_dir, "gate.pt")
        if os.path.exists(gate_path):
            self.gate.load_state_dict(
                torch.load(gate_path, map_location=self.config.device)
            )
        prompt_path = os.path.join(ckpt_dir, "soft_prompt.pt")
        if os.path.exists(prompt_path):
            self.generator.load_prompt(prompt_path)
        logger.info("Loaded checkpoint from %s", ckpt_dir)
