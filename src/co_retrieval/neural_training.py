"""Neural Co-Training pipeline — 7-phase training specification.

Phase 0 — Build repo index (AST chunking → Jina embed → FAISS)
Phase 1 — Warm-up soft prompt (20% no-ctx, 50% oracle, 30% noisy; CE loss)
Phase 2 — Build preference data (6 strategies, teacher-forcing NLL)
Phase 3 — DPO-style train retriever + gate (combined score, reference policy)
Phase 4 — Refresh FAISS index (re-embed after retriever update)
Phase 5 — Alternating co-training rounds (P1→P2→P3→P4 repeated)
Phase 6 — Final evaluation (retrieval + generation metrics)
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from co_retrieval.chunking import CodeChunk
from co_retrieval.dense_retriever import DenseRetriever
from co_retrieval.embedding_cache import EmbeddingCache
from co_retrieval.neural_gate import NeuralGate
from co_retrieval.quality_metrics import (
    composite_quality,
    exact_match,
    edit_similarity,
    identifier_f1,
    identifier_set,
)
from co_retrieval.soft_prompt import SoftPromptLLM
from co_retrieval.training import TrainingSample, TrainingHistory

logger = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")


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

    # Phase 1 — Soft Prompt Warm-up
    warmup_steps: int = 200
    warmup_no_context_ratio: float = 0.20
    warmup_oracle_ratio: float = 0.50
    # warmup_noisy_ratio = 1.0 - no_context - oracle = 0.30

    # Phase 2 — Preference Data
    preference_margin: float = 0.1
    num_hard_negatives: int = 10
    preference_pool_top_k: int = 20

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


# ── Preference Pair ───────────────────────────────────────────────────────────


@dataclass
class PreferencePair:
    """A single DPO training pair."""

    query: str  # left_context
    chosen_chunks: List[CodeChunk]
    rejected_chunks: List[CodeChunk]
    chosen_is_stop: bool = False
    rejected_is_stop: bool = False
    chosen_nll: float = 0.0
    rejected_nll: float = 0.0


# ── NeuralCoTrainer ───────────────────────────────────────────────────────────


class NeuralCoTrainer:
    """7-phase co-training pipeline for Co-Retrieval."""

    def __init__(self, config: NeuralCoTrainingConfig) -> None:
        self.config = config
        self.rng = random.Random(config.random_seed)
        torch.manual_seed(config.random_seed)

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
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

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 1 — Warm-up Soft Prompt
    # ══════════════════════════════════════════════════════════════════════

    def phase1_warmup_soft_prompt(
        self,
        samples: Sequence[TrainingSample],
        steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Train soft prompt with mixed context (CE loss).

        Context mixing: 20% no-ctx, 50% oracle, 30% noisy/retrieved.
        Freeze: LLM, Retriever, Gate.  Update: Soft Prompt only.
        """
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

            if r < no_ctx_ratio:
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
    ) -> List[PreferencePair]:
        """Create DPO pairs using teacher-forcing NLL (no decoding).

        6 strategies: stop, BM25, jina_frozen, current, oracle, hard_neg.
        """
        logger.info("Phase 2: Building preference data (%d samples)…", len(samples))
        pairs: List[PreferencePair] = []

        for i, sample in enumerate(samples):
            candidates = list(sample.candidate_chunks or self._chunks)
            if not candidates:
                continue

            # Build 6 context strategies
            strategies: Dict[str, Tuple[List[CodeChunk], bool]] = {}

            # C_stop — no retrieval
            strategies["stop"] = ([], True)

            # C_bm25 — lexical baseline
            bm25_ctx = _bm25_retrieve(
                sample.left_context, candidates, self.config.top_k
            )
            strategies["bm25"] = (bm25_ctx, False)

            # C_jina — frozen pretrained Jina
            jina_ctx = self.retriever.retrieve_with_encoder(
                sample.left_context,
                candidates,
                top_k=self.config.top_k,
                encoder=self.retriever.initial_encoder,
            )
            strategies["jina_frozen"] = (jina_ctx, False)

            # C_current — current (trained) retriever
            current_ctx = self.retriever.retrieve_chunks(
                sample.left_context, candidates, top_k=self.config.top_k
            )
            strategies["current"] = (current_ctx, False)

            # C_oracle — oracle snippets
            oracle_ctx = _oracle_chunks(
                sample.target, candidates, self.config.top_k
            )
            strategies["oracle"] = (oracle_ctx, False)

            # C_negative — hard negatives (top-scored but irrelevant)
            hard_neg_ctx = self._mine_hard_negatives(
                sample, candidates
            )
            if hard_neg_ctx:
                strategies["hard_neg"] = (hard_neg_ctx, False)

            # Evaluate each strategy by teacher-forcing NLL
            scored: List[Tuple[str, float, List[CodeChunk], bool]] = []
            with torch.no_grad():
                for name, (ctx_chunks, is_stop) in strategies.items():
                    nll = self.generator.teacher_forcing_nll(
                        left_context=sample.left_context,
                        target=sample.target,
                        retrieved_chunks=ctx_chunks if not is_stop else None,
                        use_soft_prompt=not is_stop,
                    )
                    scored.append((name, nll.item(), ctx_chunks, is_stop))

            # Sort by NLL ascending (lower = better)
            scored.sort(key=lambda x: x[1])

            # Form preference pairs with margin
            best_name, best_nll, best_ctx, best_is_stop = scored[0]
            worst_name, worst_nll, worst_ctx, worst_is_stop = scored[-1]

            if worst_nll - best_nll >= self.config.preference_margin:
                pairs.append(
                    PreferencePair(
                        query=sample.left_context,
                        chosen_chunks=best_ctx,
                        rejected_chunks=worst_ctx,
                        chosen_is_stop=best_is_stop,
                        rejected_is_stop=worst_is_stop,
                        chosen_nll=best_nll,
                        rejected_nll=worst_nll,
                    )
                )

            if (i + 1) % 100 == 0:
                logger.info(
                    "  Phase 2: %d/%d samples, %d pairs so far",
                    i + 1, len(samples), len(pairs),
                )

        logger.info(
            "Phase 2 done: %d pairs from %d samples (%.1f%%)",
            len(pairs), len(samples),
            100 * len(pairs) / max(1, len(samples)),
        )
        return pairs

    def _mine_hard_negatives(
        self,
        sample: TrainingSample,
        candidates: Sequence[CodeChunk],
    ) -> List[CodeChunk]:
        """Find chunks that the retriever ranks high but are not relevant."""
        target_symbols = identifier_set(sample.target)
        if not target_symbols:
            return []

        top_ranked = self.retriever.retrieve_chunks(
            sample.left_context,
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
        pairs: Sequence[PreferencePair],
        steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """DPO ranking loss → update Retriever + Gate.

        Freeze: LLM, Soft Prompt.  Update: Retriever (full), Gate.
        """
        n_steps = steps or len(pairs)
        n_steps = min(n_steps, len(pairs))
        if n_steps <= 0:
            return {"phase3_dpo_loss": 0.0, "phase3_gate_loss": 0.0}

        logger.info("Phase 3: DPO training (%d pairs)…", n_steps)

        # Snapshot reference at start
        self.retriever.refresh_reference()

        total_dpo_loss = 0.0
        total_gate_loss = 0.0
        pair_list = list(pairs)
        self.rng.shuffle(pair_list)

        for i, pair in enumerate(pair_list[:n_steps]):
            # DPO loss → update retriever + gate
            self.retriever_opt.zero_grad()
            self.gate_opt.zero_grad()

            dpo_loss = self.retriever.dpo_loss(
                query_text=pair.query,
                chosen_chunks=pair.chosen_chunks,
                rejected_chunks=pair.rejected_chunks,
                gate=self.gate,
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
                torch.nn.utils.clip_grad_norm_(
                    self.gate.parameters(),
                    self.config.grad_clip_norm,
                )
                self.retriever_opt.step()
                self.gate_opt.step()
                total_dpo_loss += dpo_loss.item()

            # Gate supervised loss (optional reinforcement)
            with torch.no_grad():
                q_vec = self.retriever.encode_query(pair.query)
            g = self.gate(q_vec.detach())
            retrieve_better = not pair.chosen_is_stop
            g_loss = self.gate.gate_loss(g, retrieve_better)
            if not torch.isnan(g_loss):
                self.gate_opt.zero_grad()
                g_loss.backward()
                self.gate_opt.step()
                total_gate_loss += g_loss.item()

            if (i + 1) % 50 == 0:
                logger.info(
                    "  Phase 3 step %d/%d  dpo=%.4f  gate=%.4f",
                    i + 1, n_steps,
                    total_dpo_loss / (i + 1),
                    total_gate_loss / (i + 1),
                )

        denom = max(1, n_steps)
        result = {
            "phase3_dpo_loss": total_dpo_loss / denom,
            "phase3_gate_loss": total_gate_loss / denom,
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
            pairs = self.phase2_build_preference_data(samples)

            # P3: DPO train retriever + gate
            p3 = self.phase3_dpo_training(
                pairs, steps=self.config.steps_per_round_dpo
            )

            # P4: Refresh index
            self.phase4_refresh_index()

            round_result = {
                "round": r,
                **p1,
                "num_dpo_pairs": len(pairs),
                **p3,
            }
            round_history.append(round_result)
            logger.info("Round %d result: %s", r, round_result)

        return round_history

    # ══════════════════════════════════════════════════════════════════════
    #  Phase 6 — Final Evaluation
    # ══════════════════════════════════════════════════════════════════════

    def phase6_evaluate(
        self,
        test_samples: Sequence[TrainingSample],
    ) -> Dict[str, float]:
        """Evaluate retrieval + generation + end-to-end."""
        logger.info("Phase 6: Evaluating on %d samples…", len(test_samples))

        total_em = 0.0
        total_edit_sim = 0.0
        total_id_f1 = 0.0
        total_retrieve_count = 0
        total_nll_with = 0.0
        total_nll_without = 0.0
        n = 0

        for sample in test_samples:
            candidates = list(sample.candidate_chunks or self._chunks)

            with torch.no_grad():
                q_vec = self.retriever.encode_query(sample.left_context)
                should_retrieve = self.gate.should_retrieve(q_vec)

            if should_retrieve and candidates:
                ctx = self.retriever.retrieve_chunks(
                    sample.left_context, candidates, top_k=self.config.top_k
                )
                pred = self.generator.generate(
                    sample.left_context,
                    max_new_tokens=self.config.max_new_tokens,
                    retrieved_chunks=ctx,
                    use_soft_prompt=True,
                )
                total_retrieve_count += 1
            else:
                ctx = []
                pred = self.generator.generate(
                    sample.left_context,
                    max_new_tokens=self.config.max_new_tokens,
                    use_soft_prompt=False,
                )

            total_em += exact_match(pred, sample.target)
            total_edit_sim += edit_similarity(pred, sample.target)
            total_id_f1 += identifier_f1(pred, sample.target)

            # NLL comparison
            with torch.no_grad():
                nll_with = self.generator.teacher_forcing_nll(
                    sample.left_context, sample.target,
                    retrieved_chunks=ctx if ctx else None,
                    use_soft_prompt=bool(ctx),
                ).item()
                nll_without = self.generator.teacher_forcing_nll(
                    sample.left_context, sample.target,
                    use_soft_prompt=False,
                ).item()
            total_nll_with += nll_with
            total_nll_without += nll_without
            n += 1

        denom = max(1, n)
        metrics = {
            "exact_match": total_em / denom,
            "edit_similarity": total_edit_sim / denom,
            "identifier_f1": total_id_f1 / denom,
            "retrieval_rate": total_retrieve_count / denom,
            "avg_nll_with_retrieval": total_nll_with / denom,
            "avg_nll_without_retrieval": total_nll_without / denom,
            "nll_improvement": (total_nll_without - total_nll_with) / denom,
            "num_samples": n,
        }
        logger.info("Phase 6 results: %s", {k: round(v, 4) for k, v in metrics.items()})
        return metrics

    # ══════════════════════════════════════════════════════════════════════
    #  Full pipeline
    # ══════════════════════════════════════════════════════════════════════

    def train(
        self,
        samples: Sequence[TrainingSample],
        chunks: Optional[Sequence[CodeChunk]] = None,
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

        # Phase 1 — Initial soft prompt warm-up
        p1 = self.phase1_warmup_soft_prompt(samples)

        # Phase 5 — Alternating co-training rounds (contains P1-P4)
        round_history = self.phase5_co_training(samples)

        # Save final checkpoint
        self._save_checkpoint(round_history)

        return {
            "status": "ok",
            "initial_warmup": p1,
            "rounds": round_history,
            "num_samples": len(samples),
            "num_chunks": len(all_chunks),
        }

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, sample: TrainingSample) -> str:
        """Single-sample inference with trained pipeline."""
        candidates = list(sample.candidate_chunks or self._chunks)

        with torch.no_grad():
            q_vec = self.retriever.encode_query(sample.left_context)

        if self.gate.should_retrieve(q_vec) and candidates:
            ctx = self.retriever.retrieve_chunks(
                sample.left_context, candidates, top_k=self.config.top_k
            )
            return self.generator.generate(
                sample.left_context,
                max_new_tokens=self.config.max_new_tokens,
                retrieved_chunks=ctx,
                use_soft_prompt=True,
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
