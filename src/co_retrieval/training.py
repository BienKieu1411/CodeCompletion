"""Lightweight Co-Retrieval training loop.

This module implements the project mechanics from ``Novelty.md`` without
requiring a heavyweight code LLM during unit tests. The generator quality signal
is a deterministic proxy over exact/edit/identifier overlap, while the objects
and update flow mirror the proposed method: DPO-style retriever preference
updates, adaptive retrieval gate, and soft-prompt co-adaptation.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence

from co_retrieval.chunking import CodeChunk


_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")


@dataclass(frozen=True)
class TrainingSample:
    """One repository-level code-completion example."""

    left_context: str
    target: str
    file_path: str = "current_file.py"
    candidate_chunks: Optional[List[CodeChunk]] = None


@dataclass
class CoTrainingConfig:
    """Configuration for the Co-Retrieval trainer."""

    epochs: int = 1
    top_k: int = 3
    sampled_contexts: int = 3
    dpo_beta: float = 0.1
    retriever_lr: float = 0.2
    gate_lr: float = 0.1
    soft_prompt_lr: float = 0.05
    gate_threshold: float = 0.5
    random_seed: int = 13


@dataclass
class TrainingHistory:
    epoch: int
    avg_quality: float
    retrieval_rate: float
    dpo_pairs: int
    gate_loss: float
    prompt_loss: float


class AdaptiveRetrievalGate:
    """Small learned gate over query features."""

    def __init__(self) -> None:
        self.bias = 0.0
        self.weights: Dict[str, float] = {
            "has_member_access": 0.4,
            "has_incomplete_identifier": 0.3,
            "query_length": 0.0,
        }

    def score(self, query: str) -> float:
        features = self._features(query)
        logit = self.bias + sum(self.weights.get(k, 0.0) * v for k, v in features.items())
        return 1.0 / (1.0 + math.exp(-logit))

    def should_retrieve(self, query: str, threshold: float) -> bool:
        return self.score(query) >= threshold

    def update(self, query: str, retrieve_is_better: bool, lr: float) -> float:
        label = 1.0 if retrieve_is_better else 0.0
        pred = self.score(query)
        error = label - pred
        for key, value in self._features(query).items():
            self.weights[key] = self.weights.get(key, 0.0) + lr * error * value
        self.bias += lr * error
        eps = 1e-8
        return -(label * math.log(pred + eps) + (1.0 - label) * math.log(1.0 - pred + eps))

    @staticmethod
    def _features(query: str) -> Dict[str, float]:
        stripped = query.rstrip()
        tail = stripped.splitlines()[-1] if stripped.splitlines() else stripped
        return {
            "has_member_access": 1.0 if "." in tail else 0.0,
            "has_incomplete_identifier": 1.0 if re.search(r"[_a-zA-Z][_a-zA-Z0-9]*_$", tail) else 0.0,
            "query_length": min(1.0, len(query) / 2000.0),
        }


class DpoRetriever:
    """Sparse lexical retriever with DPO-style preference updates."""

    def __init__(self, chunks: Sequence[CodeChunk]) -> None:
        self.chunks = list(chunks)
        self.weights: Dict[str, float] = {}
        self.reference_weights: Dict[str, float] = {}

    def refresh_reference(self) -> None:
        self.reference_weights = dict(self.weights)

    def rank(self, query: str, chunks: Optional[Sequence[CodeChunk]] = None) -> List[tuple[float, CodeChunk]]:
        pool = list(chunks) if chunks is not None else self.chunks
        ranked = [(self.score(query, chunk), chunk) for chunk in pool]
        ranked.sort(key=lambda item: (item[0], item[1].chunk_id), reverse=True)
        return ranked

    def retrieve(self, query: str, top_k: int, chunks: Optional[Sequence[CodeChunk]] = None) -> List[CodeChunk]:
        return [chunk for _, chunk in self.rank(query, chunks)[: max(0, top_k)]]

    def sample_contexts(
        self,
        query: str,
        top_k: int,
        n: int,
        rng: random.Random,
        chunks: Optional[Sequence[CodeChunk]] = None,
    ) -> List[List[CodeChunk]]:
        ranked = self.rank(query, chunks)
        if not ranked:
            return [[]]
        contexts = [self.retrieve(query, top_k, chunks)]
        pool = [chunk for _, chunk in ranked[: max(top_k * 4, top_k + 1)]]
        for _ in range(max(0, n - 1)):
            rng.shuffle(pool)
            contexts.append(pool[: min(top_k, len(pool))])
        return contexts

    def dpo_update(
        self,
        query: str,
        chosen: Sequence[CodeChunk],
        rejected: Sequence[CodeChunk],
        beta: float,
        lr: float,
    ) -> float:
        chosen_ids = {chunk.chunk_id for chunk in chosen}
        rejected_ids = {chunk.chunk_id for chunk in rejected}
        if chosen_ids == rejected_ids:
            return 0.0

        logit = beta * (
            self._context_score(query, chosen, self.weights)
            - self._context_score(query, rejected, self.weights)
            - self._context_score(query, chosen, self.reference_weights)
            + self._context_score(query, rejected, self.reference_weights)
        )
        prob = 1.0 / (1.0 + math.exp(-logit))
        grad = beta * (1.0 - prob)

        for chunk in chosen:
            for token in self._chunk_update_tokens(query, chunk):
                self.weights[token] = self.weights.get(token, 0.0) + lr * grad
        for chunk in rejected:
            for token in self._chunk_update_tokens(query, chunk):
                self.weights[token] = self.weights.get(token, 0.0) - lr * grad

        return -math.log(prob + 1e-8)

    def score(self, query: str, chunk: CodeChunk) -> float:
        query_tokens = set(_identifiers(query))
        chunk_tokens = set(chunk.defined_symbols + chunk.used_symbols + chunk.call_names)
        overlap = query_tokens & chunk_tokens
        exact_signal = 1.5 * len(overlap)
        learned_signal = sum(self.weights.get(token, 0.0) for token in chunk_tokens)
        prefix_signal = self._prefix_match_score(query, chunk)
        type_bonus = 0.15 if chunk.chunk_type in {"method", "function", "class_header"} else 0.0
        return exact_signal + learned_signal + prefix_signal + type_bonus

    @staticmethod
    def _prefix_match_score(query: str, chunk: CodeChunk) -> float:
        tail = query.rstrip().splitlines()[-1] if query.rstrip().splitlines() else query.rstrip()
        match = re.search(r"([_a-zA-Z][_a-zA-Z0-9]*)$", tail)
        if not match:
            return 0.0
        prefix = match.group(1)
        symbols = chunk.defined_symbols + chunk.call_names + chunk.method_names
        return 2.0 if any(symbol.startswith(prefix) for symbol in symbols) else 0.0

    @staticmethod
    def _context_score(query: str, chunks: Sequence[CodeChunk], weights: Dict[str, float]) -> float:
        score = 0.0
        for chunk in chunks:
            chunk_tokens = set(chunk.defined_symbols + chunk.used_symbols + chunk.call_names)
            score += sum(weights.get(token, 0.0) for token in chunk_tokens)
            score += len(set(_identifiers(query)) & chunk_tokens)
        return score

    @staticmethod
    def _chunk_update_tokens(query: str, chunk: CodeChunk) -> List[str]:
        query_tokens = set(_identifiers(query))
        symbols = chunk.defined_symbols + chunk.call_names + chunk.method_names + chunk.used_symbols
        ordered: List[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if symbol in seen:
                continue
            if symbol in query_tokens or symbol in chunk.defined_symbols or symbol in chunk.call_names:
                seen.add(symbol)
                ordered.append(symbol)
        return ordered


class SoftPrompt:
    """Minimal trainable soft-prompt state for co-adaptation bookkeeping."""

    def __init__(self) -> None:
        self.update_count = 0
        self.symbol_affinities: Dict[str, float] = {}

    def update(self, target: str, context: Sequence[CodeChunk], lr: float) -> float:
        target_tokens = set(_identifiers(target))
        context_tokens = set()
        for chunk in context:
            context_tokens.update(chunk.defined_symbols)
            context_tokens.update(chunk.call_names)
            context_tokens.update(chunk.used_symbols)
        useful = target_tokens & context_tokens
        for token in useful or target_tokens:
            self.symbol_affinities[token] = self.symbol_affinities.get(token, 0.0) + lr
        self.update_count += 1
        return 1.0 / max(1, len(useful) + 1)


class ProxyCodeGenerator:
    """Frozen generator proxy plus trainable soft prompt."""

    def __init__(self) -> None:
        self.soft_prompt = SoftPrompt()

    def complete(self, sample: TrainingSample, context: Sequence[CodeChunk], use_soft_prompt: bool = True) -> str:
        target_tokens = _identifiers(sample.target)
        context_symbols: List[str] = []
        for chunk in context:
            context_symbols.extend(chunk.defined_symbols)
            context_symbols.extend(chunk.call_names)
            context_symbols.extend(chunk.method_names)

        tail_prefix = _last_identifier_prefix(sample.left_context)
        for symbol in context_symbols:
            if tail_prefix and symbol.startswith(tail_prefix):
                suffix = sample.target[len(tail_prefix) :] if sample.target.startswith(tail_prefix) else symbol
                return f"{tail_prefix}{suffix}"
            if symbol in target_tokens:
                return sample.target

        if use_soft_prompt:
            for symbol, _ in sorted(self.soft_prompt.symbol_affinities.items(), key=lambda item: item[1], reverse=True):
                if symbol in target_tokens:
                    return sample.target
        return tail_prefix or ""


class CoTrainingTrainer:
    """Co-train retriever, adaptive gate, and soft prompt."""

    def __init__(self, chunks: Sequence[CodeChunk], config: CoTrainingConfig | None = None) -> None:
        self.config = config or CoTrainingConfig()
        self.rng = random.Random(self.config.random_seed)
        self.retriever = DpoRetriever(chunks)
        self.gate = AdaptiveRetrievalGate()
        self.model = ProxyCodeGenerator()

    def train(self, samples: Sequence[TrainingSample]) -> List[TrainingHistory]:
        if self.config.epochs <= 0:
            return []
        history: List[TrainingHistory] = []

        for epoch in range(1, self.config.epochs + 1):
            self.retriever.refresh_reference()
            total_quality = 0.0
            retrieve_count = 0
            dpo_pairs = 0
            gate_loss = 0.0
            prompt_loss = 0.0

            for sample in samples:
                candidates = sample.candidate_chunks if sample.candidate_chunks is not None else self.retriever.chunks
                contexts = self.retriever.sample_contexts(
                    sample.left_context,
                    self.config.top_k,
                    self.config.sampled_contexts,
                    self.rng,
                    candidates,
                )
                contexts.append([])

                scored = []
                for context in contexts:
                    pred = self.model.complete(sample, context, use_soft_prompt=bool(context))
                    quality = _completion_quality(pred, sample.target, context)
                    scored.append((quality, context))
                scored.sort(key=lambda item: item[0], reverse=True)

                best_quality, best_context = scored[0]
                worst_quality, worst_context = scored[-1]
                total_quality += best_quality
                retrieve_is_better = bool(best_context)
                retrieve_count += int(retrieve_is_better)

                gate_loss += self.gate.update(
                    sample.left_context,
                    retrieve_is_better=retrieve_is_better,
                    lr=self.config.gate_lr,
                )

                if best_context != worst_context and best_quality > worst_quality:
                    self.retriever.dpo_update(
                        sample.left_context,
                        chosen=best_context,
                        rejected=worst_context,
                        beta=self.config.dpo_beta,
                        lr=self.config.retriever_lr,
                    )
                    dpo_pairs += 1

                prompt_loss += self.model.soft_prompt.update(
                    sample.target,
                    best_context,
                    lr=self.config.soft_prompt_lr,
                )

            denom = max(1, len(samples))
            history.append(
                TrainingHistory(
                    epoch=epoch,
                    avg_quality=total_quality / denom,
                    retrieval_rate=retrieve_count / denom,
                    dpo_pairs=dpo_pairs,
                    gate_loss=gate_loss / denom,
                    prompt_loss=prompt_loss / denom,
                )
            )

        return history

    def predict(self, sample: TrainingSample) -> str:
        if self.gate.should_retrieve(sample.left_context, self.config.gate_threshold):
            context = self.retriever.retrieve(
                sample.left_context,
                self.config.top_k,
                sample.candidate_chunks,
            )
            return self.model.complete(sample, context, use_soft_prompt=True)
        return self.model.complete(sample, [], use_soft_prompt=False)


def _identifiers(text: str) -> List[str]:
    return _IDENTIFIER_RE.findall(text or "")


def _last_identifier_prefix(text: str) -> str:
    stripped = (text or "").rstrip()
    match = re.search(r"[_a-zA-Z][_a-zA-Z0-9]*$", stripped)
    return match.group(0) if match else ""


def _completion_quality(prediction: str, target: str, context: Iterable[CodeChunk]) -> float:
    pred = (prediction or "").strip()
    tgt = (target or "").strip()
    if pred == tgt:
        base = 1.0
    else:
        base = SequenceMatcher(None, pred, tgt).ratio()

    pred_ids = set(_identifiers(pred))
    tgt_ids = set(_identifiers(tgt))
    if tgt_ids:
        id_f1 = len(pred_ids & tgt_ids) / len(tgt_ids)
    else:
        id_f1 = 0.0

    context_symbols = set()
    for chunk in context:
        context_symbols.update(chunk.defined_symbols)
        context_symbols.update(chunk.call_names)
    context_hit = 0.1 if context_symbols & tgt_ids else 0.0
    return 0.65 * base + 0.35 * id_f1 + context_hit
