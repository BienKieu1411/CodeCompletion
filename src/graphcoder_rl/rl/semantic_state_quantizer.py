"""
Semantic state quantizer (lightweight VQ-style).

Maps continuous graph embeddings into discrete semantic state IDs for
stable RL traversal states.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


def _l2_sq(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


def _normalize(vec: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(v * v for v in vec))
    if n < 1e-9:
        return [0.0 for _ in vec]
    return [v / n for v in vec]


def _seeded_codebook_vector(dim: int, idx: int, seed_text: str) -> List[float]:
    """Generate deterministic pseudo-random vectors without external deps."""
    raw = hashlib.sha256(f"{seed_text}:{idx}".encode("utf8")).digest()
    vals: List[float] = []
    for i in range(dim):
        byte = raw[i % len(raw)]
        vals.append((byte / 255.0) * 2.0 - 1.0)
    return _normalize(vals)


@dataclass
class QuantizedState:
    state_id: int
    state_label: str
    distance: float


class SemanticStateQuantizer:
    """
    Deterministic vector quantizer.

    Supports:
    - hard quantization: nearest codebook index
    - soft quantization: top-k weighted interpolation (for analysis)
    """

    def __init__(self, dim: int = 16, n_codes: int = 16, seed: str = "graphcoderrl-vq"):
        self.dim = max(8, dim)
        self.n_codes = max(4, n_codes)
        self.seed = seed
        self.codebook: List[List[float]] = [
            _seeded_codebook_vector(self.dim, i, self.seed)
            for i in range(self.n_codes)
        ]

    @staticmethod
    def _state_label(code_id: int) -> str:
        labels = [
            "generic",
            "controller",
            "service",
            "repository",
            "model",
            "utility",
            "api",
            "schema",
            "config",
            "io",
            "domain",
            "validation",
            "query",
            "pipeline",
            "interface",
            "other",
        ]
        return labels[code_id % len(labels)]

    def quantize(self, embedding: Sequence[float]) -> QuantizedState:
        emb = _normalize(list(embedding)[: self.dim] + [0.0] * max(0, self.dim - len(embedding)))
        best_idx = 0
        best_dist = float("inf")
        for i, code in enumerate(self.codebook):
            d = _l2_sq(emb, code)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return QuantizedState(state_id=best_idx, state_label=self._state_label(best_idx), distance=best_dist)

    def soft_quantize(self, embedding: Sequence[float], top_k: int = 3) -> List[Tuple[int, float]]:
        emb = _normalize(list(embedding)[: self.dim] + [0.0] * max(0, self.dim - len(embedding)))
        dists = [(i, _l2_sq(emb, code)) for i, code in enumerate(self.codebook)]
        dists.sort(key=lambda x: x[1])
        top = dists[: max(1, min(top_k, len(dists)))]

        # Convert distances to normalized similarity weights.
        sims = [1.0 / (1e-6 + d) for _, d in top]
        z = sum(sims) or 1.0
        return [(idx, sim / z) for (idx, _), sim in zip(top, sims)]

    def quantize_batch(self, embeddings: Dict[str, Sequence[float]]) -> Dict[str, QuantizedState]:
        return {node_id: self.quantize(vec) for node_id, vec in embeddings.items()}
