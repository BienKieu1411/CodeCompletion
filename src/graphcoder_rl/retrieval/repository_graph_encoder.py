"""
Lightweight Graph Encoder for repository graphs.

Implements a small message-passing encoder inspired by GraphSAGE/GGNN ideas
without heavyweight framework dependencies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


_IDENTIFIER_RE = re.compile(r"[_a-zA-Z][_a-zA-Z0-9]*")


def _safe_log1p(x: float) -> float:
    return math.log(1.0 + max(0.0, x))


def _l2_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _normalize(vec: Sequence[float]) -> List[float]:
    n = _l2_norm(vec)
    if n < 1e-9:
        return [0.0 for _ in vec]
    return [v / n for v in vec]


def _mean(vectors: Iterable[Sequence[float]], dim: int) -> List[float]:
    total = [0.0] * dim
    count = 0
    for vec in vectors:
        if len(vec) != dim:
            continue
        count += 1
        for i in range(dim):
            total[i] += float(vec[i])
    if count == 0:
        return [0.0] * dim
    return [x / count for x in total]


@dataclass
class NodeFeature:
    node_type: str
    text: str
    edge_counts: Dict[str, int]
    defined_symbols: int
    used_symbols: int
    call_count: int


class LightweightGraphEncoder:
    """
    Computes graph-aware node embeddings via simple iterative aggregation.

    - Initial features combine lexical/statistical + node-type priors.
    - Each layer aggregates neighbor representations by edge-type weighting.
    """

    _NODE_TYPE_PRIOR = {
        "file": [1.0, 0.0, 0.0],
        "chunk": [0.0, 1.0, 0.0],
        "anchor": [0.0, 0.0, 1.0],
        "symbol": [0.35, 0.35, 0.3],
    }

    _EDGE_WEIGHTS = {
        "calls": 1.0,
        "imports": 0.9,
        "contains": 0.45,
        "inside": 0.3,
        "adjacent_chunk": 0.25,
        "mentions": 0.5,
        "defines": 0.5,
        "uses_type": 0.65,
        "inherits": 0.7,
        "data_dependency": 0.75,
        "control_dependency": 0.7,
        "overrides": 0.72,
    }

    def __init__(self, hidden_dim: int = 16, n_layers: int = 2):
        self.hidden_dim = max(8, hidden_dim)
        self.n_layers = max(1, n_layers)

    def _base_feature_vector(self, feat: NodeFeature) -> List[float]:
        tokens = _IDENTIFIER_RE.findall(feat.text or "")
        token_count = len(tokens)
        unique_ratio = (len(set(tokens)) / max(1, token_count)) if token_count else 0.0

        pri = self._NODE_TYPE_PRIOR.get(feat.node_type, [0.0, 0.0, 0.0])

        base = [
            _safe_log1p(len(feat.text)),
            _safe_log1p(token_count),
            unique_ratio,
            _safe_log1p(feat.defined_symbols),
            _safe_log1p(feat.used_symbols),
            _safe_log1p(feat.call_count),
            _safe_log1p(sum(feat.edge_counts.values())),
            _safe_log1p(feat.edge_counts.get("imports", 0)),
            _safe_log1p(feat.edge_counts.get("calls", 0)),
            _safe_log1p(feat.edge_counts.get("adjacent_chunk", 0)),
            pri[0],
            pri[1],
            pri[2],
        ]

        # Pad to hidden_dim in a deterministic way.
        if len(base) < self.hidden_dim:
            pad = [0.0] * (self.hidden_dim - len(base))
            base = base + pad
        else:
            base = base[: self.hidden_dim]

        return _normalize(base)

    def encode(
        self,
        node_features: Dict[str, NodeFeature],
        adjacency: Dict[str, List[Tuple[str, str]]],
    ) -> Dict[str, List[float]]:
        """
        Args:
            node_features: node_id -> NodeFeature
            adjacency: node_id -> List[(neighbor_id, edge_type)]
        Returns:
            node_embeddings: node_id -> embedding list[float]
        """
        h: Dict[str, List[float]] = {
            node_id: self._base_feature_vector(feat)
            for node_id, feat in node_features.items()
        }

        for _ in range(self.n_layers):
            next_h: Dict[str, List[float]] = {}
            for node_id, self_vec in h.items():
                neighbors = adjacency.get(node_id, [])
                if not neighbors:
                    next_h[node_id] = self_vec
                    continue

                weighted_neighbors: List[List[float]] = []
                for nbr_id, edge_type in neighbors:
                    if nbr_id not in h:
                        continue
                    w = self._EDGE_WEIGHTS.get(edge_type, 0.2)
                    weighted_neighbors.append([w * x for x in h[nbr_id]])

                if not weighted_neighbors:
                    agg = [0.0] * self.hidden_dim
                else:
                    agg = _mean(weighted_neighbors, self.hidden_dim)

                merged = [0.6 * self_vec[i] + 0.4 * agg[i] for i in range(self.hidden_dim)]
                next_h[node_id] = _normalize(merged)

            h = next_h

        return h
