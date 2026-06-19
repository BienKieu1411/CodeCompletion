"""Adaptive retrieval gate (neural version).

The gate receives the *query embedding* produced by the retriever encoder and
outputs a scalar probability g ∈ [0, 1] indicating whether cross-file
retrieval is likely to help for the current completion.

Design
------
* **MLP backbone** — two-layer MLP with GELU activation and dropout.
* **Log-probability interface** — ``log_probs`` returns both
  ``log P(continue|q)`` and ``log P(stop|q)`` for use in the combined
  DPO score ``S(q, C) = gate_score + retrieval_score``.
* **Entropy regularisation** — discourages gate collapse to always-on/off.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NeuralGate(nn.Module):
    """Adaptive retrieval gate operating on dense query embeddings.

    Parameters
    ----------
    input_dim : int
        Dimension of the query embedding (must match retriever output).
    hidden_dim : int
        Hidden layer size.
    dropout : float
        Dropout rate between MLP layers.
    entropy_weight : float
        Weight for the entropy regularisation term.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        entropy_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.entropy_weight = entropy_weight

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Initialise bias towards retrieval (safer default).
        with torch.no_grad():
            self.mlp[-1].bias.fill_(0.2)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, query_vec: torch.Tensor) -> torch.Tensor:
        """Compute gate probability g = P(continue/retrieve | query).

        Parameters
        ----------
        query_vec : (*, input_dim)

        Returns
        -------
        g : (*, 1) tensor with values in [0, 1].
        """
        return torch.sigmoid(self.mlp(query_vec))

    # ── Log-probabilities for combined DPO scoring ────────────────────────

    def log_probs(
        self, query_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (log P_gate(continue|q), log P_gate(stop|q)).

        Used in the combined score:
            S(q, C_retrieved) = log P_gate(continue|q) + mean(sim(q, snippet_i))
            S(q, C_stop)      = log P_gate(stop|q)
        """
        logit = self.mlp(query_vec)  # raw logit before sigmoid
        log_continue = F.logsigmoid(logit)          # log σ(x)
        log_stop = F.logsigmoid(-logit)              # log(1 - σ(x)) = log σ(-x)
        return log_continue.squeeze(-1), log_stop.squeeze(-1)

    # ── Decision ──────────────────────────────────────────────────────────

    def should_retrieve(
        self,
        query_vec: torch.Tensor,
        threshold: float = 0.5,
    ) -> bool:
        """Deterministic decision for inference."""
        with torch.no_grad():
            g = self.forward(query_vec)
        return bool(g.item() >= threshold)

    # ── Loss ──────────────────────────────────────────────────────────────

    def gate_loss(
        self,
        g: torch.Tensor,
        retrieve_is_better: bool,
    ) -> torch.Tensor:
        """Binary cross-entropy + entropy regularisation.

        Parameters
        ----------
        g : scalar tensor — gate output probability.
        retrieve_is_better : bool — supervision label from NLL comparison.
        """
        label = torch.tensor(
            [1.0 if retrieve_is_better else 0.0],
            device=g.device,
            dtype=g.dtype,
        )
        g_clamped = g.view(-1).clamp(1e-7, 1.0 - 1e-7)

        bce = F.binary_cross_entropy(g_clamped, label)

        # Entropy regularisation: maximise H(g) to prevent collapse
        entropy = -(
            g_clamped * torch.log(g_clamped)
            + (1.0 - g_clamped) * torch.log(1.0 - g_clamped)
        )
        return bce - self.entropy_weight * entropy.mean()

    # ── Monitoring ────────────────────────────────────────────────────────

    def gate_stats(self, query_vecs: torch.Tensor) -> Dict[str, float]:
        """Compute gate statistics over a batch for monitoring."""
        with torch.no_grad():
            gs = self.forward(query_vecs).squeeze(-1)
        return {
            "mean": float(gs.mean()),
            "std": float(gs.std()) if gs.numel() > 1 else 0.0,
            "retrieve_rate": float((gs >= 0.5).float().mean()),
        }
