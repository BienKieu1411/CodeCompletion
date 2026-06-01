"""
Graph action policy head for RL traversal.

Takes fixed action features from graph retriever and outputs action logits.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTraversalPolicy(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_features: (N_actions, input_dim)
        Returns:
            logits: (N_actions,)
        """
        if action_features.dim() != 2:
            raise ValueError("action_features must be 2D")
        return self.net(action_features).squeeze(-1)

    def logprobs(self, action_features: torch.Tensor) -> torch.Tensor:
        logits = self.forward(action_features)
        return F.log_softmax(logits, dim=-1)

    def selected_logprobs(
        self,
        action_features: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            selected_logprobs: (K,)
            all_logprobs: (N,)
        """
        all_lps = self.logprobs(action_features)
        if selected_indices.numel() == 0:
            return all_lps[:1], all_lps
        return all_lps[selected_indices], all_lps
