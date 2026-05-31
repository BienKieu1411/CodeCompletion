"""
PPO Trainer with Value Head + GAE + CAHM Masking

Key improvements:
- Value head (Critic) for proper advantage estimation
- Generalized Advantage Estimation (GAE-λ)
- Entropy bonus for exploration
- CAHM mask integration into advantages
- Gradient clipping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """
    Value head (Critic) that predicts baseline reward from UniXCoder embeddings.
    Attached on top of the retriever's CLS/mean-pooled embedding.
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, hidden_dim) — from retriever encoder
        Returns:
            values: (batch,) — predicted baseline values
        """
        return self.net(embeddings).squeeze(-1)


class GraphFRLPPOTrainer:
    """
    PPO trainer with CAHM masking for fine-grained credit assignment.

    Implements:
    - Clipped surrogate objective (PPO-Clip)
    - CAHM-masked advantages (Theorem 1: variance reduction)
    - Entropy bonus for exploration
    - Value head training (critic loss)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        cahm_engine,
        value_head: ValueHead = None,
        value_optimizer: torch.optim.Optimizer = None,
        clip_param: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
    ):
        self.model = model
        self.optimizer = optimizer
        self.cahm_engine = cahm_engine
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        # Value head (Critic)
        if value_head is not None:
            self.value_head = value_head
            self.value_optimizer = value_optimizer or torch.optim.AdamW(
                value_head.parameters(), lr=1e-4
            )
        else:
            self.value_head = None
            self.value_optimizer = None

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation.

        For retrieval actions, we treat each action independently
        (single-step MDP), so GAE simplifies to: A = R - V.

        The CAHM mask is applied AFTER advantage computation
        to zero out noisy actions.
        """
        # Single-step advantage (retrieval is essentially a bandit problem)
        advantages = rewards - values.detach()

        # Apply CAHM mask — Theorem 1: Var(A_masked) < Var(A_full)
        masked_advantages = advantages * mask

        return masked_advantages

    def step(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        mask_k: torch.Tensor,
    ) -> dict:
        """
        One PPO update step.

        Args:
            logprobs: Current policy log-probabilities (requires_grad=True)
            old_logprobs: Old policy log-probabilities (detached)
            values: Value estimates (from value head or zeros)
            rewards: Reward signals
            mask_k: CAHM mask (binary)

        Returns:
            dict with loss components for logging
        """
        self.optimizer.zero_grad()
        if self.value_optimizer:
            self.value_optimizer.zero_grad()

        # 1. Compute masked advantages
        advantages = self.compute_gae(rewards, values, mask_k)

        # Normalize advantages (critical for stable PPO)
        if advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 2. Policy loss (PPO-Clip)
        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 3. Entropy bonus (encourage exploration)
        # For log-softmax outputs, entropy ≈ -sum(p * log(p))
        # We approximate using the logprobs directly
        entropy = -(logprobs.exp() * logprobs).mean()
        entropy_loss = -self.entropy_coef * entropy

        # 4. Value loss (if value head exists)
        value_loss = torch.tensor(0.0, device=logprobs.device)
        if self.value_head is not None:
            value_loss = self.value_coef * F.mse_loss(values, rewards)

        # Total loss
        total_loss = policy_loss + entropy_loss + value_loss

        # Backpropagate
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.value_head is not None and self.value_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.max_grad_norm)
            self.value_optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "value_loss": value_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "clip_fraction": ((ratio - 1.0).abs() > self.clip_param).float().mean().item(),
        }
