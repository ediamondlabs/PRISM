"""
Actor Network for PPO with dual-head architecture.

Implements decentralized actor with:
- WHO head: Select k neighbors for communication
- WHAT head: Select representation type (W=Weights, E=Embeddings)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPOActor(nn.Module):
    """PPO Actor with WHO and WHAT heads for communication decisions."""

    def __init__(
        self,
        state_dim: int = 106,
        shared_hidden_dim: int = 128,
        when_hidden_dim: int = 32,
        who_hidden_dim: int = 64,
        what_hidden_dim: int = 64,
        max_neighbors: int = 20,
        k_neighbors: int = 2,
        num_repr_types: int = 2,
        dropout: float = 0.1,
        use_when_head: bool = False,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.max_neighbors = max_neighbors
        self.k_neighbors = k_neighbors
        self.num_repr_types = num_repr_types
        self.use_when_head = use_when_head

        # Shared encoder: processes state into common representation
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, shared_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(shared_hidden_dim, shared_hidden_dim),
            nn.ReLU(),
        )

        # WHEN head: Decide whether to communicate (binary: 0=skip, 1=communicate)
        if use_when_head:
            self.when_head = nn.Sequential(
                nn.Linear(shared_hidden_dim, when_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(when_hidden_dim, 2),  # Binary decision
            )

        # WHO head: Select k neighbors
        # Output: scores for each neighbor (use top-k for selection)
        self.who_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, who_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(who_hidden_dim, max_neighbors),
        )

        # WHAT head: Select representation type
        # Output: logits for categories (2 for E/P, 3 for W/E/P)
        self.what_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, what_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(what_hidden_dim, num_repr_types),
        )

    def forward(
        self, state: torch.Tensor, neighbor_mask: Optional[torch.Tensor] = None
    ) -> Tuple:
        """Forward pass through actor.

        Args:
            state: State tensor (batch, state_dim)
            neighbor_mask: Boolean mask for valid neighbors (batch, max_neighbors)

        Returns:
            Tuple of (when_logits, who_logits, what_logits) if use_when_head=True
            Tuple of (who_logits, what_logits) if use_when_head=False
        """
        # Shared encoding
        shared_features = self.shared_encoder(state)

        # WHEN head: communication decision (optional)
        if self.use_when_head:
            when_logits = self.when_head(shared_features)
        else:
            when_logits = None

        # WHO head: neighbor selection scores
        who_logits = self.who_head(shared_features)

        # Apply mask to invalid neighbors (set to large negative value)
        if neighbor_mask is not None:
            who_logits = who_logits.masked_fill(~neighbor_mask, -1e9)

        # WHAT head: representation type logits
        what_logits = self.what_head(shared_features)

        if self.use_when_head:
            return when_logits, who_logits, what_logits
        else:
            return who_logits, what_logits

    def get_action(
        self,
        state: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Sample action from policy.

        Args:
            state: State tensor (batch, state_dim)
            neighbor_mask: Boolean mask for valid neighbors (batch, max_neighbors)
            deterministic: If True, select argmax; otherwise sample

        Returns:
            Dictionary with:
                - when_action: Communicate decision (batch,) [optional if use_when_head=True]
                - who_action: Selected neighbor indices (batch, k_neighbors)
                - what_action: Selected representation type (batch,)
                - when_log_prob: Log probability of WHEN action (batch,) [optional]
                - who_log_prob: Log probability of WHO action (batch,)
                - what_log_prob: Log probability of WHAT action (batch,)
                - when_logits: Communication decision logits (batch, 2) [optional]
                - who_logits: All neighbor scores (batch, max_neighbors)
                - what_logits: All repr type logits (batch, num_repr_types)
        """
        forward_output = self.forward(state, neighbor_mask)

        if self.use_when_head:
            when_logits, who_logits, what_logits = forward_output
        else:
            who_logits, what_logits = forward_output
            when_logits = None

        # WHEN action: Decide whether to communicate (optional)
        if self.use_when_head:
            when_dist = torch.distributions.Categorical(logits=when_logits)
            if deterministic:
                when_action = when_logits.argmax(dim=-1)
            else:
                when_action = when_dist.sample()
            when_log_prob = when_dist.log_prob(when_action)
        else:
            when_action = None
            when_log_prob = None

        # WHO action: Select top-k neighbors
        if deterministic:
            # Deterministic: select top-k
            _, who_action = torch.topk(who_logits, k=self.k_neighbors, dim=-1)
        else:
            # Stochastic: sample from Gumbel-Softmax or use categorical sampling
            # For simplicity, use top-k sampling with Gumbel noise
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(who_logits) + 1e-8) + 1e-8)
            noisy_logits = who_logits + gumbel_noise
            _, who_action = torch.topk(noisy_logits, k=self.k_neighbors, dim=-1)

        # Compute WHO log probability
        # Use categorical distribution over all neighbors, then sum log probs of selected
        who_probs = F.softmax(who_logits, dim=-1)
        who_log_probs = torch.log(who_probs + 1e-8)

        # Sum log probs of selected neighbors (treating as independent selections)
        batch_idx = torch.arange(who_action.size(0)).unsqueeze(1)
        selected_log_probs = who_log_probs[batch_idx, who_action]
        who_log_prob = selected_log_probs.sum(dim=-1)

        # WHAT action: Select representation type
        what_dist = torch.distributions.Categorical(logits=what_logits)
        if deterministic:
            what_action = what_logits.argmax(dim=-1)
        else:
            what_action = what_dist.sample()

        what_log_prob = what_dist.log_prob(what_action)

        result = {
            "who_action": who_action,
            "what_action": what_action,
            "who_log_prob": who_log_prob,
            "what_log_prob": what_log_prob,
            "who_logits": who_logits,
            "what_logits": what_logits,
        }

        # Add WHEN action if used
        if self.use_when_head:
            result["when_action"] = when_action
            result["when_log_prob"] = when_log_prob
            result["when_logits"] = when_logits

        return result

    def evaluate_actions(
        self,
        state: torch.Tensor,
        who_action: torch.Tensor,
        what_action: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None,
        when_action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate log probabilities and entropy of given actions.

        Args:
            state: State tensor (batch, state_dim)
            who_action: WHO actions to evaluate (batch, k_neighbors)
            what_action: WHAT actions to evaluate (batch,)
            neighbor_mask: Boolean mask for valid neighbors (batch, max_neighbors)
            when_action: WHEN actions to evaluate (batch,) [optional if use_when_head=True]

        Returns:
            Dictionary with log_probs and entropies
        """
        forward_output = self.forward(state, neighbor_mask)

        if self.use_when_head:
            when_logits, who_logits, what_logits = forward_output
        else:
            who_logits, what_logits = forward_output
            when_logits = None

        # WHEN log probability and entropy (optional)
        if self.use_when_head and when_action is not None:
            when_dist = torch.distributions.Categorical(logits=when_logits)
            when_log_prob = when_dist.log_prob(when_action)
            when_entropy = when_dist.entropy()
        else:
            when_log_prob = torch.zeros_like(who_action[:, 0])  # (batch,)
            when_entropy = torch.zeros_like(who_action[:, 0])

        # WHO log probability
        who_probs = F.softmax(who_logits, dim=-1)
        who_log_probs = torch.log(who_probs + 1e-8)

        batch_idx = torch.arange(who_action.size(0)).unsqueeze(1)
        selected_log_probs = who_log_probs[batch_idx, who_action]
        who_log_prob = selected_log_probs.sum(dim=-1)

        # WHO entropy (use entropy of categorical distribution)
        who_entropy = -(who_probs * who_log_probs).sum(dim=-1)

        # WHAT log probability and entropy
        what_dist = torch.distributions.Categorical(logits=what_logits)
        what_log_prob = what_dist.log_prob(what_action)
        what_entropy = what_dist.entropy()

        # Total log probability (sum of independent decisions)
        total_log_prob = when_log_prob + who_log_prob + what_log_prob

        result = {
            "log_prob": total_log_prob,
            "who_log_prob": who_log_prob,
            "what_log_prob": what_log_prob,
            "entropy": when_entropy + who_entropy + what_entropy,
            "who_entropy": who_entropy,
            "what_entropy": what_entropy,
        }

        # Add WHEN stats if used
        if self.use_when_head:
            result["when_log_prob"] = when_log_prob
            result["when_entropy"] = when_entropy

        return result

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
