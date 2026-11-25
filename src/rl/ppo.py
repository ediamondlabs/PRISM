"""
Proximal Policy Optimization (PPO) algorithm with Generalized Advantage Estimation (GAE).

Implements PPO with clipped objective and GAE for advantage computation.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.rl.actor import PPOActor
from src.rl.critic import CentralizedCritic


class PPO:
    """PPO algorithm with GAE."""

    def __init__(
        self,
        actor: PPOActor,
        critic: nn.Module,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize PPO.

        Args:
            actor: Actor network
            critic: Critic network (centralized or decentralized)
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device for computation
        """
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: Rewards (batch, seq_len)
            values: State values (batch, seq_len)
            dones: Done flags (batch, seq_len)
            next_values: Next state values (batch, seq_len)

        Returns:
            Tuple of (advantages, returns)
        """
        batch_size, seq_len = rewards.shape

        advantages = torch.zeros_like(rewards)
        gae = 0

        # Compute GAE backwards through time
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = next_values[:, t]
            else:
                next_value = values[:, t + 1]

            # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]

            # GAE: A_t = δ_t + γ*λ*A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[:, t]) * gae
            advantages[:, t] = gae

        # Returns: R_t = A_t + V(s_t)
        returns = advantages + values

        return advantages, returns

    def update(
        self,
        states: torch.Tensor,
        who_actions: torch.Tensor,
        what_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        global_states: Optional[torch.Tensor] = None,
        neighbor_masks: Optional[torch.Tensor] = None,
        num_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Update actor and critic networks.

        Args:
            states: States (N, state_dim)
            who_actions: WHO actions (N, k_neighbors)
            what_actions: WHAT actions (N,)
            old_log_probs: Old log probabilities (N,)
            advantages: Advantages (N,)
            returns: Returns (N,)
            global_states: Global states for centralized critic (N, num_vehicles, state_dim)
            neighbor_masks: Neighbor masks (N, max_neighbors)
            num_epochs: Number of update epochs
            batch_size: Mini-batch size

        Returns:
            Dictionary with training statistics
        """
        num_samples = states.size(0)
        indices = torch.arange(num_samples)

        stats = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
        }

        for epoch in range(num_epochs):
            # Shuffle indices
            perm = torch.randperm(num_samples)
            indices = indices[perm]

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                # Get batch
                batch_states = states[batch_indices]
                batch_who_actions = who_actions[batch_indices]
                batch_what_actions = what_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                batch_neighbor_masks = None
                if neighbor_masks is not None:
                    batch_neighbor_masks = neighbor_masks[batch_indices]

                # Normalize advantages
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                    batch_advantages.std() + 1e-8
                )

                # Evaluate actions
                eval_results = self.actor.evaluate_actions(
                    batch_states,
                    batch_who_actions,
                    batch_what_actions,
                    batch_neighbor_masks,
                )

                log_probs = eval_results["log_prob"]
                entropy = eval_results["entropy"].mean()

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Add entropy bonus
                actor_loss = actor_loss - self.entropy_coef * entropy

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update
                if isinstance(self.critic, CentralizedCritic):
                    assert global_states is not None, "Global states required for centralized critic"
                    batch_global_states = global_states[batch_indices]
                    values = self.critic(batch_global_states).squeeze(-1)
                else:
                    values = self.critic(batch_states).squeeze(-1)

                critic_loss = F.mse_loss(values, batch_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Track statistics
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                stats["actor_loss"] += actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += entropy.item()
                stats["clip_fraction"] += clip_fraction.item()

        # Average statistics
        num_updates = num_epochs * ((num_samples + batch_size - 1) // batch_size)
        for key in stats:
            stats[key] /= num_updates

        return stats

    def get_value(
        self, state: torch.Tensor, global_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get state value from critic.

        Args:
            state: Local state (batch, state_dim)
            global_state: Global state for centralized critic (batch, num_vehicles, state_dim)

        Returns:
            State value (batch, 1)
        """
        if isinstance(self.critic, CentralizedCritic):
            assert global_state is not None, "Global state required for centralized critic"
            return self.critic(global_state)
        else:
            return self.critic(state)

    def save(self, path: str):
        """Save actor and critic models.

        Args:
            path: Save path (will save as path_actor.pt and path_critic.pt)
        """
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load actor and critic models.

        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
