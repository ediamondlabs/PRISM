"""
Replay Buffer for PPO trajectory storage.

Stores trajectories and computes advantages using GAE.
"""

from typing import Dict, List, Optional

import numpy as np
import torch


class RolloutBuffer:
    """Buffer for storing PPO rollouts."""

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        num_vehicles: int,
        max_neighbors: int,
        k_neighbors: int,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize rollout buffer.

        Args:
            buffer_size: Maximum buffer size
            state_dim: State dimension
            num_vehicles: Number of vehicles
            max_neighbors: Maximum number of neighbors
            k_neighbors: Number of neighbors selected (k=2)
            device: Device for tensors
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.num_vehicles = num_vehicles
        self.max_neighbors = max_neighbors
        self.k_neighbors = k_neighbors
        self.device = device

        # Storage
        self.states = torch.zeros(buffer_size, state_dim, device=device)
        self.who_actions = torch.zeros(buffer_size, k_neighbors, dtype=torch.long, device=device)
        self.what_actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, device=device)

        # For centralized critic
        self.global_states = torch.zeros(
            buffer_size, num_vehicles, state_dim, device=device
        )

        # Neighbor masks
        self.neighbor_masks = torch.zeros(
            buffer_size, max_neighbors, dtype=torch.bool, device=device
        )

        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        who_action: np.ndarray,
        what_action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        global_state: Optional[np.ndarray] = None,
        neighbor_mask: Optional[np.ndarray] = None,
    ):
        """Add transition to buffer.

        Args:
            state: State (state_dim,)
            who_action: WHO action (k_neighbors,)
            what_action: WHAT action (scalar)
            log_prob: Log probability
            reward: Reward
            value: State value
            done: Done flag
            global_state: Global state (num_vehicles, state_dim)
            neighbor_mask: Neighbor mask (max_neighbors,)
        """
        self.states[self.ptr] = torch.from_numpy(state).float()
        self.who_actions[self.ptr] = torch.from_numpy(who_action).long()
        self.what_actions[self.ptr] = what_action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)

        if global_state is not None:
            self.global_states[self.ptr] = torch.from_numpy(global_state).float()

        if neighbor_mask is not None:
            self.neighbor_masks[self.ptr] = torch.from_numpy(neighbor_mask).bool()

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def get(
        self, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """Get all data and compute advantages.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Dictionary with all buffer data and computed advantages/returns
        """
        assert self.size > 0, "Buffer is empty"

        # Get all data
        indices = torch.arange(self.size, device=self.device)

        states = self.states[indices]
        who_actions = self.who_actions[indices]
        what_actions = self.what_actions[indices]
        log_probs = self.log_probs[indices]
        rewards = self.rewards[indices]
        values = self.values[indices]
        dones = self.dones[indices]
        global_states = self.global_states[indices]
        neighbor_masks = self.neighbor_masks[indices]

        # Compute advantages using GAE
        advantages = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]

            # TD error
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        # Returns
        returns = advantages + values

        return {
            "states": states,
            "who_actions": who_actions,
            "what_actions": what_actions,
            "log_probs": log_probs,
            "rewards": rewards,
            "values": values,
            "advantages": advantages,
            "returns": returns,
            "global_states": global_states,
            "neighbor_masks": neighbor_masks,
        }

    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.size = 0


class MultiAgentRolloutBuffer:
    """Buffer for multi-agent PPO rollouts."""

    def __init__(
        self,
        buffer_size: int,
        state_dim: int,
        num_vehicles: int,
        max_neighbors: int,
        k_neighbors: int,
        device: torch.device = torch.device("cpu"),
        vehicle_ids: Optional[List[int]] = None,
    ):
        """Initialize multi-agent rollout buffer.

        Args:
            buffer_size: Maximum buffer size per agent
            state_dim: State dimension
            num_vehicles: Number of vehicles
            max_neighbors: Maximum number of neighbors
            k_neighbors: Number of neighbors selected
            device: Device for tensors
            vehicle_ids: List of actual vehicle IDs (if None, uses range(num_vehicles))
        """
        self.num_vehicles = num_vehicles
        self.device = device
        self.vehicle_ids = vehicle_ids if vehicle_ids is not None else list(range(num_vehicles))

        # Create buffer for each vehicle
        self.buffers = {
            vid: RolloutBuffer(
                buffer_size=buffer_size,
                state_dim=state_dim,
                num_vehicles=num_vehicles,
                max_neighbors=max_neighbors,
                k_neighbors=k_neighbors,
                device=device,
            )
            for vid in self.vehicle_ids
        }

    def add(
        self,
        vehicle_id: int,
        state: np.ndarray,
        who_action: np.ndarray,
        what_action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        global_state: Optional[np.ndarray] = None,
        neighbor_mask: Optional[np.ndarray] = None,
    ):
        """Add transition for a specific vehicle.

        Args:
            vehicle_id: Vehicle ID
            state: State
            who_action: WHO action
            what_action: WHAT action
            log_prob: Log probability
            reward: Reward
            value: State value
            done: Done flag
            global_state: Global state
            neighbor_mask: Neighbor mask
        """
        self.buffers[vehicle_id].add(
            state=state,
            who_action=who_action,
            what_action=what_action,
            log_prob=log_prob,
            reward=reward,
            value=value,
            done=done,
            global_state=global_state,
            neighbor_mask=neighbor_mask,
        )

    def get_all(
        self, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """Get all data from all vehicles and combine.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            Dictionary with combined data from all vehicles
        """
        all_data = []

        for vid in self.vehicle_ids:
            if self.buffers[vid].size > 0:
                data = self.buffers[vid].get(gamma, gae_lambda)
                all_data.append(data)

        if len(all_data) == 0:
            raise ValueError("All buffers are empty")

        # Combine data
        combined = {
            key: torch.cat([data[key] for data in all_data], dim=0)
            for key in all_data[0].keys()
        }

        return combined

    def clear(self):
        """Clear all buffers."""
        for buffer in self.buffers.values():
            buffer.clear()

    def __len__(self) -> int:
        """Get total number of samples across all buffers."""
        return sum(buffer.size for buffer in self.buffers.values())
