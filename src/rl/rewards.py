"""
Reward function for V2V communication RL agent.

r_total = βi × r_immediate + βg × r_global

where:
- r_immediate = βc × comm_success + βs × neighbor_stability
- r_global = perf_scale × (loss_prev - loss_curr) / loss_prev
"""

from typing import Dict, List, Optional

import numpy as np


class RewardFunction:
    """Reward function for V2V communication based on training loss improvement."""

    def __init__(
        self,
        immediate_weight: float = 0.3,
        global_weight: float = 0.7,
        perf_scale: float = 100.0,
        communication_success_weight: float = 0.5,
        neighbor_stability_weight: float = 0.5,
        **kwargs,
    ):
        self.immediate_weight = immediate_weight
        self.global_weight = global_weight
        self.perf_scale = perf_scale
        self.communication_success_weight = communication_success_weight
        self.neighbor_stability_weight = neighbor_stability_weight

        # Track previous training loss for percentage improvement
        self.prev_train_loss = None

    def compute_immediate_reward(
        self,
        vehicle_id: int,
        what_action: int,
        communication_success: bool,
        anomaly_detected: bool,  # Ignored - kept for API compatibility
        neighbor_stability: float,
    ) -> float:
        """Compute immediate reward for a single vehicle.

        Simplified formula:
        r_immediate = 0.5 × comm_success + 0.5 × neighbor_stability

        Args:
            vehicle_id: Vehicle ID (unused)
            what_action: WHAT action (unused in new formula)
            communication_success: Whether communication succeeded
            anomaly_detected: Ignored (was fake/constant)
            neighbor_stability: Stability metric (0-1, higher is better)

        Returns:
            Immediate reward in range [0, 1]
        """
        # Communication success: 1.0 if success, 0.0 otherwise
        comm_reward = 1.0 if communication_success else 0.0

        # Neighbor stability: already in [0, 1]
        stability_reward = neighbor_stability

        # Weighted combination
        immediate = (
            self.communication_success_weight * comm_reward
            + self.neighbor_stability_weight * stability_reward
        )

        return immediate

    def compute_global_reward(
        self,
        current_train_loss: float,
        comm_cost_mb: float = 0.0,  # Ignored - kept for API compatibility
        avg_staleness_seconds: float = 0.0,  # Ignored
    ) -> float:
        """Compute global reward based on training loss improvement.

        Formula:
        r_global = perf_scale × (loss_prev - loss_curr) / loss_prev

        This gives percentage improvement:
        - 10% loss reduction → reward = +10
        - 5% loss increase → reward = -5

        NOTE: prev_train_loss is only updated when loss actually CHANGES.
        This ensures ALL steps within an episode receive the same global reward
        based on episode-to-episode improvement, not step-to-step.

        Args:
            current_train_loss: Current training loss (lower is better)
            comm_cost_mb: Ignored (logged separately)
            avg_staleness_seconds: Ignored

        Returns:
            Global reward based on percentage loss improvement
        """
        # First call: initialize and return 0
        if self.prev_train_loss is None:
            self.prev_train_loss = current_train_loss
            return 0.0

        # Prevent division by zero
        if self.prev_train_loss == 0.0:
            self.prev_train_loss = current_train_loss
            return 0.0

        # Percentage improvement in loss
        # Positive when loss decreases, negative when increases
        percentage_improvement = (self.prev_train_loss - current_train_loss) / self.prev_train_loss

        # Only update prev_train_loss when the loss actually CHANGES
        # (i.e., at the start of a new episode after LSTM training)
        # This ensures all steps in an episode get the same global reward
        if abs(current_train_loss - self.prev_train_loss) > 1e-6:
            self.prev_train_loss = current_train_loss

        return self.perf_scale * percentage_improvement

    def compute_total_reward(
        self,
        vehicle_id: int,
        what_action: int,
        communication_success: bool,
        anomaly_detected: bool,
        neighbor_stability: float,
        current_train_loss: float,
        comm_cost_mb: float = 0.0,
        avg_staleness_seconds: float = 0.0,
    ) -> Dict[str, float]:
        """Compute total reward combining immediate and global components.

        Formula:
        r_total = 0.3 × r_immediate + 0.7 × r_global

        Args:
            vehicle_id: Vehicle ID
            what_action: WHAT action
            communication_success: Communication success flag
            anomaly_detected: Ignored (kept for API compatibility)
            neighbor_stability: Neighbor stability metric
            current_train_loss: Current training loss
            comm_cost_mb: Ignored
            avg_staleness_seconds: Ignored

        Returns:
            Dictionary with reward components and total
        """
        # Immediate reward
        immediate = self.compute_immediate_reward(
            vehicle_id=vehicle_id,
            what_action=what_action,
            communication_success=communication_success,
            anomaly_detected=anomaly_detected,
            neighbor_stability=neighbor_stability,
        )

        # Global reward (training loss improvement)
        global_r = self.compute_global_reward(current_train_loss=current_train_loss)

        # Total reward
        total = self.immediate_weight * immediate + self.global_weight * global_r

        return {
            "immediate": immediate,
            "global": global_r,
            "total": total,
        }

    def reset(self):
        """Reset reward function state.

        NOTE: We intentionally do NOT reset prev_train_loss here.
        This allows global reward to track episode-to-episode improvement.
        The loss only changes between episodes (after LSTM training),
        so resetting would make global_reward always 0.
        """
        # Do NOT reset prev_train_loss - we need it to persist across episodes
        # to compute episode-to-episode loss improvement
        pass


class MultiAgentRewardTracker:
    """Track rewards for multiple agents."""

    def __init__(
        self, num_vehicles: int, reward_fn: RewardFunction, vehicle_ids: Optional[List[int]] = None
    ):
        """Initialize multi-agent reward tracker.

        Args:
            num_vehicles: Number of vehicles
            reward_fn: Reward function instance
            vehicle_ids: List of actual vehicle IDs (if None, uses range(num_vehicles))
        """
        self.num_vehicles = num_vehicles
        self.reward_fn = reward_fn
        self.vehicle_ids = vehicle_ids if vehicle_ids is not None else list(range(num_vehicles))

        # Track cumulative rewards
        self.cumulative_rewards = {vid: 0.0 for vid in self.vehicle_ids}
        self.episode_rewards = []

    def compute_rewards(
        self,
        vehicle_actions: Dict[int, Dict],
        communication_results: Dict[int, bool],
        anomaly_detections: Dict[int, bool],
        neighbor_stabilities: Dict[int, float],
        current_train_loss: float,
        total_comm_cost_mb: float,
        staleness_metrics: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        """Compute rewards for all vehicles.

        Args:
            vehicle_actions: Actions per vehicle {vid: {who_action, what_action}}
            communication_results: Communication success per vehicle
            anomaly_detections: Anomaly detection per vehicle (ignored)
            neighbor_stabilities: Neighbor stability per vehicle
            current_train_loss: Global training loss (lower is better)
            total_comm_cost_mb: Total communication cost in MB (logged only)
            staleness_metrics: Ignored

        Returns:
            Rewards per vehicle {vid: reward}
        """
        rewards = {}

        for vid in self.vehicle_ids:
            # Get vehicle-specific info
            what_action = vehicle_actions.get(vid, {}).get("what_action", 1)
            comm_success = communication_results.get(vid, False)
            anomaly = anomaly_detections.get(vid, False)  # Ignored but passed for API
            stability = neighbor_stabilities.get(vid, 0.5)

            # Compute reward
            reward_dict = self.reward_fn.compute_total_reward(
                vehicle_id=vid,
                what_action=what_action,
                communication_success=comm_success,
                anomaly_detected=anomaly,
                neighbor_stability=stability,
                current_train_loss=current_train_loss,
            )

            rewards[vid] = reward_dict["total"]
            self.cumulative_rewards[vid] += reward_dict["total"]

        return rewards

    def end_episode(self):
        """End current episode and record rewards."""
        episode_reward = sum(self.cumulative_rewards.values())
        self.episode_rewards.append(episode_reward)

        # Reset cumulative
        self.cumulative_rewards = {vid: 0.0 for vid in self.vehicle_ids}

    def get_statistics(self) -> Dict[str, float]:
        """Get reward statistics.

        Returns:
            Dictionary with mean, std, min, max of episode rewards
        """
        if len(self.episode_rewards) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        return {
            "mean": float(np.mean(self.episode_rewards)),
            "std": float(np.std(self.episode_rewards)),
            "min": float(np.min(self.episode_rewards)),
            "max": float(np.max(self.episode_rewards)),
        }


def estimate_communication_cost(
    representation_type: str, data_size_params: Optional[Dict[str, float]] = None
) -> float:
    """Estimate communication cost in MB for different representation types.

    Args:
        representation_type: Type of representation (E/W)
        data_size_params: Optional parameters for data size estimation

    Returns:
        Estimated cost in MB
    """
    if data_size_params is None:
        data_size_params = {
            "E": 0.1,   # Embeddings: ~100 KB
            "W": 10.0,  # Weights: ~10 MB
        }

    return data_size_params.get(representation_type, 0.1)


def compute_neighbor_stability(
    previous_neighbors: set, current_neighbors: set
) -> float:
    """Compute neighbor stability metric.

    Args:
        previous_neighbors: Set of previous neighbor IDs
        current_neighbors: Set of current neighbor IDs

    Returns:
        Stability score in [0, 1] (higher is more stable)
    """
    if len(previous_neighbors) == 0 and len(current_neighbors) == 0:
        return 1.0

    if len(previous_neighbors) == 0 or len(current_neighbors) == 0:
        return 0.0

    # Jaccard similarity
    intersection = len(previous_neighbors & current_neighbors)
    union = len(previous_neighbors | current_neighbors)

    stability = intersection / union if union > 0 else 0.0

    return stability
