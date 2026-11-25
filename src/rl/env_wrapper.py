"""
Environment wrapper for RL training with V2VEnvironment.

Wraps V2VEnvironment to provide RL-compatible interface:
- Constructs state from observations
- Executes WHO/WHAT actions
- Computes rewards
- Handles multi-agent coordination
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.models import extract_neighbor_features
from src.rl.rewards import (
    MultiAgentRewardTracker,
    RewardFunction,
    compute_neighbor_stability,
    estimate_communication_cost,
)
from src.simulation import V2VEnvironment


class RLV2VWrapper:
    """Wrapper for V2VEnvironment to enable RL training with optional DFL support."""

    def __init__(
        self,
        env: V2VEnvironment,
        reward_fn: RewardFunction,
        use_neighbor_encoder: bool = False,
        device: torch.device = torch.device("cpu"),
        dfl_protocols: Optional[Dict] = None,
        use_dfl_features: bool = False,
    ):
        """Initialize RL wrapper.

        Args:
            env: V2VEnvironment instance
            reward_fn: Reward function
            use_neighbor_encoder: Whether to use neighbor encoder
            device: Device for computation
            dfl_protocols: Dict mapping vehicle_id to DFLProtocol (optional for DFL training)
            use_dfl_features: Whether to include DFL staleness features in state (110-dim vs 106-dim)
        """
        self.env = env
        self.reward_fn = reward_fn
        self.use_neighbor_encoder = use_neighbor_encoder
        self.device = device
        self.dfl_protocols = dfl_protocols or {}
        self.use_dfl_features = use_dfl_features

        self.num_vehicles = env.num_vehicles
        # State dim: 106 (base) + 4 (DFL features) = 110 if use_dfl_features
        self.state_dim = 110 if use_dfl_features else 106

        # Track previous neighbors for stability
        self.previous_neighbors = {vid: set() for vid in env.vehicle_ids}

        # Track previous velocities for acceleration computation
        self._prev_velocities = {vid: (0.0, 0.0) for vid in env.vehicle_ids}

        # Reward tracker
        self.reward_tracker = MultiAgentRewardTracker(
            num_vehicles=len(env.vehicle_ids),
            reward_fn=reward_fn,
            vehicle_ids=list(env.vehicle_ids),
        )

    def reset(self) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Reset environment and get initial states.

        Returns:
            Tuple of (states, global_state):
                - states: Dictionary mapping vehicle_id to state (state_dim,)
                - global_state: Global state array (num_vehicles, state_dim)
        """
        # Reset environment
        obs = self.env.reset()

        # Reset reward function (only the F1 tracking, not the episode rewards list)
        self.reward_fn.reset()

        # Reset cumulative rewards for new episode, but preserve episode_rewards list
        self.reward_tracker.cumulative_rewards = {vid: 0.0 for vid in self.reward_tracker.vehicle_ids}

        # Reset neighbor and velocity tracking
        for vid in self.env.vehicle_ids:
            self.previous_neighbors[vid] = set()
            self._prev_velocities[vid] = (0.0, 0.0)

        # Construct states
        states, global_state = self._construct_states(obs)

        return states, global_state

    def step(
        self, actions: Dict[int, Dict]
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float], bool, Dict]:
        """Execute actions and get next states.

        Args:
            actions: Dictionary mapping vehicle_id to action dict
                    {who_action: np.ndarray(k,), what_action: int}

        Returns:
            Tuple of (states, rewards, done, info):
                - states: Dictionary of next states
                - rewards: Dictionary of rewards per vehicle
                - done: Episode termination flag
                - info: Additional information
        """
        # Step environment (advances time, updates neighbors, computes embeddings)
        obs, _, done, env_info = self.env.step()

        # Simulate communication (placeholder - actual communication in Layer 4)
        communication_results = self._simulate_communication(actions)

        # Compute anomaly detections
        anomaly_detections = self._compute_anomaly_detections(obs)

        # Compute neighbor stability
        neighbor_stabilities = self._compute_neighbor_stabilities()

        # Estimate communication costs
        total_comm_cost = self._estimate_total_comm_cost(actions, communication_results)

        # Compute global training loss (average across all vehicles)
        current_train_loss = self._compute_training_loss()

        # Compute staleness metrics (for DFL training)
        staleness_metrics = self._compute_staleness_metrics() if self.use_dfl_features else None

        # Compute rewards
        rewards = self.reward_tracker.compute_rewards(
            vehicle_actions=actions,
            communication_results=communication_results,
            anomaly_detections=anomaly_detections,
            neighbor_stabilities=neighbor_stabilities,
            current_train_loss=current_train_loss,
            total_comm_cost_mb=total_comm_cost,
            staleness_metrics=staleness_metrics,
        )

        # Construct next states
        states, global_state = self._construct_states(obs)

        # Update previous neighbors
        for vid in self.env.vehicle_ids:
            self.previous_neighbors[vid] = self.env.vehicles[vid].neighbors.copy()

        # Additional info
        info = {
            **env_info,
            "global_state": global_state,
            "communication_results": communication_results,
            "anomaly_detections": anomaly_detections,
            "train_loss": current_train_loss,
            "comm_cost_mb": total_comm_cost,
        }

        return states, rewards, done, info

    def _construct_states(
        self, observations: Dict[int, Dict]
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """Construct state representations from observations.

        Args:
            observations: Observations from environment

        Returns:
            Tuple of (states, global_state)
        """
        states = {}
        global_state_list = []

        for vid in sorted(self.env.vehicle_ids):
            obs = observations[vid]

            # Local features (10-dim): paper state vector s_i
            vx, vy = obs["velocity"][0], obs["velocity"][1]
            heading = np.arctan2(vy, vx)  # Heading angle from velocity
            # Acceleration magnitude (finite difference with previous velocity)
            prev_vel = self._prev_velocities.get(vid, (0.0, 0.0))
            accel_mag = np.sqrt((vx - prev_vel[0]) ** 2 + (vy - prev_vel[1]) ** 2) / self.env.time_step_duration
            self._prev_velocities[vid] = (vx, vy)

            local_features = np.array(
                [
                    obs["position"][0],
                    obs["position"][1],
                    vx,
                    vy,
                    obs["speed"],
                    obs["anomaly_score"] if obs["anomaly_score"] is not None else 0.0,
                    obs["num_neighbors"],
                    float(obs["timestep"]) / self.env.max_timesteps,
                    heading,
                    accel_mag,
                ],
                dtype=np.float32,
            )

            # Neighbor encoding (96-dim)
            if self.use_neighbor_encoder and "neighborhood_encoding" in obs:
                # Use Layer 2 encoding if available
                neighbor_encoding = obs["neighborhood_encoding"]
                if neighbor_encoding is None:
                    neighbor_encoding = np.zeros(32, dtype=np.float32)

                # Expand to 96-dim by concatenating mean/max/std
                # (This is before the ρ network, so we have aggregated features)
                neighbor_features_96 = np.tile(neighbor_encoding, 3)[:96]
            else:
                # Manual neighbor encoding: extract full 40-dim features,
                # then aggregate via mean/max/std over φ-output-sized chunks
                vehicle = self.env.vehicles[vid]
                neighbors = [self.env.vehicles[nid] for nid in vehicle.neighbors]

                if len(neighbors) > 0:
                    nf = extract_neighbor_features(
                        vehicle, neighbors, self.env.comm_range
                    )
                    if len(nf) > 0:
                        # Use all 40 features, pad to 32 for each pool
                        mean_pool = nf.mean(axis=0)[:32]
                        max_pool = nf.max(axis=0)[:32]
                        std_pool = nf.std(axis=0)[:32]
                        neighbor_features_96 = np.concatenate(
                            [mean_pool, max_pool, std_pool]
                        ).astype(np.float32)
                    else:
                        neighbor_features_96 = np.zeros(96, dtype=np.float32)
                else:
                    neighbor_features_96 = np.zeros(96, dtype=np.float32)

            # Concatenate to form base state (106-dim)
            base_state = np.concatenate([local_features, neighbor_features_96]).astype(
                np.float32
            )

            # Add DFL features if enabled (4 additional dims -> 110 total)
            if self.use_dfl_features and vid in self.dfl_protocols:
                dfl_features = self.dfl_protocols[vid].get_state_features()
                dfl_array = np.array(
                    [
                        dfl_features["time_since_last_update"],
                        dfl_features["avg_neighbor_staleness"],
                        dfl_features["pending_updates_count"],
                        dfl_features["model_divergence_estimate"],
                    ],
                    dtype=np.float32,
                )
                state = np.concatenate([base_state, dfl_array])
            else:
                state = base_state

            states[vid] = state
            global_state_list.append(state)

        # Global state (num_vehicles, state_dim)
        global_state = np.stack(global_state_list, axis=0)

        return states, global_state

    def _simulate_communication(
        self, actions: Dict[int, Dict]
    ) -> Dict[int, bool]:
        """Simulate communication success (placeholder).

        In Layer 4, this will involve actual data transfer.
        For now, assume communication succeeds if vehicle has neighbors.

        Args:
            actions: Vehicle actions

        Returns:
            Communication success per vehicle
        """
        results = {}

        for vid in self.env.vehicle_ids:
            vehicle = self.env.vehicles[vid]
            # Communication succeeds if vehicle has neighbors
            has_neighbors = len(vehicle.neighbors) > 0
            results[vid] = has_neighbors

        return results

    def _compute_anomaly_detections(
        self, observations: Dict[int, Dict]
    ) -> Dict[int, bool]:
        """Compute anomaly detections based on threshold.

        Args:
            observations: Environment observations

        Returns:
            Anomaly detection flags per vehicle
        """
        # Use threshold based on reconstruction error statistics
        # Anomaly scores (MSE) range from ~0.003 to ~0.09
        # Use 95th percentile (0.049) as threshold for anomaly detection
        threshold = 0.05  # Updated from 0.5 - matches actual score distribution

        detections = {}
        for vid in self.env.vehicle_ids:
            obs = observations[vid]
            anomaly_score = obs.get("anomaly_score", 0.0)
            # Handle None values - treat as 0.0 (no anomaly) until window is full
            if anomaly_score is None:
                anomaly_score = 0.0
            detections[vid] = anomaly_score > threshold

        return detections

    def _compute_neighbor_stabilities(self) -> Dict[int, float]:
        """Compute neighbor stability for all vehicles.

        Returns:
            Stability scores per vehicle
        """
        stabilities = {}

        for vid in self.env.vehicle_ids:
            vehicle = self.env.vehicles[vid]
            current = vehicle.neighbors
            previous = self.previous_neighbors.get(vid, set())

            stability = compute_neighbor_stability(previous, current)
            stabilities[vid] = stability

        return stabilities

    def _estimate_total_comm_cost(
        self, actions: Dict[int, Dict], comm_results: Dict[int, bool]
    ) -> float:
        """Estimate total communication cost in MB.

        Args:
            actions: Vehicle actions
            comm_results: Communication success flags

        Returns:
            Total cost in MB
        """
        total_cost = 0.0
        repr_types = ["E", "W"]  # 0=Embeddings (~100KB), 1=Weights (~10MB)

        for vid in self.env.vehicle_ids:
            if not comm_results.get(vid, False):
                continue

            what_action = actions.get(vid, {}).get("what_action", 0)  # Default: E
            repr_type = repr_types[what_action]

            cost = estimate_communication_cost(repr_type)
            total_cost += cost

        return total_cost

    def _compute_training_loss(self) -> float:
        """Compute global training loss (average across all vehicles).

        Uses the local_training_loss from each vehicle's DFL protocol.
        This provides a REAL performance metric that improves as models get better.

        Returns:
            Average training loss across all vehicles (lower is better)
        """
        if not self.dfl_protocols:
            # No DFL protocols (e.g., testing without DFL) - return default
            return 1.0

        total_loss = 0.0
        num_vehicles = 0

        for vehicle_id, protocol in self.dfl_protocols.items():
            # Get the local training loss from most recent local training
            total_loss += protocol.local_training_loss
            num_vehicles += 1

        avg_loss = total_loss / num_vehicles if num_vehicles > 0 else 1.0
        return avg_loss

    def _compute_staleness_metrics(self) -> Dict[int, float]:
        """Compute average staleness per vehicle (for DFL training).

        Returns:
            Dictionary mapping vehicle_id to average staleness in seconds
        """
        staleness_metrics = {}

        for vid in self.env.vehicle_ids:
            if vid in self.dfl_protocols:
                # Get staleness from DFL protocol
                staleness_dict = self.dfl_protocols[vid].get_staleness_metrics()
                avg_staleness = (
                    sum(staleness_dict.values()) / len(staleness_dict)
                    if len(staleness_dict) > 0
                    else 0.0
                )
                staleness_metrics[vid] = avg_staleness
            else:
                staleness_metrics[vid] = 0.0

        return staleness_metrics

    def get_neighbor_masks(self) -> Dict[int, np.ndarray]:
        """Get neighbor masks for all vehicles.

        Returns:
            Dictionary mapping vehicle_id to neighbor mask (max_neighbors,)
        """
        masks = {}

        for vid in self.env.vehicle_ids:
            vehicle = self.env.vehicles[vid]
            mask = np.zeros(self.env.sim_config.get("max_neighbors", 20), dtype=bool)

            # Set True for valid neighbors
            for i, neighbor_id in enumerate(sorted(vehicle.neighbors)):
                if i < len(mask):
                    mask[i] = True

            masks[vid] = mask

        return masks
