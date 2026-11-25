"""Heuristic communication policies for DFL baselines.

These policies replace the RL (PPO) actor with fixed rules for WHO/WHAT
decisions, keeping the DFL infrastructure identical. Used to benchmark
whether the learned policy adds genuine value over simple heuristics.
"""

from abc import ABC, abstractmethod

import numpy as np


class HeuristicPolicy(ABC):
    """Base class for heuristic communication policies.

    All policies share the same WHO logic (random k neighbors) and differ
    only in WHAT representation to send (0=Embeddings, 1=Weights).
    """

    def __init__(self, k_neighbors: int = 2, max_neighbors: int = 5, seed: int = 42) -> None:
        self.k_neighbors = k_neighbors
        self.max_neighbors = max_neighbors
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def select_what(self, episode: int, timestep: int, vehicle_id: int, **kwargs) -> int:
        """Return 0 (Embeddings) or 1 (Weights)."""

    def select_who(self, neighbors: list[int]) -> np.ndarray:
        """Select k random neighbor indices from available neighbors.

        Returns indices into the neighbors list (not actual neighbor IDs),
        matching the PPOActor output format expected by simulate_dfl_communication.
        """
        n_available = len(neighbors)
        if n_available == 0:
            return np.array([], dtype=np.int64)
        k = min(self.k_neighbors, n_available)
        indices = self.rng.choice(n_available, size=k, replace=False)
        return np.sort(indices)

    def select_action(
        self,
        vehicle_id: int,
        neighbors: list[int],
        episode: int,
        timestep: int,
        **kwargs,
    ) -> dict:
        """Select WHO and WHAT actions for a vehicle.

        Args:
            vehicle_id: ID of the acting vehicle.
            neighbors: Sorted list of current neighbor IDs.
            episode: Current training episode.
            timestep: Current timestep within the episode.
            **kwargs: Extra context (e.g., current_loss for ThresholdPolicy).

        Returns:
            Action dict matching PPOActor format:
            {"who_action": np.ndarray of indices, "what_action": int}
        """
        who = self.select_who(neighbors)
        what = self.select_what(episode, timestep, vehicle_id, **kwargs)
        return {"who_action": who, "what_action": what}

    @property
    def name(self) -> str:
        return self.__class__.__name__


class WeightsOnlyPolicy(HeuristicPolicy):
    """Always send full model weights. Ablation: no embeddings ever."""

    def select_what(self, _episode: int, _timestep: int, _vehicle_id: int, **kwargs) -> int:
        return 1


class EmbeddingsOnlyPolicy(HeuristicPolicy):
    """Always send lightweight embeddings. Ablation: no weight sharing ever."""

    def select_what(self, _episode: int, _timestep: int, _vehicle_id: int, **kwargs) -> int:
        return 0


class FixedSwitchPolicy(HeuristicPolicy):
    """Weights for the first N episodes, then Embeddings after.

    Tests whether "weights early, embeddings late" is sufficient
    without RL-learned timing.
    """

    def __init__(
        self,
        switch_episode: int = 500,
        k_neighbors: int = 2,
        max_neighbors: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(k_neighbors=k_neighbors, max_neighbors=max_neighbors, seed=seed)
        self.switch_episode = switch_episode

    def select_what(self, episode: int, timestep: int, vehicle_id: int, **kwargs) -> int:
        return 1 if episode < self.switch_episode else 0


class PeriodicPolicy(HeuristicPolicy):
    """Alternate Weights and Embeddings every `period` episodes.

    Tests whether regular switching at a fixed cadence can match
    the RL agent's adaptive switching.
    """

    def __init__(
        self,
        period: int = 100,
        k_neighbors: int = 2,
        max_neighbors: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(k_neighbors=k_neighbors, max_neighbors=max_neighbors, seed=seed)
        self.period = period

    def select_what(self, episode: int, timestep: int, vehicle_id: int, **kwargs) -> int:
        return 1 if (episode // self.period) % 2 == 0 else 0


class ThresholdPolicy(HeuristicPolicy):
    """Send Weights when loss is high, Embeddings when loss is low.

    The most "intelligent" heuristic -- adapts based on model quality,
    but uses a fixed threshold rather than a learned policy.
    """

    def __init__(
        self,
        loss_threshold: float = 0.01,
        k_neighbors: int = 2,
        max_neighbors: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(k_neighbors=k_neighbors, max_neighbors=max_neighbors, seed=seed)
        self.loss_threshold = loss_threshold

    def select_what(self, episode: int, timestep: int, vehicle_id: int, **kwargs) -> int:
        current_loss = kwargs.get("current_loss", 0.0)
        return 1 if current_loss > self.loss_threshold else 0


class Fixed7030Policy(HeuristicPolicy):
    """70% probability Weights, 30% probability Embeddings.

    Tests whether the RL-learned ~70/30 W/E ratio matters, or if
    random assignment at that ratio is sufficient.
    """

    def __init__(
        self,
        weights_ratio: float = 0.7,
        k_neighbors: int = 2,
        max_neighbors: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(k_neighbors=k_neighbors, max_neighbors=max_neighbors, seed=seed)
        self.weights_ratio = weights_ratio

    def select_what(self, episode: int, timestep: int, vehicle_id: int, **kwargs) -> int:
        return 1 if self.rng.random() < self.weights_ratio else 0


POLICY_REGISTRY = {
    "weights_only": WeightsOnlyPolicy,
    "embeddings_only": EmbeddingsOnlyPolicy,
    "fixed_switch": FixedSwitchPolicy,
    "periodic": PeriodicPolicy,
    "threshold": ThresholdPolicy,
    "fixed_7030": Fixed7030Policy,
}


def create_heuristic_policy(name: str, **kwargs) -> HeuristicPolicy:
    """Factory function to create a heuristic policy by name.

    Args:
        name: One of: weights_only, embeddings_only, fixed_switch,
              periodic, threshold, fixed_7030
        **kwargs: Policy-specific parameters (k_neighbors, max_neighbors,
                  seed, switch_episode, period, loss_threshold, weights_ratio)

    Returns:
        Instantiated HeuristicPolicy subclass.

    Raises:
        ValueError: If name is not in the registry.
    """
    if name not in POLICY_REGISTRY:
        valid = ", ".join(sorted(POLICY_REGISTRY.keys()))
        raise ValueError(f"Unknown heuristic policy '{name}'. Valid: {valid}")
    return POLICY_REGISTRY[name](**kwargs)
