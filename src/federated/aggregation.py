"""
Asynchronous aggregation for decentralized federated learning.

Implements staleness-weighted aggregation per paper Algorithm 1:
    z̄_k = (1/Σw_j) × Σ(w_j × z_jk)  for j ∈ E_{i,k} ∪ {i}

where w_j = exp(-λ × Δt_j) is the staleness weight.
Local data is included with staleness=0 (weight=1.0).
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np


class AsynchronousAggregator:
    """Asynchronous aggregator with staleness-weighted averaging."""

    def __init__(
        self,
        alpha: float = 0.3,
        staleness_decay: float = 0.1,
        max_staleness_seconds: float = 30.0,
    ):
        """Initialize asynchronous aggregator.

        Args:
            alpha: Base aggregation weight (0.3 = 30% weight to neighbors)
            staleness_decay: Decay rate λ for staleness weighting
            max_staleness_seconds: Maximum allowed staleness (older data ignored)
        """
        self.alpha = alpha
        self.staleness_decay = staleness_decay
        self.max_staleness_seconds = max_staleness_seconds

    def compute_staleness_weights(
        self,
        timestamps: Dict[int, float],
        current_time: Optional[float] = None,
    ) -> Dict[int, float]:
        """Compute staleness weights for neighbor updates.

        Args:
            timestamps: Dict mapping neighbor_id to timestamp of their update
            current_time: Current timestamp (default: time.time())

        Returns:
            Dict mapping neighbor_id to staleness weight w_i ∈ [0, 1]
        """
        if current_time is None:
            current_time = time.time()

        weights = {}

        for neighbor_id, timestamp in timestamps.items():
            # Compute staleness in seconds
            staleness = current_time - timestamp

            # Ignore updates that are too old
            if staleness > self.max_staleness_seconds:
                weights[neighbor_id] = 0.0
                continue

            # Exponential decay: w = exp(-λ × staleness)
            weight = np.exp(-self.staleness_decay * staleness)
            weights[neighbor_id] = weight

        return weights

    def aggregate_embeddings(
        self,
        local_embeddings: np.ndarray,
        neighbor_embeddings: Dict[int, np.ndarray],
        timestamps: Dict[int, float],
        current_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Aggregate embeddings with pure staleness-weighted averaging.

        Paper formula: z̄_k = (1/Σw_j) × Σ(w_j × z_jk)  for j ∈ E_{i,k} ∪ {i}
        Local data is included with w_local = 1.0 (staleness=0).

        Args:
            local_embeddings: Local embeddings (num_samples, embedding_dim)
            neighbor_embeddings: Dict mapping neighbor_id to embeddings
            timestamps: Dict mapping neighbor_id to update timestamp
            current_time: Current time (default: time.time())

        Returns:
            Tuple of (aggregated_embeddings, aggregation_stats)
        """
        if len(neighbor_embeddings) == 0:
            return local_embeddings, {"num_neighbors": 0, "total_weight": 1.0}

        staleness_weights = self.compute_staleness_weights(timestamps, current_time)

        valid_neighbors = {
            nid: emb
            for nid, emb in neighbor_embeddings.items()
            if staleness_weights.get(nid, 0.0) > 0.0
        }

        if len(valid_neighbors) == 0:
            return local_embeddings, {
                "num_neighbors": 0,
                "total_weight": 1.0,
                "num_stale": len(neighbor_embeddings),
            }

        # Local weight is 1.0 (staleness=0 → exp(0)=1)
        local_w = 1.0
        neighbor_w = {nid: staleness_weights[nid] for nid in valid_neighbors}
        total_weight = local_w + sum(neighbor_w.values())

        # Weighted average: start with local contribution
        aggregated = (local_w / total_weight) * local_embeddings

        num_local_samples = local_embeddings.shape[0]
        for nid, emb in valid_neighbors.items():
            num_samples = min(num_local_samples, emb.shape[0])
            aggregated[:num_samples] += (neighbor_w[nid] / total_weight) * emb[:num_samples]

        stats = {
            "num_neighbors": len(valid_neighbors),
            "total_weight": total_weight,
            "local_weight": local_w / total_weight,
            "num_stale": len(neighbor_embeddings) - len(valid_neighbors),
            "neighbor_weights": neighbor_w,
        }

        return aggregated, stats

    def aggregate_prototypes(
        self,
        local_prototypes: np.ndarray,
        neighbor_prototypes: Dict[int, np.ndarray],
        timestamps: Dict[int, float],
        current_time: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Aggregate prototypes with pure staleness-weighted averaging.

        Same formula as embeddings but applied to cluster centroids.
        """
        if len(neighbor_prototypes) == 0:
            return local_prototypes, {"num_neighbors": 0, "total_weight": 1.0}

        staleness_weights = self.compute_staleness_weights(timestamps, current_time)

        valid_neighbors = {
            nid: proto
            for nid, proto in neighbor_prototypes.items()
            if staleness_weights.get(nid, 0.0) > 0.0
        }

        if len(valid_neighbors) == 0:
            return local_prototypes, {
                "num_neighbors": 0,
                "total_weight": 1.0,
                "num_stale": len(neighbor_prototypes),
            }

        local_w = 1.0
        neighbor_w = {nid: staleness_weights[nid] for nid in valid_neighbors}
        total_weight = local_w + sum(neighbor_w.values())

        aggregated = (local_w / total_weight) * local_prototypes
        for nid, proto in valid_neighbors.items():
            aggregated += (neighbor_w[nid] / total_weight) * proto

        aggregated = np.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0)

        stats = {
            "num_neighbors": len(valid_neighbors),
            "total_weight": total_weight,
            "local_weight": local_w / total_weight,
            "num_stale": len(neighbor_prototypes) - len(valid_neighbors),
            "neighbor_weights": neighbor_w,
        }

        return aggregated, stats

    def aggregate_weights(
        self,
        local_weights: Dict,
        neighbor_weights: Dict[int, Dict],
        timestamps: Dict[int, float],
        current_time: Optional[float] = None,
    ) -> Tuple[Dict, Dict]:
        """Aggregate model weights with pure staleness-weighted averaging.

        Implements paper Algorithm 1 (Phase 4):
            θ_new = (1/Σw_j) × Σ(w_j × θ_j)  for j ∈ W_i ∪ {i}

        where w_j = exp(-λ × Δt_j). The local model is included with
        staleness=0 (weight=1.0).

        Args:
            local_weights: Local model state_dict
            neighbor_weights: Dict mapping neighbor_id to state_dict
            timestamps: Dict mapping neighbor_id to update timestamp
            current_time: Current time

        Returns:
            Tuple of (aggregated_state_dict, aggregation_stats)
        """
        import torch

        if len(neighbor_weights) == 0:
            return local_weights, {"num_neighbors": 0, "total_weight": 1.0}

        staleness_weights = self.compute_staleness_weights(timestamps, current_time)

        valid_neighbors = {
            nid: sd
            for nid, sd in neighbor_weights.items()
            if staleness_weights.get(nid, 0.0) > 0.0
        }

        if len(valid_neighbors) == 0:
            return local_weights, {
                "num_neighbors": 0,
                "total_weight": 1.0,
                "num_stale": len(neighbor_weights),
            }

        # Local weight is 1.0 (staleness=0 → exp(0)=1)
        local_w = 1.0
        total_weight = local_w + sum(staleness_weights[nid] for nid in valid_neighbors)

        # Weighted average over state_dict keys
        aggregated = {}
        for key in local_weights:
            param = local_weights[key]
            if not isinstance(param, torch.Tensor):
                aggregated[key] = param
                continue

            weighted_sum = local_w * param.float()
            for nid, sd in valid_neighbors.items():
                weighted_sum = weighted_sum + staleness_weights[nid] * sd[key].float()

            aggregated[key] = (weighted_sum / total_weight).to(param.dtype)

        neighbor_w_map = {nid: staleness_weights[nid] for nid in valid_neighbors}
        stats = {
            "num_neighbors": len(valid_neighbors),
            "total_weight": total_weight,
            "local_weight": local_w / total_weight,
            "num_stale": len(neighbor_weights) - len(valid_neighbors),
            "neighbor_weights": neighbor_w_map,
        }

        return aggregated, stats

    def compute_divergence(
        self,
        local_data: np.ndarray,
        aggregated_data: np.ndarray,
    ) -> float:
        """Compute divergence between local and aggregated data.

        Useful for measuring how much the aggregation changed the local model.

        Args:
            local_data: Local representation
            aggregated_data: Aggregated representation

        Returns:
            L2 distance between local and aggregated
        """
        # Sanitize inputs to prevent NaN/Inf in computation
        local_data = np.nan_to_num(local_data, nan=0.0, posinf=0.0, neginf=0.0)
        aggregated_data = np.nan_to_num(aggregated_data, nan=0.0, posinf=0.0, neginf=0.0)

        # L2 norm of difference
        diff = local_data - aggregated_data
        divergence = np.linalg.norm(diff)

        return float(divergence)

    def should_aggregate(
        self,
        timestamps: Dict[int, float],
        min_neighbors: int = 1,
        current_time: Optional[float] = None,
    ) -> bool:
        """Determine if aggregation should be performed.

        Args:
            timestamps: Dict of neighbor timestamps
            min_neighbors: Minimum number of valid (non-stale) neighbors required
            current_time: Current time

        Returns:
            True if aggregation should proceed
        """
        # Compute staleness weights
        staleness_weights = self.compute_staleness_weights(timestamps, current_time)

        # Count valid neighbors (non-stale)
        num_valid = sum(1 for w in staleness_weights.values() if w > 0.0)

        return num_valid >= min_neighbors

    def get_aggregation_metrics(
        self,
        stats: Dict,
    ) -> Dict[str, float]:
        """Extract key metrics from aggregation stats."""
        metrics = {
            "num_neighbors": stats.get("num_neighbors", 0),
            "total_weight": stats.get("total_weight", 1.0),
            "local_weight": stats.get("local_weight", 1.0),
            "num_stale": stats.get("num_stale", 0),
        }

        neighbor_weights = stats.get("neighbor_weights", {})
        if neighbor_weights:
            metrics["avg_neighbor_weight"] = np.mean(list(neighbor_weights.values()))
            metrics["max_neighbor_weight"] = np.max(list(neighbor_weights.values()))
            metrics["min_neighbor_weight"] = np.min(list(neighbor_weights.values()))

        return metrics
