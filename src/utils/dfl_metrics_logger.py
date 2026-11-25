"""
DFL-specific metrics logging for comprehensive evaluation.

Tracks federated learning dynamics, aggregation quality, and communication patterns.
"""

from typing import Dict, Optional
import numpy as np
from collections import defaultdict


class DFLMetricsLogger:
    """Logger for DFL-specific metrics during training."""

    def __init__(self):
        """Initialize metrics logger."""
        self.reset()

    def reset(self):
        """Reset all metrics."""
        # Aggregation metrics
        self.aggregation_stats = defaultdict(list)

        # Communication metrics
        self.what_actions = defaultdict(int)
        self.who_actions = defaultdict(list)
        self.comm_success = []

        # Performance metrics
        self.local_performance = []
        self.aggregated_performance = []

        # Representation quality
        self.embedding_stats = defaultdict(list)
        self.prototype_stats = defaultdict(list)

        # Network topology
        self.topology_stats = defaultdict(list)

    def log_aggregation(self, vehicle_id: int, agg_stats: Dict):
        """Log aggregation statistics for a vehicle.

        Args:
            vehicle_id: Vehicle ID
            agg_stats: Aggregation statistics from AsynchronousAggregator
        """
        self.aggregation_stats[vehicle_id].append({
            'num_neighbors': agg_stats.get('num_neighbors', 0),
            'total_weight': agg_stats.get('total_weight', 1.0),
            'local_weight': agg_stats.get('local_weight', 1.0),
            'num_stale': agg_stats.get('num_stale', 0),
        })

    def log_communication(self, vehicle_id: int, action: Dict, success: bool):
        """Log communication action and result.

        Args:
            vehicle_id: Vehicle ID
            action: Action dict with 'who_action' and 'what_action'
            success: Whether communication succeeded
        """
        repr_types = ['E', 'W']  # 0=Embeddings, 1=Weights
        what_idx = action.get('what_action', 0)
        if what_idx < len(repr_types):
            self.what_actions[repr_types[what_idx]] += 1

        # Track WHO actions (neighbor selection)
        self.who_actions[vehicle_id].append(action.get('who_action', []))

        # Track success rate
        self.comm_success.append(1 if success else 0)

    def log_representation_quality(
        self,
        vehicle_id: int,
        embeddings: Optional[np.ndarray] = None,
        prototypes: Optional[np.ndarray] = None,
    ):
        """Log representation quality metrics.

        Args:
            vehicle_id: Vehicle ID
            embeddings: Embedding vectors (N, dim)
            prototypes: Prototype vectors (K, dim)
        """
        if embeddings is not None:
            self.embedding_stats[vehicle_id].append({
                'mean_norm': float(np.linalg.norm(embeddings, axis=1).mean()),
                'std_norm': float(np.linalg.norm(embeddings, axis=1).std()),
                'has_nan': bool(np.isnan(embeddings).any()),
                'has_inf': bool(np.isinf(embeddings).any()),
            })

        if prototypes is not None:
            # Compute prototype separation (inter-cluster distance)
            if len(prototypes) > 1:
                dists = []
                for i in range(len(prototypes)):
                    for j in range(i + 1, len(prototypes)):
                        dist = np.linalg.norm(prototypes[i] - prototypes[j])
                        dists.append(dist)
                separation = np.mean(dists)
            else:
                separation = 0.0

            self.prototype_stats[vehicle_id].append({
                'mean_norm': float(np.linalg.norm(prototypes, axis=1).mean()),
                'separation': float(separation),
                'has_nan': bool(np.isnan(prototypes).any()),
                'has_inf': bool(np.isinf(prototypes).any()),
            })

    def log_topology(self, network_stats: Dict):
        """Log network topology statistics.

        Args:
            network_stats: Dict with topology metrics
        """
        for key, value in network_stats.items():
            self.topology_stats[key].append(value)

    def get_episode_summary(self) -> Dict:
        """Get summary statistics for current episode.

        Returns:
            Dictionary with aggregated metrics
        """
        summary = {}

        # Aggregation summary
        if self.aggregation_stats:
            all_neighbors = [
                stat['num_neighbors']
                for stats in self.aggregation_stats.values()
                for stat in stats
            ]
            all_local_weights = [
                stat['local_weight']
                for stats in self.aggregation_stats.values()
                for stat in stats
            ]

            summary['agg/avg_neighbors'] = np.mean(all_neighbors) if all_neighbors else 0
            summary['agg/avg_local_weight'] = np.mean(all_local_weights) if all_local_weights else 1.0
            summary['agg/num_vehicles_aggregated'] = len(self.aggregation_stats)

        # Communication summary
        total_actions = sum(self.what_actions.values())
        if total_actions > 0:
            summary['comm/embedding_pct'] = self.what_actions['E'] / total_actions
            summary['comm/prototype_pct'] = self.what_actions['P'] / total_actions
            summary['comm/weight_pct'] = self.what_actions['W'] / total_actions

        if self.comm_success:
            summary['comm/success_rate'] = np.mean(self.comm_success)

        # Representation quality summary
        if self.embedding_stats:
            all_norms = [
                stat['mean_norm']
                for stats in self.embedding_stats.values()
                for stat in stats
            ]
            all_nans = [
                stat['has_nan']
                for stats in self.embedding_stats.values()
                for stat in stats
            ]

            summary['repr/embedding_norm_mean'] = np.mean(all_norms) if all_norms else 0
            summary['repr/embedding_has_nan'] = any(all_nans)

        if self.prototype_stats:
            all_separations = [
                stat['separation']
                for stats in self.prototype_stats.values()
                for stat in stats
            ]

            summary['repr/prototype_separation'] = np.mean(all_separations) if all_separations else 0

        # Topology summary
        for key, values in self.topology_stats.items():
            if values:
                summary[f'topology/{key}'] = np.mean(values)

        return summary

    def compute_dfl_benefit(
        self,
        local_f1: float,
        aggregated_f1: float,
    ) -> Dict:
        """Compute benefit of federated learning.

        Args:
            local_f1: F1 using only local model
            aggregated_f1: F1 using aggregated model

        Returns:
            Dictionary with benefit metrics
        """
        improvement = aggregated_f1 - local_f1
        relative_improvement = improvement / local_f1 if local_f1 > 0 else 0

        return {
            'dfl/local_f1': local_f1,
            'dfl/aggregated_f1': aggregated_f1,
            'dfl/improvement': improvement,
            'dfl/relative_improvement': relative_improvement,
        }

    def compute_efficiency_metrics(
        self,
        f1_score: float,
        comm_cost_mb: float,
        baseline_comm_cost: float = 100.0,
    ) -> Dict:
        """Compute communication efficiency metrics.

        Args:
            f1_score: Current F1 score
            comm_cost_mb: Communication cost in MB
            baseline_comm_cost: Baseline cost for "always send" strategy

        Returns:
            Dictionary with efficiency metrics
        """
        # F1 per MB of communication
        efficiency = f1_score / comm_cost_mb if comm_cost_mb > 0 else 0

        # Relative efficiency vs baseline
        relative_efficiency = (
            (f1_score / comm_cost_mb) / (f1_score / baseline_comm_cost)
            if comm_cost_mb > 0 and baseline_comm_cost > 0
            else 1.0
        )

        return {
            'efficiency/f1_per_mb': efficiency,
            'efficiency/relative_to_baseline': relative_efficiency,
            'efficiency/bandwidth_reduction': (baseline_comm_cost - comm_cost_mb) / baseline_comm_cost if baseline_comm_cost > 0 else 0,
        }
