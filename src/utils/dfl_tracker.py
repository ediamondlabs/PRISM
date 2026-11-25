"""
Comprehensive DFL tracking system with JSON persistence.

Tracks all metrics during DFL training and saves to structured JSON files
for later analysis, visualization, and paper writing.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
from collections import defaultdict
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class DFLComprehensiveTracker:
    """
    Comprehensive tracker for all DFL metrics with JSON persistence.

    Saves metrics hierarchically:
    - metadata.json: Run configuration and info
    - episodes_summary.json: Episode-level aggregated metrics
    - vehicles_detail/: Per-vehicle detailed metrics (saved periodically)
    - checkpoints/: Periodic snapshots for crash recovery
    """

    def __init__(
        self,
        output_dir: str = "outputs/dfl_metrics",
        run_name: Optional[str] = None,
        save_interval: int = 50,
        vehicle_detail_interval: int = 100,
    ):
        """
        Initialize comprehensive tracker.

        Args:
            output_dir: Base directory for saving metrics
            run_name: Name for this training run (auto-generated if None)
            save_interval: Save episode summary every N episodes
            vehicle_detail_interval: Save vehicle details every N episodes
        """
        # Setup directories
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(output_dir) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "vehicles_detail").mkdir(exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        # Save intervals
        self.save_interval = save_interval
        self.vehicle_detail_interval = vehicle_detail_interval

        # Metadata
        self.metadata = {
            "run_name": self.run_name,
            "start_time": datetime.now().isoformat(),
            "config": {},
            "model_info": {},
            "dataset_info": {},
        }
        self.start_timestamp = time.time()

        # Episode-level metrics (accumulated)
        self.episodes = []
        self.current_episode = None

        # Vehicle-level metrics (current episode)
        self.current_vehicle_metrics = defaultdict(lambda: defaultdict(list))

        # Timestep-level metrics (sampled to avoid huge files)
        self.sampled_timesteps = []
        self.sample_every = 10  # Sample every 10th timestep

        # Aggregation statistics
        self.aggregation_history = []

        # Communication patterns
        self.communication_history = []

        print(f"[DFLTracker] Initialized: {self.output_dir}")

    def set_metadata(self, config: Dict = None, model_info: Dict = None, dataset_info: Dict = None):
        """Set run metadata (config, model architecture, dataset info)."""
        if config:
            self.metadata["config"] = self._convert_to_serializable(config)
        if model_info:
            self.metadata["model_info"] = self._convert_to_serializable(model_info)
        if dataset_info:
            self.metadata["dataset_info"] = self._convert_to_serializable(dataset_info)

        # Save metadata immediately
        self._save_metadata()

    def start_episode(self, episode_id: int):
        """Start tracking a new episode."""
        self.current_episode = {
            "episode_id": episode_id,
            "timestamp": time.time() - self.start_timestamp,

            # Episode metrics (will be updated during episode)
            "reward": 0.0,
            "f1_score": 0.0,
            "comm_cost_mb": 0.0,

            # RL metrics
            "actor_loss": None,
            "critic_loss": None,
            "entropy": None,
            "clip_fraction": None,

            # DFL metrics
            "num_aggregations": 0,
            "avg_neighbors_per_agg": 0.0,
            "avg_staleness_seconds": 0.0,
            "avg_divergence": 0.0,

            # Communication metrics (updated: removed Prototypes)
            "what_actions": {"E": 0, "W": 0},
            "comm_success_rate": 0.0,

            # Representation quality
            "avg_embedding_norm": 0.0,
            "avg_prototype_separation": 0.0,
            "num_nan_inf": 0,

            # Performance tracking
            "avg_anomaly_score": 0.0,
            "detection_rate": 0.0,

            # Timing
            "episode_duration_seconds": 0.0,
        }

        # Reset vehicle metrics for this episode
        self.current_vehicle_metrics.clear()

    def log_timestep(
        self,
        timestep: int,
        observations: Dict[int, Dict],
        actions: Dict[int, Dict],
        rewards: Dict[int, float],
        dfl_info: Optional[Dict] = None,
    ):
        """
        Log timestep-level data.

        Args:
            timestep: Timestep number
            observations: Vehicle observations
            actions: Vehicle actions
            rewards: Vehicle rewards
            dfl_info: DFL-specific info (aggregation stats, etc.)
        """
        # Sample timesteps to avoid huge files
        if timestep % self.sample_every == 0:
            sampled_data = {
                "episode_id": self.current_episode["episode_id"],
                "timestep": timestep,
                "anomaly_scores": {},
                "actions": {},
                "rewards": {},
            }

            for vid in observations.keys():
                sampled_data["anomaly_scores"][str(vid)] = observations[vid].get("anomaly_score")
                sampled_data["actions"][str(vid)] = {
                    "what": int(actions[vid]["what_action"]),
                    "who": [int(x) for x in actions[vid]["who_action"]] if "who_action" in actions[vid] else []
                }
                sampled_data["rewards"][str(vid)] = float(rewards.get(vid, 0.0))

            self.sampled_timesteps.append(sampled_data)

    def log_vehicle_action(
        self,
        vehicle_id: int,
        action: Dict,
        communication_success: bool,
        neighbors_info: Optional[Dict] = None,
    ):
        """
        Log vehicle-level action and communication.

        Args:
            vehicle_id: Vehicle ID
            action: Action dict with who_action and what_action
            communication_success: Whether communication succeeded
            neighbors_info: Info about selected neighbors
        """
        vid_str = str(vehicle_id)

        # Track actions (updated: removed Prototypes)
        what_idx = action.get("what_action", 0)
        repr_type = ["E", "W"][what_idx]  # 0=Embeddings, 1=Weights
        self.current_episode["what_actions"][repr_type] += 1

        # Track vehicle-specific metrics
        self.current_vehicle_metrics[vid_str]["actions"].append({
            "what": repr_type,
            "who": [int(x) for x in action.get("who_action", [])],
            "success": communication_success,
        })

    def log_aggregation(
        self,
        vehicle_id: int,
        aggregation_stats: Dict,
        representations: Optional[Dict] = None,
    ):
        """
        Log federated aggregation event.

        Args:
            vehicle_id: Vehicle ID
            aggregation_stats: Stats from AsynchronousAggregator
            representations: Embeddings/prototypes before and after
        """
        vid_str = str(vehicle_id)

        agg_event = {
            "vehicle_id": vehicle_id,
            "episode_id": self.current_episode["episode_id"],
            "num_neighbors": aggregation_stats.get("num_neighbors", 0),
            "local_weight": aggregation_stats.get("local_weight", 1.0),
            "total_weight": aggregation_stats.get("total_weight", 1.0),
            "num_stale": aggregation_stats.get("num_stale", 0),
        }

        # Add to aggregation history
        self.aggregation_history.append(agg_event)

        # Update episode aggregation stats
        self.current_episode["num_aggregations"] += 1

        # Track vehicle-specific aggregation
        self.current_vehicle_metrics[vid_str]["aggregations"].append(agg_event)

    def log_representation_quality(
        self,
        vehicle_id: int,
        embeddings: Optional[np.ndarray] = None,
        prototypes: Optional[np.ndarray] = None,
    ):
        """
        Log representation quality metrics.

        Args:
            vehicle_id: Vehicle ID
            embeddings: Embedding vectors
            prototypes: Prototype vectors
        """
        vid_str = str(vehicle_id)
        quality = {}

        if embeddings is not None:
            norms = np.linalg.norm(embeddings, axis=1)
            quality["embedding_norm_mean"] = float(norms.mean())
            quality["embedding_norm_std"] = float(norms.std())
            quality["has_nan"] = bool(np.isnan(embeddings).any())
            quality["has_inf"] = bool(np.isinf(embeddings).any())

            if quality["has_nan"] or quality["has_inf"]:
                self.current_episode["num_nan_inf"] += 1

        if prototypes is not None and len(prototypes) > 1:
            # Compute pairwise distances
            dists = []
            for i in range(len(prototypes)):
                for j in range(i + 1, len(prototypes)):
                    dist = np.linalg.norm(prototypes[i] - prototypes[j])
                    dists.append(dist)
            quality["prototype_separation"] = float(np.mean(dists))

        if quality:
            self.current_vehicle_metrics[vid_str]["quality"].append(quality)

    def end_episode(
        self,
        final_reward: float,
        final_train_loss: float,
        comm_cost_mb: float,
        rl_stats: Optional[Dict] = None,
        duration_seconds: float = 0.0,
        lstm_loss: Optional[float] = None,
    ):
        """
        Finalize episode and compute summary statistics.

        Args:
            final_reward: Total episode reward
            final_train_loss: Final training loss (average across vehicles)
            comm_cost_mb: Total communication cost
            rl_stats: RL training stats (losses, entropy, etc.)
            duration_seconds: Episode duration
            lstm_loss: Average LSTM training loss (if training from scratch)
        """
        if self.current_episode is None:
            return

        # Update final metrics
        self.current_episode["reward"] = float(final_reward)
        self.current_episode["train_loss"] = float(final_train_loss)
        self.current_episode["comm_cost_mb"] = float(comm_cost_mb)
        self.current_episode["episode_duration_seconds"] = float(duration_seconds)

        # Update LSTM loss if provided
        if lstm_loss is not None:
            self.current_episode["lstm_loss"] = float(lstm_loss)

        # Update RL stats if provided
        if rl_stats:
            self.current_episode["actor_loss"] = float(rl_stats.get("actor_loss", 0))
            self.current_episode["critic_loss"] = float(rl_stats.get("critic_loss", 0))
            self.current_episode["entropy"] = float(rl_stats.get("entropy", 0))
            self.current_episode["clip_fraction"] = float(rl_stats.get("clip_fraction", 0))

        # Compute aggregation statistics
        recent_aggs = [
            agg for agg in self.aggregation_history
            if agg["episode_id"] == self.current_episode["episode_id"]
        ]

        if recent_aggs:
            self.current_episode["avg_neighbors_per_agg"] = np.mean([
                agg["num_neighbors"] for agg in recent_aggs
            ])
            self.current_episode["avg_divergence"] = 0.0  # TODO: track divergence

        # Compute vehicle-level summaries
        vehicle_summaries = {}
        for vid, metrics in self.current_vehicle_metrics.items():
            summary = {
                "num_actions": len(metrics.get("actions", [])),
                "num_aggregations": len(metrics.get("aggregations", [])),
                "comm_success_rate": np.mean([
                    a["success"] for a in metrics.get("actions", [])
                ]) if metrics.get("actions") else 0.0,
            }

            if metrics.get("quality"):
                quality_list = metrics["quality"]
                if any("embedding_norm_mean" in q for q in quality_list):
                    summary["avg_embedding_norm"] = np.mean([
                        q["embedding_norm_mean"] for q in quality_list
                        if "embedding_norm_mean" in q
                    ])
                if any("prototype_separation" in q for q in quality_list):
                    summary["avg_prototype_separation"] = np.mean([
                        q["prototype_separation"] for q in quality_list
                        if "prototype_separation" in q
                    ])

            vehicle_summaries[vid] = summary

        # Add vehicle summaries to episode
        self.current_episode["vehicle_summaries"] = vehicle_summaries

        # Append to episodes list
        self.episodes.append(self.current_episode)

        # Periodic saving
        episode_id = self.current_episode["episode_id"]

        # Save episode summary
        if (episode_id + 1) % self.save_interval == 0:
            self._save_episodes_summary()
            print(f"[DFLTracker] Saved episodes summary (up to episode {episode_id})")

        # Save detailed vehicle metrics
        if (episode_id + 1) % self.vehicle_detail_interval == 0:
            self._save_vehicle_details(episode_id)
            print(f"[DFLTracker] Saved vehicle details (episode {episode_id})")

        # Save checkpoint
        if (episode_id + 1) % (self.save_interval * 2) == 0:
            self._save_checkpoint(episode_id)

        # Clear current episode
        self.current_episode = None

    def finalize(self):
        """Finalize tracking and save all remaining data."""
        print("\n[DFLTracker] Finalizing and saving all metrics...")

        # Update metadata with end time
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["total_episodes"] = len(self.episodes)
        self.metadata["total_duration_seconds"] = time.time() - self.start_timestamp

        # Save everything
        self._save_metadata()
        self._save_episodes_summary()
        self._save_aggregation_history()
        self._save_sampled_timesteps()

        if self.current_episode:
            self._save_vehicle_details(self.current_episode["episode_id"])

        # Create summary statistics
        self._create_summary_statistics()

        print(f"[DFLTracker] All metrics saved to: {self.output_dir}")
        print(f"[DFLTracker] Total episodes: {len(self.episodes)}")

    def _save_metadata(self):
        """Save metadata to JSON."""
        path = self.output_dir / "metadata.json"
        with open(path, "w") as f:
            json.dump(self.metadata, f, indent=2, cls=NumpyEncoder)

    def _save_episodes_summary(self):
        """Save episode summary to JSON."""
        path = self.output_dir / "episodes_summary.json"
        with open(path, "w") as f:
            json.dump(self.episodes, f, indent=2, cls=NumpyEncoder)

    def _save_vehicle_details(self, episode_id: int):
        """Save detailed vehicle metrics for recent episodes."""
        # Save vehicle details for episodes in this batch
        start_id = max(0, episode_id - self.vehicle_detail_interval + 1)
        end_id = episode_id

        vehicle_data = {
            "episode_range": [start_id, end_id],
            "vehicles": dict(self.current_vehicle_metrics),
        }

        path = self.output_dir / "vehicles_detail" / f"episodes_{start_id:06d}_{end_id:06d}.json"
        with open(path, "w") as f:
            json.dump(vehicle_data, f, indent=2, cls=NumpyEncoder)

    def _save_aggregation_history(self):
        """Save complete aggregation history."""
        path = self.output_dir / "aggregation_history.json"
        with open(path, "w") as f:
            json.dump(self.aggregation_history, f, indent=2, cls=NumpyEncoder)

    def _save_sampled_timesteps(self):
        """Save sampled timestep data."""
        if self.sampled_timesteps:
            path = self.output_dir / "timesteps_sampled.json"
            with open(path, "w") as f:
                json.dump(self.sampled_timesteps, f, indent=2, cls=NumpyEncoder)

    def _save_checkpoint(self, episode_id: int):
        """Save checkpoint for crash recovery."""
        checkpoint = {
            "episode_id": episode_id,
            "timestamp": datetime.now().isoformat(),
            "num_episodes_completed": len(self.episodes),
        }
        path = self.output_dir / "checkpoints" / f"checkpoint_{episode_id:06d}.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _create_summary_statistics(self):
        """Create overall summary statistics."""
        if not self.episodes:
            return

        summary = {
            "training": {
                "total_episodes": len(self.episodes),
                "total_duration_hours": (time.time() - self.start_timestamp) / 3600,
            },
            "performance": {
                "initial_train_loss": self.episodes[0]["train_loss"],
                "final_train_loss": self.episodes[-1]["train_loss"],
                "best_train_loss": min(ep["train_loss"] for ep in self.episodes),
                "avg_train_loss": np.mean([ep["train_loss"] for ep in self.episodes]),
            },
            "communication": {
                "initial_comm_cost": self.episodes[0]["comm_cost_mb"],
                "final_comm_cost": self.episodes[-1]["comm_cost_mb"],
                "avg_comm_cost": np.mean([ep["comm_cost_mb"] for ep in self.episodes]),
                "total_comm_cost": sum(ep["comm_cost_mb"] for ep in self.episodes),
            },
            "actions": {
                "total_E": sum(ep["what_actions"]["E"] for ep in self.episodes),
                "total_W": sum(ep["what_actions"]["W"] for ep in self.episodes),
            },
            "efficiency": {
                "loss_improvement": self.episodes[0]["train_loss"] - self.episodes[-1]["train_loss"],
                "loss_improvement_pct": 100 * (
                    (self.episodes[0]["train_loss"] - self.episodes[-1]["train_loss"]) / self.episodes[0]["train_loss"]
                ) if self.episodes[0]["train_loss"] > 0 else 0,
                "bandwidth_reduction_pct": 100 * (
                    1 - self.episodes[-1]["comm_cost_mb"] / self.episodes[0]["comm_cost_mb"]
                ) if self.episodes[0]["comm_cost_mb"] > 0 else 0,
            }
        }

        # Calculate action distribution (updated: removed Prototypes)
        total_actions = sum(summary["actions"].values())
        if total_actions > 0:
            summary["actions"]["E_pct"] = 100 * summary["actions"]["total_E"] / total_actions
            summary["actions"]["W_pct"] = 100 * summary["actions"]["total_W"] / total_actions

        path = self.output_dir / "summary_statistics.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

        print(f"\n{'='*60}")
        print("DFL TRAINING SUMMARY")
        print('='*60)
        print("\nPerformance:")
        print(f"  Train Loss: {summary['performance']['initial_train_loss']:.4f} → {summary['performance']['final_train_loss']:.4f}")
        print(f"  Best Train Loss: {summary['performance']['best_train_loss']:.4f}")
        print(f"  Loss Improvement: {summary['efficiency']['loss_improvement_pct']:.1f}%")

        print("\nCommunication:")
        print(f"  Cost: {summary['communication']['initial_comm_cost']:.1f} → {summary['communication']['final_comm_cost']:.1f} MB")
        print(f"  Reduction: {summary['efficiency']['bandwidth_reduction_pct']:.1f}%")

        print("\nActions:")
        if total_actions > 0:
            print(f"  E (Embeddings): {summary['actions']['E_pct']:.1f}%")
            print(f"  W (Weights): {summary['actions']['W_pct']:.1f}%")
        print('='*60)

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
