"""
V2V Environment for multi-agent simulation.

Implements a hybrid Gym-like + custom API for V2V network simulation:
- Gym methods: reset(), step(), render()
- Custom methods: advance_time(), discover_neighbors(), get_neighbor_graph()

Manages 20 vehicles moving through recorded trajectories with dynamic topology.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from src.models import (
    create_autoencoder_from_config,
    create_neighbor_batch,
    extract_neighbor_features,
)
from src.simulation.vehicle_node import VehicleNode
from src.utils.logging_utils import setup_logger
from src.utils.reproducibility import get_device

logger = setup_logger(__name__)


class V2VEnvironment:
    """V2V network environment for multi-agent RL simulation."""

    def __init__(self, config: DictConfig):
        """Initialize V2V environment.

        Args:
            config: Hydra configuration
        """
        self.config = config
        self.sim_config = config.simulation

        # Environment parameters
        self.num_vehicles = self.sim_config.num_vehicles
        self.comm_range = self.sim_config.comm_range_m
        self.time_step_duration = self.sim_config.time_step_duration_s
        self.max_timesteps = self.sim_config.max_episode_timesteps

        # Current state
        self.current_timestep: int = 0
        self.episode_count: int = 0

        # Device for model computations
        self.device = get_device(config.get("device", "auto"))

        # Load Layer 1 model for embeddings
        self.lstm_model = self._load_layer1_model()

        # Load Layer 2 model for neighborhood encoding (optional)
        self.use_neighbor_encoder = self.sim_config.get("use_neighbor_encoder", False)
        self.neighbor_encoder = None
        if self.use_neighbor_encoder:
            self.neighbor_encoder = self._load_layer2_model()

        # Load trajectory data
        self.trajectory_data, self.vehicle_ids = self._load_trajectory_data()

        # Initialize vehicles
        self.vehicles: Dict[int, VehicleNode] = {}
        self._initialize_vehicles()

        # Neighbor graph (adjacency list)
        self.neighbor_graph: Dict[int, Set[int]] = {
            vid: set() for vid in self.vehicle_ids
        }

        logger.info(f"Initialized V2VEnvironment with {self.num_vehicles} vehicles")
        logger.info(f"Communication range: {self.comm_range}m")
        logger.info(f"Max timesteps: {self.max_timesteps}")

    def _load_layer1_model(self) -> torch.nn.Module:
        """Load trained LSTM Autoencoder from checkpoint or create fresh model.

        Returns:
            Loaded model in eval mode (if pre-trained) or train mode (if training from scratch)
        """
        train_layer1 = self.config.get("train_layer1", False)
        checkpoint_path = self.sim_config.get("layer1_checkpoint", None)

        # If training from scratch, create fresh model
        if train_layer1:
            logger.info("train_layer1=True: Creating fresh LSTM autoencoder for training from scratch")
            model = create_autoencoder_from_config(self.config, device=self.device)
            model.train()  # Set to training mode
            return model

        # Otherwise, load pre-trained checkpoint
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            if checkpoint_path is not None:
                logger.warning(
                    f"Layer 1 checkpoint not found: {checkpoint_path}. "
                    "Using freshly initialized model in eval mode."
                )
            model = create_autoencoder_from_config(self.config, device=self.device)
            model.eval()
            return model

        # Load model
        model = create_autoencoder_from_config(self.config, device=self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logger.info(f"Loaded Layer 1 model from {checkpoint_path}")
        return model

    def _load_layer2_model(self) -> torch.nn.Module:
        """Load trained Neighbor Encoder from checkpoint.

        Returns:
            Loaded model in eval mode
        """
        from src.models import create_neighbor_encoder_from_config

        checkpoint_path = self.sim_config.get("layer2_checkpoint", None)

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            if checkpoint_path is not None:
                logger.warning(
                    f"Layer 2 checkpoint not found: {checkpoint_path}. "
                    "Using freshly initialized model."
                )
            model = create_neighbor_encoder_from_config(self.config, device=self.device)
            model.eval()
            return model

        # Load model
        model = create_neighbor_encoder_from_config(self.config, device=self.device)
        checkpoint = torch.load(
            Path(checkpoint_path), map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logger.info(f"Loaded Layer 2 model from {checkpoint_path}")
        return model

    def _load_trajectory_data(self) -> Tuple[Dict[int, np.ndarray], List[int]]:
        """Load preprocessed trajectory data from Phase 1.

        Returns:
            Tuple of (trajectory_data_dict, vehicle_ids)
        """
        data_path = Path(self.sim_config.data.preprocessed_path)

        # Try to load from .npz file
        npz_file = data_path / "preprocessed_data.npz"
        if npz_file.exists():
            data = np.load(npz_file, allow_pickle=True)

            # Get data and vehicle IDs
            # Concatenate all splits for full trajectories
            all_data = np.concatenate([
                data["train_data"],
                data["val_data"],
                data["test_data"]
            ], axis=0)

            all_vehicle_ids = np.concatenate([
                data["train_vehicle_ids"],
                data["val_vehicle_ids"],
                data["test_vehicle_ids"]
            ], axis=0)

            # Group by vehicle ID
            unique_vehicle_ids = np.unique(all_vehicle_ids)
            trajectory_dict = {}

            for vid in unique_vehicle_ids:
                mask = all_vehicle_ids == vid
                vehicle_windows = all_data[mask]
                # Reconstruct full trajectory from overlapping windows
                # For now, just use windows directly
                trajectory_dict[int(vid)] = vehicle_windows

            # Select subset if needed
            selected_ids = sorted(unique_vehicle_ids)[:self.num_vehicles]

            logger.info(f"Loaded trajectory data for {len(selected_ids)} vehicles")
            logger.info(f"Total windows: {all_data.shape[0]}")

            return trajectory_dict, [int(vid) for vid in selected_ids]

        else:
            raise FileNotFoundError(
                f"Preprocessed data not found at {npz_file}. "
                "Run train.py first to generate preprocessed data."
            )

    def _initialize_vehicles(self):
        """Initialize all vehicle nodes with trajectory data."""
        window_size = self.config.data.window_size
        num_features = len(self.config.data.features)
        embedding_dim = self.config.model.architecture.embedding_dim

        for vehicle_id in self.vehicle_ids:
            # Create vehicle node
            vehicle = VehicleNode(
                vehicle_id=vehicle_id,
                window_size=window_size,
                num_features=num_features,
                embedding_dim=embedding_dim,
            )

            # Get vehicle's trajectory windows
            vehicle_windows = self.trajectory_data[vehicle_id]

            # Flatten windows to get trajectory sequence
            # Since windows overlap (stride < window_size), we need to reconstruct
            # For simplicity, use the first window's sequence
            if len(vehicle_windows) > 0:
                # Use all windows as timesteps
                vehicle.set_trajectory_data(
                    trajectory_data=vehicle_windows,
                    start_idx=0,
                    length=len(vehicle_windows),
                )

            self.vehicles[vehicle_id] = vehicle

        logger.info(f"Initialized {len(self.vehicles)} vehicles")

    def reset(self) -> Dict[int, Dict]:
        """Reset environment for new episode (Gym API).

        Returns:
            Dictionary mapping vehicle_id to initial observation
        """
        self.current_timestep = 0
        self.episode_count += 1

        # Reset all vehicles
        for vehicle in self.vehicles.values():
            vehicle.reset()
            vehicle.update_state(0)

        # Discover initial neighbors
        self.discover_neighbors()

        # Compute initial embeddings and anomaly scores
        self._update_all_embeddings()
        self._compute_all_anomaly_scores()

        # Compute neighborhood encodings (Layer 2)
        neighborhood_encodings = self._compute_neighborhood_encodings()

        # Get initial observations
        observations = {
            vid: vehicle.get_observation()
            for vid, vehicle in self.vehicles.items()
        }

        # Add neighborhood encodings to observations if available
        if neighborhood_encodings:
            for vid, obs in observations.items():
                obs["neighborhood_encoding"] = neighborhood_encodings.get(vid)

        logger.info(f"Episode {self.episode_count} started")
        return observations

    def step(
        self, actions: Optional[Dict[int, Dict]] = None
    ) -> Tuple[Dict[int, Dict], Dict[int, float], bool, Dict]:
        """Execute one environment step (Gym API).

        Args:
            actions: Dictionary mapping vehicle_id to action dict
                    For now, actions are optional (mobility-only simulation)

        Returns:
            Tuple of (observations, rewards, done, info)
        """
        # Advance time
        self.advance_time()

        # Discover neighbors based on new positions
        self.discover_neighbors()

        # Update embeddings and anomaly scores
        self._update_all_embeddings()
        self._compute_all_anomaly_scores()

        # Compute neighborhood encodings (Layer 2)
        neighborhood_encodings = self._compute_neighborhood_encodings()

        # Process actions if provided (placeholder for Layer 3 RL)
        if actions is not None:
            self._process_actions(actions)

        # Get observations
        observations = {
            vid: vehicle.get_observation()
            for vid, vehicle in self.vehicles.items()
        }

        # Add neighborhood encodings to observations if available
        if neighborhood_encodings:
            for vid, obs in observations.items():
                obs["neighborhood_encoding"] = neighborhood_encodings.get(vid)

        # Compute rewards (placeholder for Layer 3)
        rewards = {vid: 0.0 for vid in self.vehicle_ids}

        # Check if episode is done
        done = self.current_timestep >= self.max_timesteps

        # Info dictionary
        info = {
            "timestep": self.current_timestep,
            "episode": self.episode_count,
            "neighbor_graph": self.get_neighbor_graph(),
        }

        return observations, rewards, done, info

    def advance_time(self):
        """Advance simulation by one timestep (custom method)."""
        self.current_timestep += 1

        # Update all vehicle states
        for vehicle in self.vehicles.values():
            vehicle.update_state(self.current_timestep)

    def discover_neighbors(self):
        """Update neighbor lists based on current positions (custom method).

        Uses O(n²) brute force distance checks (efficient for 20 vehicles).
        """
        # Compute all pairwise distances
        for vid1, vehicle1 in self.vehicles.items():
            new_neighbors = set()

            for vid2, vehicle2 in self.vehicles.items():
                if vid1 == vid2:
                    continue

                # Compute distance
                distance = self._compute_distance(vehicle1, vehicle2)

                # Add as neighbor if within comm range
                if distance <= self.comm_range:
                    new_neighbors.add(vid2)

            # Update vehicle's neighbor list
            vehicle1.update_neighbors(new_neighbors)

        # Update global neighbor graph
        self.neighbor_graph = {
            vid: vehicle.neighbors.copy()
            for vid, vehicle in self.vehicles.items()
        }

    def get_neighbor_graph(self) -> Dict[int, List[int]]:
        """Get current neighbor graph (custom method).

        Returns:
            Dictionary mapping vehicle_id to list of neighbor_ids
        """
        return {vid: list(neighbors) for vid, neighbors in self.neighbor_graph.items()}

    def get_vehicle_states(self) -> Dict[int, Dict]:
        """Get state dictionaries for all vehicles (custom method).

        Returns:
            Dictionary mapping vehicle_id to state dict
        """
        return {vid: vehicle.get_state_dict() for vid, vehicle in self.vehicles.items()}

    def _compute_distance(
        self, vehicle1: VehicleNode, vehicle2: VehicleNode
    ) -> float:
        """Compute Euclidean distance between two vehicles.

        Args:
            vehicle1: First vehicle
            vehicle2: Second vehicle

        Returns:
            Distance in meters
        """
        x1, y1 = vehicle1.position
        x2, y2 = vehicle2.position
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _update_all_embeddings(self):
        """Update trajectory embeddings for all vehicles using Layer 1 model."""
        for vehicle in self.vehicles.values():
            if vehicle.current_timestep >= vehicle.window_size:
                vehicle.compute_embedding(self.lstm_model, self.device)

    def _compute_all_anomaly_scores(self):
        """Compute anomaly scores for all vehicles using Layer 1 model."""
        for vehicle in self.vehicles.values():
            if vehicle.current_timestep >= vehicle.window_size:
                vehicle.compute_anomaly_score(self.lstm_model, self.device)

    def _compute_neighborhood_encodings(self) -> Dict[int, np.ndarray]:
        """Compute neighborhood encodings for all vehicles using Layer 2 model.

        Returns:
            Dictionary mapping vehicle_id to neighborhood encoding (32-dim)
        """
        if not self.use_neighbor_encoder or self.neighbor_encoder is None:
            return {}

        # Extract neighbor features for all vehicles
        neighbor_features_list = []
        vehicle_ids_ordered = []

        for vehicle_id, vehicle in self.vehicles.items():
            # Get neighbors
            neighbor_ids = vehicle.neighbors
            neighbor_vehicles = [self.vehicles[nid] for nid in neighbor_ids]

            # Extract features
            features = extract_neighbor_features(
                vehicle, neighbor_vehicles, self.comm_range
            )
            neighbor_features_list.append(features)
            vehicle_ids_ordered.append(vehicle_id)

        # Create batch
        if len(neighbor_features_list) == 0:
            return {}

        features_batch, mask_batch = create_neighbor_batch(neighbor_features_list)
        features_batch = features_batch.to(self.device)
        mask_batch = mask_batch.to(self.device)

        # Encode neighborhoods
        self.neighbor_encoder.eval()
        with torch.no_grad():
            encodings = self.neighbor_encoder(features_batch, mask_batch)
            encodings_np = encodings.cpu().numpy()

        # Create dictionary
        neighborhood_encodings = {
            vid: encodings_np[i] for i, vid in enumerate(vehicle_ids_ordered)
        }

        return neighborhood_encodings

    def _process_actions(self, actions: Dict[int, Dict]):
        """Process communication actions from RL agents (placeholder).

        Args:
            actions: Dictionary mapping vehicle_id to action dict
        """
        # Placeholder for Layer 3 RL integration
        # Will implement communication actions in future increments
        pass

    def render(self, mode: str = "human"):
        """Render environment (optional, placeholder).

        Args:
            mode: Render mode ('human', 'rgb_array', etc.)
        """
        if mode == "human":
            print(f"\n=== Timestep {self.current_timestep}/{self.max_timesteps} ===")
            print(f"Vehicles: {self.num_vehicles}")

            # Print neighbor statistics
            neighbor_counts = [len(v.neighbors) for v in self.vehicles.values()]
            avg_neighbors = np.mean(neighbor_counts)
            print(f"Average neighbors: {avg_neighbors:.2f}")

            # Print anomaly statistics (if available)
            anomaly_scores = [
                v.anomaly_score
                for v in self.vehicles.values()
                if v.anomaly_score is not None
            ]
            if anomaly_scores:
                print(f"Anomaly scores: mean={np.mean(anomaly_scores):.4f}, "
                      f"std={np.std(anomaly_scores):.4f}")

    def get_metrics(self) -> Dict:
        """Get environment metrics for analysis.

        Returns:
            Dictionary of metrics
        """
        neighbor_counts = [len(v.neighbors) for v in self.vehicles.values()]
        anomaly_scores = [
            v.anomaly_score
            for v in self.vehicles.values()
            if v.anomaly_score is not None
        ]

        metrics = {
            "timestep": self.current_timestep,
            "num_vehicles": len(self.vehicles),
            "avg_neighbors": float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
            "min_neighbors": int(np.min(neighbor_counts)) if neighbor_counts else 0,
            "max_neighbors": int(np.max(neighbor_counts)) if neighbor_counts else 0,
        }

        if anomaly_scores:
            metrics.update({
                "avg_anomaly_score": float(np.mean(anomaly_scores)),
                "max_anomaly_score": float(np.max(anomaly_scores)),
            })

        return metrics
