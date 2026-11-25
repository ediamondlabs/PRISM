"""
Vehicle node class for V2V simulation.

Each VehicleNode represents a single vehicle in the V2V network with:
- State tracking (position, velocity, trajectory)
- Neighbor management (connection/disconnection events)
- Layer 1 integration (embeddings, anomaly scores)
- Communication history for RL rewards
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


class VehicleNode:
    """Represents a single vehicle node in the V2V network."""

    def __init__(
        self,
        vehicle_id: int,
        window_size: int = 50,
        num_features: int = 10,
        embedding_dim: int = 256,
    ):
        """Initialize vehicle node.

        Args:
            vehicle_id: Unique identifier for this vehicle
            window_size: Size of trajectory window for Layer 1 model
            num_features: Number of trajectory features
            embedding_dim: Dimension of embedding from Layer 1 model
        """
        self.vehicle_id = vehicle_id
        self.window_size = window_size
        self.num_features = num_features
        self.embedding_dim = embedding_dim

        # Current state
        self.current_timestep: int = 0
        self.position: Tuple[float, float] = (0.0, 0.0)
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.speed: float = 0.0

        # Trajectory window (last window_size timesteps)
        self.trajectory_window: np.ndarray = np.zeros(
            (window_size, num_features), dtype=np.float32
        )

        # Layer 1 outputs
        self.embedding: Optional[np.ndarray] = None
        self.anomaly_score: Optional[float] = None

        # Neighbor tracking
        self.neighbors: Set[int] = set()
        self.neighbor_history: List[Tuple[int, str, int]] = []

        # Communication history: (timestep, action_type, neighbor_id, data_dict)
        self.communication_history: List[Tuple[int, str, int, Dict]] = []

        # Full trajectory data reference (set by environment)
        self._trajectory_data: Optional[np.ndarray] = None
        self._trajectory_start_idx: int = 0
        self._trajectory_length: int = 0

    def set_trajectory_data(
        self,
        trajectory_data: np.ndarray,
        start_idx: int,
        length: int,
    ):
        """Set the trajectory data for this vehicle.

        Args:
            trajectory_data: Full trajectory array (all_timesteps, num_features)
            start_idx: Start index in trajectory data for this vehicle
            length: Number of timesteps in trajectory
        """
        self._trajectory_data = trajectory_data
        self._trajectory_start_idx = start_idx
        self._trajectory_length = length

    def update_state(self, timestep: int):
        """Update vehicle state for given timestep.

        Args:
            timestep: Current simulation timestep
        """
        if self._trajectory_data is None:
            raise ValueError("Trajectory data not set. Call set_trajectory_data() first.")

        if timestep >= self._trajectory_length:
            return

        self.current_timestep = timestep

        # Get current state from trajectory data
        # Handle both formats: 2D (timesteps, features) and 3D (windows, window_size, features)
        if self._trajectory_data.ndim == 2:
            # Flat trajectory data (timesteps, features)
            data_idx = self._trajectory_start_idx + timestep
            current_state = self._trajectory_data[data_idx]

            # Update trajectory window (shift left and add new)
            if timestep < self.window_size:
                # Still filling initial window
                self.trajectory_window[timestep] = current_state
            else:
                # Shift window and add new state
                self.trajectory_window[:-1] = self.trajectory_window[1:]
                self.trajectory_window[-1] = current_state

        else:
            # Windowed trajectory data (windows, window_size, features)
            # Use the window at current timestep (or closest available)
            window_idx = min(timestep, self._trajectory_length - 1)
            current_window = self._trajectory_data[window_idx]

            # Extract current state (last timestep of the window represents current state)
            current_state = current_window[-1]

            # Update trajectory window to use the current window
            self.trajectory_window = current_window.copy()

        # Extract velocity and speed from current state
        self.velocity = (float(current_state[0]), float(current_state[1]))
        self.speed = float(current_state[2])

        # Simple position update (integrate velocity)
        # In a real scenario, you'd use actual GPS coordinates from the dataset
        dt = 0.1  # 10 Hz sampling
        self.position = (
            self.position[0] + self.velocity[0] * dt,
            self.position[1] + self.velocity[1] * dt,
        )

    def compute_embedding(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> np.ndarray:
        """Compute trajectory embedding using Layer 1 encoder.

        Args:
            model: LSTM Autoencoder model from Layer 1
            device: Device for computation

        Returns:
            Embedding vector (embedding_dim,)
        """
        if self.current_timestep < self.window_size:
            # Not enough data for full window yet
            self.embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            return self.embedding

        # Convert trajectory window to tensor
        window_tensor = torch.from_numpy(self.trajectory_window).unsqueeze(0).float()
        window_tensor = window_tensor.to(device)

        # Get embedding from encoder
        model.eval()
        with torch.no_grad():
            embedding = model.encode(window_tensor)
            self.embedding = embedding.cpu().numpy().flatten()

        return self.embedding

    def compute_anomaly_score(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> float:
        """Compute anomaly score (reconstruction error) using Layer 1 model.

        Args:
            model: LSTM Autoencoder model from Layer 1
            device: Device for computation

        Returns:
            Anomaly score (reconstruction MSE)
        """
        if self.current_timestep < self.window_size:
            # Not enough data for full window yet
            self.anomaly_score = 0.0
            return self.anomaly_score

        # Convert trajectory window to tensor
        window_tensor = torch.from_numpy(self.trajectory_window).unsqueeze(0).float()
        window_tensor = window_tensor.to(device)

        # Compute reconstruction error
        model.eval()
        with torch.no_grad():
            error = model.compute_reconstruction_error(window_tensor, reduction="mean")
            self.anomaly_score = float(error.item())

        return self.anomaly_score

    def update_neighbors(self, new_neighbors: Set[int]):
        """Update neighbor list and track connection/disconnection events.

        Args:
            new_neighbors: New set of neighbor vehicle IDs
        """
        # Find connections (new neighbors)
        connected = new_neighbors - self.neighbors
        for neighbor_id in connected:
            self.neighbor_history.append(
                (self.current_timestep, "connected", neighbor_id)
            )

        # Find disconnections (neighbors that left)
        disconnected = self.neighbors - new_neighbors
        for neighbor_id in disconnected:
            self.neighbor_history.append(
                (self.current_timestep, "disconnected", neighbor_id)
            )

        # Update current neighbors
        self.neighbors = new_neighbors.copy()

    def log_communication(
        self,
        action_type: str,
        neighbor_id: int,
        data: Optional[Dict] = None,
    ):
        """Log a communication event.

        Args:
            action_type: Type of communication ('send', 'receive', 'broadcast')
            neighbor_id: ID of neighbor involved in communication
            data: Optional dictionary with communication data
        """
        self.communication_history.append(
            (self.current_timestep, action_type, neighbor_id, data or {})
        )

    def get_observation(self) -> Dict:
        """Get current observation for RL agent.

        Returns:
            Observation dictionary with current state
        """
        return {
            "vehicle_id": self.vehicle_id,
            "timestep": self.current_timestep,
            "position": np.array(self.position, dtype=np.float32),
            "velocity": np.array(self.velocity, dtype=np.float32),
            "speed": self.speed,
            "trajectory_window": self.trajectory_window.copy(),
            "embedding": self.embedding.copy() if self.embedding is not None else None,
            "anomaly_score": self.anomaly_score,
            "num_neighbors": len(self.neighbors),
            "neighbor_ids": list(self.neighbors),
        }

    def reset(self):
        """Reset vehicle state for new episode."""
        self.current_timestep = 0
        self.position = (0.0, 0.0)
        self.velocity = (0.0, 0.0)
        self.speed = 0.0
        self.trajectory_window = np.zeros(
            (self.window_size, self.num_features), dtype=np.float32
        )
        self.embedding = None
        self.anomaly_score = None
        self.neighbors = set()
        self.neighbor_history = []
        self.communication_history = []

    def get_state_dict(self) -> Dict:
        """Get complete state as dictionary.

        Returns:
            Dictionary with all state information
        """
        return {
            "vehicle_id": self.vehicle_id,
            "current_timestep": self.current_timestep,
            "position": self.position,
            "velocity": self.velocity,
            "speed": self.speed,
            "embedding": self.embedding,
            "anomaly_score": self.anomaly_score,
            "num_neighbors": len(self.neighbors),
            "neighbors": list(self.neighbors),
            "neighbor_history_length": len(self.neighbor_history),
            "communication_history_length": len(self.communication_history),
        }
