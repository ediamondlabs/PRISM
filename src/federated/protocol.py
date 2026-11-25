"""
DFL Protocol for asynchronous decentralized federated learning.

Manages communication and aggregation for a single vehicle:
- Execute RL-driven communication actions (WHEN/WHO/WHAT)
- Receive representations from neighbors
- Aggregate with staleness weighting
- Track communication history and metrics
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.federated.aggregation import AsynchronousAggregator
from src.federated.representations import RepresentationExtractor


class DFLProtocol:
    """DFL protocol manager for one vehicle."""

    def __init__(
        self,
        vehicle_id: int,
        local_model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        representation_extractor: RepresentationExtractor,
        aggregator: AsynchronousAggregator,
        device: torch.device = torch.device("cpu"),
        train_model: bool = False,
        model_lr: float = 0.001,
        training_device: torch.device = None,
    ):
        """Initialize DFL protocol for a vehicle.

        Args:
            vehicle_id: ID of this vehicle
            local_model: Local LSTM autoencoder (copy)
            data_loader: Private data loader for this vehicle
            representation_extractor: Extractor for E/P representations
            aggregator: Async aggregator with staleness weighting
            device: Device for computations (inference)
            train_model: If True, train the local_model (for from-scratch training)
            model_lr: Learning rate for model training (if train_model=True)
            training_device: Device for training LSTM (can be different from device)
        """
        self.vehicle_id = vehicle_id
        self.local_model = local_model
        self.data_loader = data_loader
        self.representation_extractor = representation_extractor
        self.aggregator = aggregator
        self.device = device
        self.train_model = train_model
        self.training_device = training_device if training_device is not None else device

        # Optimizer for local model (if training from scratch)
        self.model_optimizer = None
        if train_model:
            self.model_optimizer = torch.optim.Adam(local_model.parameters(), lr=model_lr)

        # Buffer for received representations
        # Key: neighbor_id, Value: (repr_data, repr_type, timestamp)
        self.received_buffer: Dict[int, Tuple[np.ndarray, str, float]] = {}

        # Last update time (for staleness tracking)
        self.last_update_time = time.time()
        self.last_aggregation_time = time.time()

        # Communication history
        # Each entry: (timestamp, action_type, neighbor_id, repr_type, size_mb, success)
        self.communication_log: List[Tuple[float, str, int, str, float, bool]] = []

        # Statistics
        self.updates_received = 0
        self.updates_sent = 0
        self.total_data_sent_mb = 0.0
        self.total_data_received_mb = 0.0
        self.local_training_loss = 0.0  # Track LSTM training loss

    def train_local_model(self, num_epochs: int = 1) -> float:
        """Train local LSTM autoencoder on private data.

        Note: Model is moved to training_device, trained, then moved back to CPU for storage.
        This allows training on GPU without keeping all 50 models in GPU memory.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Average training loss
        """
        if not self.train_model or self.model_optimizer is None:
            return 0.0

        # Check if data loader has any data
        if len(self.data_loader) == 0:
            return 0.0

        # Move model to training device temporarily
        original_device = next(self.local_model.parameters()).device
        self.local_model.to(self.training_device)

        self.local_model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            for batch in self.data_loader:
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch

                # Skip empty batches
                if x.size(0) == 0:
                    continue

                # Move data to training device
                x = x.to(self.training_device)

                # Forward pass
                self.model_optimizer.zero_grad()
                x_recon, _ = self.local_model(x)  # Model returns (reconstruction, embedding)
                loss = torch.nn.functional.mse_loss(x_recon, x)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)
                self.model_optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.local_training_loss = avg_loss

        # Move model back to CPU for storage (save GPU memory)
        self.local_model.cpu()
        self.local_model.eval()

        return avg_loss

    def execute_communication_action(
        self,
        who_action: List[int],
        what_action: str,
        neighbors: Set[int],
        num_embedding_samples: int = 100,
    ) -> Tuple[bool, Dict]:
        """Execute communication action (send representation to neighbors).

        Args:
            who_action: List of neighbor IDs to send to (from RL)
            what_action: Representation type 'E' or 'P' (from RL)
            neighbors: Current neighbors (from environment)
            num_embedding_samples: Number of embedding samples (for 'E')

        Returns:
            Tuple of (success, communication_info)
            where communication_info contains:
                - data: Representation data package
                - size_mb: Data size in MB
                - recipients: List of neighbor IDs who received it
        """
        # Filter WHO action to only include actual neighbors
        valid_recipients = [nid for nid in who_action if nid in neighbors]

        if len(valid_recipients) == 0:
            # No valid neighbors to send to
            return False, {
                "data": None,
                "size_mb": 0.0,
                "recipients": [],
                "reason": "no_valid_neighbors",
            }

        try:
            # Temporarily move model to GPU for fast inference
            # (Models are stored on CPU to save memory)
            original_device = next(self.local_model.parameters()).device
            if original_device.type == 'cpu':
                self.local_model.to(self.training_device)

            # Extract representation
            repr_package = self.representation_extractor.extract_and_package(
                model=self.local_model,
                data_loader=self.data_loader,
                repr_type=what_action,
                num_samples=num_embedding_samples,
                compress=False,  # Optional: can enable compression
            )

            # Move model back to CPU for storage
            if original_device.type == 'cpu':
                self.local_model.cpu()

            data = repr_package["data"]
            size_mb = repr_package["size_mb"]

            # Log communication
            timestamp = time.time()
            for recipient_id in valid_recipients:
                self.communication_log.append(
                    (timestamp, "send", recipient_id, what_action, size_mb, True)
                )

            # Update statistics
            self.updates_sent += len(valid_recipients)
            self.total_data_sent_mb += size_mb * len(valid_recipients)

            comm_info = {
                "data": repr_package,
                "size_mb": size_mb,
                "recipients": valid_recipients,
                "timestamp": timestamp,
                "success": True,
            }

            return True, comm_info

        except Exception as e:
            # Log failure
            timestamp = time.time()
            for recipient_id in valid_recipients:
                self.communication_log.append(
                    (timestamp, "send_failed", recipient_id, what_action, 0.0, False)
                )

            return False, {
                "data": None,
                "size_mb": 0.0,
                "recipients": [],
                "error": str(e),
                "success": False,
            }

    def receive_representation(
        self,
        neighbor_id: int,
        repr_package: Dict,
    ):
        """Receive representation from a neighbor.

        Stores in buffer for later aggregation.

        Args:
            neighbor_id: ID of neighbor who sent the data
            repr_package: Package dict with keys:
                - type: 'E' or 'P'
                - data: Representation data
                - size_mb: Size in MB
                - timestamp: When it was sent
        """
        # Extract data
        repr_type = repr_package["type"]
        data = repr_package["data"]
        size_mb = repr_package["size_mb"]
        timestamp = repr_package.get("timestamp", time.time())

        # Store in buffer (overwrites old data from same neighbor)
        self.received_buffer[neighbor_id] = (data, repr_type, timestamp)

        # Log reception
        current_time = time.time()
        self.communication_log.append(
            (current_time, "receive", neighbor_id, repr_type, size_mb, True)
        )

        # Update statistics
        self.updates_received += 1
        self.total_data_received_mb += size_mb

    def aggregate_received_data(
        self,
        force: bool = False,
    ) -> Tuple[bool, Dict]:
        """Aggregate received representations with local model.

        Args:
            force: Force aggregation even if buffer is empty

        Returns:
            Tuple of (success, aggregation_info)
        """
        if len(self.received_buffer) == 0 and not force:
            return False, {"reason": "empty_buffer"}

        # Separate by representation type
        embedding_neighbors = {}
        weight_neighbors = {}
        timestamps = {}

        for neighbor_id, (data, repr_type, timestamp) in self.received_buffer.items():
            timestamps[neighbor_id] = timestamp

            if repr_type == "E":
                embedding_neighbors[neighbor_id] = data
            elif repr_type == "W":
                weight_neighbors[neighbor_id] = data

        current_time = time.time()
        aggregation_info = {
            "timestamp": current_time,
            "num_neighbors": len(self.received_buffer),
            "embedding_aggregation": None,
            "weight_aggregation": None,
        }

        original_device = next(self.local_model.parameters()).device
        if original_device.type == 'cpu':
            self.local_model.to(self.training_device)

        # Aggregate weights if any (paper Algorithm 1, Phase 4)
        if len(weight_neighbors) > 0:
            local_weights = self.representation_extractor.extract_weights(self.local_model)

            aggregated_weights, weight_stats = self.aggregator.aggregate_weights(
                local_weights=local_weights,
                neighbor_weights=weight_neighbors,
                timestamps=timestamps,
                current_time=current_time,
            )

            # Apply aggregated weights to local model
            self.local_model.load_state_dict(aggregated_weights)

            aggregation_info["weight_aggregation"] = {
                "stats": weight_stats,
            }

        # Aggregate embeddings if any
        if len(embedding_neighbors) > 0:
            local_embeddings = self.representation_extractor.extract_embeddings(
                model=self.local_model,
                data_loader=self.data_loader,
                num_samples=100,
            )

            aggregated_emb, emb_stats = self.aggregator.aggregate_embeddings(
                local_embeddings=local_embeddings,
                neighbor_embeddings=embedding_neighbors,
                timestamps=timestamps,
                current_time=current_time,
            )

            aggregation_info["embedding_aggregation"] = {
                "aggregated_data": aggregated_emb,
                "stats": emb_stats,
                "divergence": self.aggregator.compute_divergence(local_embeddings, aggregated_emb),
            }

        if original_device.type == 'cpu':
            self.local_model.cpu()

        # Clear buffer after aggregation
        self.received_buffer.clear()

        # Update last aggregation time
        self.last_aggregation_time = current_time

        return True, aggregation_info

    def has_pending_updates(self) -> bool:
        """Check if there are pending updates to aggregate."""
        return len(self.received_buffer) > 0

    def get_staleness_metrics(self) -> Dict[int, float]:
        """Get staleness (time since last update) for each neighbor in buffer.

        Returns:
            Dict mapping neighbor_id to staleness in seconds
        """
        current_time = time.time()
        staleness = {}

        for neighbor_id, (_, _, timestamp) in self.received_buffer.items():
            staleness[neighbor_id] = current_time - timestamp

        return staleness

    def get_overall_staleness(self) -> float:
        """Get overall staleness (time since last aggregation).

        Returns:
            Staleness in seconds
        """
        return time.time() - self.last_aggregation_time

    def get_update_statistics(self) -> Dict[str, float]:
        """Get communication and update statistics.

        Returns:
            Dict with statistics:
                - updates_sent: Total updates sent
                - updates_received: Total updates received
                - data_sent_mb: Total data sent in MB
                - data_received_mb: Total data received in MB
                - avg_staleness: Average staleness of pending updates
                - num_pending: Number of pending updates
        """
        # Compute average staleness of pending updates
        staleness_dict = self.get_staleness_metrics()
        avg_staleness = np.mean(list(staleness_dict.values())) if staleness_dict else 0.0

        stats = {
            "updates_sent": self.updates_sent,
            "updates_received": self.updates_received,
            "data_sent_mb": self.total_data_sent_mb,
            "data_received_mb": self.total_data_received_mb,
            "avg_staleness": avg_staleness,
            "num_pending": len(self.received_buffer),
            "overall_staleness": self.get_overall_staleness(),
        }

        return stats

    def get_communication_history(
        self,
        last_n: Optional[int] = None,
    ) -> List[Tuple]:
        """Get communication history.

        Args:
            last_n: Return only last N entries (None = all)

        Returns:
            List of communication log entries
        """
        if last_n is None:
            return self.communication_log
        else:
            return self.communication_log[-last_n:]

    def reset_statistics(self):
        """Reset communication statistics (useful for new episode)."""
        self.updates_sent = 0
        self.updates_received = 0
        self.total_data_sent_mb = 0.0
        self.total_data_received_mb = 0.0
        self.communication_log.clear()

    def clear_buffer(self):
        """Clear received representation buffer."""
        self.received_buffer.clear()

    def get_state_features(self) -> Dict[str, float]:
        """Get DFL state features for RL agent.

        Returns features related to communication staleness and pending updates
        that can be added to the RL state.

        Returns:
            Dict with features:
                - time_since_last_update: Seconds since last aggregation
                - avg_neighbor_staleness: Average staleness of pending updates
                - pending_updates_count: Number of pending updates in buffer
                - model_divergence_estimate: Estimate of model divergence
        """
        staleness_dict = self.get_staleness_metrics()

        features = {
            "time_since_last_update": self.get_overall_staleness(),
            "avg_neighbor_staleness": np.mean(list(staleness_dict.values()))
            if staleness_dict
            else 0.0,
            "pending_updates_count": len(self.received_buffer),
            # Model divergence: rough estimate based on pending updates
            # More pending updates + higher staleness = likely higher divergence
            "model_divergence_estimate": len(self.received_buffer)
            * (1.0 + np.mean(list(staleness_dict.values())) / 10.0)
            if staleness_dict
            else 0.0,
        }

        return features
