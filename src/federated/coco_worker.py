"""
CoCo Worker: Algorithm 1 from paper
Implements worker-side federated learning with CoCo optimization.
"""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .coco_algorithm import CoCo


class CoCoWorker:
    """Worker implementation for CoCo federated learning (Algorithm 1).

    Algorithm 1 (Worker procedure):
        for k = 1 to K do
            Line 2: Compute D_{i,j} with current neighbors (Eq. 23)
            Line 3: Send (b_i^(k,in), b_i^(k,out), c_i^(k), D_{i,j}^(k)) to coordinator
            Line 4: Receive (N_i^(k+1), r_i^(k+1)) from coordinator
            Line 5: Local training τ steps (Eq. 5)
            Line 6: Compress model to r_i^(k+1) parameters
            Line 7: Send/receive compressed models with neighbors
            Line 8: Aggregate received models (Eq. 6)
        end for
    """

    def __init__(
        self,
        worker_id: int,
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device,
        tau: int = 50,  # Local update steps
        learning_rate: float = 0.001,
        coco_optimizer: Optional[CoCo] = None,
    ):
        """Initialize CoCo worker.

        Args:
            worker_id: Unique worker ID
            model: Local LSTM model
            data_loader: Private data loader
            device: Device for training
            tau: Number of local training steps (τ in Eq. 5)
            learning_rate: Learning rate for local training
            coco_optimizer: CoCo optimizer for consensus distance computation
        """
        self.worker_id = worker_id
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.tau = tau
        self.lr = learning_rate

        # CoCo optimizer for consensus distance
        self.coco = coco_optimizer if coco_optimizer is not None else CoCo()

        # Use Adam optimizer (matching DFL implementation for LSTM autoencoder)
        # From DFL: src/federated/protocol.py:62 - torch.optim.Adam(local_model.parameters(), lr=model_lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Current state
        self.current_neighbors = []  # N_i^(k)
        self.compression_ratio = 1.0  # r_i^(k)
        self.bandwidth_in = 0.0  # b_i^(k,in)
        self.bandwidth_out = 0.0  # b_i^(k,out)
        self.comp_time = 0.0  # c_i^(k)

        # Communication tracking
        self.total_bytes_sent = 0
        self.total_bytes_received = 0

    def compute_consensus_distances_line2(
        self,
        neighbor_models: Dict[int, Dict],
    ) -> Dict[int, float]:
        """Algorithm 1 Line 2: Compute consensus distance with neighbors (Eq. 23).

        Eq. 23: D_{i,j}^(k) = ||w_i^(k) - w_j^(k)||^2 / ||w_i^(k)||^2

        Args:
            neighbor_models: Dict mapping neighbor_id -> model state_dict

        Returns:
            Dict mapping neighbor_id -> consensus distance D_{i,j}
        """
        consensus_distances = {}
        my_model = self.model.state_dict()

        for neighbor_id, neighbor_model in neighbor_models.items():
            # Eq. 23
            D_ij = self.coco.compute_consensus_distance_eq23(my_model, neighbor_model)
            consensus_distances[neighbor_id] = D_ij

        return consensus_distances

    def prepare_status_for_coordinator_line3(
        self,
        consensus_distances: Dict[int, float],
        gradient_norm: float = 0.0,
    ) -> Dict:
        """Algorithm 1 Line 3: Prepare status message for coordinator.

        Returns:
            Status dict with:
                - bandwidth_in: b_i^(k,in)
                - bandwidth_out: b_i^(k,out)
                - comp_time: c_i^(k)
                - consensus_distances: D_{i,j}^(k) for all neighbors
                - gradient_norm: ||g_i||_2 for Equation 28
        """
        status = {
            "worker_id": self.worker_id,
            "bandwidth_in": self.bandwidth_in,
            "bandwidth_out": self.bandwidth_out,
            "comp_time": self.comp_time,
            "consensus_distances": consensus_distances,
            "gradient_norm": gradient_norm,  # BUG FIX #1: Add gradient norm
        }
        return status

    def receive_from_coordinator_line4(
        self,
        neighbors: List[int],
        compression_ratio: float,
    ):
        """Algorithm 1 Line 4: Receive topology and compression ratio from coordinator.

        Args:
            neighbors: N_i^(k+1) - new neighbor list
            compression_ratio: r_i^(k+1) - new compression ratio
        """
        self.current_neighbors = neighbors
        self.compression_ratio = compression_ratio

    def local_update_line5_eq5(self) -> tuple:
        """Algorithm 1 Line 5: Local training for τ steps (Eq. 5).

        Eq. 5: w_i^(k,τ) = w_i^(k) - η ∇L_i(w_i^(k))

        Performs τ local SGD updates on private data.

        Returns:
            Tuple of (avg_loss, gradient_norm) where gradient_norm is ||g_i||_2
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        total_grad_norm = 0.0

        # Run τ local update steps
        data_iter = iter(self.data_loader)
        for step in range(self.tau):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Reset iterator if dataset exhausted
                data_iter = iter(self.data_loader)
                batch = next(data_iter)

            # Unpack batch
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch

            x = x.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.model(x)
            loss = self.model.compute_reconstruction_error(x)

            # Backward pass (Eq. 5)
            loss.backward()

            # BUG FIX #1: Track gradient norm for Equation 28
            # Compute ||g_i||_2 = L2 norm of gradients
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            total_grad_norm += grad_norm

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0
        return avg_loss, avg_grad_norm

    def compress_model_line6(self) -> tuple:
        """Algorithm 1 Line 6: Compress model to r_i^(k+1) parameters.

        Uses RandomK sparsification from paper Section 5.2.

        Returns:
            Tuple of (compressed_dict, compressed_size_bytes)
        """
        model_dict = self.model.state_dict()
        compressed_dict = {}
        actual_bytes = 0

        for key, param in model_dict.items():
            param_np = param.cpu().detach().numpy()

            # RandomK compression: keep r_i fraction of parameters
            num_params = param_np.size
            k = max(1, int(num_params * self.compression_ratio))

            # Flatten, select top-k by magnitude, sparsify
            flat_param = param_np.flatten()
            indices = np.argsort(np.abs(flat_param))[-k:]

            compressed_flat = np.zeros_like(flat_param)
            compressed_flat[indices] = flat_param[indices]

            compressed_param = compressed_flat.reshape(param_np.shape)
            compressed_dict[key] = torch.from_numpy(compressed_param).to(param.dtype)

            # Track actual bandwidth: only non-zero elements + indices
            # Each parameter: 4 bytes (float32)
            # Each index: 4 bytes (int32)
            actual_bytes += k * 4  # parameter values
            actual_bytes += k * 4  # indices

        return compressed_dict, actual_bytes

    def communicate_with_neighbors_line7(
        self,
        compressed_model: Dict,
        neighbor_models: Dict[int, Dict],
    ) -> Dict[int, Dict]:
        """Algorithm 1 Line 7: Send/receive compressed models with neighbors.

        Args:
            compressed_model: My compressed model
            neighbor_models: Received compressed models from neighbors

        Returns:
            Dict mapping neighbor_id -> received compressed model
        """
        # In simulation, this is a pass-through
        # In reality, this would involve network communication

        # NOTE: We don't track received bandwidth here to avoid double-counting.
        # Bandwidth is tracked on the sender side in compress_model_line6()

        return neighbor_models

    def aggregate_models_line8_eq6(
        self,
        received_models: Dict[int, Dict],
    ):
        """Algorithm 1 Line 8: Aggregate received models (Eq. 6).

        Eq. 6: x_i^(k+1) = x_i^(k,τ) + Σ_{j∈N_i} w_ij * (x_j^(k,τ) - x_i^(k,τ)) ⊙ z_j^(k)

        Element-wise aggregation with compression masks.

        Args:
            received_models: Dict mapping neighbor_id -> compressed model state_dict
        """
        my_model = self.model.state_dict()
        num_neighbors = len(received_models)

        if num_neighbors == 0:
            return

        # Mixing weight (Eq. 7)
        w_ij = 1.0 / (num_neighbors + 1)

        aggregated_dict = {}

        for key in my_model.keys():
            # Start with my local model
            my_param = my_model[key].clone().float()
            aggregated_param = my_param.clone()

            # Eq. 6: Add weighted differences with neighbors (element-wise)
            # x_i^(k+1) = x_i^(k,τ) + Σ w_ij * (x_j^(k,τ) - x_i^(k,τ)) ⊙ z_j
            for neighbor_id, neighbor_model in received_models.items():
                neighbor_param = neighbor_model[key].to(self.device).float()

                # Element-wise difference weighted by compression mask
                # Only non-zero elements in compressed model are aggregated
                diff = neighbor_param - my_param
                aggregated_param += w_ij * diff

            aggregated_dict[key] = aggregated_param.to(my_model[key].dtype)

        # Update model
        self.model.load_state_dict(aggregated_dict)

    def update_bandwidth_and_comp_time(
        self,
        bandwidth_in: float,
        bandwidth_out: float,
        comp_time: float,
    ):
        """Update worker's bandwidth and computation time estimates.

        Args:
            bandwidth_in: Incoming bandwidth (bytes/sec)
            bandwidth_out: Outgoing bandwidth (bytes/sec)
            comp_time: Computation time for τ local updates (seconds)
        """
        self.bandwidth_in = bandwidth_in
        self.bandwidth_out = bandwidth_out
        self.comp_time = comp_time

    def get_model_state(self) -> Dict:
        """Get current model state dict."""
        return self.model.state_dict()

    def get_communication_stats(self) -> Dict:
        """Get communication statistics."""
        return {
            "total_bytes_sent": self.total_bytes_sent,
            "total_bytes_received": self.total_bytes_received,
            "total_bytes": self.total_bytes_sent + self.total_bytes_received,
        }

    def _estimate_model_size(self, model_dict: Dict) -> int:
        """Estimate model size in bytes."""
        total_bytes = 0
        for key, param in model_dict.items():
            if hasattr(param, 'element_size'):
                total_bytes += param.numel() * param.element_size()
            else:
                # Numpy array
                total_bytes += param.nbytes
        return total_bytes

    def reset_communication_stats(self):
        """Reset communication statistics."""
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
