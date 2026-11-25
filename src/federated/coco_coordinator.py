"""
CoCo Coordinator: Algorithm 2 from paper
Implements coordinator-side topology construction and compression optimization.
"""

from typing import Dict, List, Tuple
import numpy as np

from .coco_algorithm import CoCo


class CoCoCoordinator:
    """Coordinator implementation for CoCo federated learning (Algorithm 2).

    Algorithm 2 (Coordinator procedure):
        for k = 1 to K do
            Line 2: Receive status from all workers
            Line 3: Run CoCo (Algorithm 3) → get A^(k), r^(k)
            Line 4: Send (A^(k), r^(k)) to all workers
        end for
    """

    def __init__(
        self,
        num_vehicles: int = 20,
        base_topology: str = "complete",  # Base topology type
        beta_1: float = 0.5,  # EMA for consensus distance (Eq. 26)
        beta_2: float = 0.1,  # EMA for threshold (Eq. 28)
        initial_D_max: float = 0.1,  # Initial threshold
        model_size_bytes: float = 1e6,  # Model size for bandwidth calculation
    ):
        """Initialize CoCo coordinator.

        Args:
            num_vehicles: Number of workers (m)
            base_topology: Base topology type ("complete", "ring", "star")
            beta_1: EMA coefficient for consensus distance (Eq. 26)
            beta_2: EMA coefficient for threshold (Eq. 28)
            initial_D_max: Initial D_max threshold
            model_size_bytes: Model size B in bytes
        """
        self.m = num_vehicles
        self.base_topology_type = base_topology
        self.initial_D_max = initial_D_max

        # Create CoCo optimizer (Algorithm 3)
        self.coco = CoCo(
            num_vehicles=num_vehicles,
            beta_1=beta_1,
            beta_2=beta_2,
            model_size_bytes=model_size_bytes,
        )

        # Generate base topology A_b
        self.A_base = self._create_base_topology()

        # Current round
        self.round = 0

    def receive_worker_status_line2(
        self,
        worker_statuses: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Algorithm 2 Line 2: Receive status from all workers.

        Args:
            worker_statuses: List of status dicts from workers, each containing:
                - worker_id: int
                - bandwidth_in: float
                - bandwidth_out: float
                - comp_time: float
                - consensus_distances: Dict[int, float]
                - gradient_norm: float (for Equation 28)

        Returns:
            Tuple of (bandwidth_in, bandwidth_out, comp_time, consensus_distances, gradient_norms):
                - bandwidth_in: (m,) array
                - bandwidth_out: (m,) array
                - comp_time: (m,) array
                - consensus_distances: (m, m) matrix
                - gradient_norms: (m,) array
        """
        # Initialize arrays
        bandwidth_in = np.zeros(self.m)
        bandwidth_out = np.zeros(self.m)
        comp_time = np.zeros(self.m)
        consensus_distances = np.zeros((self.m, self.m))
        gradient_norms = np.zeros(self.m)

        # Collect status from each worker
        for status in worker_statuses:
            worker_id = status["worker_id"]
            bandwidth_in[worker_id] = status["bandwidth_in"]
            bandwidth_out[worker_id] = status["bandwidth_out"]
            comp_time[worker_id] = status["comp_time"]
            gradient_norms[worker_id] = status.get("gradient_norm", 0.0)

            # Fill consensus distance matrix
            for neighbor_id, distance in status["consensus_distances"].items():
                consensus_distances[worker_id, neighbor_id] = distance

        return bandwidth_in, bandwidth_out, comp_time, consensus_distances, gradient_norms

    def run_coco_line3(
        self,
        bandwidth_in: np.ndarray,
        bandwidth_out: np.ndarray,
        comp_time: np.ndarray,
        consensus_distances: np.ndarray,
        gradient_norms: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Algorithm 2 Line 3: Run CoCo optimization (Algorithm 3).

        Args:
            bandwidth_in: Incoming bandwidth per worker (m,)
            bandwidth_out: Outgoing bandwidth per worker (m,)
            comp_time: Computation time per worker (m,)
            consensus_distances: Consensus distance matrix (m, m)
            gradient_norms: Gradient L2 norms per worker (m,) - for Eq. 28

        Returns:
            Tuple of (A_k, r_k):
                - A_k: Optimized adjacency matrix (m, m)
                - r_k: Compression ratios (m,)
        """
        # Determine D_max for this round
        if self.coco.D_max is None:
            D_max_k = self.initial_D_max
        else:
            # Use adaptive threshold from CoCo (Eq. 28)
            D_max_k = self.coco.D_max

        # Run Algorithm 3
        A_k, r_k = self.coco.optimize(
            bandwidth_in=bandwidth_in,
            bandwidth_out=bandwidth_out,
            comp_time=comp_time,
            consensus_distances=consensus_distances,
            D_max_k=D_max_k,
            A_base=self.A_base,
            gradient_norms=gradient_norms,
        )

        return A_k, r_k

    def send_topology_and_ratios_line4(
        self,
        A_k: np.ndarray,
        r_k: np.ndarray,
    ) -> Dict[int, Dict]:
        """Algorithm 2 Line 4: Send topology and compression ratios to workers.

        Args:
            A_k: Adjacency matrix (m, m)
            r_k: Compression ratios (m,)

        Returns:
            Dict mapping worker_id -> assignment dict:
                - neighbors: List of neighbor IDs
                - compression_ratio: float
        """
        assignments = {}

        for i in range(self.m):
            # Extract neighbors from adjacency matrix
            neighbors = np.where(A_k[i, :] == 1)[0].tolist()

            # Create assignment
            assignments[i] = {
                "neighbors": neighbors,
                "compression_ratio": float(r_k[i]),
            }

        return assignments

    def execute_round(
        self,
        worker_statuses: List[Dict],
    ) -> Dict[int, Dict]:
        """Execute one CoCo round (Algorithm 2 Lines 2-4).

        Args:
            worker_statuses: List of status dicts from all workers

        Returns:
            Dict mapping worker_id -> assignment (neighbors, compression_ratio)
        """
        # Line 2: Receive status (including gradient norms for Equation 28)
        bandwidth_in, bandwidth_out, comp_time, consensus_distances, gradient_norms = \
            self.receive_worker_status_line2(worker_statuses)

        # Line 3: Run CoCo (with gradient norms for Bug Fix #1)
        A_k, r_k = self.run_coco_line3(
            bandwidth_in, bandwidth_out, comp_time, consensus_distances, gradient_norms
        )

        # Line 4: Send assignments
        assignments = self.send_topology_and_ratios_line4(A_k, r_k)

        self.round += 1

        return assignments

    def _create_base_topology(self) -> np.ndarray:
        """Create base topology A_b.

        Returns:
            Adjacency matrix (m, m)
        """
        if self.base_topology_type == "complete":
            # Complete graph: all connected
            A_base = np.ones((self.m, self.m))
            np.fill_diagonal(A_base, 0)  # No self-loops

        elif self.base_topology_type == "ring":
            # Ring topology
            A_base = np.zeros((self.m, self.m))
            for i in range(self.m):
                A_base[i, (i + 1) % self.m] = 1
                A_base[i, (i - 1) % self.m] = 1

        elif self.base_topology_type == "star":
            # Star topology: vehicle 0 is center
            A_base = np.zeros((self.m, self.m))
            for i in range(1, self.m):
                A_base[0, i] = 1
                A_base[i, 0] = 1

        else:
            # Default to complete
            A_base = np.ones((self.m, self.m))
            np.fill_diagonal(A_base, 0)

        return A_base

    def get_current_topology(self) -> np.ndarray:
        """Get current base topology."""
        return self.A_base.copy()

    def get_round(self) -> int:
        """Get current round number."""
        return self.round
