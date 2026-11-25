"""
CoCo Algorithm: Joint Topology Construction and Model Compression
Exact implementation from paper "Accelerating Decentralized Federated Learning in Heterogeneous Edge Computing"

Algorithm 3: CoCo (Page 8, Lines 1-26)
Equations: 17 (link time), 23 (consensus distance connected),
           25 (consensus distance unconnected), 26 (EMA),
           27 (LP optimization), 28 (adaptive threshold)
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import linprog
from scipy.sparse.csgraph import floyd_warshall


class CoCo:
    """CoCo: Joint Topology Construction and Model Compression Optimization.

    Implements Algorithm 3 from paper (Page 8, Lines 1-26).
    """

    def __init__(
        self,
        num_vehicles: int = 20,
        beta_1: float = 0.5,  # Eq. 26 - EMA for consensus distance
        beta_2: float = 0.1,  # Eq. 28 - EMA for threshold
        model_size_bytes: float = 1e6,  # B in Eq. 17
    ):
        """Initialize CoCo optimizer.

        Args:
            num_vehicles: Number of workers (m)
            beta_1: EMA coefficient for consensus distance (Eq. 26)
            beta_2: EMA coefficient for threshold (Eq. 28)
            model_size_bytes: Model size B for transmission (Eq. 17)
        """
        self.m = num_vehicles
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.B = model_size_bytes

        # EMA tracking (Eq. 26, 28)
        self.D_bar = None  # D̄_{i,j}^(k) - smoothed consensus distance
        self.D_max = None  # D_max^(k) - adaptive threshold

    def optimize(
        self,
        bandwidth_in: np.ndarray,   # b_i^(k,in) - shape (m,)
        bandwidth_out: np.ndarray,  # b_i^(k,out) - shape (m,)
        comp_time: np.ndarray,      # c_i^(k) - shape (m,)
        consensus_distances: np.ndarray,  # D_{i,j}^(k) - shape (m, m)
        D_max_k: float,             # D_max^(k) - threshold
        A_base: np.ndarray,         # A_b - base topology
        gradient_norms: np.ndarray = None,  # BUG FIX #1: ||g_i||_2 - shape (m,)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Algorithm 3: CoCo optimization (Lines 1-26).

        Input (from paper):
            b_i^(k,out), b_i^(k,in): Bandwidth constraints
            c_i^(k): Computation time
            D_{i,j}^(k), ∀i,j ∈ M: Consensus distances
            D_max^(k): Threshold
            A_b: Base adjacent matrix

        Output:
            A^(k): Optimized topology
            r^(k): Compression ratios

        Args:
            bandwidth_in: Incoming bandwidth per vehicle (m,)
            bandwidth_out: Outgoing bandwidth per vehicle (m,)
            comp_time: Computation time per vehicle (m,)
            consensus_distances: Consensus distance matrix (m, m)
            D_max_k: Threshold for consensus distance
            A_base: Base topology adjacency matrix (m, m)

        Returns:
            Tuple of (A_k, r_k) where:
                A_k: Optimized adjacency matrix (m, m)
                r_k: Compression ratios (m,)
        """
        # Update EMA consensus distances (Eq. 26)
        D_ij_k = consensus_distances
        if self.D_bar is None:
            # Initialize
            self.D_bar = D_ij_k.copy()
        else:
            # Eq. 26: D̄_{i,j}^(k) = β_1 * D̄_{i,j}^(k-1) + (1-β_1) * D_{i,j}^(k)
            self.D_bar = self.beta_1 * self.D_bar + (1 - self.beta_1) * D_ij_k

        # Update adaptive threshold (Eq. 28)
        if self.D_max is None:
            self.D_max = D_max_k
        else:
            # BUG FIX #1: Eq. 28 uses gradient norms, not consensus distance
            # Eq. 28: D_max^(k) = (1-β₂)D_max^(k-1) + β₂ · (1/m) Σ||g_i||₂
            if gradient_norms is not None:
                gradient_norm_avg = np.mean(gradient_norms)  # (1/m) Σ||g_i||₂
                self.D_max = (1 - self.beta_2) * self.D_max + self.beta_2 * gradient_norm_avg
            else:
                # Fallback to old behavior if gradient_norms not provided
                D_bar_prev = np.mean(self.D_bar)
                self.D_max = self.beta_2 * self.D_max + (1 - self.beta_2) * D_bar_prev

        # Line 1: Initialize adjacent matrix A^(k) = A_b, search step s = m and Flag = True
        A_k = A_base.copy()
        s = self.m
        Flag = True

        # Line 2: Solve Eq. (27) with A^(k), and obtain the initialization of r^(k) and t^(k)
        r_k, t_k = self._solve_eq27(A_k, self.D_bar, self.D_max,
                                     bandwidth_in, bandwidth_out, comp_time)

        # Line 3: while True do
        while True:
            # Line 4: Obtain the transmission time t_{i,j}^(k) of each link (i,j) based on Eq. (17)
            t_ij = self._compute_link_times_eq17(A_k, r_k, comp_time, bandwidth_out)

            # Line 5-9: Adjust search step s
            if Flag:
                # Line 6: s = sqrt(sum(A^(k)))
                s = int(np.sqrt(np.sum(A_k)))
            else:
                # Line 8: s = s/2
                s = s // 2

            # Line 10: E = select s links with the longest transmission times
            E = self._select_slowest_links(t_ij, s)

            # Line 11: A^(k), r^(k), t^(k), Flag = AdjustCR(A^(k), E, t^(k), r^(k), Flag, ...)
            A_k, r_k, t_k, Flag = self._adjust_CR(
                A_k, E, t_k, r_k, Flag,
                self.D_bar, self.D_max,
                bandwidth_in, bandwidth_out, comp_time
            )

            # Line 12-14: Break condition
            if not Flag and s == 1:
                break

        # Output: A^(k), r^(k)
        return A_k, r_k

    def _adjust_CR(
        self,
        A_k: np.ndarray,
        E: List[Tuple[int, int]],
        t_k: float,
        r_k: np.ndarray,
        Flag: bool,
        D_bar: np.ndarray,
        D_max: float,
        bandwidth_in: np.ndarray,
        bandwidth_out: np.ndarray,
        comp_time: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """function AdjustCR (Lines 13-26 in Algorithm 3).

        Args:
            A_k: Current adjacency matrix
            E: Links to remove (list of (i,j) tuples)
            t_k: Current round time
            r_k: Current compression ratios
            Flag: Search flag
            D_bar: Smoothed consensus distances
            D_max: Threshold
            bandwidth_in: Incoming bandwidth
            bandwidth_out: Outgoing bandwidth
            comp_time: Computation time

        Returns:
            Tuple of (A_k, r_k, t_k, Flag)
        """
        # Line 13: Set Flag = False
        Flag = False

        # Line 14: A' = A^(k)
        A_prime = A_k.copy()

        # Line 15-20: for e_{i,j} ∈ E do
        for e_ij in E:
            i, j = e_ij

            # Line 16: A'_{i,j} = 0
            A_prime[i, j] = 0

            # Line 17: if λ_2(L(A')) == 0 then
            if not self._is_connected(A_prime):
                # Line 18: A'_{i,j} = 1
                A_prime[i, j] = 1

        # Line 21: Solve Eq. (27) with A', obtain r' and t'
        r_prime, t_prime = self._solve_eq27(
            A_prime, D_bar, D_max,
            bandwidth_in, bandwidth_out, comp_time
        )

        # Line 22-26: Compare and return
        if t_prime < t_k:
            # Line 23: Return A', r', t', True
            return A_prime, r_prime, t_prime, True
        else:
            # Line 25: Return A^(k), r^(k), t^(k), False
            return A_k, r_k, t_k, False

    def _solve_eq27(
        self,
        A_k: np.ndarray,
        D_bar: np.ndarray,
        D_max: float,
        bandwidth_in: np.ndarray,
        bandwidth_out: np.ndarray,
        comp_time: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Solve Linear Programming problem (Eq. 27).

        Eq. 27:
            min t^(k)
            s.t. (1/m²) Σ_i Σ_j (1 - a_{i,j}^(k)*r_j^(k)) * D̄_{i,j}^(k) ≤ D_max^(k)
                 0 < r_i^(k) ≤ 1, ∀i ∈ M
                 t^(k) ≥ t_i^(k,cp) + r_i^(k)*B/b_{i,j}^(k), ∀i,j with a_{i,j}=1

        Args:
            A_k: Adjacency matrix (m, m)
            D_bar: Smoothed consensus distances (m, m)
            D_max: Threshold
            bandwidth_in: Incoming bandwidth (m,)
            bandwidth_out: Outgoing bandwidth (m,)
            comp_time: Computation time (m,)

        Returns:
            Tuple of (r_k, t_k) where:
                r_k: Compression ratios (m,)
                t_k: Minimum round time
        """
        m = self.m

        # Variables: [r_0, r_1, ..., r_{m-1}, t]
        # Total: m + 1 variables

        # Objective: min t  =>  c = [0, 0, ..., 0, 1]
        c = np.zeros(m + 1)
        c[-1] = 1.0  # Minimize t

        # Constraints
        A_ub = []
        b_ub = []

        # Constraint 1 (Eq. 27 first constraint):
        # (1/m²) Σ_i Σ_j (1 - a_{i,j}^(k)*r_j^(k)) * D̄_{i,j}^(k) ≤ D_max^(k)
        # Rewrite: Σ_i Σ_j D̄_{i,j}^(k) - Σ_i Σ_j a_{i,j}^(k)*r_j^(k)*D̄_{i,j}^(k) ≤ m²*D_max^(k)
        # Linear form: -Σ_j [Σ_i a_{i,j}^(k)*D̄_{i,j}^(k)] * r_j ≤ m²*D_max - Σ_i Σ_j D̄_{i,j}^(k)

        constraint_coeff = np.zeros(m + 1)
        for j in range(m):
            constraint_coeff[j] = -np.sum(A_k[:, j] * D_bar[:, j])

        rhs = m**2 * D_max - np.sum(D_bar)
        A_ub.append(constraint_coeff)
        b_ub.append(rhs)

        # Constraint 2 (Eq. 17 - link time constraints):
        # t^(k) ≥ t_i^(k,cp) + r_i^(k)*B/b_{i,j}^(k), ∀i,j with a_{i,j}=1
        # Rewrite: -r_i * B/b_{i,j} + t ≥ t_i^(k,cp)
        # Linear form: -r_i * B/b_{i,j} + t ≥ c_i  =>  r_i * B/b_{i,j} - t ≤ -c_i

        for i in range(m):
            # Find neighbors of i
            neighbors = np.where(A_k[i, :] == 1)[0]
            if len(neighbors) == 0:
                continue

            # Use minimum bandwidth to neighbors (bottleneck)
            b_ij_min = bandwidth_out[i]  # Simplified: use outgoing bandwidth

            constraint_coeff = np.zeros(m + 1)
            constraint_coeff[i] = self.B / (b_ij_min + 1e-9)  # r_i coefficient
            constraint_coeff[-1] = -1.0  # -t coefficient

            rhs_i = -comp_time[i]
            A_ub.append(constraint_coeff)
            b_ub.append(rhs_i)

        # Bounds: 0 < r_i ≤ 1, t ≥ 0
        bounds = [(0.01, 1.0) for _ in range(m)] + [(0, None)]

        # Solve LP
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs'
        )

        if result.success:
            r_k = result.x[:m]
            t_k = result.x[-1]
        else:
            # Fallback: uniform compression ratios
            r_k = np.ones(m) * 0.5
            t_k = np.max(comp_time + 0.5 * self.B / (bandwidth_out + 1e-9))

        return r_k, t_k

    def _compute_link_times_eq17(
        self,
        A_k: np.ndarray,
        r_k: np.ndarray,
        comp_time: np.ndarray,
        bandwidth_out: np.ndarray,
    ) -> np.ndarray:
        """Compute link transmission times (Eq. 17).

        Eq. 17: t_{i,j}^(k) = t_i^(k,cp) + r_i^(k)*B/b_{i,j}^(k)

        Args:
            A_k: Adjacency matrix (m, m)
            r_k: Compression ratios (m,)
            comp_time: Computation time (m,)
            bandwidth_out: Outgoing bandwidth (m,)

        Returns:
            Link times t_{i,j}^(k) (m, m)
        """
        m = self.m
        t_ij = np.zeros((m, m))

        for i in range(m):
            for j in range(m):
                if A_k[i, j] == 1:
                    # Eq. 17: t_{i,j}^(k) = t_i^(k,cp) + r_i^(k)*B/b_{i,j}^(k)
                    t_cp = comp_time[i]
                    t_comm = r_k[i] * self.B / (bandwidth_out[i] + 1e-9)
                    t_ij[i, j] = t_cp + t_comm

        return t_ij

    def _select_slowest_links(
        self,
        t_ij: np.ndarray,
        s: int,
    ) -> List[Tuple[int, int]]:
        """Select s links with longest transmission times (Line 10).

        Args:
            t_ij: Link times (m, m)
            s: Number of links to select

        Returns:
            List of (i, j) tuples for slowest links
        """
        # Flatten and get indices of links with non-zero times
        flat_times = []
        link_indices = []

        for i in range(self.m):
            for j in range(self.m):
                if t_ij[i, j] > 0:
                    flat_times.append(t_ij[i, j])
                    link_indices.append((i, j))

        if len(flat_times) == 0:
            return []

        # Sort by time (descending) and select top s
        sorted_indices = np.argsort(flat_times)[::-1]
        s = min(s, len(sorted_indices))

        E = [link_indices[idx] for idx in sorted_indices[:s]]
        return E

    def _is_connected(self, A: np.ndarray) -> bool:
        """Check if topology is connected using spectral graph theory (Line 17).

        A graph is connected if λ_2(L(A)) > 0, where L is the Laplacian matrix.

        Args:
            A: Adjacency matrix (m, m)

        Returns:
            True if connected, False otherwise
        """
        # Make symmetric for undirected graph
        A_sym = np.maximum(A, A.T)

        # Compute Laplacian: L = D - A
        D = np.diag(np.sum(A_sym, axis=1))
        L = D - A_sym

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(L)
        eigenvalues = np.sort(eigenvalues)

        # λ_2 > 0 means connected (algebraic connectivity)
        lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0

        return lambda_2 > 1e-6  # Numerical tolerance

    def compute_consensus_distance_eq23(
        self,
        model_i: Dict,
        model_j: Dict,
    ) -> float:
        """Compute consensus distance between two models (Eq. 23).

        Eq. 23: D_{i,j}^(k) = ||w_i^(k) - w_j^(k)||^2 / ||w_i^(k)||^2

        Args:
            model_i: Model parameters for vehicle i (state_dict)
            model_j: Model parameters for vehicle j (state_dict)

        Returns:
            Consensus distance D_{i,j}
        """
        # Flatten model parameters to vectors
        w_i = self._flatten_model(model_i)
        w_j = self._flatten_model(model_j)

        # Eq. 23: D_{i,j} = ||w_i - w_j||^2 / ||w_i||^2
        numerator = np.sum((w_i - w_j) ** 2)
        denominator = np.sum(w_i ** 2) + 1e-9  # Avoid division by zero

        D_ij = numerator / denominator
        return D_ij

    def compute_consensus_distance_unconnected_eq25(
        self,
        A_k: np.ndarray,
        consensus_distances: np.ndarray,
        i: int,
        j: int,
    ) -> float:
        """Compute consensus distance for unconnected workers (Eq. 25).

        Eq. 25: D_{i,j}^(k) = average of all D_{p,q}^(k) on shortest path from i to j

        Uses Floyd-Warshall algorithm to find shortest paths.

        Args:
            A_k: Adjacency matrix (m, m)
            consensus_distances: Known consensus distances (m, m)
            i: Source vehicle
            j: Target vehicle

        Returns:
            Estimated consensus distance D_{i,j}
        """
        # Use Floyd-Warshall to find shortest path
        # Convert adjacency to distance matrix (1 for edges, inf for no edge)
        dist_matrix = np.where(A_k > 0, 1, np.inf)
        np.fill_diagonal(dist_matrix, 0)

        # Run Floyd-Warshall
        shortest_paths, predecessors = floyd_warshall(
            dist_matrix,
            return_predecessors=True
        )

        # If no path exists, return max consensus distance
        if np.isinf(shortest_paths[i, j]):
            return np.max(consensus_distances)

        # Reconstruct path and average consensus distances along it
        path = self._reconstruct_path(predecessors, i, j)

        if len(path) < 2:
            return consensus_distances[i, j]

        # Average consensus distances along path
        path_distances = []
        for p in range(len(path) - 1):
            p_idx = path[p]
            q_idx = path[p + 1]
            path_distances.append(consensus_distances[p_idx, q_idx])

        D_ij = np.mean(path_distances) if len(path_distances) > 0 else 0.0
        return D_ij

    def _reconstruct_path(
        self,
        predecessors: np.ndarray,
        i: int,
        j: int,
    ) -> List[int]:
        """Reconstruct shortest path from predecessors matrix."""
        if predecessors[i, j] == -9999:  # No path
            return []

        path = [j]
        while j != i:
            j = predecessors[i, j]
            if j == -9999:
                return []
            path.append(j)

        path.reverse()
        return path

    def _flatten_model(self, model_dict: Dict) -> np.ndarray:
        """Flatten model parameters to 1D vector."""
        params = []
        for key in sorted(model_dict.keys()):
            param = model_dict[key]
            if hasattr(param, 'cpu'):
                param = param.cpu().detach().numpy()
            params.append(param.flatten())

        return np.concatenate(params)
