"""
Neighbor Encoder (Layer 2) for V2V network.

Implements permutation-invariant encoding of neighbor information using DeepSets architecture.
For each vehicle, encodes information from all neighbors into a fixed-size representation.

Architecture:
    1. Extract features for each neighbor (distance, velocity, embeddings, etc.)
    2. Encode each neighbor individually: φ(neighbor_features)
    3. Aggregate across neighbors: ρ(mean, max, std)
    4. Output fixed-size neighborhood encoding
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class NeighborEncoder(nn.Module):
    """Permutation-invariant neighbor encoder using DeepSets architecture."""

    def __init__(
        self,
        neighbor_feature_dim: int = 40,
        phi_hidden_dim: int = 64,
        phi_output_dim: int = 32,
        rho_hidden_dim: int = 64,
        rho_output_dim: int = 32,
        dropout: float = 0.0,
    ):
        """Initialize neighbor encoder.

        Args:
            neighbor_feature_dim: Dimension of features per neighbor
            phi_hidden_dim: Hidden dimension for φ network (individual encoding)
            phi_output_dim: Output dimension for φ network
            rho_hidden_dim: Hidden dimension for ρ network (set encoding)
            rho_output_dim: Output dimension for ρ network (final neighborhood encoding)
            dropout: Dropout rate
        """
        super().__init__()

        self.neighbor_feature_dim = neighbor_feature_dim
        self.phi_output_dim = phi_output_dim
        self.rho_output_dim = rho_output_dim

        # φ network: encode individual neighbors
        # Input: neighbor_features (40-dim)
        # Output: encoded_neighbor (32-dim)
        self.phi = nn.Sequential(
            nn.Linear(neighbor_feature_dim, phi_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(phi_hidden_dim, phi_output_dim),
        )

        # ρ network: encode aggregated neighborhood
        # Input: concatenated [mean, max, std] pooling (3 * phi_output_dim)
        # Output: neighborhood_encoding (32-dim)
        aggregated_dim = 3 * phi_output_dim
        self.rho = nn.Sequential(
            nn.Linear(aggregated_dim, rho_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rho_hidden_dim, rho_output_dim),
        )

    def forward(
        self, neighbor_features: torch.Tensor, neighbor_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode neighborhood information.

        Args:
            neighbor_features: Tensor of shape (batch, max_neighbors, feature_dim)
            neighbor_mask: Optional mask of shape (batch, max_neighbors)
                          True for valid neighbors, False for padding

        Returns:
            Neighborhood encoding of shape (batch, rho_output_dim)
        """
        batch_size, max_neighbors, feature_dim = neighbor_features.shape
        assert feature_dim == self.neighbor_feature_dim

        # Encode each neighbor individually using φ
        # Shape: (batch, max_neighbors, phi_output_dim)
        encoded_neighbors = self.phi(neighbor_features)

        # Apply mask if provided (zero out padded neighbors)
        if neighbor_mask is not None:
            mask_expanded = neighbor_mask.unsqueeze(-1).float()
            encoded_neighbors = encoded_neighbors * mask_expanded

        # Aggregate across neighbors using mean, max, std pooling
        if neighbor_mask is not None:
            # Masked aggregation
            num_neighbors = neighbor_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            mean_pool = encoded_neighbors.sum(dim=1) / num_neighbors

            # For max pooling, replace masked values with large negative number
            masked_encoded = encoded_neighbors.clone()
            masked_encoded[~neighbor_mask.unsqueeze(-1).expand_as(encoded_neighbors)] = -1e9
            max_pool, _ = masked_encoded.max(dim=1)

            # For std pooling, compute over valid neighbors only
            variance = ((encoded_neighbors - mean_pool.unsqueeze(1)) ** 2).sum(
                dim=1
            ) / num_neighbors
            std_pool = torch.sqrt(variance.clamp(min=1e-8))
        else:
            # Standard aggregation (no masking)
            mean_pool = encoded_neighbors.mean(dim=1)
            max_pool, _ = encoded_neighbors.max(dim=1)
            std_pool = encoded_neighbors.std(dim=1)

        # Concatenate aggregated features
        # Shape: (batch, 3 * phi_output_dim)
        aggregated = torch.cat([mean_pool, max_pool, std_pool], dim=1)

        # Encode aggregated neighborhood using ρ
        # Shape: (batch, rho_output_dim)
        neighborhood_encoding = self.rho(aggregated)

        return neighborhood_encoding

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def project_embedding(embedding: np.ndarray, target_dim: int = 32) -> np.ndarray:
    """Project high-dimensional embedding to target dimension via truncation.

    For learned projection, use EmbeddingProjection nn.Module instead.
    This is a simple fallback for inference without a trained projection layer.

    Args:
        embedding: Raw embedding array (embedding_dim,)
        target_dim: Target dimension for neighbor features

    Returns:
        Projected embedding of shape (target_dim,)
    """
    if len(embedding) == target_dim:
        return embedding
    if len(embedding) > target_dim:
        return embedding[:target_dim]
    # Pad if smaller (shouldn't happen in practice)
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: len(embedding)] = embedding
    return padded


class EmbeddingProjection(nn.Module):
    """Learned linear projection from autoencoder embedding to neighbor feature space."""

    def __init__(self, input_dim: int = 256, output_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# Default projection dimension for neighbor features
NEIGHBOR_EMBEDDING_DIM = 32


def extract_neighbor_features(
    ego_vehicle: "VehicleNode",
    neighbor_vehicles: List["VehicleNode"],
    comm_range: float,
    embedding_projection: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract feature vector for each neighbor.

    When autoencoder embedding_dim > 32 (e.g., 256), embeddings are projected
    down to 32-dim to maintain df=40 neighbor feature dimension per the paper.

    Args:
        ego_vehicle: The ego vehicle (self)
        neighbor_vehicles: List of neighbor vehicles
        comm_range: Communication range in meters
        embedding_projection: Optional projection matrix (32, embedding_dim)
                            for learned projection. If None, uses truncation.

    Returns:
        Array of shape (num_neighbors, 40) with neighbor features
    """
    if len(neighbor_vehicles) == 0:
        return np.zeros((0, 40), dtype=np.float32)

    neighbor_features_list = []

    ego_pos = np.array(ego_vehicle.position, dtype=np.float32)
    ego_vel = np.array(ego_vehicle.velocity, dtype=np.float32)

    for neighbor in neighbor_vehicles:
        neighbor_pos = np.array(neighbor.position, dtype=np.float32)
        relative_pos = neighbor_pos - ego_pos
        distance = float(np.linalg.norm(relative_pos))
        relative_features = np.array(
            [relative_pos[0], relative_pos[1], distance], dtype=np.float32
        )

        neighbor_vel = np.array(neighbor.velocity, dtype=np.float32)
        relative_vel = neighbor_vel - ego_vel
        relative_speed = float(np.linalg.norm(relative_vel))
        velocity_features = np.array(
            [relative_vel[0], relative_vel[1], relative_speed], dtype=np.float32
        )

        link_quality = np.float32(max(0.0, 1.0 - distance / comm_range))

        anomaly_score = np.float32(
            neighbor.anomaly_score if neighbor.anomaly_score is not None else 0.0
        )

        # Project embedding to 32-dim for neighbor features (paper: df=40)
        if neighbor.embedding is not None:
            raw_emb = neighbor.embedding.astype(np.float32)
            if embedding_projection is not None:
                embedding = embedding_projection @ raw_emb
            else:
                embedding = project_embedding(raw_emb, NEIGHBOR_EMBEDDING_DIM)
        else:
            embedding = np.zeros(NEIGHBOR_EMBEDDING_DIM, dtype=np.float32)

        # 3 + 3 + 1 + 1 + 32 = 40 features
        neighbor_feature = np.concatenate(
            [
                relative_features,
                velocity_features,
                np.array([link_quality], dtype=np.float32),
                np.array([anomaly_score], dtype=np.float32),
                embedding,
            ]
        ).astype(np.float32)

        neighbor_features_list.append(neighbor_feature)

    neighbor_features = np.stack(neighbor_features_list, axis=0)

    return neighbor_features


def create_neighbor_batch(
    neighbor_features_list: List[np.ndarray], max_neighbors: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create batched neighbor features with padding.

    Args:
        neighbor_features_list: List of neighbor feature arrays
                               Each array has shape (num_neighbors_i, feature_dim)
        max_neighbors: Maximum number of neighbors (for padding)
                      If None, uses the max in the batch

    Returns:
        Tuple of:
            - neighbor_features: Tensor of shape (batch, max_neighbors, feature_dim)
            - neighbor_mask: Boolean tensor of shape (batch, max_neighbors)
    """
    batch_size = len(neighbor_features_list)
    feature_dim = 40

    # Determine max_neighbors
    if max_neighbors is None:
        max_neighbors = max(len(nf) for nf in neighbor_features_list)
        max_neighbors = max(max_neighbors, 1)  # At least 1

    # Initialize tensors
    neighbor_features = torch.zeros(
        batch_size, max_neighbors, feature_dim, dtype=torch.float32
    )
    neighbor_mask = torch.zeros(batch_size, max_neighbors, dtype=torch.bool)

    # Fill in features
    for i, nf in enumerate(neighbor_features_list):
        num_neighbors = len(nf)
        if num_neighbors > 0:
            num_to_use = min(num_neighbors, max_neighbors)
            neighbor_features[i, :num_to_use] = torch.from_numpy(nf[:num_to_use])
            neighbor_mask[i, :num_to_use] = True

    return neighbor_features, neighbor_mask


def create_neighbor_encoder_from_config(
    config, device: Optional[torch.device] = None
) -> NeighborEncoder:
    """Create neighbor encoder from Hydra config.

    Args:
        config: Hydra configuration object
        device: Device to place model on

    Returns:
        Initialized NeighborEncoder model
    """
    if device is None:
        device = torch.device("cpu")

    # Get architecture parameters from config
    arch_config = config.model.neighbor_encoder.architecture

    model = NeighborEncoder(
        neighbor_feature_dim=arch_config.neighbor_feature_dim,
        phi_hidden_dim=arch_config.phi_hidden_dim,
        phi_output_dim=arch_config.phi_output_dim,
        rho_hidden_dim=arch_config.rho_hidden_dim,
        rho_output_dim=arch_config.rho_output_dim,
        dropout=arch_config.get("dropout", 0.0),
    )

    model = model.to(device)
    return model
