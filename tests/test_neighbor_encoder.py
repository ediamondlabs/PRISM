"""
Tests for NeighborEncoder (Layer 2).
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.models import (
    NeighborEncoder,
    create_neighbor_batch,
    create_neighbor_encoder_from_config,
    extract_neighbor_features,
)
from src.simulation import VehicleNode


@pytest.fixture
def neighbor_encoder():
    """Create a simple NeighborEncoder for testing."""
    return NeighborEncoder(
        neighbor_feature_dim=40,
        phi_hidden_dim=64,
        phi_output_dim=32,
        rho_hidden_dim=64,
        rho_output_dim=32,
        dropout=0.0,
    )


@pytest.fixture
def test_config():
    """Create test configuration for neighbor encoder."""
    config_dict = {
        "model": {
            "neighbor_encoder": {
                "architecture": {
                    "neighbor_feature_dim": 40,
                    "phi_hidden_dim": 64,
                    "phi_output_dim": 32,
                    "rho_hidden_dim": 64,
                    "rho_output_dim": 32,
                    "dropout": 0.0,
                }
            }
        }
    }
    return OmegaConf.create(config_dict)


@pytest.fixture
def sample_neighbor_features():
    """Create sample neighbor features."""
    # 5 neighbors, 40 features each
    return torch.randn(1, 5, 40)


@pytest.fixture
def sample_vehicle_nodes():
    """Create sample vehicle nodes for feature extraction."""
    nodes = []
    for i in range(4):
        node = VehicleNode(vehicle_id=i, window_size=50, num_features=10, embedding_dim=32)
        # Set some state
        node.position = (i * 10.0, i * 5.0)
        node.velocity = (1.0 + i * 0.5, 0.5 + i * 0.2)
        node.speed = 2.0 + i * 0.5
        node.embedding = np.random.randn(32).astype(np.float32)
        node.anomaly_score = 0.1 * i
        nodes.append(node)
    return nodes


# ==================== Model Tests ====================


def test_neighbor_encoder_initialization(neighbor_encoder):
    """Test NeighborEncoder initialization."""
    assert neighbor_encoder.neighbor_feature_dim == 40
    assert neighbor_encoder.phi_output_dim == 32
    assert neighbor_encoder.rho_output_dim == 32
    assert isinstance(neighbor_encoder.phi, torch.nn.Sequential)
    assert isinstance(neighbor_encoder.rho, torch.nn.Sequential)


def test_neighbor_encoder_forward_pass(neighbor_encoder, sample_neighbor_features):
    """Test forward pass with simple input."""
    neighbor_encoder.eval()

    with torch.no_grad():
        output = neighbor_encoder(sample_neighbor_features)

    assert output.shape == (1, 32)  # batch=1, rho_output_dim=32
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_neighbor_encoder_forward_with_mask(neighbor_encoder):
    """Test forward pass with masking (variable number of neighbors)."""
    # Batch of 3 vehicles with 5, 3, 1 neighbors respectively
    batch_size = 3
    max_neighbors = 5
    neighbor_features = torch.randn(batch_size, max_neighbors, 40)

    # Create mask: [True, True, True, True, True],
    #              [True, True, True, False, False],
    #              [True, False, False, False, False]
    neighbor_mask = torch.zeros(batch_size, max_neighbors, dtype=torch.bool)
    neighbor_mask[0, :5] = True
    neighbor_mask[1, :3] = True
    neighbor_mask[2, :1] = True

    neighbor_encoder.eval()
    with torch.no_grad():
        output = neighbor_encoder(neighbor_features, neighbor_mask)

    assert output.shape == (batch_size, 32)
    assert not torch.isnan(output).any()


def test_neighbor_encoder_no_neighbors(neighbor_encoder):
    """Test forward pass with zero neighbors."""
    # Batch with no neighbors (all masked out)
    neighbor_features = torch.randn(2, 3, 40)
    neighbor_mask = torch.zeros(2, 3, dtype=torch.bool)

    neighbor_encoder.eval()
    with torch.no_grad():
        output = neighbor_encoder(neighbor_features, neighbor_mask)

    assert output.shape == (2, 32)
    # Output should not be NaN even with no neighbors
    assert not torch.isnan(output).any()


def test_neighbor_encoder_permutation_invariance(neighbor_encoder):
    """Test that encoder is permutation-invariant across neighbors."""
    neighbor_encoder.eval()

    # Create neighbor features for 4 neighbors
    neighbor_features = torch.randn(1, 4, 40)

    with torch.no_grad():
        # Original order
        output1 = neighbor_encoder(neighbor_features)

        # Permute neighbors
        perm_indices = torch.tensor([2, 0, 3, 1])
        neighbor_features_perm = neighbor_features[:, perm_indices, :]
        output2 = neighbor_encoder(neighbor_features_perm)

    # Outputs should be identical (or very close due to floating point)
    assert torch.allclose(output1, output2, atol=1e-6)


def test_neighbor_encoder_batch_processing(neighbor_encoder):
    """Test batch processing with multiple vehicles."""
    batch_size = 8
    max_neighbors = 10
    neighbor_features = torch.randn(batch_size, max_neighbors, 40)

    neighbor_encoder.eval()
    with torch.no_grad():
        output = neighbor_encoder(neighbor_features)

    assert output.shape == (batch_size, 32)
    assert not torch.isnan(output).any()


def test_neighbor_encoder_get_num_parameters(neighbor_encoder):
    """Test parameter counting."""
    num_params = neighbor_encoder.get_num_parameters()
    assert num_params > 0
    assert isinstance(num_params, int)

    # Manually count parameters
    manual_count = sum(p.numel() for p in neighbor_encoder.parameters() if p.requires_grad)
    assert num_params == manual_count


# ==================== Feature Extraction Tests ====================


def test_extract_neighbor_features(sample_vehicle_nodes):
    """Test neighbor feature extraction."""
    ego_vehicle = sample_vehicle_nodes[0]
    neighbor_vehicles = sample_vehicle_nodes[1:]

    features = extract_neighbor_features(ego_vehicle, neighbor_vehicles, comm_range=300.0)

    assert features.shape == (3, 40)  # 3 neighbors, 40 features each
    assert features.dtype == np.float32
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()


def test_extract_neighbor_features_no_neighbors(sample_vehicle_nodes):
    """Test feature extraction with no neighbors."""
    ego_vehicle = sample_vehicle_nodes[0]
    neighbor_vehicles = []

    features = extract_neighbor_features(ego_vehicle, neighbor_vehicles, comm_range=300.0)

    assert features.shape == (0, 40)


def test_extract_neighbor_features_content(sample_vehicle_nodes):
    """Test that extracted features contain expected information."""
    ego_vehicle = sample_vehicle_nodes[0]
    neighbor_vehicle = sample_vehicle_nodes[1]

    features = extract_neighbor_features(ego_vehicle, [neighbor_vehicle], comm_range=300.0)

    # Check feature structure
    # Features: [relative_pos (3), relative_vel (3), link_quality (1), anomaly_score (1), embedding (32)]
    relative_pos = features[0, :3]
    relative_vel = features[0, 3:6]
    link_quality = features[0, 6]
    anomaly_score = features[0, 7]
    embedding = features[0, 8:]

    # Relative position should be neighbor_pos - ego_pos
    expected_rel_pos = np.array(neighbor_vehicle.position) - np.array(ego_vehicle.position)
    assert np.allclose(relative_pos[:2], expected_rel_pos, atol=1e-5)

    # Distance should be positive
    assert relative_pos[2] > 0

    # Link quality should be in [0, 1]
    assert 0.0 <= link_quality <= 1.0

    # Anomaly score should match
    assert np.isclose(anomaly_score, neighbor_vehicle.anomaly_score)

    # Embedding should match
    assert np.allclose(embedding, neighbor_vehicle.embedding)


def test_extract_neighbor_features_link_quality(sample_vehicle_nodes):
    """Test link quality computation based on distance."""
    ego_vehicle = sample_vehicle_nodes[0]

    # Create neighbor at exact communication range
    neighbor = VehicleNode(vehicle_id=10, window_size=50, num_features=10, embedding_dim=32)
    neighbor.position = (300.0, 0.0)  # 300m away
    neighbor.velocity = (0.0, 0.0)
    neighbor.embedding = np.zeros(32, dtype=np.float32)
    neighbor.anomaly_score = 0.0

    ego_vehicle.position = (0.0, 0.0)

    features = extract_neighbor_features(ego_vehicle, [neighbor], comm_range=300.0)

    link_quality = features[0, 6]
    # At max range, link quality should be close to 0
    assert link_quality < 0.1


# ==================== Batch Creation Tests ====================


def test_create_neighbor_batch():
    """Test creating batched neighbor features."""
    # Create neighbor features for 3 vehicles
    nf1 = np.random.randn(5, 40).astype(np.float32)  # 5 neighbors
    nf2 = np.random.randn(3, 40).astype(np.float32)  # 3 neighbors
    nf3 = np.random.randn(7, 40).astype(np.float32)  # 7 neighbors

    neighbor_features_list = [nf1, nf2, nf3]

    features, mask = create_neighbor_batch(neighbor_features_list)

    # Check shapes
    assert features.shape == (3, 7, 40)  # batch=3, max_neighbors=7, features=40
    assert mask.shape == (3, 7)

    # Check mask
    assert mask[0].sum() == 5
    assert mask[1].sum() == 3
    assert mask[2].sum() == 7

    # Check that features match for valid neighbors
    assert torch.allclose(features[0, :5], torch.from_numpy(nf1))
    assert torch.allclose(features[1, :3], torch.from_numpy(nf2))


def test_create_neighbor_batch_with_max_neighbors():
    """Test batch creation with specified max_neighbors."""
    nf1 = np.random.randn(5, 40).astype(np.float32)
    nf2 = np.random.randn(3, 40).astype(np.float32)

    features, mask = create_neighbor_batch([nf1, nf2], max_neighbors=10)

    assert features.shape == (2, 10, 40)
    assert mask.shape == (2, 10)
    assert mask[0].sum() == 5
    assert mask[1].sum() == 3


def test_create_neighbor_batch_no_neighbors():
    """Test batch creation with vehicles that have no neighbors."""
    nf1 = np.zeros((0, 40), dtype=np.float32)  # No neighbors
    nf2 = np.random.randn(2, 40).astype(np.float32)

    features, mask = create_neighbor_batch([nf1, nf2])

    assert features.shape[0] == 2
    assert mask[0].sum() == 0  # No neighbors for first vehicle
    assert mask[1].sum() == 2


# ==================== Config Integration Tests ====================


def test_create_neighbor_encoder_from_config(test_config):
    """Test creating neighbor encoder from Hydra config."""
    device = torch.device("cpu")
    model = create_neighbor_encoder_from_config(test_config, device=device)

    assert isinstance(model, NeighborEncoder)
    assert model.neighbor_feature_dim == 40
    assert model.phi_output_dim == 32
    assert model.rho_output_dim == 32
    assert next(model.parameters()).device == device


# ==================== Integration Tests ====================


def test_end_to_end_encoding(sample_vehicle_nodes):
    """Test end-to-end neighbor encoding pipeline."""
    # Create encoder
    encoder = NeighborEncoder(
        neighbor_feature_dim=40,
        phi_hidden_dim=64,
        phi_output_dim=32,
        rho_hidden_dim=64,
        rho_output_dim=32,
    )
    encoder.eval()

    # Extract features for each vehicle
    neighbor_features_list = []
    for i, ego_vehicle in enumerate(sample_vehicle_nodes):
        # Get neighbors (all other vehicles)
        neighbors = [v for j, v in enumerate(sample_vehicle_nodes) if j != i]
        features = extract_neighbor_features(ego_vehicle, neighbors, comm_range=300.0)
        neighbor_features_list.append(features)

    # Create batch
    features_batch, mask_batch = create_neighbor_batch(neighbor_features_list)

    # Encode neighborhoods
    with torch.no_grad():
        encodings = encoder(features_batch, mask_batch)

    assert encodings.shape == (len(sample_vehicle_nodes), 32)
    assert not torch.isnan(encodings).any()


def test_gradient_flow(neighbor_encoder, sample_neighbor_features):
    """Test that gradients flow through the model."""
    neighbor_encoder.train()

    output = neighbor_encoder(sample_neighbor_features)
    loss = output.sum()
    loss.backward()

    # Check that gradients exist
    for name, param in neighbor_encoder.named_parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
