"""
Tests for V2V simulation module (VehicleNode and V2VEnvironment).
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.models import LSTMAutoencoder
from src.simulation import V2VEnvironment, VehicleNode


@pytest.fixture
def vehicle_node():
    """Create a VehicleNode for testing."""
    return VehicleNode(vehicle_id=1, window_size=50, num_features=10, embedding_dim=16)


@pytest.fixture
def simple_trajectory():
    """Create simple trajectory data for testing."""
    # 100 timesteps, 10 features
    trajectory = np.random.randn(100, 10).astype(np.float32)
    # Make first two features (velocity_x, velocity_y) reasonable
    trajectory[:, 0] = np.sin(np.linspace(0, 4 * np.pi, 100)) * 5  # velocity_x
    trajectory[:, 1] = np.cos(np.linspace(0, 4 * np.pi, 100)) * 5  # velocity_y
    trajectory[:, 2] = np.sqrt(trajectory[:, 0]**2 + trajectory[:, 1]**2)  # speed
    return trajectory


@pytest.fixture
def simple_model():
    """Create a simple LSTM Autoencoder for testing."""
    model = LSTMAutoencoder(
        input_size=10,
        hidden_size=32,
        num_layers=1,
        embedding_dim=16,
        seq_len=50,
    )
    model.eval()
    return model


@pytest.fixture
def test_config():
    """Create test configuration."""
    config_dict = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "window_size": 50,
            "features": [
                "velocity_x", "velocity_y", "speed",
                "accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z", "yaw"
            ],
        },
        "model": {
            "name": "lstm_autoencoder",
            "architecture": {
                "input_size": 10,
                "hidden_size": 32,
                "num_layers": 1,
                "embedding_dim": 16,
            },
        },
        "simulation": {
            "num_vehicles": 3,
            "comm_range_m": 100.0,
            "time_step_duration_s": 0.1,
            "max_episode_timesteps": 100,
            "layer1_checkpoint": "nonexistent.pt",  # Will use fresh model
            "data": {
                "use_preprocessed": True,
                "preprocessed_path": "outputs/preprocessed",
            },
        },
    }
    return OmegaConf.create(config_dict)


# ==================== VehicleNode Tests ====================


def test_vehicle_node_initialization(vehicle_node):
    """Test VehicleNode initialization."""
    assert vehicle_node.vehicle_id == 1
    assert vehicle_node.window_size == 50
    assert vehicle_node.num_features == 10
    assert vehicle_node.current_timestep == 0
    assert vehicle_node.position == (0.0, 0.0)
    assert vehicle_node.trajectory_window.shape == (50, 10)
    assert vehicle_node.embedding is None
    assert vehicle_node.anomaly_score is None
    assert len(vehicle_node.neighbors) == 0


def test_vehicle_node_set_trajectory_data(vehicle_node, simple_trajectory):
    """Test setting trajectory data."""
    vehicle_node.set_trajectory_data(simple_trajectory, start_idx=0, length=100)
    assert vehicle_node._trajectory_data is not None
    assert vehicle_node._trajectory_length == 100


def test_vehicle_node_state_update(vehicle_node, simple_trajectory):
    """Test state update from trajectory data."""
    vehicle_node.set_trajectory_data(simple_trajectory, start_idx=0, length=100)

    # Update to timestep 10
    vehicle_node.update_state(10)
    assert vehicle_node.current_timestep == 10
    assert vehicle_node.position != (0.0, 0.0)  # Position should have changed
    assert vehicle_node.velocity != (0.0, 0.0)

    # Check trajectory window is updated
    assert not np.all(vehicle_node.trajectory_window == 0)


def test_vehicle_node_embedding_computation(vehicle_node, simple_trajectory, simple_model):
    """Test embedding computation using LSTM model."""
    vehicle_node.set_trajectory_data(simple_trajectory, start_idx=0, length=100)

    # Update to have enough data for full window
    vehicle_node.update_state(50)

    # Compute embedding
    device = torch.device("cpu")
    embedding = vehicle_node.compute_embedding(simple_model, device)

    assert embedding is not None
    assert len(embedding) == 16  # embedding_dim
    assert vehicle_node.embedding is not None
    assert np.array_equal(vehicle_node.embedding, embedding)


def test_vehicle_node_embedding_early_timestep(vehicle_node, simple_trajectory, simple_model):
    """Test embedding computation before window is full."""
    vehicle_node.set_trajectory_data(simple_trajectory, start_idx=0, length=100)

    # Update to early timestep (before window is full)
    vehicle_node.update_state(20)

    # Compute embedding
    device = torch.device("cpu")
    embedding = vehicle_node.compute_embedding(simple_model, device)

    assert embedding is not None
    assert len(embedding) == 16
    assert np.all(embedding == 0)  # Should be zeros for early timesteps


def test_vehicle_node_anomaly_score_computation(vehicle_node, simple_trajectory, simple_model):
    """Test anomaly score computation using LSTM model."""
    vehicle_node.set_trajectory_data(simple_trajectory, start_idx=0, length=100)

    # Update to have enough data
    vehicle_node.update_state(50)

    # Compute anomaly score
    device = torch.device("cpu")
    score = vehicle_node.compute_anomaly_score(simple_model, device)

    assert score is not None
    assert score >= 0  # MSE is always non-negative
    assert vehicle_node.anomaly_score == score


def test_vehicle_node_neighbor_tracking(vehicle_node):
    """Test neighbor connection/disconnection tracking."""
    # Initial state
    assert len(vehicle_node.neighbors) == 0
    assert len(vehicle_node.neighbor_history) == 0

    # Add neighbors
    new_neighbors = {2, 3, 4}
    vehicle_node.update_neighbors(new_neighbors)

    assert vehicle_node.neighbors == {2, 3, 4}
    assert len(vehicle_node.neighbor_history) == 3  # 3 connections
    assert all(event[1] == "connected" for event in vehicle_node.neighbor_history)

    # Remove one, add one
    vehicle_node.current_timestep = 10
    new_neighbors = {2, 3, 5}  # Lost 4, gained 5
    vehicle_node.update_neighbors(new_neighbors)

    assert vehicle_node.neighbors == {2, 3, 5}
    assert len(vehicle_node.neighbor_history) == 5  # 3 + 1 disconnection + 1 connection

    # Check for disconnection event
    disconnection_events = [e for e in vehicle_node.neighbor_history if e[1] == "disconnected"]
    assert len(disconnection_events) == 1
    assert disconnection_events[0][2] == 4  # Vehicle 4 disconnected


def test_vehicle_node_communication_logging(vehicle_node):
    """Test communication event logging."""
    assert len(vehicle_node.communication_history) == 0

    # Log some communications
    vehicle_node.log_communication("send", 2, {"type": "update", "data": [1, 2, 3]})
    vehicle_node.current_timestep = 5
    vehicle_node.log_communication("receive", 3, {"type": "ack"})

    assert len(vehicle_node.communication_history) == 2
    assert vehicle_node.communication_history[0][1] == "send"
    assert vehicle_node.communication_history[1][1] == "receive"


def test_vehicle_node_get_observation(vehicle_node, simple_trajectory):
    """Test observation dictionary generation."""
    vehicle_node.set_trajectory_data(simple_trajectory, start_idx=0, length=100)
    vehicle_node.update_state(10)
    vehicle_node.neighbors = {2, 3}

    obs = vehicle_node.get_observation()

    assert obs["vehicle_id"] == 1
    assert obs["timestep"] == 10
    assert "position" in obs
    assert "velocity" in obs
    assert "trajectory_window" in obs
    assert obs["num_neighbors"] == 2
    assert set(obs["neighbor_ids"]) == {2, 3}


def test_vehicle_node_reset(vehicle_node, simple_trajectory):
    """Test vehicle reset."""
    vehicle_node.set_trajectory_data(simple_trajectory, start_idx=0, length=100)
    vehicle_node.update_state(50)
    vehicle_node.neighbors = {2, 3}
    vehicle_node.embedding = np.ones(16)

    # Reset
    vehicle_node.reset()

    assert vehicle_node.current_timestep == 0
    assert vehicle_node.position == (0.0, 0.0)
    assert vehicle_node.embedding is None
    assert len(vehicle_node.neighbors) == 0
    assert len(vehicle_node.neighbor_history) == 0


# ==================== V2VEnvironment Tests ====================


@pytest.mark.skipif(
    not (Path("outputs/preprocessed/preprocessed_data.npz")).exists(),
    reason="Preprocessed data not available. Run train.py first.",
)
def test_v2v_environment_initialization(test_config):
    """Test V2VEnvironment initialization."""
    env = V2VEnvironment(test_config)

    assert env.num_vehicles == 3
    assert env.comm_range == 100.0
    assert env.current_timestep == 0
    assert len(env.vehicles) > 0
    assert env.lstm_model is not None


@pytest.mark.skipif(
    not (Path("outputs/preprocessed/preprocessed_data.npz")).exists(),
    reason="Preprocessed data not available",
)
def test_v2v_environment_reset(test_config):
    """Test environment reset."""
    env = V2VEnvironment(test_config)

    observations = env.reset()

    assert isinstance(observations, dict)
    assert len(observations) > 0
    assert env.current_timestep == 0
    assert env.episode_count == 1

    # Check observation structure
    for vehicle_id, obs in observations.items():
        assert "vehicle_id" in obs
        assert "position" in obs
        assert "velocity" in obs


@pytest.mark.skipif(
    not (Path("outputs/preprocessed/preprocessed_data.npz")).exists(),
    reason="Preprocessed data not available",
)
def test_v2v_environment_step(test_config):
    """Test environment step."""
    env = V2VEnvironment(test_config)
    env.reset()

    # Take one step
    obs, rewards, done, info = env.step()

    assert isinstance(obs, dict)
    assert isinstance(rewards, dict)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert env.current_timestep == 1
    assert "timestep" in info
    assert "neighbor_graph" in info


@pytest.mark.skipif(
    not (Path("outputs/preprocessed/preprocessed_data.npz")).exists(),
    reason="Preprocessed data not available",
)
def test_v2v_environment_neighbor_discovery(test_config):
    """Test neighbor discovery mechanism."""
    env = V2VEnvironment(test_config)
    env.reset()

    # Manually set positions to test neighbor discovery
    vehicle_ids = list(env.vehicles.keys())
    if len(vehicle_ids) >= 2:
        v1 = env.vehicles[vehicle_ids[0]]
        v2 = env.vehicles[vehicle_ids[1]]

        # Place vehicles close together (within range)
        v1.position = (0.0, 0.0)
        v2.position = (50.0, 0.0)  # 50m away

        env.discover_neighbors()

        # Should be neighbors (within 100m range)
        assert vehicle_ids[1] in v1.neighbors
        assert vehicle_ids[0] in v2.neighbors


@pytest.mark.skipif(
    not (Path("outputs/preprocessed/preprocessed_data.npz")).exists(),
    reason="Preprocessed data not available",
)
def test_v2v_environment_neighbor_discovery_out_of_range(test_config):
    """Test neighbor discovery for vehicles out of range."""
    env = V2VEnvironment(test_config)
    env.reset()

    vehicle_ids = list(env.vehicles.keys())
    if len(vehicle_ids) >= 2:
        v1 = env.vehicles[vehicle_ids[0]]
        v2 = env.vehicles[vehicle_ids[1]]

        # Place vehicles far apart (out of range)
        v1.position = (0.0, 0.0)
        v2.position = (200.0, 0.0)  # 200m away (> 100m range)

        env.discover_neighbors()

        # Should NOT be neighbors
        assert vehicle_ids[1] not in v1.neighbors
        assert vehicle_ids[0] not in v2.neighbors


@pytest.mark.skipif(
    not (Path("outputs/preprocessed/preprocessed_data.npz")).exists(),
    reason="Preprocessed data not available",
)
def test_v2v_environment_get_metrics(test_config):
    """Test metrics collection."""
    env = V2VEnvironment(test_config)
    env.reset()

    metrics = env.get_metrics()

    assert "timestep" in metrics
    assert "num_vehicles" in metrics
    assert "avg_neighbors" in metrics
    assert metrics["num_vehicles"] == len(env.vehicles)


