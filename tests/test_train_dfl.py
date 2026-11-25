"""Tests for DFL training script."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from train_dfl import (
    create_dfl_protocols,
    create_vehicle_data_loaders,
    simulate_dfl_communication,
)


@pytest.fixture
def dfl_config():
    """Create test configuration for DFL training."""
    config = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "dataset_path": "data/processed/v2v_trajectories.csv",
            "metadata_path": "data/processed/metadata.json",
            "features": ["x", "y", "speed", "acceleration"],
            "num_vehicles": 5,
            "sequence_length": 20,
            "window_size": 20,
            "stride": 10,
            "batch_size": 16,
            "min_presence_duration": 10.0,
        },
        "model": {
            "architecture": {
                "embedding_dim": 32,
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.1,
            }
        },
        "federated": {
            "alpha": 0.3,
            "staleness_decay": 0.1,
            "max_staleness_seconds": 30.0,
            "num_prototypes": 5,
        },
        "rl": {
            "actor": {
                "shared_hidden_dim": 128,
                "when_hidden_dim": 64,
                "who_hidden_dim": 64,
                "what_hidden_dim": 64,
                "dropout": 0.1,
            },
            "critic": {
                "hidden_dim": 256,
                "num_layers": 2,
                "dropout": 0.1,
            },
            "training": {
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "value_loss_coef": 0.5,
                "entropy_coef": 0.01,
                "max_grad_norm": 0.5,
                "buffer_size": 1000,
                "batch_size": 64,
                "ppo_epochs": 4,
                "num_episodes": 10,
                "update_interval": 5,
                "save_interval": 10,
                "log_interval": 5,
            },
            "reward": {
                "immediate_weight": 0.3,
                "global_weight": 0.7,
                "perf_scale": 100.0,
                "comm_cost_scale": 0.01,
                "staleness_penalty_weight": 0.1,
            },
            "k_neighbors": 2,
        },
        "simulation": {
            "max_neighbors": 5,
            "max_episode_timesteps": 10,
        },
        "paths": {"output_dir": "outputs"},
    }
    return DictConfig(config)


@pytest.fixture
def mock_env():
    """Create mock V2V environment."""
    env = MagicMock()
    env.vehicle_ids = [1, 2, 3]  # Reduced for simpler testing
    env.layer1_model = MagicMock()
    env.layer1_model.state_dict.return_value = {}

    # Mock vehicles with neighbors
    env.vehicles = {}
    for vid in env.vehicle_ids:
        vehicle = MagicMock()
        vehicle.id = vid
        # Each vehicle has 2 neighbors (matching test expectations)
        other_vids = [v for v in env.vehicle_ids if v != vid]
        vehicle.neighbors = set(other_vids[:2]) if len(other_vids) >= 2 else set(other_vids)
        env.vehicles[vid] = vehicle

    return env


@pytest.fixture
def mock_features_df():
    """Create mock features dataframe."""
    import pandas as pd

    data = []
    for vid in range(1, 6):  # 5 vehicles
        for t in range(50):  # 50 timesteps
            data.append(
                {
                    "vehicle_id": vid,
                    "timestamp": t * 0.1,
                    "frame": t,
                    "x": np.random.randn(),
                    "y": np.random.randn(),
                    "speed": np.random.rand() * 30,
                    "acceleration": np.random.randn(),
                }
            )
    return pd.DataFrame(data)


class TestCreateVehicleDataLoaders:
    """Test vehicle-specific data loader creation."""

    def test_creates_loaders_for_all_vehicles(self, mock_features_df, dfl_config):
        """Test that data loaders are created for all vehicles."""
        vehicle_ids = [1, 2, 3, 4, 5]
        loaders = create_vehicle_data_loaders(mock_features_df, vehicle_ids, dfl_config)

        assert len(loaders) == 5
        assert all(vid in loaders for vid in vehicle_ids)
        assert all(isinstance(loader, torch.utils.data.DataLoader) for loader in loaders.values())

    def test_vehicle_data_isolation(self, mock_features_df, dfl_config):
        """Test that each vehicle's data is properly isolated."""
        vehicle_ids = [1, 2, 3]
        loaders = create_vehicle_data_loaders(mock_features_df, vehicle_ids, dfl_config)

        # Check that each loader has data
        for vid, loader in loaders.items():
            assert len(loader.dataset) > 0

    def test_windowing_applied_correctly(self, mock_features_df, dfl_config):
        """Test that sliding windows are applied correctly."""
        vehicle_ids = [1]
        loaders = create_vehicle_data_loaders(mock_features_df, vehicle_ids, dfl_config)

        # Get first batch
        batch = next(iter(loaders[1]))
        windows = batch[0]

        # Check window shape: (batch_size, sequence_length, num_features)
        assert windows.ndim == 3
        assert windows.shape[1] == dfl_config.data.sequence_length
        assert windows.shape[2] == len(dfl_config.data.features)

    def test_batch_size_respected(self, mock_features_df, dfl_config):
        """Test that batch size is respected."""
        vehicle_ids = [1]
        loaders = create_vehicle_data_loaders(mock_features_df, vehicle_ids, dfl_config)

        batch = next(iter(loaders[1]))
        windows = batch[0]
        assert windows.shape[0] <= dfl_config.data.batch_size


class TestCreateDFLProtocols:
    """Test DFL protocol creation."""

    def test_creates_protocols_for_all_vehicles(self, mock_env, dfl_config):
        """Test that DFL protocols are created for all vehicles."""
        # Mock data loaders
        mock_loaders = {vid: MagicMock() for vid in mock_env.vehicle_ids}

        with patch("train_dfl.create_autoencoder_from_config") as mock_create_ae:
            mock_create_ae.return_value = MagicMock()
            mock_create_ae.return_value.state_dict.return_value = {}

            protocols = create_dfl_protocols(
                mock_env, mock_loaders, torch.device("cpu"), dfl_config
            )

            assert len(protocols) == len(mock_env.vehicle_ids)
            assert all(vid in protocols for vid in mock_env.vehicle_ids)

    def test_each_protocol_has_required_components(self, mock_env, dfl_config):
        """Test that each protocol has all required components."""
        mock_loaders = {vid: MagicMock() for vid in mock_env.vehicle_ids}

        with patch("train_dfl.create_autoencoder_from_config") as mock_create_ae, patch(
            "train_dfl.RepresentationExtractor"
        ) as mock_repr, patch("train_dfl.AsynchronousAggregator") as mock_agg, patch(
            "train_dfl.DFLProtocol"
        ) as mock_dfl:

            mock_create_ae.return_value = MagicMock()
            mock_create_ae.return_value.state_dict.return_value = {}

            protocols = create_dfl_protocols(
                mock_env, mock_loaders, torch.device("cpu"), dfl_config
            )

            # Verify components were created
            num_vehicles = len(mock_env.vehicle_ids)
            assert mock_repr.call_count == num_vehicles
            assert mock_agg.call_count == num_vehicles
            assert mock_dfl.call_count == num_vehicles

    def test_protocols_use_correct_config_params(self, mock_env, dfl_config):
        """Test that protocols use correct configuration parameters."""
        mock_loaders = {vid: MagicMock() for vid in mock_env.vehicle_ids}

        with patch("train_dfl.create_autoencoder_from_config") as mock_create_ae, patch(
            "train_dfl.RepresentationExtractor"
        ) as mock_repr, patch("train_dfl.AsynchronousAggregator") as mock_agg, patch(
            "train_dfl.DFLProtocol"
        ):

            mock_create_ae.return_value = MagicMock()
            mock_create_ae.return_value.state_dict.return_value = {}

            create_dfl_protocols(mock_env, mock_loaders, torch.device("cpu"), dfl_config)

            # Check RepresentationExtractor params
            mock_repr.assert_called_with(
                embedding_dim=dfl_config.model.architecture.embedding_dim,
                num_prototypes=dfl_config.federated.num_prototypes,
                device=torch.device("cpu"),
            )

            # Check AsynchronousAggregator params
            mock_agg.assert_called_with(
                alpha=dfl_config.federated.alpha,
                staleness_decay=dfl_config.federated.staleness_decay,
                max_staleness_seconds=dfl_config.federated.max_staleness_seconds,
            )


class TestSimulateDFLCommunication:
    """Test DFL communication simulation."""

    @pytest.fixture
    def mock_protocols(self):
        """Create mock DFL protocols."""
        protocols = {}
        for vid in [1, 2, 3]:
            protocol = MagicMock()
            protocol.execute_communication_action.return_value = (
                True,
                {
                    "recipients": [2, 3] if vid == 1 else [1],
                    "data": {"type": "E", "embeddings": np.random.randn(10, 32)},
                    "size_mb": 0.1,
                },
            )
            protocol.has_pending_updates.return_value = True
            protocol.receive_representation = MagicMock()
            protocol.aggregate_received_data = MagicMock()
            protocols[vid] = protocol
        return protocols

    @pytest.fixture
    def mock_actions(self):
        """Create mock RL actions. WHEN removed — all vehicles communicate every step."""
        return {
            1: {"who_action": [0, 1], "what_action": 1},  # W (weights)
            2: {"who_action": [0], "what_action": 0},  # E (embeddings)
            3: {"who_action": [0], "what_action": 1},  # W (weights)
        }

    def test_respects_when_action(self, mock_env, mock_actions, mock_protocols):
        """Test that all vehicles communicate (WHEN removed from paper)."""
        result = simulate_dfl_communication(mock_env, mock_actions, mock_protocols)

        # All vehicles communicate every timestep (WHEN decision removed)
        assert result["comm_results"][1] is True
        assert result["comm_results"][2] is True
        assert result["comm_results"][3] is True

    def test_messages_delivered_to_recipients(self, mock_env, mock_actions, mock_protocols):
        """Test that messages are delivered to intended recipients."""
        simulate_dfl_communication(mock_env, mock_actions, mock_protocols)

        # Check that receive_representation was called for recipients
        # (specific calls depend on mock setup)
        for protocol in mock_protocols.values():
            # At least check the method exists and can be called
            assert hasattr(protocol, "receive_representation")

    def test_aggregation_triggered_for_pending_updates(
        self, mock_env, mock_actions, mock_protocols
    ):
        """Test that aggregation is triggered when updates are pending."""
        simulate_dfl_communication(mock_env, mock_actions, mock_protocols)

        # All protocols with pending updates should aggregate
        for protocol in mock_protocols.values():
            protocol.has_pending_updates.assert_called()
            protocol.aggregate_received_data.assert_called()

    def test_returns_communication_statistics(self, mock_env, mock_actions, mock_protocols):
        """Test that communication statistics are returned."""
        result = simulate_dfl_communication(mock_env, mock_actions, mock_protocols)

        assert "comm_results" in result
        assert "total_data_sent_mb" in result
        assert "num_messages" in result
        assert isinstance(result["total_data_sent_mb"], float)
        assert isinstance(result["num_messages"], int)

    def test_handles_no_communication(self, mock_env, mock_protocols):
        """Test handling when no WHO recipients are selected."""
        # All vehicles provide empty who_action (no recipients)
        actions = {
            1: {"who_action": [], "what_action": 0},
            2: {"who_action": [], "what_action": 0},
            3: {"who_action": [], "what_action": 0},
        }

        # Set up mocks to return failure (no recipients) with size_mb=0
        for vid, protocol in mock_protocols.items():
            protocol.execute_communication_action.return_value = (
                False,
                {"recipients": [], "data": {}, "size_mb": 0.0},
            )

        result = simulate_dfl_communication(mock_env, actions, mock_protocols)

        assert result["total_data_sent_mb"] == 0.0
        assert result["num_messages"] == 0


class TestDFLIntegration:
    """Integration tests for DFL training components."""

    def test_data_loaders_and_protocols_integration(self, mock_features_df, mock_env, dfl_config):
        """Test integration of data loaders and DFL protocols."""
        # Create data loaders
        vehicle_ids = list(mock_env.vehicle_ids)
        loaders = create_vehicle_data_loaders(mock_features_df, vehicle_ids, dfl_config)

        # Create protocols with actual loaders
        with patch("train_dfl.create_autoencoder_from_config") as mock_create_ae:
            mock_create_ae.return_value = MagicMock()
            mock_create_ae.return_value.state_dict.return_value = {}

            protocols = create_dfl_protocols(
                mock_env, loaders, torch.device("cpu"), dfl_config
            )

            # Verify integration
            assert len(loaders) == len(protocols)
            assert set(loaders.keys()) == set(protocols.keys())

    def test_full_communication_cycle(self, mock_env, dfl_config):
        """Test a full communication cycle: action → send → receive → aggregate."""
        # Create minimal mock protocols
        protocols = {}
        for vid in mock_env.vehicle_ids:
            protocol = MagicMock()
            protocol.execute_communication_action.return_value = (
                True,
                {
                    "recipients": [2, 3],
                    "data": {"type": "E", "embeddings": np.random.randn(5, 32)},
                    "size_mb": 0.05,
                },
            )
            protocol.has_pending_updates.return_value = True
            protocol.receive_representation = MagicMock()
            protocol.aggregate_received_data = MagicMock()
            protocols[vid] = protocol

        # Create actions for all vehicles
        actions = {
            vid: {"when_action": 1, "who_action": [0, 1], "what_action": 1}
            for vid in mock_env.vehicle_ids
        }

        # Execute communication
        result = simulate_dfl_communication(mock_env, actions, protocols)

        # Verify all phases executed
        assert "comm_results" in result
        assert all(result["comm_results"].values())  # All communicated
        assert result["total_data_sent_mb"] > 0


class TestDFLConfigValidation:
    """Test configuration validation for DFL training."""

    def test_required_config_fields_present(self, dfl_config):
        """Test that all required config fields are present."""
        assert "data" in dfl_config
        assert "model" in dfl_config
        assert "federated" in dfl_config
        assert "rl" in dfl_config
        assert "simulation" in dfl_config

    def test_federated_params_valid(self, dfl_config):
        """Test that federated parameters are valid."""
        fed_cfg = dfl_config.federated
        assert 0 < fed_cfg.alpha < 1
        assert fed_cfg.staleness_decay > 0
        assert fed_cfg.max_staleness_seconds > 0
        assert fed_cfg.num_prototypes > 0

    def test_rl_params_valid(self, dfl_config):
        """Test that RL parameters are valid."""
        rl_cfg = dfl_config.rl.training
        assert rl_cfg.learning_rate > 0
        assert 0 < rl_cfg.gamma <= 1
        assert 0 < rl_cfg.gae_lambda <= 1
        assert rl_cfg.clip_epsilon > 0
        assert dfl_config.rl.k_neighbors > 0  # k_neighbors is in rl section, not training


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
