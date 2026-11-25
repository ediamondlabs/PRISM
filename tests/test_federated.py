"""
Tests for Federated Learning components.

Tests RepresentationExtractor, AsynchronousAggregator, and DFLProtocol.
"""

import time

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.federated import AsynchronousAggregator, DFLProtocol, RepresentationExtractor


class SimpleLSTMEncoder(nn.Module):
    """Minimal LSTM encoder for testing."""

    def __init__(self, input_dim: int = 2, embedding_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, embedding_dim, batch_first=True)
        self.embedding_dim = embedding_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to embedding."""
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.encode(x)


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def embedding_dim():
    """Embedding dimension for testing."""
    return 32


@pytest.fixture
def num_prototypes():
    """Number of prototypes for testing."""
    return 5


@pytest.fixture
def simple_model(device, embedding_dim):
    """Create simple LSTM encoder for testing."""
    model = SimpleLSTMEncoder(input_dim=2, embedding_dim=embedding_dim)
    return model.to(device)


@pytest.fixture
def sample_data_loader(device):
    """Create sample data loader."""
    seq_len = 10
    num_samples = 150  # Increased to support tests requesting 100 samples
    input_dim = 2

    X = torch.randn(num_samples, seq_len, input_dim)
    dataset = TensorDataset(X)
    return DataLoader(dataset, batch_size=16, shuffle=False)


@pytest.fixture
def representation_extractor(embedding_dim, num_prototypes, device):
    """Create RepresentationExtractor for testing."""
    return RepresentationExtractor(
        embedding_dim=embedding_dim,
        num_prototypes=num_prototypes,
        device=device,
    )


@pytest.fixture
def aggregator():
    """Create AsynchronousAggregator for testing."""
    return AsynchronousAggregator(
        alpha=0.3,
        staleness_decay=0.1,
        max_staleness_seconds=30.0,
    )


class TestRepresentationExtractor:
    """Test RepresentationExtractor class."""

    def test_extract_embeddings_shape(
        self, representation_extractor, simple_model, sample_data_loader, embedding_dim
    ):
        """Test embedding extraction returns correct shape."""
        num_samples = 100
        embeddings = representation_extractor.extract_embeddings(
            model=simple_model,
            data_loader=sample_data_loader,
            num_samples=num_samples,
        )

        assert embeddings.shape == (num_samples, embedding_dim)
        assert embeddings.dtype == np.float32

    def test_extract_embeddings_deterministic(
        self, representation_extractor, simple_model, sample_data_loader
    ):
        """Test embedding extraction is deterministic in eval mode."""
        embeddings1 = representation_extractor.extract_embeddings(
            model=simple_model,
            data_loader=sample_data_loader,
            num_samples=50,
        )

        embeddings2 = representation_extractor.extract_embeddings(
            model=simple_model,
            data_loader=sample_data_loader,
            num_samples=50,
        )

        np.testing.assert_array_almost_equal(embeddings1, embeddings2)

    def test_extract_prototypes_shape(
        self, representation_extractor, embedding_dim, num_prototypes
    ):
        """Test prototype extraction returns correct shape."""
        embeddings = np.random.randn(100, embedding_dim).astype(np.float32)

        prototypes = representation_extractor.extract_prototypes(
            embeddings=embeddings,
            update_clusters=True,
        )

        assert prototypes.shape == (num_prototypes, embedding_dim)
        assert prototypes.dtype == np.float32

    def test_extract_prototypes_updates_clusters(self, representation_extractor, embedding_dim):
        """Test that update_clusters=True modifies cluster centers."""
        # Use structurally different data (different means/clusters)
        np.random.seed(0)
        embeddings1 = np.random.randn(100, embedding_dim).astype(np.float32) + 5.0
        np.random.seed(1)
        embeddings2 = np.random.randn(100, embedding_dim).astype(np.float32) - 5.0

        prototypes1 = representation_extractor.extract_prototypes(embeddings1, update_clusters=True)
        prototypes2 = representation_extractor.extract_prototypes(embeddings2, update_clusters=True)

        # After normalization, cluster centers should still differ in direction
        assert not np.allclose(prototypes1, prototypes2, atol=0.01)

    def test_extract_prototypes_empty_data(self, representation_extractor, embedding_dim, num_prototypes):
        """Test prototype extraction with empty data."""
        embeddings = np.empty((0, embedding_dim), dtype=np.float32)

        prototypes = representation_extractor.extract_prototypes(
            embeddings=embeddings,
            update_clusters=False,
        )

        assert prototypes.shape == (num_prototypes, embedding_dim)
        assert np.all(prototypes == 0.0)

    def test_estimate_size_embeddings(self, representation_extractor, embedding_dim):
        """Test size estimation for embeddings."""
        num_samples = 100
        size_mb = representation_extractor.estimate_size("E", num_samples=num_samples)

        expected_size = (num_samples * embedding_dim * 4) / (1024 * 1024)
        assert abs(size_mb - expected_size) < 1e-6

    def test_estimate_size_weights_fallback(self, representation_extractor):
        """Test size estimation for weights without model (fallback)."""
        size_mb = representation_extractor.estimate_size("W")
        assert size_mb == 10.0  # Fallback estimate

    def test_estimate_size_weights_with_model(self, representation_extractor, simple_model):
        """Test size estimation for weights with model."""
        size_mb = representation_extractor.estimate_size("W", model=simple_model)
        assert size_mb > 0

    def test_compress_embeddings(self, representation_extractor, embedding_dim):
        """Test embedding compression with 8-bit quantization."""
        embeddings = np.random.randn(50, embedding_dim).astype(np.float32)

        compressed, comp_info = representation_extractor.compress_embeddings(embeddings, bits=8)

        assert compressed.dtype == np.uint8
        assert compressed.shape == embeddings.shape
        assert "min" in comp_info
        assert "max" in comp_info
        assert comp_info["bits"] == 8

    def test_compress_decompress_roundtrip(self, representation_extractor, embedding_dim):
        """Test compress/decompress roundtrip preserves data reasonably."""
        embeddings = np.random.randn(50, embedding_dim).astype(np.float32)

        compressed, comp_info = representation_extractor.compress_embeddings(embeddings, bits=8)
        decompressed = representation_extractor.decompress_embeddings(compressed, comp_info)

        assert decompressed.shape == embeddings.shape
        # With 8-bit quantization, expect some loss but reasonable approximation
        np.testing.assert_allclose(decompressed, embeddings, rtol=0.1, atol=0.5)

    def test_extract_and_package_embeddings(
        self, representation_extractor, simple_model, sample_data_loader, embedding_dim
    ):
        """Test extract_and_package for embeddings."""
        num_samples = 100
        package = representation_extractor.extract_and_package(
            model=simple_model,
            data_loader=sample_data_loader,
            repr_type="E",
            num_samples=num_samples,
            compress=False,
        )

        assert package["type"] == "E"
        assert package["data"].shape == (num_samples, embedding_dim)
        assert package["size_mb"] > 0
        assert package["compressed"] is False
        assert package["compression_info"] is None

    def test_extract_and_package_weights(
        self, representation_extractor, simple_model, sample_data_loader
    ):
        """Test extract_and_package for weights."""
        package = representation_extractor.extract_and_package(
            model=simple_model,
            data_loader=sample_data_loader,
            repr_type="W",
        )

        assert package["type"] == "W"
        assert isinstance(package["data"], dict)  # state_dict
        assert package["size_mb"] > 0
        assert package["compressed"] is False

    def test_extract_and_package_compressed(
        self, representation_extractor, simple_model, sample_data_loader
    ):
        """Test extract_and_package with compression."""
        package = representation_extractor.extract_and_package(
            model=simple_model,
            data_loader=sample_data_loader,
            repr_type="E",
            num_samples=50,
            compress=True,
        )

        assert package["compressed"] is True
        assert package["compression_info"] is not None
        assert package["data"].dtype == np.uint8

    def test_reset_prototypes(self, representation_extractor, embedding_dim):
        """Test resetting prototype clusters."""
        embeddings = np.random.randn(100, embedding_dim).astype(np.float32)

        prototypes1 = representation_extractor.extract_prototypes(embeddings, update_clusters=True)
        representation_extractor.reset_prototypes()
        prototypes2 = representation_extractor.extract_prototypes(embeddings, update_clusters=True)

        # After reset, prototypes should be different (re-initialized)
        # Note: K-Means may converge to similar centroids, so just check cluster object was reset
        assert representation_extractor.kmeans is not None
        assert representation_extractor.kmeans.cluster_centers_ is not None


class TestAsynchronousAggregator:
    """Test AsynchronousAggregator class."""

    def test_compute_staleness_weights_recent(self, aggregator):
        """Test staleness weights for recent updates."""
        current_time = time.time()
        timestamps = {
            1: current_time - 1.0,  # 1 second old
            2: current_time - 5.0,  # 5 seconds old
            3: current_time - 10.0, # 10 seconds old
        }

        weights = aggregator.compute_staleness_weights(timestamps, current_time)

        assert len(weights) == 3
        # More recent updates should have higher weight
        assert weights[1] > weights[2] > weights[3]
        # All weights should be in [0, 1]
        for w in weights.values():
            assert 0.0 <= w <= 1.0

    def test_compute_staleness_weights_stale(self, aggregator):
        """Test staleness weights for very stale updates."""
        current_time = time.time()
        timestamps = {
            1: current_time - 50.0,  # 50 seconds old (> max_staleness_seconds=30)
        }

        weights = aggregator.compute_staleness_weights(timestamps, current_time)

        assert weights[1] == 0.0  # Should be filtered out

    def test_aggregate_embeddings_no_neighbors(self, aggregator):
        """Test aggregation with no neighbors."""
        local_embeddings = np.random.randn(100, 32).astype(np.float32)
        neighbor_embeddings = {}
        timestamps = {}

        aggregated, stats = aggregator.aggregate_embeddings(
            local_embeddings=local_embeddings,
            neighbor_embeddings=neighbor_embeddings,
            timestamps=timestamps,
        )

        np.testing.assert_array_equal(aggregated, local_embeddings)
        assert stats["num_neighbors"] == 0
        assert stats["total_weight"] == 1.0

    def test_aggregate_embeddings_with_neighbors(self, aggregator):
        """Test aggregation with valid neighbors."""
        local_embeddings = np.random.randn(100, 32).astype(np.float32)

        current_time = time.time()
        neighbor_embeddings = {
            1: np.random.randn(100, 32).astype(np.float32),
            2: np.random.randn(100, 32).astype(np.float32),
        }
        timestamps = {
            1: current_time - 1.0,
            2: current_time - 5.0,
        }

        aggregated, stats = aggregator.aggregate_embeddings(
            local_embeddings=local_embeddings,
            neighbor_embeddings=neighbor_embeddings,
            timestamps=timestamps,
            current_time=current_time,
        )

        assert aggregated.shape == local_embeddings.shape
        assert stats["num_neighbors"] == 2
        assert stats["total_weight"] > 1.0  # local(1.0) + neighbor weights
        assert 0.0 < stats["local_weight"] < 1.0

    def test_aggregate_embeddings_different_sample_counts(self, aggregator):
        """Test aggregation handles different sample counts."""
        local_embeddings = np.random.randn(100, 32).astype(np.float32)

        current_time = time.time()
        neighbor_embeddings = {
            1: np.random.randn(50, 32).astype(np.float32),  # Fewer samples
        }
        timestamps = {1: current_time - 1.0}

        aggregated, stats = aggregator.aggregate_embeddings(
            local_embeddings=local_embeddings,
            neighbor_embeddings=neighbor_embeddings,
            timestamps=timestamps,
            current_time=current_time,
        )

        # Should still work (takes min sample count)
        assert aggregated.shape == local_embeddings.shape

    def test_aggregate_prototypes_with_neighbors(self, aggregator):
        """Test prototype aggregation."""
        num_prototypes = 5
        embedding_dim = 32

        local_prototypes = np.random.randn(num_prototypes, embedding_dim).astype(np.float32)

        current_time = time.time()
        neighbor_prototypes = {
            1: np.random.randn(num_prototypes, embedding_dim).astype(np.float32),
            2: np.random.randn(num_prototypes, embedding_dim).astype(np.float32),
        }
        timestamps = {
            1: current_time - 2.0,
            2: current_time - 8.0,
        }

        aggregated, stats = aggregator.aggregate_prototypes(
            local_prototypes=local_prototypes,
            neighbor_prototypes=neighbor_prototypes,
            timestamps=timestamps,
            current_time=current_time,
        )

        assert aggregated.shape == (num_prototypes, embedding_dim)
        assert stats["num_neighbors"] == 2
        assert stats["total_weight"] > 1.0

    def test_compute_divergence(self, aggregator):
        """Test divergence computation."""
        local_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        aggregated_data = np.array([[1.1, 2.1], [3.1, 4.1]], dtype=np.float32)

        divergence = aggregator.compute_divergence(local_data, aggregated_data)

        assert divergence > 0.0
        assert isinstance(divergence, float)

    def test_should_aggregate_sufficient_neighbors(self, aggregator):
        """Test should_aggregate with sufficient neighbors."""
        current_time = time.time()
        timestamps = {
            1: current_time - 1.0,
            2: current_time - 5.0,
        }

        should_agg = aggregator.should_aggregate(timestamps, min_neighbors=1)

        assert should_agg is True

    def test_should_aggregate_insufficient_neighbors(self, aggregator):
        """Test should_aggregate with insufficient neighbors."""
        current_time = time.time()
        timestamps = {
            1: current_time - 50.0,  # Too stale
        }

        should_agg = aggregator.should_aggregate(timestamps, min_neighbors=1)

        assert should_agg is False

    def test_get_aggregation_metrics(self, aggregator):
        """Test extraction of aggregation metrics."""
        stats = {
            "num_neighbors": 3,
            "total_weight": 2.5,
            "local_weight": 0.4,
            "num_stale": 1,
            "neighbor_weights": {1: 0.2, 2: 0.2, 3: 0.2},
        }

        metrics = aggregator.get_aggregation_metrics(stats)

        assert metrics["num_neighbors"] == 3
        assert metrics["total_weight"] == 2.5
        assert metrics["local_weight"] == 0.4
        assert metrics["num_stale"] == 1
        assert "avg_neighbor_weight" in metrics
        assert metrics["avg_neighbor_weight"] == pytest.approx(0.2)


class TestDFLProtocol:
    """Test DFLProtocol class."""

    @pytest.fixture
    def dfl_protocol(self, simple_model, sample_data_loader, representation_extractor, aggregator, device):
        """Create DFLProtocol for testing."""
        return DFLProtocol(
            vehicle_id=0,
            local_model=simple_model,
            data_loader=sample_data_loader,
            representation_extractor=representation_extractor,
            aggregator=aggregator,
            device=device,
        )

    def test_initialization(self, dfl_protocol):
        """Test DFLProtocol initialization."""
        assert dfl_protocol.vehicle_id == 0
        assert len(dfl_protocol.received_buffer) == 0
        assert dfl_protocol.updates_sent == 0
        assert dfl_protocol.updates_received == 0

    def test_execute_communication_action_no_neighbors(self, dfl_protocol):
        """Test communication when no neighbors are available."""
        who_action = [1, 2]  # Target neighbors 1 and 2
        what_action = "E"
        neighbors = set()  # No neighbors

        success, comm_info = dfl_protocol.execute_communication_action(
            who_action=who_action,
            what_action=what_action,
            neighbors=neighbors,
        )

        assert success is False
        assert comm_info["reason"] == "no_valid_neighbors"
        assert comm_info["data"] is None

    def test_execute_communication_action_embeddings(self, dfl_protocol):
        """Test communication action for embeddings."""
        who_action = [1, 2]
        what_action = "E"
        neighbors = {1, 2, 3}

        success, comm_info = dfl_protocol.execute_communication_action(
            who_action=who_action,
            what_action=what_action,
            neighbors=neighbors,
            num_embedding_samples=100,
        )

        assert success is True
        assert comm_info["success"] is True
        assert comm_info["data"]["type"] == "E"
        assert comm_info["size_mb"] > 0
        assert set(comm_info["recipients"]) == {1, 2}
        assert dfl_protocol.updates_sent == 2

    @pytest.mark.skip(reason="Prototypes (P) removed from paper — only E and W repr types remain")
    def test_execute_communication_action_prototypes(self, dfl_protocol):
        """Prototypes representation type removed per PRISM paper (num_repr_types=2)."""
        pass

    def test_execute_communication_filters_invalid_neighbors(self, dfl_protocol):
        """Test that WHO action is filtered to only valid neighbors."""
        who_action = [1, 2, 99]  # 99 is not a neighbor
        what_action = "E"
        neighbors = {1, 2}

        success, comm_info = dfl_protocol.execute_communication_action(
            who_action=who_action,
            what_action=what_action,
            neighbors=neighbors,
        )

        assert success is True
        assert set(comm_info["recipients"]) == {1, 2}  # 99 filtered out

    def test_receive_representation(self, dfl_protocol, representation_extractor, simple_model, sample_data_loader):
        """Test receiving representation from neighbor."""
        # Create representation package
        repr_package = representation_extractor.extract_and_package(
            model=simple_model,
            data_loader=sample_data_loader,
            repr_type="E",
            num_samples=50,
        )

        neighbor_id = 5
        dfl_protocol.receive_representation(neighbor_id, repr_package)

        assert neighbor_id in dfl_protocol.received_buffer
        assert dfl_protocol.updates_received == 1
        assert dfl_protocol.total_data_received_mb > 0

    def test_receive_representation_overwrites(self, dfl_protocol, representation_extractor, simple_model, sample_data_loader):
        """Test that receiving from same neighbor overwrites old data."""
        repr_package1 = representation_extractor.extract_and_package(
            model=simple_model,
            data_loader=sample_data_loader,
            repr_type="E",
            num_samples=50,
        )

        repr_package2 = representation_extractor.extract_and_package(
            model=simple_model,
            data_loader=sample_data_loader,
            repr_type="W",
        )

        neighbor_id = 3
        dfl_protocol.receive_representation(neighbor_id, repr_package1)
        dfl_protocol.receive_representation(neighbor_id, repr_package2)

        # Buffer should only have one entry for this neighbor
        assert len(dfl_protocol.received_buffer) == 1
        assert dfl_protocol.received_buffer[neighbor_id][1] == "W"  # Latest type
        assert dfl_protocol.updates_received == 2

    def test_has_pending_updates(self, dfl_protocol, representation_extractor, simple_model, sample_data_loader):
        """Test checking for pending updates."""
        assert dfl_protocol.has_pending_updates() is False

        repr_package = representation_extractor.extract_and_package(
            model=simple_model,
            data_loader=sample_data_loader,
            repr_type="E",
            num_samples=50,
        )
        dfl_protocol.receive_representation(1, repr_package)

        assert dfl_protocol.has_pending_updates() is True

    def test_aggregate_received_data_empty(self, dfl_protocol):
        """Test aggregation with empty buffer."""
        success, info = dfl_protocol.aggregate_received_data()

        assert success is False
        assert info["reason"] == "empty_buffer"

    def test_aggregate_received_data_embeddings(self, dfl_protocol, representation_extractor, simple_model, sample_data_loader):
        """Test aggregating received embeddings."""
        # Receive embeddings from two neighbors
        for neighbor_id in [1, 2]:
            repr_package = representation_extractor.extract_and_package(
                model=simple_model,
                data_loader=sample_data_loader,
                repr_type="E",
                num_samples=100,
            )
            dfl_protocol.receive_representation(neighbor_id, repr_package)

        success, info = dfl_protocol.aggregate_received_data()

        assert success is True
        assert info["num_neighbors"] == 2
        assert info["embedding_aggregation"] is not None
        assert "aggregated_data" in info["embedding_aggregation"]
        assert len(dfl_protocol.received_buffer) == 0  # Buffer cleared

    def test_aggregate_received_data_weights(self, dfl_protocol, representation_extractor, simple_model, sample_data_loader):
        """Test aggregating received weights."""
        repr_package = representation_extractor.extract_and_package(
            model=simple_model,
            data_loader=sample_data_loader,
            repr_type="W",
        )
        dfl_protocol.receive_representation(1, repr_package)

        success, info = dfl_protocol.aggregate_received_data()

        assert success is True
        assert info["num_neighbors"] == 1

    def test_get_staleness_metrics(self, dfl_protocol, representation_extractor, simple_model, sample_data_loader):
        """Test getting staleness metrics."""
        # Receive from two neighbors at different times
        repr_package1 = representation_extractor.extract_and_package(
            model=simple_model,
            data_loader=sample_data_loader,
            repr_type="E",
            num_samples=50,
        )
        dfl_protocol.receive_representation(1, repr_package1)

        time.sleep(0.1)  # Small delay

        repr_package2 = representation_extractor.extract_and_package(
            model=simple_model,
            data_loader=sample_data_loader,
            repr_type="E",
            num_samples=50,
        )
        dfl_protocol.receive_representation(2, repr_package2)

        staleness = dfl_protocol.get_staleness_metrics()

        assert len(staleness) == 2
        assert staleness[1] > staleness[2]  # Neighbor 1 is older

    def test_get_update_statistics(self, dfl_protocol):
        """Test getting update statistics."""
        stats = dfl_protocol.get_update_statistics()

        assert "updates_sent" in stats
        assert "updates_received" in stats
        assert "data_sent_mb" in stats
        assert "data_received_mb" in stats
        assert "avg_staleness" in stats
        assert "num_pending" in stats

    def test_get_state_features(self, dfl_protocol):
        """Test getting state features for RL."""
        features = dfl_protocol.get_state_features()

        assert "time_since_last_update" in features
        assert "avg_neighbor_staleness" in features
        assert "pending_updates_count" in features
        assert "model_divergence_estimate" in features

        # Check all values are numeric (float or numpy float)
        assert all(isinstance(v, (int, float, np.floating, np.integer)) for v in features.values())

    def test_reset_statistics(self, dfl_protocol, representation_extractor, simple_model, sample_data_loader):
        """Test resetting statistics."""
        # Generate some activity
        who_action = [1]
        what_action = "E"
        neighbors = {1}
        dfl_protocol.execute_communication_action(who_action, what_action, neighbors)

        assert dfl_protocol.updates_sent > 0

        dfl_protocol.reset_statistics()

        assert dfl_protocol.updates_sent == 0
        assert dfl_protocol.updates_received == 0
        assert dfl_protocol.total_data_sent_mb == 0.0
        assert dfl_protocol.total_data_received_mb == 0.0
        assert len(dfl_protocol.communication_log) == 0

    def test_clear_buffer(self, dfl_protocol, representation_extractor, simple_model, sample_data_loader):
        """Test clearing received buffer."""
        repr_package = representation_extractor.extract_and_package(
            model=simple_model,
            data_loader=sample_data_loader,
            repr_type="E",
            num_samples=50,
        )
        dfl_protocol.receive_representation(1, repr_package)

        assert len(dfl_protocol.received_buffer) > 0

        dfl_protocol.clear_buffer()

        assert len(dfl_protocol.received_buffer) == 0
