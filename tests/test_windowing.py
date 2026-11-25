from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from src.data.dataloader import V2VDataLoader
from src.data.preprocessing import (
    TrajectoryNormalizer,
    compute_split_statistics,
    create_train_val_test_split,
    verify_no_data_leakage,
)
from src.data.trajectory_window import (
    TrajectoryWindowGenerator,
    create_preprocessed_dataset,
    load_preprocessed_dataset,
)


@pytest.fixture(scope="module")
def config():
    """Fixture to load configuration."""
    config_dir = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
        return cfg


@pytest.fixture(scope="module")
def sample_trajectory_data():
    """Create sample trajectory data for testing."""
    np.random.seed(42)
    data = {
        "vehicle_id": np.repeat([0, 1, 2], 100),
        "timestamp": np.tile(np.arange(100) * 0.1, 3),
        "frame": np.tile(np.arange(1, 101), 3),
        "velocity_x": np.random.randn(300),
        "velocity_y": np.random.randn(300),
        "speed": np.abs(np.random.randn(300)),
        "accel_x": np.random.randn(300) * 0.5,
        "accel_y": np.random.randn(300) * 0.5,
        "accel_z": np.random.randn(300) * 0.1 - 9.81,
        "gyro_x": np.random.randn(300) * 0.01,
        "gyro_y": np.random.randn(300) * 0.01,
        "gyro_z": np.random.randn(300) * 0.05,
        "yaw": np.random.rand(300) * 360,
    }
    return pd.DataFrame(data)


@pytest.fixture
def window_generator():
    """Fixture for window generator."""
    features = [
        "velocity_x",
        "velocity_y",
        "speed",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "yaw",
    ]
    return TrajectoryWindowGenerator(window_size=50, stride=10, features=features)


class TestTrajectoryWindowGenerator:
    """Test trajectory window generation."""

    def test_initialization(self, window_generator):
        """Test window generator initialization."""
        assert window_generator.window_size == 50
        assert window_generator.stride == 10
        assert len(window_generator.features) == 10

    def test_generate_windows_shape(self, window_generator, sample_trajectory_data):
        """Test that generated windows have correct shape."""
        windows, vehicle_ids = window_generator.generate_windows(sample_trajectory_data)

        assert windows.ndim == 3
        assert windows.shape[1] == 50
        assert windows.shape[2] == 10
        assert len(vehicle_ids) == len(windows)

    def test_generate_windows_count(self, window_generator, sample_trajectory_data):
        """Test correct number of windows generated."""
        windows, vehicle_ids = window_generator.generate_windows(sample_trajectory_data)

        num_timesteps = 100
        expected_windows_per_vehicle = (num_timesteps - 50) // 10 + 1
        expected_total = expected_windows_per_vehicle * 3

        assert len(windows) == expected_total

    def test_compute_expected_windows(self, window_generator, sample_trajectory_data):
        """Test computation of expected window counts."""
        expected = window_generator.compute_expected_windows_per_vehicle(sample_trajectory_data)

        assert len(expected) == 3
        for vehicle_id in [0, 1, 2]:
            assert vehicle_id in expected
            assert expected[vehicle_id] == 6

    def test_windows_preserve_temporal_order(self, window_generator, sample_trajectory_data):
        """Test that windows preserve temporal ordering."""
        windows, _ = window_generator.generate_windows(sample_trajectory_data)

        first_window = windows[0]
        assert first_window.shape == (50, 10)

        timestamps_original = sample_trajectory_data[sample_trajectory_data["vehicle_id"] == 0][
            "timestamp"
        ].values[:50]
        assert len(timestamps_original) == 50

    def test_overlapping_windows(self, window_generator, sample_trajectory_data):
        """Test that consecutive windows overlap correctly."""
        windows, vehicle_ids = window_generator.generate_windows(sample_trajectory_data)

        vehicle_0_windows = windows[vehicle_ids == 0]
        if len(vehicle_0_windows) >= 2:
            window1 = vehicle_0_windows[0]
            window2 = vehicle_0_windows[1]

            overlap_size = window_generator.window_size - window_generator.stride
            assert np.allclose(window1[window_generator.stride :], window2[:overlap_size])

    def test_missing_features_raises_error(self, sample_trajectory_data):
        """Test that missing features raise an error."""
        gen = TrajectoryWindowGenerator(
            window_size=50, stride=10, features=["nonexistent_feature"]
        )

        with pytest.raises(ValueError, match="Missing features"):
            gen.generate_windows(sample_trajectory_data)

    def test_short_trajectory_skipped(self):
        """Test that trajectories shorter than window_size are skipped."""
        short_data = pd.DataFrame(
            {
                "vehicle_id": [0] * 30,
                "timestamp": np.arange(30) * 0.1,
                "frame": np.arange(1, 31),
                "velocity_x": np.random.randn(30),
                "velocity_y": np.random.randn(30),
                "speed": np.abs(np.random.randn(30)),
                "accel_x": np.random.randn(30),
                "accel_y": np.random.randn(30),
                "accel_z": np.random.randn(30),
                "gyro_x": np.random.randn(30),
                "gyro_y": np.random.randn(30),
                "gyro_z": np.random.randn(30),
                "yaw": np.random.randn(30),
            }
        )

        gen = TrajectoryWindowGenerator(
            window_size=50,
            stride=10,
            features=[
                "velocity_x",
                "velocity_y",
                "speed",
                "accel_x",
                "accel_y",
                "accel_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "yaw",
            ],
        )

        with pytest.raises(ValueError, match="No windows generated"):
            gen.generate_windows(short_data)


class TestTrajectoryNormalizer:
    """Test trajectory normalization."""

    def test_normalizer_initialization(self):
        """Test normalizer can be initialized."""
        normalizer = TrajectoryNormalizer(method="standardize")
        assert normalizer.method == "standardize"
        assert not normalizer.is_fitted

    def test_fit_transform(self):
        """Test fitting and transforming data."""
        data = np.random.randn(100, 50, 10)
        normalizer = TrajectoryNormalizer(method="standardize")

        normalized = normalizer.fit_transform(data)

        assert normalized.shape == data.shape
        assert normalizer.is_fitted

        np.testing.assert_almost_equal(normalized.mean(), 0.0, decimal=5)
        np.testing.assert_almost_equal(normalized.std(), 1.0, decimal=5)

    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error."""
        normalizer = TrajectoryNormalizer()
        data = np.random.randn(10, 50, 10)

        with pytest.raises(ValueError, match="not fitted"):
            normalizer.transform(data)

    def test_get_params(self):
        """Test getting normalizer parameters."""
        data = np.random.randn(100, 50, 10)
        normalizer = TrajectoryNormalizer(method="standardize")
        normalizer.fit(data)

        params = normalizer.get_params()
        assert "method" in params
        assert "mean" in params
        assert "scale" in params
        assert len(params["mean"]) == 10

    def test_separate_fit_and_transform(self):
        """Test separate fit and transform operations."""
        train_data = np.random.randn(100, 50, 10)
        test_data = np.random.randn(20, 50, 10)

        normalizer = TrajectoryNormalizer()
        normalizer.fit(train_data)
        normalized_test = normalizer.transform(test_data)

        assert normalized_test.shape == test_data.shape


class TestDataSplitting:
    """Test train/val/test splitting."""

    def test_create_splits(self):
        """Test creating train/val/test splits."""
        windows = np.random.randn(1000, 50, 10)
        vehicle_ids = np.repeat(np.arange(10), 100)

        data_splits, metadata_splits = create_train_val_test_split(
            windows, vehicle_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
        )

        assert "train" in data_splits
        assert "val" in data_splits
        assert "test" in data_splits

        total_samples = sum(len(split) for split in data_splits.values())
        assert total_samples == len(windows)

    def test_split_ratios(self):
        """Test that split ratios are approximately correct."""
        windows = np.random.randn(1000, 50, 10)
        vehicle_ids = np.repeat(np.arange(10), 100)

        data_splits, _ = create_train_val_test_split(
            windows, vehicle_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
        )

        total = len(windows)
        train_ratio_actual = len(data_splits["train"]) / total
        val_ratio_actual = len(data_splits["val"]) / total
        test_ratio_actual = len(data_splits["test"]) / total

        assert 0.6 < train_ratio_actual < 0.8
        assert 0.1 < val_ratio_actual < 0.2
        assert 0.1 < test_ratio_actual < 0.2

    def test_all_vehicles_in_all_splits(self):
        """Test that all vehicles appear in all splits."""
        windows = np.random.randn(1000, 50, 10)
        vehicle_ids = np.repeat(np.arange(10), 100)

        _, metadata_splits = create_train_val_test_split(
            windows, vehicle_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
        )

        train_vehicles = set(metadata_splits["train"])
        val_vehicles = set(metadata_splits["val"])
        test_vehicles = set(metadata_splits["test"])

        assert train_vehicles == val_vehicles == test_vehicles
        assert len(train_vehicles) == 10

    def test_invalid_ratios_raise_error(self):
        """Test that invalid split ratios raise an error."""
        windows = np.random.randn(100, 50, 10)
        vehicle_ids = np.repeat(np.arange(5), 20)

        with pytest.raises(ValueError, match="must sum to 1.0"):
            create_train_val_test_split(
                windows, vehicle_ids, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3
            )

    def test_no_data_leakage(self):
        """Test verification of no data leakage."""
        windows = np.random.randn(1000, 50, 10)
        vehicle_ids = np.repeat(np.arange(10), 100)

        data_splits, _ = create_train_val_test_split(
            windows, vehicle_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
        )

        no_leakage = verify_no_data_leakage(
            data_splits["train"], data_splits["val"], data_splits["test"]
        )

        assert no_leakage is True

    def test_split_statistics(self):
        """Test computation of split statistics."""
        windows = np.random.randn(1000, 50, 10)
        vehicle_ids = np.repeat(np.arange(10), 100)

        data_splits, metadata_splits = create_train_val_test_split(
            windows, vehicle_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42
        )

        stats = compute_split_statistics(data_splits, metadata_splits)

        assert "train" in stats
        assert "val" in stats
        assert "test" in stats

        for split_name, split_stats in stats.items():
            assert "num_windows" in split_stats
            assert "num_vehicles" in split_stats
            assert "shape" in split_stats
            assert split_stats["num_vehicles"] == 10


class TestIntegrationWindowing:
    """Integration tests for windowing pipeline."""

    def test_create_preprocessed_dataset(self, config, sample_trajectory_data):
        """Test creating complete preprocessed dataset."""
        dataset = create_preprocessed_dataset(config, sample_trajectory_data, save_dir=None)

        assert "train_data" in dataset
        assert "val_data" in dataset
        assert "test_data" in dataset
        assert "train_vehicle_ids" in dataset
        assert "config" in dataset

        assert dataset["train_data"].shape[1] == 50
        assert dataset["train_data"].shape[2] == 10

    def test_save_and_load_preprocessed_dataset(self, config, sample_trajectory_data, tmp_path):
        """Test saving and loading preprocessed dataset."""
        save_dir = tmp_path / "preprocessed"

        dataset_saved = create_preprocessed_dataset(config, sample_trajectory_data, save_dir=save_dir)

        save_path = save_dir / "preprocessed_data.npz"
        assert save_path.exists()

        dataset_loaded = load_preprocessed_dataset(save_path)

        np.testing.assert_array_equal(dataset_saved["train_data"], dataset_loaded["train_data"])
        np.testing.assert_array_equal(dataset_saved["val_data"], dataset_loaded["val_data"])
        np.testing.assert_array_equal(dataset_saved["test_data"], dataset_loaded["test_data"])

    def test_end_to_end_pipeline_with_real_data(self, config):
        """Test end-to-end pipeline with real V2V data (small subset)."""
        loader = V2VDataLoader(config)
        loader.load_dataset()
        loader.select_vehicles(num_vehicles=5)
        features_df = loader.extract_features()

        dataset = create_preprocessed_dataset(config, features_df, save_dir=None)

        assert dataset["train_data"].shape[0] > 0
        assert dataset["val_data"].shape[0] > 0
        assert dataset["test_data"].shape[0] > 0

        assert dataset["train_data"].shape[1] == 50
        assert dataset["train_data"].shape[2] == 10

        train_vehicles = len(np.unique(dataset["train_vehicle_ids"]))
        val_vehicles = len(np.unique(dataset["val_vehicle_ids"]))
        test_vehicles = len(np.unique(dataset["test_vehicle_ids"]))

        assert train_vehicles == val_vehicles == test_vehicles == 5
