from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from src.data.dataloader import V2VDataLoader
from src.data.schemas import DatasetMetadata, TrajectoryFeatures, VehicleRecord, VehicleStats


@pytest.fixture(scope="module")
def config():
    """Fixture to load configuration."""
    config_dir = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
        return cfg


@pytest.fixture(scope="module")
def dataloader(config):
    """Fixture to create data loader instance."""
    return V2VDataLoader(config)


class TestSchemas:
    """Test Pydantic validation schemas."""

    def test_vehicle_record_valid(self):
        """Test valid vehicle record passes validation."""
        record = VehicleRecord(
            frame=1,
            timestamp=0.1,
            vehicle_id=0,
            latitude=41.385,
            longitude=2.180,
            altitude=50.0,
            x=613.6,
            y=-1.6,
            z=0.3,
            roll=0.0,
            pitch=0.0,
            yaw=90.0,
            velocity_x=0.0,
            velocity_y=13.12,
            velocity_z=0.0,
            speed=13.12,
            accel_x=0.0,
            accel_y=-0.65,
            accel_z=-9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            compass=1.55,
        )
        assert record.vehicle_id == 0
        assert record.speed >= 0.0

    def test_vehicle_record_negative_speed_fails(self):
        """Test that negative speed fails validation."""
        with pytest.raises(ValidationError):
            VehicleRecord(
                frame=1,
                timestamp=0.1,
                vehicle_id=0,
                latitude=41.385,
                longitude=2.180,
                altitude=50.0,
                x=613.6,
                y=-1.6,
                z=0.3,
                roll=0.0,
                pitch=0.0,
                yaw=90.0,
                velocity_x=0.0,
                velocity_y=13.12,
                velocity_z=0.0,
                speed=-5.0,
                accel_x=0.0,
                accel_y=-0.65,
                accel_z=-9.81,
                gyro_x=0.0,
                gyro_y=0.0,
                gyro_z=0.0,
                compass=1.55,
            )

    def test_trajectory_features_valid(self):
        """Test valid trajectory features."""
        features = TrajectoryFeatures(
            velocity_x=[1.0] * 50,
            velocity_y=[2.0] * 50,
            speed=[3.0] * 50,
            accel_x=[0.1] * 50,
            accel_y=[0.2] * 50,
            accel_z=[-9.81] * 50,
            gyro_x=[0.01] * 50,
            gyro_y=[0.01] * 50,
            gyro_z=[0.01] * 50,
            yaw=[90.0] * 50,
        )
        assert len(features.velocity_x) == 50
        assert len(features.speed) == 50

    def test_trajectory_features_empty_fails(self):
        """Test that empty feature sequences fail validation."""
        with pytest.raises(ValidationError):
            TrajectoryFeatures(
                velocity_x=[],
                velocity_y=[],
                speed=[],
                accel_x=[],
                accel_y=[],
                accel_z=[],
                gyro_x=[],
                gyro_y=[],
                gyro_z=[],
                yaw=[],
            )

    def test_dataset_metadata_valid(self):
        """Test valid dataset metadata."""
        metadata = DatasetMetadata(
            session_id="20251017_124247",
            total_records=4048814,
            unique_vehicles=3999,
            duration_seconds=3599.9,
            sampling_rate_hz=10,
            sensor_types=["GPS", "IMU", "Compass"],
            gps_noise_std_meters=2.0,
            imu_accel_noise_std=0.1,
            imu_gyro_noise_std=0.01,
        )
        assert metadata.total_records > 0
        assert metadata.sampling_rate_hz == 10

    def test_vehicle_stats_valid(self):
        """Test valid vehicle statistics."""
        stats = VehicleStats(
            vehicle_id=0,
            num_records=3585,
            presence_duration=358.4,
            avg_speed=9.5,
            max_speed=19.1,
            trajectory_length=3405.6,
        )
        assert stats.presence_duration > 0
        assert stats.avg_speed >= 0

    def test_vehicle_stats_invalid_presence(self):
        """Test that invalid presence duration fails."""
        with pytest.raises(ValidationError):
            VehicleStats(
                vehicle_id=0,
                num_records=100,
                presence_duration=0.0,
                avg_speed=10.0,
                max_speed=20.0,
                trajectory_length=1000.0,
            )


class TestDataLoader:
    """Test V2VDataLoader functionality."""

    def test_dataloader_initialization(self, config):
        """Test data loader can be initialized."""
        loader = V2VDataLoader(config)
        assert loader.config is not None
        assert loader.features == config.data.features
        assert len(loader.features) == 10

    def test_load_dataset(self, dataloader):
        """Test dataset loading."""
        df = dataloader.load_dataset()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "vehicle_id" in df.columns
        assert "timestamp" in df.columns

    def test_load_metadata(self, dataloader):
        """Test metadata loading and validation."""
        metadata = dataloader.load_metadata()
        assert isinstance(metadata, DatasetMetadata)
        assert metadata.total_records == 4048814
        assert metadata.unique_vehicles == 3999
        assert metadata.sampling_rate_hz == 10

    def test_compute_vehicle_statistics(self, dataloader):
        """Test vehicle statistics computation."""
        if dataloader.df is None:
            dataloader.load_dataset()

        stats = dataloader.compute_vehicle_statistics()
        assert isinstance(stats, dict)
        assert len(stats) > 0

        for vehicle_id, vehicle_stats in stats.items():
            assert isinstance(vehicle_stats, VehicleStats)
            assert vehicle_stats.vehicle_id == vehicle_id
            assert vehicle_stats.num_records > 0
            assert vehicle_stats.presence_duration > 0

    def test_select_vehicles(self, dataloader):
        """Test vehicle selection based on presence duration."""
        if dataloader.df is None:
            dataloader.load_dataset()

        selected = dataloader.select_vehicles(num_vehicles=20)
        assert isinstance(selected, list)
        assert len(selected) == 20
        assert all(isinstance(vid, int) for vid in selected)

        vehicle_stats = dataloader.compute_vehicle_statistics()
        for vehicle_id in selected:
            assert vehicle_stats[vehicle_id].presence_duration > dataloader.min_presence_duration

    def test_select_vehicles_sorted_by_duration(self, dataloader):
        """Test that selected vehicles are sorted by presence duration."""
        if dataloader.df is None:
            dataloader.load_dataset()

        selected = dataloader.select_vehicles(num_vehicles=10)
        vehicle_stats = dataloader.compute_vehicle_statistics()

        durations = [vehicle_stats[vid].presence_duration for vid in selected]
        assert durations == sorted(durations, reverse=True)

    def test_extract_features(self, dataloader):
        """Test feature extraction."""
        if dataloader.df is None:
            dataloader.load_dataset()

        if dataloader.selected_vehicles is None:
            dataloader.select_vehicles(num_vehicles=5)

        features_df = dataloader.extract_features()
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) > 0

        expected_cols = ["vehicle_id", "timestamp", "frame"] + dataloader.features
        for col in expected_cols:
            assert col in features_df.columns

        assert len(features_df.columns) == len(expected_cols)

    def test_extract_features_correct_vehicles(self, dataloader):
        """Test that extracted features contain only selected vehicles."""
        if dataloader.df is None:
            dataloader.load_dataset()

        selected = dataloader.select_vehicles(num_vehicles=3)
        features_df = dataloader.extract_features()

        extracted_vehicles = features_df["vehicle_id"].unique()
        assert set(extracted_vehicles) == set(selected)

    def test_extract_features_has_ten_features(self, dataloader):
        """Test that exactly 10 trajectory features are extracted."""
        if dataloader.df is None:
            dataloader.load_dataset()

        if dataloader.selected_vehicles is None:
            dataloader.select_vehicles(num_vehicles=2)

        features_df = dataloader.extract_features()

        feature_cols = [col for col in features_df.columns if col not in ["vehicle_id", "timestamp", "frame"]]
        assert len(feature_cols) == 10

        expected_features = ["velocity_x", "velocity_y", "speed", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "yaw"]
        assert set(feature_cols) == set(expected_features)

    def test_get_data_summary(self, dataloader):
        """Test data summary generation."""
        if dataloader.df is None:
            dataloader.load_dataset()

        summary = dataloader.get_data_summary()
        assert isinstance(summary, dict)
        assert "total_records" in summary
        assert "unique_vehicles" in summary
        assert "features" in summary
        assert summary["num_features"] == 10

    def test_data_sorted_by_timestamp(self, dataloader):
        """Test that extracted data is sorted by vehicle and timestamp."""
        if dataloader.df is None:
            dataloader.load_dataset()

        dataloader.select_vehicles(num_vehicles=2)
        features_df = dataloader.extract_features()

        for vehicle_id in features_df["vehicle_id"].unique():
            vehicle_data = features_df[features_df["vehicle_id"] == vehicle_id]
            timestamps = vehicle_data["timestamp"].values
            assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))


class TestIntegration:
    """Integration tests for full data loading pipeline."""

    def test_full_pipeline(self, config):
        """Test complete data loading pipeline."""
        loader = V2VDataLoader(config)

        loader.load_dataset()
        assert loader.df is not None

        loader.load_metadata()
        assert loader.metadata is not None

        selected_vehicles = loader.select_vehicles(num_vehicles=20)
        assert len(selected_vehicles) == 20

        features_df = loader.extract_features()
        assert len(features_df) > 0
        assert set(features_df["vehicle_id"].unique()) == set(selected_vehicles)

        summary = loader.get_data_summary()
        assert summary["selected_vehicles"] == 20
        assert summary["num_features"] == 10

    def test_pipeline_reproducibility(self, config):
        """Test that pipeline produces consistent results."""
        loader1 = V2VDataLoader(config)
        loader1.load_dataset()
        selected1 = loader1.select_vehicles(num_vehicles=10)
        features1 = loader1.extract_features()

        loader2 = V2VDataLoader(config)
        loader2.load_dataset()
        selected2 = loader2.select_vehicles(num_vehicles=10)
        features2 = loader2.extract_features()

        assert selected1 == selected2
        pd.testing.assert_frame_equal(features1, features2)
