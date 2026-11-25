import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data.schemas import DatasetMetadata, VehicleStats
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class V2VDataLoader:
    """Data loader for V2V trajectory dataset."""

    def __init__(self, config: DictConfig):
        """Initialize data loader with configuration.

        Args:
            config: Hydra configuration object
        """
        self.config = config
        self.data_path = Path(config.data.dataset_path)
        self.metadata_path = Path(config.data.metadata_path)
        self.features = config.data.features
        self.min_presence_duration = config.data.min_presence_duration

        self.df: pd.DataFrame = None
        self.metadata: DatasetMetadata = None
        self.selected_vehicles: List[int] = None

    def load_dataset(self) -> pd.DataFrame:
        """Load full dataset from CSV.

        Returns:
            DataFrame containing all vehicle records
        """
        logger.info(f"Loading dataset from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.df):,} records from {self.df['vehicle_id'].nunique()} vehicles")

        return self.df

    def load_metadata(self) -> DatasetMetadata:
        """Load and validate dataset metadata.

        Returns:
            Validated metadata object
        """
        logger.info(f"Loading metadata from {self.metadata_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")

        with open(self.metadata_path) as f:
            metadata_dict = json.load(f)

        self.metadata = DatasetMetadata(**metadata_dict)
        logger.info(f"Metadata validated: {self.metadata.total_records:,} records, "
                    f"{self.metadata.unique_vehicles} vehicles, "
                    f"{self.metadata.duration_seconds:.1f}s duration")

        return self.metadata

    def compute_vehicle_statistics(self) -> Dict[int, VehicleStats]:
        """Compute statistics for each vehicle in the dataset.

        Returns:
            Dictionary mapping vehicle_id to VehicleStats
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        logger.info("Computing vehicle statistics")
        vehicle_stats = {}

        for vehicle_id, group in self.df.groupby("vehicle_id"):
            group_sorted = group.sort_values("timestamp")

            num_records = len(group)
            presence_duration = group_sorted["timestamp"].max() - group_sorted["timestamp"].min()

            avg_speed = group["speed"].mean()
            max_speed = group["speed"].max()

            dx = group_sorted["x"].diff().fillna(0)
            dy = group_sorted["y"].diff().fillna(0)
            trajectory_length = np.sqrt(dx**2 + dy**2).sum()

            stats = VehicleStats(
                vehicle_id=int(vehicle_id),
                num_records=num_records,
                presence_duration=float(presence_duration),
                avg_speed=float(avg_speed),
                max_speed=float(max_speed),
                trajectory_length=float(trajectory_length),
            )
            vehicle_stats[int(vehicle_id)] = stats

        logger.info(f"Computed statistics for {len(vehicle_stats)} vehicles")
        return vehicle_stats

    def select_vehicles(self, num_vehicles: int = None) -> List[int]:
        """Select top vehicles by record count with presence duration filter.

        Args:
            num_vehicles: Number of vehicles to select (default from config)

        Returns:
            List of selected vehicle IDs (sorted by record count, descending)
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        if num_vehicles is None:
            num_vehicles = self.config.data.num_vehicles

        logger.info(f"Selecting top {num_vehicles} vehicles by record count (min presence >{self.min_presence_duration}s)")

        vehicle_stats = self.compute_vehicle_statistics()

        eligible_vehicles = [
            (vid, stats) for vid, stats in vehicle_stats.items()
            if stats.presence_duration > self.min_presence_duration
        ]

        if len(eligible_vehicles) < num_vehicles:
            logger.warning(
                f"Only {len(eligible_vehicles)} vehicles meet criteria "
                f"(presence >{self.min_presence_duration}s), but {num_vehicles} requested"
            )

        eligible_vehicles.sort(key=lambda x: x[1].num_records, reverse=True)
        self.selected_vehicles = [vid for vid, _ in eligible_vehicles[:num_vehicles]]

        total_records = sum(vehicle_stats[vid].num_records for vid in self.selected_vehicles)
        logger.info(f"Selected top {len(self.selected_vehicles)} vehicles by record count (total: {total_records:,} records)")
        logger.info(f"Selected vehicle IDs: {self.selected_vehicles[:5]}...")

        return self.selected_vehicles

    def extract_features(self, vehicle_ids: List[int] = None) -> pd.DataFrame:
        """Extract specified features for selected vehicles.

        Args:
            vehicle_ids: List of vehicle IDs to extract (default: selected vehicles)

        Returns:
            DataFrame with extracted features
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        if vehicle_ids is None:
            if self.selected_vehicles is None:
                raise ValueError("No vehicles selected. Call select_vehicles() first.")
            vehicle_ids = self.selected_vehicles

        logger.info(f"Extracting {len(self.features)} features for {len(vehicle_ids)} vehicles")

        required_cols = ["vehicle_id", "timestamp", "frame"] + self.features
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        filtered_df = self.df[self.df["vehicle_id"].isin(vehicle_ids)][required_cols].copy()
        filtered_df = filtered_df.sort_values(["vehicle_id", "timestamp"])

        logger.info(f"Extracted {len(filtered_df):,} records with {len(self.features)} features")

        return filtered_df

    def get_data_summary(self) -> Dict:
        """Get summary statistics of loaded data.

        Returns:
            Dictionary containing summary statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        summary = {
            "total_records": len(self.df),
            "unique_vehicles": self.df["vehicle_id"].nunique(),
            "time_range": (self.df["timestamp"].min(), self.df["timestamp"].max()),
            "duration_seconds": self.df["timestamp"].max() - self.df["timestamp"].min(),
            "features": self.features,
            "num_features": len(self.features),
        }

        if self.selected_vehicles is not None:
            summary["selected_vehicles"] = len(self.selected_vehicles)
            summary["selected_vehicle_ids"] = self.selected_vehicles

        return summary
