from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data.preprocessing import TrajectoryNormalizer, create_train_val_test_split
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class TrajectoryWindowGenerator:
    """Generate sliding windows from vehicle trajectories."""

    def __init__(
        self,
        window_size: int = 50,
        stride: int = 10,
        features: list = None,
    ):
        """Initialize window generator.

        Args:
            window_size: Number of timesteps per window
            stride: Stride for sliding window
            features: List of feature names to extract
        """
        self.window_size = window_size
        self.stride = stride
        self.features = features or []

    def generate_windows(
        self,
        data: pd.DataFrame,
        vehicle_id_col: str = "vehicle_id",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sliding windows from trajectory data.

        Args:
            data: DataFrame with trajectory data
            vehicle_id_col: Column name for vehicle ID

        Returns:
            Tuple of (windows, vehicle_ids) where windows has shape
            (num_windows, window_size, num_features)
        """
        if not self.features:
            raise ValueError("No features specified for window generation")

        missing_features = [f for f in self.features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")

        windows_list = []
        vehicle_ids_list = []

        for vehicle_id, group in data.groupby(vehicle_id_col):
            group_sorted = group.sort_values("timestamp")
            feature_data = group_sorted[self.features].values

            vehicle_windows = self._create_windows_for_vehicle(feature_data)

            if len(vehicle_windows) > 0:
                windows_list.append(vehicle_windows)
                vehicle_ids_list.extend([vehicle_id] * len(vehicle_windows))

        if not windows_list:
            raise ValueError("No windows generated from data")

        windows = np.concatenate(windows_list, axis=0)
        vehicle_ids = np.array(vehicle_ids_list)

        logger.info(
            f"Generated {len(windows)} windows from {len(data[vehicle_id_col].unique())} vehicles"
        )
        logger.info(f"Window shape: {windows.shape}")

        return windows, vehicle_ids

    def _create_windows_for_vehicle(self, data: np.ndarray) -> np.ndarray:
        """Create sliding windows for a single vehicle.

        Args:
            data: Feature array of shape (num_timesteps, num_features)

        Returns:
            Windows array of shape (num_windows, window_size, num_features)
        """
        num_timesteps = len(data)

        if num_timesteps < self.window_size:
            logger.warning(
                f"Vehicle has only {num_timesteps} timesteps, "
                f"less than window_size={self.window_size}. Skipping."
            )
            return np.array([])

        windows = []
        for start_idx in range(0, num_timesteps - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            window = data[start_idx:end_idx]
            windows.append(window)

        return np.array(windows)

    def compute_expected_windows_per_vehicle(
        self,
        data: pd.DataFrame,
        vehicle_id_col: str = "vehicle_id",
    ) -> Dict[int, int]:
        """Compute expected number of windows for each vehicle.

        Args:
            data: DataFrame with trajectory data
            vehicle_id_col: Column name for vehicle ID

        Returns:
            Dictionary mapping vehicle_id to expected window count
        """
        expected_windows = {}

        for vehicle_id, group in data.groupby(vehicle_id_col):
            num_timesteps = len(group)
            if num_timesteps >= self.window_size:
                num_windows = (num_timesteps - self.window_size) // self.stride + 1
            else:
                num_windows = 0
            expected_windows[int(vehicle_id)] = num_windows

        return expected_windows


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
) -> np.ndarray:
    """Create sliding windows from trajectory data.

    Helper function for creating windows from a single vehicle's data.

    Args:
        data: Feature array of shape (num_timesteps, num_features)
        window_size: Number of timesteps per window
        stride: Stride for sliding window (default: 1)

    Returns:
        Windows array of shape (num_windows, window_size, num_features)

    Raises:
        ValueError: If data has fewer timesteps than window_size
    """
    num_timesteps = len(data)

    if num_timesteps < window_size:
        raise ValueError(
            f"Data has {num_timesteps} timesteps, less than window_size={window_size}"
        )

    windows = []
    for start_idx in range(0, num_timesteps - window_size + 1, stride):
        end_idx = start_idx + window_size
        window = data[start_idx:end_idx]
        windows.append(window)

    return np.array(windows)


def create_preprocessed_dataset(
    config: DictConfig,
    data: pd.DataFrame,
    save_dir: Path = None,
) -> Dict[str, np.ndarray]:
    """Create complete preprocessed dataset with windows, splits, and normalization.

    Args:
        config: Hydra configuration
        data: Raw trajectory data
        save_dir: Optional directory to save preprocessed data

    Returns:
        Dictionary containing all preprocessed data
    """
    logger.info("Creating preprocessed dataset")

    window_gen = TrajectoryWindowGenerator(
        window_size=config.data.window_size,
        stride=config.data.stride,
        features=config.data.features,
    )

    windows, vehicle_ids = window_gen.generate_windows(data)

    data_splits, metadata_splits = create_train_val_test_split(
        windows=windows,
        vehicle_ids=vehicle_ids,
        train_ratio=config.data.split_ratios.train,
        val_ratio=config.data.split_ratios.val,
        test_ratio=config.data.split_ratios.test,
        random_state=config.seed,
    )

    if config.data.preprocessing.normalize:
        logger.info("Normalizing data")
        normalizer = TrajectoryNormalizer(method=config.data.preprocessing.normalization_method)

        data_splits["train"] = normalizer.fit_transform(data_splits["train"])

        if len(data_splits["val"]) > 0:
            data_splits["val"] = normalizer.transform(data_splits["val"])

        if len(data_splits["test"]) > 0:
            data_splits["test"] = normalizer.transform(data_splits["test"])

        norm_params = normalizer.get_params()
    else:
        norm_params = None

    dataset = {
        "train_data": data_splits["train"],
        "val_data": data_splits["val"],
        "test_data": data_splits["test"],
        "train_vehicle_ids": metadata_splits["train"],
        "val_vehicle_ids": metadata_splits["val"],
        "test_vehicle_ids": metadata_splits["test"],
        "config": {
            "window_size": config.data.window_size,
            "stride": config.data.stride,
            "features": config.data.features,
            "num_features": len(config.data.features),
            "seed": config.seed,
        },
        "normalization": norm_params,
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "preprocessed_data.npz"

        np.savez_compressed(save_path, **dataset)
        logger.info(f"Saved preprocessed dataset to {save_path}")

    logger.info("Dataset created successfully:")
    logger.info(f"  Train: {dataset['train_data'].shape}")
    logger.info(f"  Val: {dataset['val_data'].shape}")
    logger.info(f"  Test: {dataset['test_data'].shape}")

    return dataset


def load_preprocessed_dataset(file_path: Path) -> Dict[str, np.ndarray]:
    """Load preprocessed dataset from file.

    Args:
        file_path: Path to .npz file

    Returns:
        Dictionary containing preprocessed data
    """
    logger.info(f"Loading preprocessed dataset from {file_path}")

    data = np.load(file_path, allow_pickle=True)

    dataset = {key: data[key] for key in data.keys()}

    logger.info("Loaded dataset:")
    logger.info(f"  Train: {dataset['train_data'].shape}")
    logger.info(f"  Val: {dataset['val_data'].shape}")
    logger.info(f"  Test: {dataset['test_data'].shape}")

    return dataset
