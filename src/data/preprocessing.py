from typing import Dict, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class TrajectoryNormalizer:
    """Normalizer for trajectory features."""

    def __init__(self, method: str = "standardize"):
        """Initialize normalizer.

        Args:
            method: Normalization method ("standardize" or "minmax")
        """
        self.method = method
        self.scaler = StandardScaler() if method == "standardize" else None
        self.is_fitted = False

    def fit(self, data: np.ndarray) -> "TrajectoryNormalizer":
        """Fit normalizer to training data.

        Args:
            data: Training data of shape (num_samples, window_size, num_features)

        Returns:
            Self for method chaining
        """
        original_shape = data.shape
        data_reshaped = data.reshape(-1, data.shape[-1])

        if self.method == "standardize":
            self.scaler.fit(data_reshaped)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        self.is_fitted = True
        logger.info(f"Fitted {self.method} normalizer on data shape {original_shape}")
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted normalizer.

        Args:
            data: Data of shape (num_samples, window_size, num_features)

        Returns:
            Normalized data with same shape
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        original_shape = data.shape
        data_reshaped = data.reshape(-1, data.shape[-1])

        if self.method == "standardize":
            normalized = self.scaler.transform(data_reshaped)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        return normalized.reshape(original_shape)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit normalizer and transform data.

        Args:
            data: Data of shape (num_samples, window_size, num_features)

        Returns:
            Normalized data
        """
        self.fit(data)
        return self.transform(data)

    def get_params(self) -> Dict:
        """Get normalizer parameters for saving/loading.

        Returns:
            Dictionary of normalizer parameters
        """
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted")

        if self.method == "standardize":
            return {
                "method": self.method,
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
            }
        return {"method": self.method}


def create_train_val_test_split(
    windows: np.ndarray,
    vehicle_ids: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Create stratified train/val/test splits ensuring vehicles appear in all splits.

    Args:
        windows: Trajectory windows of shape (num_windows, window_size, num_features)
        vehicle_ids: Vehicle ID for each window
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (data_splits, metadata_splits)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    unique_vehicles = np.unique(vehicle_ids)
    logger.info(f"Creating splits for {len(unique_vehicles)} vehicles, {len(windows)} total windows")

    train_windows, train_ids = [], []
    val_windows, val_ids = [], []
    test_windows, test_ids = [], []

    for vehicle_id in unique_vehicles:
        mask = vehicle_ids == vehicle_id
        vehicle_windows = windows[mask]
        vehicle_window_ids = vehicle_ids[mask]

        n_samples = len(vehicle_windows)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        indices = np.random.RandomState(random_state).permutation(n_samples)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train_windows.append(vehicle_windows[train_idx])
        train_ids.append(vehicle_window_ids[train_idx])

        val_windows.append(vehicle_windows[val_idx])
        val_ids.append(vehicle_window_ids[val_idx])

        test_windows.append(vehicle_windows[test_idx])
        test_ids.append(vehicle_window_ids[test_idx])

    data_splits = {
        "train": np.concatenate(train_windows, axis=0),
        "val": np.concatenate(val_windows, axis=0),
        "test": np.concatenate(test_windows, axis=0),
    }

    metadata_splits = {
        "train": np.concatenate(train_ids, axis=0),
        "val": np.concatenate(val_ids, axis=0),
        "test": np.concatenate(test_ids, axis=0),
    }

    logger.info(f"Split sizes - Train: {len(data_splits['train'])}, "
                f"Val: {len(data_splits['val'])}, Test: {len(data_splits['test'])}")

    for split_name, vehicle_ids_split in metadata_splits.items():
        unique_in_split = len(np.unique(vehicle_ids_split))
        logger.info(f"{split_name.capitalize()} split: {unique_in_split} unique vehicles")

    return data_splits, metadata_splits


def verify_no_data_leakage(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    tolerance: float = 1e-10,
) -> bool:
    """Verify there's no data leakage between splits.

    Args:
        train_data: Training windows
        val_data: Validation windows
        test_data: Test windows
        tolerance: Numerical tolerance for comparison

    Returns:
        True if no leakage detected, False otherwise
    """
    train_flat = train_data.reshape(len(train_data), -1)
    val_flat = val_data.reshape(len(val_data), -1)
    test_flat = test_data.reshape(len(test_data), -1)

    for i, train_sample in enumerate(train_flat):
        if np.any(np.all(np.abs(val_flat - train_sample) < tolerance, axis=1)):
            logger.warning(f"Data leakage detected: train sample {i} found in validation set")
            return False
        if np.any(np.all(np.abs(test_flat - train_sample) < tolerance, axis=1)):
            logger.warning(f"Data leakage detected: train sample {i} found in test set")
            return False

    logger.info("No data leakage detected between splits")
    return True


def compute_split_statistics(
    data_splits: Dict[str, np.ndarray],
    metadata_splits: Dict[str, np.ndarray],
) -> Dict[str, Dict]:
    """Compute statistics for each data split.

    Args:
        data_splits: Dictionary of split data
        metadata_splits: Dictionary of split metadata (vehicle IDs)

    Returns:
        Dictionary of statistics for each split
    """
    statistics = {}

    for split_name in ["train", "val", "test"]:
        data = data_splits[split_name]
        vehicle_ids = metadata_splits[split_name]

        stats = {
            "num_windows": len(data),
            "num_vehicles": len(np.unique(vehicle_ids)),
            "shape": data.shape,
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }
        statistics[split_name] = stats

    return statistics
