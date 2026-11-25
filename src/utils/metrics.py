from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_reconstruction_metrics(
    reconstruction_errors: np.ndarray,
    labels: np.ndarray,
    threshold: float = None,
) -> Dict[str, float]:
    """Compute anomaly detection metrics from reconstruction errors.

    Args:
        reconstruction_errors: Reconstruction errors per sample
        labels: Binary labels (1 for anomaly, 0 for normal)
        threshold: Classification threshold (if None, use median)

    Returns:
        Dictionary of metrics
    """
    if threshold is None:
        threshold = np.median(reconstruction_errors)

    predictions = (reconstruction_errors > threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }

    if len(np.unique(labels)) > 1:
        metrics["auc"] = float(roc_auc_score(labels, reconstruction_errors))
    else:
        metrics["auc"] = 0.0

    return metrics


def find_best_threshold(
    reconstruction_errors: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
) -> float:
    """Find optimal threshold for anomaly detection.

    Args:
        reconstruction_errors: Reconstruction errors per sample
        labels: Binary labels
        metric: Metric to optimize ("f1", "precision", "recall")

    Returns:
        Optimal threshold value
    """
    if len(np.unique(labels)) < 2:
        return float(np.median(reconstruction_errors))

    precision, recall, thresholds = precision_recall_curve(labels, reconstruction_errors)

    if metric == "f1":
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
    elif metric == "precision":
        best_idx = np.argmax(precision)
    elif metric == "recall":
        best_idx = np.argmax(recall)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return float(np.median(reconstruction_errors))


def compute_batch_loss(
    model: torch.nn.Module,
    batch: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    """Compute loss for a batch.

    Args:
        model: Autoencoder model
        batch: Input batch
        criterion: Loss function

    Returns:
        Loss tensor
    """
    reconstruction, _ = model(batch)
    loss = criterion(reconstruction, batch)
    return loss


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """Update statistics with new value.

        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsTracker:
    """Track multiple metrics over training."""

    def __init__(self):
        self.metrics = {}

    def update(self, metrics_dict: Dict[str, float], epoch: int):
        """Update metrics for current epoch.

        Args:
            metrics_dict: Dictionary of metric values
            epoch: Current epoch number
        """
        for name, value in metrics_dict.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append((epoch, value))

    def get_metric(self, name: str) -> list:
        """Get all values for a specific metric.

        Args:
            name: Metric name

        Returns:
            List of (epoch, value) tuples
        """
        return self.metrics.get(name, [])

    def get_latest(self, name: str) -> float:
        """Get latest value for a metric.

        Args:
            name: Metric name

        Returns:
            Latest metric value or 0.0 if not found
        """
        values = self.metrics.get(name, [])
        return values[-1][1] if values else 0.0

    def get_best(self, name: str, mode: str = "min") -> tuple:
        """Get best value and epoch for a metric.

        Args:
            name: Metric name
            mode: "min" or "max"

        Returns:
            Tuple of (best_value, best_epoch)
        """
        values = self.metrics.get(name, [])
        if not values:
            return (float("inf") if mode == "min" else float("-inf"), -1)

        if mode == "min":
            best_epoch, best_value = min(values, key=lambda x: x[1])
        else:
            best_epoch, best_value = max(values, key=lambda x: x[1])

        return (best_value, best_epoch)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary format.

        Returns:
            Dictionary of all metrics
        """
        return dict(self.metrics)
