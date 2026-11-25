"""
Anomaly detection module for trajectory reconstruction-based anomaly detection.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


class AnomalyDetector:
    """Reconstruction-based anomaly detector with automatic threshold tuning."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        threshold: Optional[float] = None,
    ):
        """Initialize anomaly detector.

        Args:
            model: Trained autoencoder model
            device: Device for computation
            threshold: Fixed anomaly threshold (if None, needs tuning)
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.eval()

    @torch.no_grad()
    def compute_reconstruction_errors(
        self, data_loader: DataLoader
    ) -> np.ndarray:
        """Compute reconstruction errors for all samples.

        Args:
            data_loader: DataLoader with trajectory data

        Returns:
            Array of reconstruction errors (MSE) per sample
        """
        errors = []

        for batch in data_loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(self.device)
            error = self.model.compute_reconstruction_error(batch, reduction="none")
            errors.append(error.cpu().numpy())

        return np.concatenate(errors)

    def tune_threshold_percentile(
        self, errors: np.ndarray, percentile: float = 95.0
    ) -> float:
        """Tune threshold using percentile of reconstruction errors.

        Args:
            errors: Reconstruction errors from normal (training) data
            percentile: Percentile value (0-100)

        Returns:
            Computed threshold value
        """
        threshold = float(np.percentile(errors, percentile))
        self.threshold = threshold
        return threshold

    def tune_threshold_optimal(
        self, errors: np.ndarray, labels: np.ndarray, metric: str = "f1"
    ) -> Tuple[float, Dict[str, float]]:
        """Tune threshold to optimize a specific metric.

        Args:
            errors: Reconstruction errors
            labels: Binary labels (0=normal, 1=anomaly)
            metric: Metric to optimize ('f1', 'precision', 'recall')

        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        if metric not in ["f1", "precision", "recall"]:
            raise ValueError(f"Unknown metric: {metric}. Use 'f1', 'precision', or 'recall'")

        thresholds = np.linspace(errors.min(), errors.max(), 1000)
        best_score = 0.0
        best_threshold = thresholds[0]
        best_metrics = {}

        for thresh in thresholds:
            predictions = (errors >= thresh).astype(int)

            if metric == "f1":
                score = f1_score(labels, predictions, zero_division=0)
            elif metric == "precision":
                score = precision_score(labels, predictions, zero_division=0)
            elif metric == "recall":
                score = recall_score(labels, predictions, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = thresh
                best_metrics = self._compute_metrics(labels, predictions)

        self.threshold = float(best_threshold)
        return float(best_threshold), best_metrics

    def tune_threshold_youden(
        self, errors: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Tune threshold using Youden's J statistic (sensitivity + specificity - 1).

        Args:
            errors: Reconstruction errors
            labels: Binary labels (0=normal, 1=anomaly)

        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        fpr, tpr, thresholds = roc_curve(labels, errors)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = float(thresholds[best_idx])

        predictions = (errors >= best_threshold).astype(int)
        metrics = self._compute_metrics(labels, predictions)

        self.threshold = best_threshold
        return best_threshold, metrics

    def predict(
        self, data_loader: DataLoader, return_errors: bool = False
    ) -> np.ndarray:
        """Predict anomalies for data.

        Args:
            data_loader: DataLoader with trajectory data
            return_errors: If True, return (predictions, errors) tuple

        Returns:
            Binary predictions (0=normal, 1=anomaly) or tuple if return_errors=True
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call tune_threshold_* first.")

        errors = self.compute_reconstruction_errors(data_loader)
        predictions = (errors >= self.threshold).astype(int)

        if return_errors:
            return predictions, errors
        return predictions

    def evaluate(
        self, data_loader: DataLoader, labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate anomaly detection performance.

        Args:
            data_loader: DataLoader with trajectory data
            labels: True binary labels (0=normal, 1=anomaly)

        Returns:
            Dictionary of evaluation metrics
        """
        predictions, errors = self.predict(data_loader, return_errors=True)
        metrics = self._compute_metrics(labels, predictions)

        try:
            metrics["auroc"] = float(roc_auc_score(labels, errors))
        except ValueError:
            metrics["auroc"] = 0.0

        precision, recall, _ = precision_recall_curve(labels, errors)
        metrics["auprc"] = float(auc(recall, precision))

        return metrics

    def _compute_metrics(
        self, labels: np.ndarray, predictions: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification metrics.

        Args:
            labels: True binary labels
            predictions: Predicted binary labels

        Returns:
            Dictionary of metrics
        """
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
        }

    def get_anomaly_scores(self, data_loader: DataLoader) -> np.ndarray:
        """Get anomaly scores (reconstruction errors) for data.

        Args:
            data_loader: DataLoader with trajectory data

        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        return self.compute_reconstruction_errors(data_loader)

    def set_threshold(self, threshold: float) -> None:
        """Manually set anomaly detection threshold.

        Args:
            threshold: Threshold value
        """
        self.threshold = float(threshold)
