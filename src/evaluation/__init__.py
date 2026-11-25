"""Evaluation module for anomaly detection."""

from src.evaluation.anomaly_detector import AnomalyDetector
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_error_distribution,
    plot_metrics_summary,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_trajectory_comparison,
)

__all__ = [
    "AnomalyDetector",
    "plot_confusion_matrix",
    "plot_error_distribution",
    "plot_metrics_summary",
    "plot_precision_recall_curve",
    "plot_roc_curve",
    "plot_trajectory_comparison",
]
