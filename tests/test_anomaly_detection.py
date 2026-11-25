"""
Tests for anomaly detection module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import AnomalyDetector
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_error_distribution,
    plot_metrics_summary,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_trajectory_comparison,
)
from src.models import LSTMAutoencoder


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model(device):
    """Create a small autoencoder for testing."""
    model = LSTMAutoencoder(
        input_size=5,
        hidden_size=16,
        num_layers=1,
        embedding_dim=8,
        seq_len=20,
    ).to(device)
    return model


@pytest.fixture
def data_loader():
    """Create a simple data loader."""
    data = torch.randn(50, 20, 5)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=10, shuffle=False)


@pytest.fixture
def anomaly_detector(model, device):
    """Create anomaly detector instance."""
    return AnomalyDetector(model=model, device=device)


def test_anomaly_detector_initialization(model, device):
    """Test anomaly detector initialization."""
    detector = AnomalyDetector(model=model, device=device, threshold=0.5)
    assert detector.model == model
    assert detector.device == device
    assert detector.threshold == 0.5


def test_compute_reconstruction_errors(anomaly_detector, data_loader):
    """Test reconstruction error computation."""
    errors = anomaly_detector.compute_reconstruction_errors(data_loader)
    assert isinstance(errors, np.ndarray)
    assert len(errors) == 50
    assert errors.ndim == 1
    assert np.all(errors >= 0)


def test_tune_threshold_percentile(anomaly_detector, data_loader):
    """Test percentile-based threshold tuning."""
    errors = anomaly_detector.compute_reconstruction_errors(data_loader)
    threshold = anomaly_detector.tune_threshold_percentile(errors, percentile=95.0)

    assert isinstance(threshold, float)
    assert threshold > 0
    assert anomaly_detector.threshold == threshold

    # Check that approximately 95% of errors are below threshold
    below_threshold = np.sum(errors < threshold) / len(errors)
    assert 0.90 < below_threshold <= 1.0


def test_tune_threshold_optimal_f1(anomaly_detector, data_loader):
    """Test optimal F1-based threshold tuning."""
    errors = anomaly_detector.compute_reconstruction_errors(data_loader)

    # Create synthetic labels (high errors = anomaly)
    labels = (errors > np.median(errors)).astype(int)

    threshold, metrics = anomaly_detector.tune_threshold_optimal(
        errors, labels, metric="f1"
    )

    assert isinstance(threshold, float)
    assert threshold > 0
    assert anomaly_detector.threshold == threshold
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "accuracy" in metrics
    assert 0 <= metrics["f1"] <= 1


def test_tune_threshold_optimal_precision(anomaly_detector, data_loader):
    """Test optimal precision-based threshold tuning."""
    errors = anomaly_detector.compute_reconstruction_errors(data_loader)
    labels = (errors > np.median(errors)).astype(int)

    threshold, metrics = anomaly_detector.tune_threshold_optimal(
        errors, labels, metric="precision"
    )

    assert isinstance(threshold, float)
    assert "precision" in metrics
    assert 0 <= metrics["precision"] <= 1


def test_tune_threshold_optimal_recall(anomaly_detector, data_loader):
    """Test optimal recall-based threshold tuning."""
    errors = anomaly_detector.compute_reconstruction_errors(data_loader)
    labels = (errors > np.median(errors)).astype(int)

    threshold, metrics = anomaly_detector.tune_threshold_optimal(
        errors, labels, metric="recall"
    )

    assert isinstance(threshold, float)
    assert "recall" in metrics
    assert 0 <= metrics["recall"] <= 1


def test_tune_threshold_optimal_invalid_metric(anomaly_detector, data_loader):
    """Test that invalid metric raises error."""
    errors = anomaly_detector.compute_reconstruction_errors(data_loader)
    labels = np.ones(len(errors), dtype=int)

    with pytest.raises(ValueError, match="Unknown metric"):
        anomaly_detector.tune_threshold_optimal(errors, labels, metric="invalid")


def test_tune_threshold_youden(anomaly_detector, data_loader):
    """Test Youden's J statistic threshold tuning."""
    errors = anomaly_detector.compute_reconstruction_errors(data_loader)
    labels = (errors > np.median(errors)).astype(int)

    threshold, metrics = anomaly_detector.tune_threshold_youden(errors, labels)

    assert isinstance(threshold, float)
    assert threshold > 0
    assert anomaly_detector.threshold == threshold
    assert "f1" in metrics
    assert "precision" in metrics


def test_predict_without_threshold(anomaly_detector, data_loader):
    """Test that predict raises error without threshold."""
    with pytest.raises(ValueError, match="Threshold not set"):
        anomaly_detector.predict(data_loader)


def test_predict_with_threshold(anomaly_detector, data_loader):
    """Test prediction with set threshold."""
    anomaly_detector.set_threshold(0.1)
    predictions = anomaly_detector.predict(data_loader)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 50
    assert np.all((predictions == 0) | (predictions == 1))


def test_predict_return_errors(anomaly_detector, data_loader):
    """Test prediction with error return."""
    anomaly_detector.set_threshold(0.1)
    predictions, errors = anomaly_detector.predict(data_loader, return_errors=True)

    assert isinstance(predictions, np.ndarray)
    assert isinstance(errors, np.ndarray)
    assert len(predictions) == len(errors)


def test_evaluate(anomaly_detector, data_loader):
    """Test evaluation with labels."""
    errors = anomaly_detector.compute_reconstruction_errors(data_loader)
    labels = (errors > np.median(errors)).astype(int)

    anomaly_detector.tune_threshold_percentile(errors, percentile=50.0)
    metrics = anomaly_detector.evaluate(data_loader, labels)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "auroc" in metrics
    assert "auprc" in metrics

    for metric_value in metrics.values():
        assert 0 <= metric_value <= 1


def test_get_anomaly_scores(anomaly_detector, data_loader):
    """Test anomaly score retrieval."""
    scores = anomaly_detector.get_anomaly_scores(data_loader)

    assert isinstance(scores, np.ndarray)
    assert len(scores) == 50
    assert np.all(scores >= 0)


def test_set_threshold(anomaly_detector):
    """Test manual threshold setting."""
    anomaly_detector.set_threshold(0.42)
    assert anomaly_detector.threshold == 0.42


def test_plot_roc_curve(anomaly_detector, data_loader):
    """Test ROC curve plotting."""
    errors = anomaly_detector.compute_reconstruction_errors(data_loader)
    labels = (errors > np.median(errors)).astype(int)

    fig = plot_roc_curve(labels, errors)
    assert fig is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "roc_curve.png"
        fig = plot_roc_curve(labels, errors, save_path=save_path)
        assert save_path.exists()


def test_plot_precision_recall_curve(anomaly_detector, data_loader):
    """Test precision-recall curve plotting."""
    errors = anomaly_detector.compute_reconstruction_errors(data_loader)
    labels = (errors > np.median(errors)).astype(int)

    fig = plot_precision_recall_curve(labels, errors)
    assert fig is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "pr_curve.png"
        fig = plot_precision_recall_curve(labels, errors, save_path=save_path)
        assert save_path.exists()


def test_plot_confusion_matrix(anomaly_detector, data_loader):
    """Test confusion matrix plotting."""
    errors = anomaly_detector.compute_reconstruction_errors(data_loader)
    labels = (errors > np.median(errors)).astype(int)
    predictions = (errors > np.percentile(errors, 50)).astype(int)

    fig = plot_confusion_matrix(labels, predictions)
    assert fig is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "confusion_matrix.png"
        fig = plot_confusion_matrix(labels, predictions, save_path=save_path)
        assert save_path.exists()


def test_plot_error_distribution():
    """Test error distribution plotting."""
    normal_errors = np.random.randn(100) * 0.5 + 1.0
    anomaly_errors = np.random.randn(50) * 1.0 + 3.0
    threshold = 2.0

    fig = plot_error_distribution(normal_errors, anomaly_errors, threshold)
    assert fig is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "error_dist.png"
        fig = plot_error_distribution(
            normal_errors, anomaly_errors, threshold, save_path=save_path
        )
        assert save_path.exists()


def test_plot_error_distribution_no_anomalies():
    """Test error distribution plotting without anomaly data."""
    normal_errors = np.random.randn(100) * 0.5 + 1.0

    fig = plot_error_distribution(normal_errors)
    assert fig is not None


def test_plot_trajectory_comparison(model, device):
    """Test trajectory comparison plotting."""
    normal_samples = torch.randn(5, 20, 5)
    anomaly_samples = torch.randn(3, 20, 5)

    fig = plot_trajectory_comparison(model, normal_samples, anomaly_samples, device)
    assert fig is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "trajectory_comparison.png"
        fig = plot_trajectory_comparison(
            model, normal_samples, anomaly_samples, device, save_path=save_path
        )
        assert save_path.exists()


def test_plot_trajectory_comparison_normal_only(model, device):
    """Test trajectory comparison plotting with only normal samples."""
    normal_samples = torch.randn(3, 20, 5)

    fig = plot_trajectory_comparison(model, normal_samples, device=device)
    assert fig is not None


def test_plot_metrics_summary():
    """Test metrics summary plotting."""
    metrics = {
        "accuracy": 0.85,
        "precision": 0.80,
        "recall": 0.90,
        "f1": 0.85,
        "auroc": 0.88,
    }

    fig = plot_metrics_summary(metrics)
    assert fig is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "metrics_summary.png"
        fig = plot_metrics_summary(metrics, save_path=save_path)
        assert save_path.exists()
