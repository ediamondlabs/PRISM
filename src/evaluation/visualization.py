"""
Visualization utilities for anomaly detection results.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve",
) -> plt.Figure:
    """Plot ROC curve for anomaly detection.

    Args:
        labels: True binary labels (0=normal, 1=anomaly)
        scores: Anomaly scores (reconstruction errors)
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_auc_score

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_precision_recall_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    """Plot precision-recall curve for anomaly detection.

    Args:
        labels: True binary labels (0=normal, 1=anomaly)
        scores: Anomaly scores (reconstruction errors)
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import auc, average_precision_score

    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(labels, scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall,
        precision,
        color="blue",
        lw=2,
        label=f"PR (AUC = {pr_auc:.3f}, AP = {avg_precision:.3f})",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
    class_names: Tuple[str, str] = ("Normal", "Anomaly"),
) -> plt.Figure:
    """Plot confusion matrix.

    Args:
        labels: True binary labels
        predictions: Predicted binary labels
        save_path: Path to save figure
        title: Plot title
        class_names: Names for (negative, positive) classes

    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_error_distribution(
    normal_errors: np.ndarray,
    anomaly_errors: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    save_path: Optional[Path] = None,
    title: str = "Reconstruction Error Distribution",
) -> plt.Figure:
    """Plot distribution of reconstruction errors.

    Args:
        normal_errors: Reconstruction errors for normal samples
        anomaly_errors: Reconstruction errors for anomalous samples (optional)
        threshold: Detection threshold to plot as vertical line
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        normal_errors,
        bins=50,
        alpha=0.7,
        label=f"Normal (n={len(normal_errors)})",
        density=True,
        color="blue",
    )

    if anomaly_errors is not None:
        ax.hist(
            anomaly_errors,
            bins=50,
            alpha=0.7,
            label=f"Anomaly (n={len(anomaly_errors)})",
            density=True,
            color="red",
        )

    if threshold is not None:
        ax.axvline(
            threshold,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Threshold: {threshold:.4f}",
        )

    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_trajectory_comparison(
    model: torch.nn.Module,
    normal_samples: torch.Tensor,
    anomaly_samples: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
    save_path: Optional[Path] = None,
    title: str = "Trajectory Reconstruction Comparison",
    max_samples: int = 3,
) -> plt.Figure:
    """Plot comparison of original and reconstructed trajectories.

    Args:
        model: Trained autoencoder model
        normal_samples: Normal trajectory samples [batch, seq_len, features]
        anomaly_samples: Anomalous trajectory samples (optional)
        device: Device for computation
        save_path: Path to save figure
        title: Plot title
        max_samples: Maximum number of samples to plot

    Returns:
        Matplotlib figure
    """
    model.eval()

    def reconstruct(samples):
        with torch.no_grad():
            samples = samples.to(device)
            recon, _ = model(samples)
        return samples.cpu().numpy(), recon.cpu().numpy()

    normal_orig, normal_recon = reconstruct(normal_samples[:max_samples])

    n_rows = max_samples
    n_cols = 2 if anomaly_samples is None else 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(max_samples):
        # Normal samples - spatial trajectory
        axes[i, 0].plot(
            normal_orig[i, :, 0],
            normal_orig[i, :, 1],
            "b-o",
            label="Original",
            markersize=3,
            alpha=0.7,
        )
        axes[i, 0].plot(
            normal_recon[i, :, 0],
            normal_recon[i, :, 1],
            "r--x",
            label="Reconstructed",
            markersize=3,
            alpha=0.7,
        )
        axes[i, 0].set_xlabel("X Position", fontsize=10)
        axes[i, 0].set_ylabel("Y Position", fontsize=10)
        axes[i, 0].set_title(
            f"Normal Sample {i+1}: Spatial", fontsize=11, fontweight="bold"
        )
        axes[i, 0].legend(fontsize=9)
        axes[i, 0].grid(True, alpha=0.3)

        # Normal samples - velocity
        if normal_orig.shape[2] > 2:
            axes[i, 1].plot(
                normal_orig[i, :, 2],
                "b-o",
                label="Original",
                markersize=3,
                alpha=0.7,
            )
            axes[i, 1].plot(
                normal_recon[i, :, 2],
                "r--x",
                label="Reconstructed",
                markersize=3,
                alpha=0.7,
            )
            axes[i, 1].set_xlabel("Time Step", fontsize=10)
            axes[i, 1].set_ylabel("Velocity", fontsize=10)
            axes[i, 1].set_title(
                f"Normal Sample {i+1}: Velocity", fontsize=11, fontweight="bold"
            )
            axes[i, 1].legend(fontsize=9)
            axes[i, 1].grid(True, alpha=0.3)

        # Anomaly samples if provided
        if anomaly_samples is not None and i < len(anomaly_samples):
            anomaly_orig, anomaly_recon = reconstruct(anomaly_samples[i : i + 1])

            # Spatial trajectory
            axes[i, 2].plot(
                anomaly_orig[0, :, 0],
                anomaly_orig[0, :, 1],
                "b-o",
                label="Original",
                markersize=3,
                alpha=0.7,
            )
            axes[i, 2].plot(
                anomaly_recon[0, :, 0],
                anomaly_recon[0, :, 1],
                "r--x",
                label="Reconstructed",
                markersize=3,
                alpha=0.7,
            )
            axes[i, 2].set_xlabel("X Position", fontsize=10)
            axes[i, 2].set_ylabel("Y Position", fontsize=10)
            axes[i, 2].set_title(
                f"Anomaly Sample {i+1}: Spatial", fontsize=11, fontweight="bold"
            )
            axes[i, 2].legend(fontsize=9)
            axes[i, 2].grid(True, alpha=0.3)

            # Velocity
            if anomaly_orig.shape[2] > 2:
                axes[i, 3].plot(
                    anomaly_orig[0, :, 2],
                    "b-o",
                    label="Original",
                    markersize=3,
                    alpha=0.7,
                )
                axes[i, 3].plot(
                    anomaly_recon[0, :, 2],
                    "r--x",
                    label="Reconstructed",
                    markersize=3,
                    alpha=0.7,
                )
                axes[i, 3].set_xlabel("Time Step", fontsize=10)
                axes[i, 3].set_ylabel("Velocity", fontsize=10)
                axes[i, 3].set_title(
                    f"Anomaly Sample {i+1}: Velocity", fontsize=11, fontweight="bold"
                )
                axes[i, 3].legend(fontsize=9)
                axes[i, 3].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_metrics_summary(
    metrics: dict,
    save_path: Optional[Path] = None,
    title: str = "Anomaly Detection Performance",
) -> plt.Figure:
    """Plot bar chart of evaluation metrics.

    Args:
        metrics: Dictionary of metric names and values
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metric_names, metric_values, color="skyblue", edgecolor="navy")

    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
