#!/usr/bin/env python3
"""
Evaluation script for DRL V2V Anomaly Detection System

Evaluate a trained model on test data with comprehensive metrics and visualizations.

Usage:
    python evaluate.py checkpoint_path=/path/to/checkpoint.pt
    python evaluate.py checkpoint_path=/path/to/checkpoint.pt threshold=0.05
    python evaluate.py checkpoint_path=/path/to/checkpoint.pt tune_metric=f1
"""

import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.data.dataloader import V2VDataLoader
from src.data.trajectory_window import create_preprocessed_dataset
from src.evaluation import (
    AnomalyDetector,
    plot_error_distribution,
    plot_trajectory_comparison,
)
from src.models import create_autoencoder_from_config
from src.training.trainer import create_dataloaders
from src.utils.logging_utils import log_config, setup_logger
from src.utils.reproducibility import get_device, set_deterministic_mode, set_seed

logger = setup_logger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    """Main evaluation function.

    Args:
        config: Hydra configuration
    """
    logger.info("\n" + "=" * 80)
    logger.info("DRL V2V ANOMALY DETECTION - EVALUATION")
    logger.info("=" * 80 + "\n")

    log_config(logger, OmegaConf.to_container(config, resolve=True))

    set_seed(config.seed)
    set_deterministic_mode(True)
    logger.info(f"✅ Set seed to {config.seed} for reproducibility")

    device = get_device(config.device)
    logger.info(f"✅ Using device: {device}")

    checkpoint_path = config.get("checkpoint_path", None)
    if checkpoint_path is None:
        output_dir = Path(config.paths.output_dir) / "checkpoints"
        checkpoint_path = output_dir / "best_model.pt"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    logger.info(f"✅ Loading checkpoint: {checkpoint_path}")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING AND PREPROCESSING DATA")
    logger.info("=" * 80)

    loader = V2VDataLoader(config)
    loader.load_dataset()
    loader.load_metadata()

    selected_vehicles = loader.select_vehicles()
    logger.info(f"✅ Selected {len(selected_vehicles)} vehicles")

    features_df = loader.extract_features()
    logger.info(
        f"✅ Extracted {len(features_df):,} records with {len(config.data.features)} features"
    )

    preprocessed_dir = Path(config.paths.output_dir) / "preprocessed"
    dataset = create_preprocessed_dataset(config, features_df, save_dir=preprocessed_dir)

    logger.info("✅ Dataset preprocessed:")
    logger.info(f"   Train: {dataset['train_data'].shape}")
    logger.info(f"   Val: {dataset['val_data'].shape}")
    logger.info(f"   Test: {dataset['test_data'].shape}")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CREATING DATA LOADERS")
    logger.info("=" * 80)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data=dataset["train_data"],
        val_data=dataset["val_data"],
        test_data=dataset["test_data"],
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory and device.type == "cuda",
    )

    logger.info("✅ Created data loaders:")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: LOADING MODEL")
    logger.info("=" * 80)

    model = create_autoencoder_from_config(config, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("✅ Model loaded from checkpoint")
    logger.info(f"   Checkpoint epoch: {checkpoint['epoch'] + 1}")
    if 'val_loss' in checkpoint:
        logger.info(f"   Validation loss: {checkpoint['val_loss']:.6f}")
    elif 'metrics' in checkpoint and 'val_loss' in checkpoint['metrics']:
        logger.info(f"   Validation loss: {checkpoint['metrics']['val_loss']:.6f}")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: ANOMALY DETECTION SETUP")
    logger.info("=" * 80)

    detector = AnomalyDetector(model=model, device=device)

    tune_metric = config.get("tune_metric", "percentile")
    threshold = config.get("threshold", None)

    if threshold is not None:
        logger.info(f"✅ Using manual threshold: {threshold:.6f}")
        detector.set_threshold(threshold)
    else:
        logger.info(f"✅ Tuning threshold on validation set (method: {tune_metric})")

        val_errors = detector.compute_reconstruction_errors(val_loader)

        if tune_metric == "percentile":
            percentile = config.get("percentile", 95.0)
            threshold = detector.tune_threshold_percentile(val_errors, percentile=percentile)
            logger.info(f"   Threshold (95th percentile): {threshold:.6f}")

        elif tune_metric in ["f1", "precision", "recall"]:
            val_labels = np.zeros(len(val_errors), dtype=int)
            threshold, metrics = detector.tune_threshold_optimal(
                val_errors, val_labels, metric=tune_metric
            )
            logger.info(f"   Optimal threshold ({tune_metric}): {threshold:.6f}")
            logger.info(f"   Metrics: {metrics}")

        else:
            logger.warning(f"Unknown tune_metric: {tune_metric}. Using 95th percentile.")
            threshold = detector.tune_threshold_percentile(val_errors, percentile=95.0)
            logger.info(f"   Threshold (95th percentile): {threshold:.6f}")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: COMPUTING RECONSTRUCTION ERRORS")
    logger.info("=" * 80)

    train_errors = detector.compute_reconstruction_errors(train_loader)
    val_errors = detector.compute_reconstruction_errors(val_loader)
    test_errors = detector.compute_reconstruction_errors(test_loader)

    logger.info("✅ Reconstruction errors computed:")
    logger.info(
        f"   Train: mean={train_errors.mean():.6f}, std={train_errors.std():.6f}, median={np.median(train_errors):.6f}"
    )
    logger.info(
        f"   Val:   mean={val_errors.mean():.6f}, std={val_errors.std():.6f}, median={np.median(val_errors):.6f}"
    )
    logger.info(
        f"   Test:  mean={test_errors.mean():.6f}, std={test_errors.std():.6f}, median={np.median(test_errors):.6f}"
    )

    test_predictions = (test_errors >= threshold).astype(int)
    anomaly_rate = test_predictions.mean()
    logger.info(f"   Anomaly rate (test): {anomaly_rate:.2%}")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: GENERATING VISUALIZATIONS")
    logger.info("=" * 80)

    plots_dir = Path(config.paths.output_dir) / "evaluation_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig1 = plot_error_distribution(
        normal_errors=train_errors,
        threshold=threshold,
        save_path=plots_dir / "train_error_distribution.png",
        title="Training Set Reconstruction Error Distribution",
    )
    logger.info(f"✅ Saved: {plots_dir / 'train_error_distribution.png'}")
    plt.close(fig1)

    fig2 = plot_error_distribution(
        normal_errors=val_errors,
        threshold=threshold,
        save_path=plots_dir / "val_error_distribution.png",
        title="Validation Set Reconstruction Error Distribution",
    )
    logger.info(f"✅ Saved: {plots_dir / 'val_error_distribution.png'}")
    plt.close(fig2)

    fig3 = plot_error_distribution(
        normal_errors=test_errors,
        threshold=threshold,
        save_path=plots_dir / "test_error_distribution.png",
        title="Test Set Reconstruction Error Distribution",
    )
    logger.info(f"✅ Saved: {plots_dir / 'test_error_distribution.png'}")
    plt.close(fig3)

    test_batch = next(iter(test_loader))
    if isinstance(test_batch, (tuple, list)):
        test_batch = test_batch[0]
    test_samples = test_batch[:3]

    fig4 = plot_trajectory_comparison(
        model=model,
        normal_samples=test_samples,
        device=device,
        save_path=plots_dir / "test_trajectory_comparison.png",
        title="Test Set Trajectory Reconstruction",
        max_samples=3,
    )
    logger.info(f"✅ Saved: {plots_dir / 'test_trajectory_comparison.png'}")
    plt.close(fig4)

    error_stats = {
        "train_mean": float(train_errors.mean()),
        "train_std": float(train_errors.std()),
        "train_median": float(np.median(train_errors)),
        "val_mean": float(val_errors.mean()),
        "val_std": float(val_errors.std()),
        "val_median": float(np.median(val_errors)),
        "test_mean": float(test_errors.mean()),
        "test_std": float(test_errors.std()),
        "test_median": float(np.median(test_errors)),
        "threshold": float(threshold),
        "anomaly_rate": float(anomaly_rate),
    }

    results_file = plots_dir / "evaluation_results.txt"

    # Get validation loss from checkpoint (handle different formats)
    val_loss_str = "N/A"
    if 'val_loss' in checkpoint:
        val_loss_str = f"{checkpoint['val_loss']:.6f}"
    elif 'metrics' in checkpoint and 'val_loss' in checkpoint['metrics']:
        val_loss_str = f"{checkpoint['metrics']['val_loss']:.6f}"

    with open(results_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Checkpoint Epoch: {checkpoint['epoch'] + 1}\n")
        f.write(f"Checkpoint Val Loss: {val_loss_str}\n\n")
        f.write(f"Threshold: {threshold:.6f}\n")
        f.write(f"Threshold Method: {tune_metric}\n\n")
        f.write("Reconstruction Errors:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"  Train: mean={train_errors.mean():.6f}, std={train_errors.std():.6f}, median={np.median(train_errors):.6f}\n"
        )
        f.write(
            f"  Val:   mean={val_errors.mean():.6f}, std={val_errors.std():.6f}, median={np.median(val_errors):.6f}\n"
        )
        f.write(
            f"  Test:  mean={test_errors.mean():.6f}, std={test_errors.std():.6f}, median={np.median(test_errors):.6f}\n\n"
        )
        f.write(f"Anomaly Rate (Test): {anomaly_rate:.2%}\n")

    logger.info(f"✅ Saved: {results_file}")

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)

        with mlflow.start_run(run_name=f"evaluation_{checkpoint['epoch']+1}"):
            mlflow.log_params(
                {
                    "checkpoint_epoch": checkpoint["epoch"] + 1,
                    "threshold": threshold,
                    "tune_metric": tune_metric,
                }
            )

            mlflow.log_metrics(error_stats)

            for plot_file in plots_dir.glob("*.png"):
                mlflow.log_artifact(str(plot_file))

            mlflow.log_artifact(str(results_file))

        logger.info("✅ Results logged to MLflow")

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
