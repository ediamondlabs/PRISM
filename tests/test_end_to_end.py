#!/usr/bin/env python3
"""
End-to-end integration test for DRL V2V Anomaly Detection System

Tests the complete pipeline from data loading through model training and evaluation.

Usage:
    python test_end_to_end.py
"""

import shutil
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.dataloader import V2VDataLoader
from src.data.trajectory_window import create_preprocessed_dataset
from src.evaluation import AnomalyDetector
from src.models import create_autoencoder_from_config
from src.training.trainer import Trainer, create_dataloaders
from src.utils.logging_utils import setup_logger
from src.utils.reproducibility import get_device, set_deterministic_mode, set_seed

logger = setup_logger(__name__)


def create_test_config() -> DictConfig:
    """Create a minimal test configuration."""
    config_dict = {
        "seed": 42,
        "device": "cpu",
        "paths": {
            "data_dir": "data",
            "output_dir": "outputs_test_e2e",
            "checkpoint_dir": "outputs_test_e2e/checkpoints",
        },
        "mlflow": {
            "tracking_uri": "",
            "experiment_name": "test_e2e",
        },
        "data": {
            "dataset_path": "data/v2v_dataset_20251017_124247.csv",
            "metadata_path": "data/metadata_20251017_124247.json",
            "num_vehicles": 5,
            "min_presence_duration": 100.0,
            "window_size": 50,
            "stride": 10,
            "features": [
                "velocity_x",
                "velocity_y",
                "speed",
                "accel_x",
                "accel_y",
                "accel_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "yaw",
            ],
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": False,
            "preprocessing": {
                "normalize": True,
                "normalization_method": "standardize",
            },
        },
        "model": {
            "name": "lstm_autoencoder",
            "architecture": {
                "input_size": 10,
                "hidden_size": 64,
                "num_layers": 1,
                "embedding_dim": 16,
                "dropout": 0.0,
            },
        },
        "training": {
            "num_epochs": 3,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "optimizer": {
                "name": "adam",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
            "loss": {
                "name": "mse",
                "reduction": "mean",
            },
            "scheduler": {
                "name": "cosine_annealing",
                "T_max": 3,
                "eta_min": 1e-6,
            },
            "gradient_clip_norm": 1.0,
            "early_stopping": {
                "enabled": True,
                "patience": 5,
                "min_delta": 0.0,
                "monitor": "val_loss",
                "mode": "min",
            },
            "checkpointing": {
                "save_best": True,
                "save_last": True,
                "monitor": "val_loss",
                "mode": "min",
            },
            "logging": {
                "log_interval": 10,
                "log_gradients": False,
            },
        },
    }

    return OmegaConf.create(config_dict)


def test_end_to_end():
    """Run end-to-end integration test."""
    logger.info("\n" + "=" * 80)
    logger.info("END-TO-END INTEGRATION TEST")
    logger.info("=" * 80 + "\n")

    output_dir = Path("outputs_test_e2e")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    try:
        config = create_test_config()

        set_seed(config.seed)
        set_deterministic_mode(True)
        device = get_device(config.device)

        logger.info("Step 1: Data Loading and Preprocessing")
        logger.info("-" * 80)

        loader = V2VDataLoader(config)
        loader.load_dataset()
        loader.load_metadata()

        selected_vehicles = loader.select_vehicles()
        assert len(selected_vehicles) == 5, f"Expected 5 vehicles, got {len(selected_vehicles)}"
        logger.info(f"✅ Selected {len(selected_vehicles)} vehicles")

        features_df = loader.extract_features()
        assert len(features_df) > 0, "No features extracted"
        # DataFrame has 10 features + vehicle_id + timestep + timestamp = 13 columns
        assert len(features_df.columns) == 13, f"Expected 13 columns, got {len(features_df.columns)}"
        logger.info(f"✅ Extracted {len(features_df):,} records with 10 trajectory features")

        preprocessed_dir = output_dir / "preprocessed"
        dataset = create_preprocessed_dataset(config, features_df, save_dir=preprocessed_dir)

        assert "train_data" in dataset
        assert "val_data" in dataset
        assert "test_data" in dataset
        assert dataset["train_data"].shape[0] > 0, "No training data"
        assert dataset["val_data"].shape[0] > 0, "No validation data"
        assert dataset["test_data"].shape[0] > 0, "No test data"
        logger.info("✅ Dataset splits created:")
        logger.info(f"   Train: {dataset['train_data'].shape}")
        logger.info(f"   Val: {dataset['val_data'].shape}")
        logger.info(f"   Test: {dataset['test_data'].shape}")

        logger.info("\nStep 2: Model Initialization")
        logger.info("-" * 80)

        model = create_autoencoder_from_config(config, device=device)
        assert model is not None
        num_params = model.get_num_parameters()
        assert num_params > 0
        logger.info(f"✅ Model created with {num_params:,} parameters")

        logger.info("\nStep 3: Training")
        logger.info("-" * 80)

        train_loader, val_loader, test_loader = create_dataloaders(
            train_data=dataset["train_data"],
            val_data=dataset["val_data"],
            test_data=dataset["test_data"],
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            pin_memory=False,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        history = trainer.train()

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 3, f"Expected 3 epochs, got {len(history['train_loss'])}"
        assert all(loss > 0 for loss in history["train_loss"]), "Invalid training loss values"
        assert all(loss > 0 for loss in history["val_loss"]), "Invalid validation loss values"
        logger.info(f"✅ Training completed for {len(history['train_loss'])} epochs")
        logger.info(f"   Final train loss: {history['train_loss'][-1]:.6f}")
        logger.info(f"   Final val loss: {history['val_loss'][-1]:.6f}")

        best_checkpoint_path = trainer.checkpoint_manager.get_best_checkpoint_path()
        assert best_checkpoint_path is not None
        assert best_checkpoint_path.exists()
        logger.info(f"✅ Best checkpoint saved: {best_checkpoint_path}")

        logger.info("\nStep 4: Model Loading")
        logger.info("-" * 80)

        fresh_model = create_autoencoder_from_config(config, device=device)
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        fresh_model.load_state_dict(checkpoint["model_state_dict"])
        fresh_model.eval()

        assert "epoch" in checkpoint
        assert "metrics" in checkpoint
        logger.info(f"✅ Model loaded from checkpoint (epoch {checkpoint['epoch'] + 1})")

        logger.info("\nStep 5: Anomaly Detection")
        logger.info("-" * 80)

        detector = AnomalyDetector(model=fresh_model, device=device)

        train_errors = detector.compute_reconstruction_errors(train_loader)
        val_errors = detector.compute_reconstruction_errors(val_loader)
        test_errors = detector.compute_reconstruction_errors(test_loader)

        assert len(train_errors) > 0
        assert len(val_errors) > 0
        assert len(test_errors) > 0
        assert all(err >= 0 for err in train_errors), "Negative errors detected"
        logger.info("✅ Reconstruction errors computed")
        logger.info(f"   Train: mean={train_errors.mean():.6f}, std={train_errors.std():.6f}")
        logger.info(f"   Val: mean={val_errors.mean():.6f}, std={val_errors.std():.6f}")
        logger.info(f"   Test: mean={test_errors.mean():.6f}, std={test_errors.std():.6f}")

        threshold = detector.tune_threshold_percentile(train_errors, percentile=95.0)
        assert threshold > 0
        logger.info(f"✅ Threshold tuned (95th percentile): {threshold:.6f}")

        predictions = detector.predict(test_loader)
        assert len(predictions) == len(test_errors)
        assert all((p == 0) or (p == 1) for p in predictions), "Invalid predictions"

        anomaly_rate = predictions.mean()
        logger.info(f"✅ Predictions generated: {anomaly_rate:.2%} anomalies detected")

        logger.info("\nStep 6: Verification")
        logger.info("-" * 80)

        # Verify checkpoint structure
        required_keys = ["model_state_dict", "optimizer_state_dict", "epoch", "metrics"]
        for key in required_keys:
            assert key in checkpoint, f"Missing key in checkpoint: {key}"
        logger.info("✅ Checkpoint structure verified")

        # Verify files created
        assert (output_dir / "preprocessed").exists()
        assert (output_dir / "checkpoints").exists()
        assert (output_dir / "checkpoints" / "best_model.pt").exists()
        assert (output_dir / "checkpoints" / "last_model.pt").exists()
        logger.info("✅ Output files verified")

        # Verify model can make predictions
        test_batch = next(iter(test_loader))
        if isinstance(test_batch, (tuple, list)):
            test_batch = test_batch[0]
        test_batch = test_batch.to(device)

        with torch.no_grad():
            reconstruction, embedding = fresh_model(test_batch)

        assert reconstruction.shape == test_batch.shape
        assert embedding.shape[0] == test_batch.shape[0]
        assert embedding.shape[1] == config.model.architecture.embedding_dim
        logger.info("✅ Model inference verified")

        logger.info("\n" + "=" * 80)
        logger.info("END-TO-END TEST PASSED ✅")
        logger.info("=" * 80 + "\n")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False

    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
            logger.info(f"✅ Cleaned up test directory: {output_dir}")


if __name__ == "__main__":
    success = test_end_to_end()
    sys.exit(0 if success else 1)
