"""
Centralized Training (Upper Bound / Gold Standard).

Trains a single model on ALL data merged together.
This represents the best possible performance (no data distribution).

Uses IDENTICAL hyperparameters as DFL, CoCo, and Flower FL for fair comparison.
See configs/comparison/shared_config.yaml for shared settings.

Usage:
    python train_centralized.py
    python train_centralized.py --epochs 500 --device cuda:0
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.models.autoencoder import LSTMAutoencoder
from src.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_shared_config() -> Dict:
    """Load shared configuration for fair comparison."""
    config_path = Path(__file__).parent / "configs" / "comparison" / "shared_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    # Fallback defaults matching the config
    return {
        "model": {
            "input_size": 10,
            "hidden_size": 768,
            "num_layers": 4,
            "embedding_dim": 256,
            "dropout": 0.3,
            "seq_len": 50,
        },
        "training": {
            "learning_rate": 1.5e-3,
            "weight_decay": 1e-5,
            "batch_size": 512,
            "num_epochs": 1000,
            "gradient_clip": 1.0,
            "early_stopping_patience": 40,
            "min_delta": 1e-7,
        },
        "data": {
            "window_size": 50,
            "stride": 10,
            "features": [
                "velocity_x", "velocity_y", "speed",
                "accel_x", "accel_y", "accel_z",
                "gyro_x", "gyro_y", "gyro_z", "yaw"
            ],
            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        },
        "seed": 42,
    }


def load_and_merge_data(config: Dict, data_path: Path, num_vehicles: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load data from first N vehicles (matching FL clients) and merge into single global dataset."""
    logger.info(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} records from {df['vehicle_id'].nunique()} vehicles")

    features = config["data"]["features"]
    window_size = config["data"]["window_size"]
    stride = config["data"]["stride"]

    # Check which features exist
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        logger.warning(f"Using available features: {available_features}")
        # Fallback to common features if configured ones don't exist
        fallback = ["x", "y", "speed", "angle", "acceleration", "lane_position",
                    "distance_to_intersection", "time_to_collision", "relative_speed", "heading_diff"]
        available_features = [f for f in fallback if f in df.columns][:10]
        logger.info(f"Fallback features: {available_features}")

    # Create sequences from ONLY first N vehicles (matching FL clients)
    sequences = []
    all_vehicle_ids = sorted(df["vehicle_id"].unique())
    vehicle_ids = all_vehicle_ids[:num_vehicles]  # Take only first N vehicles
    logger.info(f"Using {num_vehicles} vehicles (matching FL): {vehicle_ids}")

    for vid in vehicle_ids:
        vdf = df[df["vehicle_id"] == vid].sort_values("timestamp")
        data = vdf[available_features].values

        for i in range(0, len(data) - window_size + 1, stride):
            sequences.append(data[i:i + window_size])

    if len(sequences) == 0:
        raise ValueError("No sequences created from data")

    X = np.array(sequences, dtype=np.float32)
    logger.info(f"Created {len(X):,} sequences of shape {X.shape[1:]}")

    # Normalize (global normalization across all data)
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mean) / std

    # Split into train/val/test
    np.random.shuffle(X)
    ratios = config["data"]["split_ratios"]
    n = len(X)
    train_end = int(n * ratios["train"])
    val_end = train_end + int(n * ratios["val"])

    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return X_train, X_val, X_test


def create_model(config: Dict, device: torch.device) -> LSTMAutoencoder:
    """Create LSTM Autoencoder with shared config."""
    model_cfg = config["model"]
    model = LSTMAutoencoder(
        input_size=model_cfg["input_size"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        embedding_dim=model_cfg["embedding_dim"],
        seq_len=model_cfg["seq_len"],
        dropout=model_cfg["dropout"],
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {num_params:,} trainable parameters")

    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch in train_loader:
        x = batch[0].to(device)
        optimizer.zero_grad()

        recon, _ = model(x)
        loss = criterion(recon, x)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item() * len(x)
        num_samples += len(x)

    return total_loss / num_samples


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    all_errors = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            recon, _ = model(x)
            loss = criterion(recon, x)

            # Per-sample reconstruction error
            errors = ((x - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
            all_errors.extend(errors)

            total_loss += loss.item() * len(x)
            num_samples += len(x)

    return total_loss / num_samples, np.array(all_errors)


def main():
    parser = argparse.ArgumentParser(description="Centralized Training (Upper Bound)")
    parser.add_argument("--data-path", type=str, default="outputs/preprocessed/preprocessed_data.npz")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/centralized")
    parser.add_argument("--num-vehicles", type=int, default=5, help="Number of vehicles to use (matching FL clients)")
    args = parser.parse_args()

    # Load shared config
    config = load_shared_config()
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs

    # Set seed for reproducibility
    set_seed(config["seed"])
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load preprocessed data
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found at {data_path}. Run preprocessing first!")

    logger.info(f"Loading preprocessed data from {data_path}")
    data = np.load(data_path, allow_pickle=True)

    # Filter to first N vehicles if needed
    if args.num_vehicles < 5:
        # Get vehicle IDs for first N vehicles
        all_vehicle_ids = sorted(np.unique(np.concatenate([
            data['train_vehicle_ids'],
            data['val_vehicle_ids'],
            data['test_vehicle_ids']
        ])))[:args.num_vehicles]

        # Filter datasets
        train_mask = np.isin(data['train_vehicle_ids'], all_vehicle_ids)
        val_mask = np.isin(data['val_vehicle_ids'], all_vehicle_ids)
        test_mask = np.isin(data['test_vehicle_ids'], all_vehicle_ids)

        X_train = data['train_data'][train_mask]
        X_val = data['val_data'][val_mask]
        X_test = data['test_data'][test_mask]
    else:
        # Use all preprocessed data
        X_train = data['train_data']
        X_val = data['val_data']
        X_test = data['test_data']

    logger.info(f"Loaded: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Update input_size based on actual features
    config["model"]["input_size"] = X_train.shape[2]

    # Create data loaders
    train_cfg = config["training"]
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
    )

    # Create model
    model = create_model(config, device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["num_epochs"],
        eta_min=1e-6,
    )

    # Training loop
    best_val_loss = float("inf")
    history = {
        "rounds": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "round_time_sec": [],
        "lstm_loss": [],  # For compatibility with comparison script
    }

    training_start_time = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting training for {train_cfg['num_epochs']} epochs")
    logger.info("=" * 60)

    for epoch in range(train_cfg["num_epochs"]):
        epoch_start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, train_cfg["gradient_clip"]
        )
        val_loss, _ = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start_time

        scheduler.step()
        history["rounds"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lstm_loss"].append(val_loss)  # Same as val_loss for compatibility
        history["round_time_sec"].append(epoch_time)

        # Save best model (no early stopping)
        if val_loss < best_val_loss - train_cfg["min_delta"]:
            best_val_loss = val_loss
            # Save best model
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info(
                f"Epoch {epoch+1:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Best: {best_val_loss:.6f} ✓ (new best)"
            )
        elif (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Best: {best_val_loss:.6f}"
            )

    # Load best model for final evaluation
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))

    # Final evaluation on test set
    test_loss, test_errors = evaluate(model, test_loader, criterion, device)
    history["test_loss"].append(test_loss)

    total_training_time = time.time() - training_start_time

    # Compute anomaly detection metrics (assuming top 5% are anomalies)
    threshold = np.percentile(test_errors, 95)
    predictions = (test_errors > threshold).astype(int)
    # For upper bound, we assume perfect labels would give high F1
    # In practice, you'd use real labels here

    logger.info("=" * 60)
    logger.info("CENTRALIZED TRAINING RESULTS (Upper Bound)")
    logger.info("=" * 60)
    logger.info(f"Final Test Loss (MSE): {test_loss:.6f}")
    logger.info(f"Best Val Loss: {best_val_loss:.6f}")
    logger.info(f"Total Epochs: {len(history['train_loss'])}")
    logger.info(f"Threshold (95th percentile): {threshold:.6f}")

    # Save results
    results = {
        "method": "Centralized (Upper Bound)",
        "test_loss": float(test_loss),
        "best_val_loss": float(best_val_loss),
        "final_lstm_loss": float(best_val_loss),  # For comparison script
        "total_epochs": len(history["train_loss"]),
        "num_rounds": len(history["train_loss"]),  # For comparison script
        "threshold": float(threshold),
        "total_time_sec": total_training_time,
        "config": config,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)

    torch.save(model.state_dict(), output_dir / "final_model.pt")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
