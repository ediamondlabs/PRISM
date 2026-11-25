"""
Flower FL Client for LSTM Autoencoder training.
Uses shared config for fair comparison with DFL, CoCo, and Centralized.
"""

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.autoencoder import LSTMAutoencoder
from src.utils.reproducibility import set_seed


def load_shared_config() -> dict:
    """Load shared configuration for fair comparison."""
    config_path = Path(__file__).parent.parent / "configs" / "comparison" / "shared_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
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
        },
        "data": {
            "window_size": 50,
            "stride": 10,
        },
        "seed": 42,
    }


def load_data(client_id: int, num_clients: int = 5) -> Tuple[DataLoader, DataLoader]:
    """Load and partition data for client using preprocessed data (matching DFL)."""
    config = load_shared_config()
    set_seed(config["seed"] + client_id)  # Different seed per client for shuffling

    # Load preprocessed data
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "outputs" / "preprocessed" / "preprocessed_data.npz"

    if not data_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found at {data_path}. Run preprocessing first!")

    data = np.load(data_path, allow_pickle=True)

    # Get unique vehicle IDs from the data
    all_vehicle_ids = sorted(np.unique(np.concatenate([
        data['train_vehicle_ids'],
        data['val_vehicle_ids']
    ])))

    # Each client gets ONE vehicle (matching DFL approach)
    # client_id 0 -> vehicle 0, client_id 1 -> vehicle 1, etc.
    if client_id >= len(all_vehicle_ids):
        raise ValueError(f"Client ID {client_id} exceeds number of vehicles {len(all_vehicle_ids)}")

    client_vehicle_id = all_vehicle_ids[client_id]

    # Filter data for this client's vehicle
    train_mask = data['train_vehicle_ids'] == client_vehicle_id
    val_mask = data['val_vehicle_ids'] == client_vehicle_id

    X_train = data['train_data'][train_mask]
    X_val = data['val_data'][val_mask]

    if len(X_val) == 0:
        X_val = X_train[:1]  # Fallback if no validation data

    # Create data loaders
    # Reduce batch size for parallel GPU training (5 clients on same GPU in MIG mode)
    batch_size = min(config["training"]["batch_size"], 512)  # Cap at 32 for GPU memory
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    input_size = X_train.shape[2]  # Number of features
    return train_loader, val_loader, input_size


class FlowerClient(fl.client.NumPyClient):
    """Flower client using shared config for fair comparison."""

    def __init__(self, client_id: int, num_clients: int = 5, device: str = None):
        self.client_id = client_id
        self.config = load_shared_config()
        # GPU assignment: can be specified or defaults to cuda:0
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print(f"Client {client_id} assigned to {self.device}")

        self.train_loader, self.val_loader, input_size = load_data(client_id, num_clients)

        # Create model with shared config
        model_cfg = self.config["model"]
        self.model = LSTMAutoencoder(
            input_size=input_size,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            embedding_dim=model_cfg["embedding_dim"],
            seq_len=model_cfg["seq_len"],
            dropout=model_cfg["dropout"],
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.train_cfg = self.config["training"]

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Client {client_id}: {len(self.train_loader.dataset)} train, "
              f"{len(self.val_loader.dataset)} val, {num_params:,} params")

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config) -> Tuple[List[np.ndarray], int, dict]:
        self.set_parameters(parameters)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg["weight_decay"],
        )
        self.model.train()

        # 5 local epochs (matching DFL's lstm_local_epochs)
        local_epochs = self.config["federated"]["local_epochs"]
        epoch_losses = []

        for epoch in range(local_epochs):
            total_loss = 0.0
            num_samples = 0

            for batch in self.train_loader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                recon, _ = self.model(x)
                loss = self.criterion(recon, x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * len(x)
                num_samples += len(x)

            epoch_loss = total_loss / num_samples if num_samples > 0 else 0.0
            epoch_losses.append(epoch_loss)

            # Clear CUDA cache after each epoch to prevent fragmentation
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Client {self.client_id} train loss (5 epochs avg): {avg_loss:.6f}")

        return self.get_parameters(config), num_samples, {"train_loss": avg_loss}

    def evaluate(self, parameters: List[np.ndarray], config) -> Tuple[float, int, dict]:
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch[0].to(self.device)
                recon, _ = self.model(x)
                loss = self.criterion(recon, x)
                total_loss += loss.item() * len(x)
                num_samples += len(x)

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        print(f"Client {self.client_id} val loss: {avg_loss:.6f}")

        # Clear CUDA cache after evaluation
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return avg_loss, num_samples, {"val_loss": avg_loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True, help="Client ID (0-4)")
    parser.add_argument("--num-clients", type=int, default=5, help="Total number of clients")
    parser.add_argument("--server", type=str, default="127.0.0.1:8983", help="Server address")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0, cuda:1, cpu)")
    args = parser.parse_args()

    client = FlowerClient(args.client_id, args.num_clients, device=args.device)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()