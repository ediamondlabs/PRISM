from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.encoder_decoder import LSTMDecoder, LSTMEncoder


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for trajectory anomaly detection."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        embedding_dim: int = 32,
        seq_len: int = 50,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.bidirectional = bidirectional

        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.decoder = LSTMDecoder(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=input_size,
            seq_len=seq_len,
            dropout=dropout,
            encoder_bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Tuple of (reconstruction, embedding)
            - reconstruction: Shape (batch, seq_len, input_size)
            - embedding: Shape (batch, embedding_dim)
        """
        embedding, hidden_state = self.encoder(x)

        reconstruction = self.decoder(embedding, hidden_state)

        return reconstruction, embedding

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embedding space.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Embedding of shape (batch, embedding_dim)
        """
        embedding, _ = self.encoder(x)
        return embedding

    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decode embedding to reconstructed sequence.

        Args:
            embedding: Embedding tensor of shape (batch, embedding_dim)

        Returns:
            Reconstructed sequence of shape (batch, seq_len, input_size)
        """
        return self.decoder(embedding)

    def compute_reconstruction_error(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """Compute reconstruction error (MSE).

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            reduction: Reduction method ("mean", "sum", or "none")

        Returns:
            Reconstruction error
        """
        reconstruction, _ = self.forward(x)

        if reduction == "none":
            mse = ((x - reconstruction) ** 2).mean(dim=(1, 2))
        elif reduction == "mean":
            mse = ((x - reconstruction) ** 2).mean()
        elif reduction == "sum":
            mse = ((x - reconstruction) ** 2).sum()
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")

        return mse

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_autoencoder_from_config(config: DictConfig, device: torch.device = None) -> LSTMAutoencoder:
    """Create autoencoder from Hydra configuration.

    Args:
        config: Hydra configuration object
        device: Device to place model on

    Returns:
        Initialized autoencoder model
    """
    model_cfg = config.model.architecture

    model = LSTMAutoencoder(
        input_size=model_cfg.input_size,
        hidden_size=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers,
        embedding_dim=model_cfg.embedding_dim,
        seq_len=config.data.window_size,
        dropout=model_cfg.get("dropout", 0.0),
        bidirectional=model_cfg.get("bidirectional", False),
    )

    if device is not None:
        model = model.to(device)

    num_params = model.get_num_parameters()
    print(f"Created {config.model.name} with {num_params:,} parameters")

    return model
