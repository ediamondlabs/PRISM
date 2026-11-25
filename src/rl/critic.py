"""
Centralized Critic Network for CTDE (Centralized Training, Decentralized Execution).

The critic has access to global state (all vehicles' states) during training,
but actors only see local observations during execution.
"""


import torch
import torch.nn as nn


class CentralizedCritic(nn.Module):
    """Centralized critic for CTDE architecture."""

    def __init__(
        self,
        state_dim: int = 106,
        num_vehicles: int = 20,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize centralized critic.

        Args:
            state_dim: Dimension of individual vehicle state
            num_vehicles: Number of vehicles in the network
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()

        self.state_dim = state_dim
        self.num_vehicles = num_vehicles
        self.global_state_dim = state_dim * num_vehicles

        # Build MLP layers
        layers = []
        input_dim = self.global_state_dim

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer: single value
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Compute state value.

        Args:
            global_state: Global state tensor
                         Shape: (batch, num_vehicles, state_dim) or (batch, num_vehicles * state_dim)

        Returns:
            State value (batch, 1)
        """
        # Flatten if needed
        if global_state.dim() == 3:
            batch_size = global_state.size(0)
            global_state = global_state.view(batch_size, -1)

        assert global_state.size(-1) == self.global_state_dim, (
            f"Expected global state dim {self.global_state_dim}, "
            f"got {global_state.size(-1)}"
        )

        value = self.network(global_state)
        return value

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DecentralizedCritic(nn.Module):
    """Decentralized critic (optional, for comparison with centralized)."""

    def __init__(
        self,
        state_dim: int = 106,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize decentralized critic.

        Args:
            state_dim: Dimension of individual vehicle state
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()

        self.state_dim = state_dim

        # Build MLP layers
        layers = []
        input_dim = state_dim

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer: single value
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state value.

        Args:
            state: Local state tensor (batch, state_dim)

        Returns:
            State value (batch, 1)
        """
        value = self.network(state)
        return value

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
