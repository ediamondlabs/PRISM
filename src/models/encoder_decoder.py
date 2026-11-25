from typing import Tuple

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """LSTM encoder that compresses trajectory sequences into embeddings."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        embedding_dim: int = 32,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Bidirectional doubles the output size
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Linear(fc_input_size, embedding_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode input sequence to embedding.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)

        Returns:
            Tuple of (embedding, (hidden_state, cell_state))
            - embedding: Shape (batch, embedding_dim)
            - hidden_state: Shape (num_layers * num_directions, batch, hidden_size)
            - cell_state: Shape (num_layers * num_directions, batch, hidden_size)
        """
        lstm_out, (hidden, cell) = self.lstm(x)

        if self.bidirectional:
            # Concatenate final forward and backward hidden states
            # hidden shape: (num_layers * 2, batch, hidden_size)
            # Forward final: hidden[-2], Backward final: hidden[-1]
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            last_hidden = hidden[-1]

        embedding = self.fc(last_hidden)

        return embedding, (hidden, cell)


class LSTMDecoder(nn.Module):
    """LSTM decoder that reconstructs trajectory sequences from embeddings.

    Decoder is always unidirectional. When the encoder is bidirectional,
    the encoder hidden state is projected down to match decoder dimensions.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 10,
        seq_len: int = 50,
        dropout: float = 0.0,
        encoder_bidirectional: bool = False,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.seq_len = seq_len
        self.encoder_bidirectional = encoder_bidirectional

        self.fc_embedding = nn.Linear(embedding_dim, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc_output = nn.Linear(hidden_size, output_size)

        # Project bidirectional encoder hidden state to decoder dimensions
        if encoder_bidirectional:
            self.hidden_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.cell_proj = nn.Linear(hidden_size * 2, hidden_size)

    def _project_encoder_state(
        self, hidden_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project bidirectional encoder hidden state for unidirectional decoder.

        Takes pairs of (forward, backward) hidden states per layer,
        concatenates them, and projects to decoder hidden_size.
        """
        hidden, cell = hidden_state
        # hidden shape: (num_layers * 2, batch, hidden_size)
        # Reshape to (num_layers, batch, hidden_size * 2)
        batch_size = hidden.size(1)
        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)  # (num_layers, batch, hidden_size*2)
        hidden = self.hidden_proj(hidden)

        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_size)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=-1)
        cell = self.cell_proj(cell)

        return hidden, cell

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode embedding to reconstruct sequence.

        Args:
            embedding: Input embedding of shape (batch, embedding_dim)
            hidden_state: Optional encoder hidden/cell states

        Returns:
            Reconstructed sequence of shape (batch, seq_len, output_size)
        """
        decoder_input = self.fc_embedding(embedding)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.seq_len, 1)

        if hidden_state is not None and self.encoder_bidirectional:
            hidden_state = self._project_encoder_state(hidden_state)

        if hidden_state is not None:
            lstm_out, _ = self.lstm(decoder_input, hidden_state)
        else:
            lstm_out, _ = self.lstm(decoder_input)

        output = self.fc_output(lstm_out)

        return output
