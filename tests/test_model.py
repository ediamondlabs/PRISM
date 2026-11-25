from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from src.models.autoencoder import LSTMAutoencoder, create_autoencoder_from_config
from src.models.encoder_decoder import LSTMDecoder, LSTMEncoder
from src.utils.reproducibility import set_seed


@pytest.fixture(scope="module")
def config():
    """Fixture to load configuration."""
    config_dir = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
        return cfg


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(8, 50, 10)


class TestLSTMEncoder:
    """Test LSTM encoder component."""

    def test_encoder_initialization(self):
        """Test encoder can be initialized."""
        encoder = LSTMEncoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, dropout=0.0
        )
        assert encoder.input_size == 10
        assert encoder.hidden_size == 128
        assert encoder.num_layers == 2
        assert encoder.embedding_dim == 32

    def test_encoder_forward_shape(self, sample_input):
        """Test encoder forward pass output shapes."""
        encoder = LSTMEncoder(input_size=10, hidden_size=128, num_layers=2, embedding_dim=32)

        embedding, (hidden, cell) = encoder(sample_input)

        assert embedding.shape == (8, 32)
        assert hidden.shape == (2, 8, 128)
        assert cell.shape == (2, 8, 128)

    def test_encoder_batch_independence(self):
        """Test that encoder processes batches independently."""
        encoder = LSTMEncoder(input_size=10, hidden_size=128, num_layers=2, embedding_dim=32)
        encoder.eval()

        x1 = torch.randn(1, 50, 10)
        x2 = torch.randn(1, 50, 10)
        x_batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            emb1, _ = encoder(x1)
            emb2, _ = encoder(x2)
            emb_batch, _ = encoder(x_batch)

        torch.testing.assert_close(emb_batch[0], emb1[0])
        torch.testing.assert_close(emb_batch[1], emb2[0])

    def test_encoder_no_nan(self, sample_input):
        """Test that encoder doesn't produce NaN values."""
        encoder = LSTMEncoder(input_size=10, hidden_size=128, num_layers=2, embedding_dim=32)

        embedding, _ = encoder(sample_input)

        assert not torch.isnan(embedding).any()
        assert not torch.isinf(embedding).any()


class TestLSTMDecoder:
    """Test LSTM decoder component."""

    def test_decoder_initialization(self):
        """Test decoder can be initialized."""
        decoder = LSTMDecoder(
            embedding_dim=32, hidden_size=128, num_layers=2, output_size=10, seq_len=50
        )
        assert decoder.embedding_dim == 32
        assert decoder.hidden_size == 128
        assert decoder.num_layers == 2
        assert decoder.output_size == 10
        assert decoder.seq_len == 50

    def test_decoder_forward_shape(self):
        """Test decoder forward pass output shape."""
        decoder = LSTMDecoder(
            embedding_dim=32, hidden_size=128, num_layers=2, output_size=10, seq_len=50
        )

        embedding = torch.randn(8, 32)
        output = decoder(embedding)

        assert output.shape == (8, 50, 10)

    def test_decoder_with_hidden_state(self):
        """Test decoder with provided hidden state."""
        decoder = LSTMDecoder(
            embedding_dim=32, hidden_size=128, num_layers=2, output_size=10, seq_len=50
        )

        embedding = torch.randn(8, 32)
        hidden = torch.randn(2, 8, 128)
        cell = torch.randn(2, 8, 128)

        output = decoder(embedding, (hidden, cell))

        assert output.shape == (8, 50, 10)

    def test_decoder_no_nan(self):
        """Test that decoder doesn't produce NaN values."""
        decoder = LSTMDecoder(
            embedding_dim=32, hidden_size=128, num_layers=2, output_size=10, seq_len=50
        )

        embedding = torch.randn(8, 32)
        output = decoder(embedding)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestLSTMAutoencoder:
    """Test complete LSTM autoencoder."""

    def test_autoencoder_initialization(self):
        """Test autoencoder can be initialized."""
        model = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        )
        assert model.input_size == 10
        assert model.hidden_size == 128
        assert model.embedding_dim == 32
        assert model.seq_len == 50

    def test_autoencoder_forward_shape(self, sample_input):
        """Test autoencoder forward pass output shapes."""
        model = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        )

        reconstruction, embedding = model(sample_input)

        assert reconstruction.shape == sample_input.shape
        assert embedding.shape == (8, 32)

    def test_autoencoder_encode_only(self, sample_input):
        """Test encoding without reconstruction."""
        model = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        )

        embedding = model.encode(sample_input)

        assert embedding.shape == (8, 32)

    def test_autoencoder_decode_only(self):
        """Test decoding from embedding."""
        model = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        )

        embedding = torch.randn(8, 32)
        reconstruction = model.decode(embedding)

        assert reconstruction.shape == (8, 50, 10)

    def test_autoencoder_reconstruction_error(self, sample_input):
        """Test reconstruction error computation."""
        model = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        )

        error_mean = model.compute_reconstruction_error(sample_input, reduction="mean")
        error_none = model.compute_reconstruction_error(sample_input, reduction="none")

        assert error_mean.shape == ()
        assert error_none.shape == (8,)
        assert error_mean >= 0
        assert (error_none >= 0).all()

    def test_autoencoder_get_num_parameters(self):
        """Test parameter counting."""
        model = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        )

        num_params = model.get_num_parameters()
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_autoencoder_gradient_flow(self, sample_input):
        """Test that gradients flow through the model."""
        model = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        )

        reconstruction, _ = model(sample_input)
        loss = ((sample_input - reconstruction) ** 2).mean()
        loss.backward()

        encoder_has_grad = any(p.grad is not None for p in model.encoder.parameters())
        decoder_has_grad = any(p.grad is not None for p in model.decoder.parameters())

        assert encoder_has_grad
        assert decoder_has_grad

        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()

    def test_autoencoder_reproducibility(self, sample_input):
        """Test that autoencoder produces same output with same seed."""
        set_seed(42)
        model1 = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        )

        set_seed(42)
        model2 = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        )

        with torch.no_grad():
            recon1, emb1 = model1(sample_input)
            recon2, emb2 = model2(sample_input)

        torch.testing.assert_close(recon1, recon2)
        torch.testing.assert_close(emb1, emb2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_autoencoder_on_gpu(self, sample_input):
        """Test autoencoder works on GPU."""
        model = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        ).cuda()

        sample_input_gpu = sample_input.cuda()

        reconstruction, embedding = model(sample_input_gpu)

        assert reconstruction.device.type == "cuda"
        assert embedding.device.type == "cuda"


class TestModelFactory:
    """Test model creation from configuration."""

    def test_create_from_config(self, config):
        """Test creating model from Hydra config."""
        device = torch.device("cpu")
        model = create_autoencoder_from_config(config, device=device)

        assert isinstance(model, LSTMAutoencoder)
        assert model.input_size == config.model.architecture.input_size
        assert model.hidden_size == config.model.architecture.hidden_size
        assert model.embedding_dim == config.model.architecture.embedding_dim
        assert next(model.parameters()).device.type == "cpu"

    def test_create_from_config_with_auto_device(self, config):
        """Test creating model with automatic device selection."""
        model = create_autoencoder_from_config(config, device=None)

        assert isinstance(model, LSTMAutoencoder)

    def test_model_matches_config_specs(self, config):
        """Test that created model matches all config specifications."""
        model = create_autoencoder_from_config(config)

        assert model.input_size == 10
        assert model.hidden_size == 768
        assert model.num_layers == 4
        assert model.embedding_dim == 256
        assert model.seq_len == 50
        assert model.bidirectional is True


class TestModelIntegration:
    """Integration tests for model with real data shapes."""

    def test_model_with_realistic_batch_sizes(self):
        """Test model with various realistic batch sizes."""
        model = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        )

        for batch_size in [1, 16, 32, 64]:
            x = torch.randn(batch_size, 50, 10)
            reconstruction, embedding = model(x)

            assert reconstruction.shape == (batch_size, 50, 10)
            assert embedding.shape == (batch_size, 32)

    def test_training_step(self):
        """Test a single training step."""
        model = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(32, 50, 10)

        initial_params = [p.clone() for p in model.parameters()]

        reconstruction, _ = model(x)
        loss = ((x - reconstruction) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        updated_params = list(model.parameters())

        params_changed = any(
            not torch.equal(p1, p2) for p1, p2 in zip(initial_params, updated_params)
        )
        assert params_changed

    def test_eval_vs_train_mode(self, sample_input):
        """Test that model behaves consistently in eval vs train mode."""
        model = LSTMAutoencoder(
            input_size=10, hidden_size=128, num_layers=2, embedding_dim=32, seq_len=50
        )

        model.train()
        with torch.no_grad():
            recon_train, _ = model(sample_input)

        model.eval()
        with torch.no_grad():
            recon_eval, _ = model(sample_input)

        torch.testing.assert_close(recon_train, recon_eval)
