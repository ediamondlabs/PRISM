from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


@pytest.fixture(scope="module")
def config_dir():
    """Fixture to provide absolute path to configs directory."""
    return str(Path(__file__).parent.parent / "configs")


@pytest.fixture
def base_config(config_dir):
    """Fixture to load base configuration."""
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
        return cfg


def test_config_loads_successfully(base_config):
    """Test that the base configuration loads without errors."""
    assert isinstance(base_config, DictConfig)
    assert base_config is not None


def test_config_has_required_sections(base_config):
    """Test that all required configuration sections are present."""
    required_sections = ["data", "model", "training", "paths", "mlflow", "seed", "device"]
    for section in required_sections:
        assert section in base_config, f"Missing required section: {section}"


def test_data_config_structure(base_config):
    """Test data configuration has correct structure."""
    data_cfg = base_config.data
    assert data_cfg.name == "v2v_trajectories"
    assert data_cfg.num_vehicles == 50
    assert data_cfg.window_size == 50
    assert data_cfg.stride == 10
    assert len(data_cfg.features) == 10
    assert data_cfg.batch_size == 512


def test_model_config_structure(base_config):
    """Test model configuration has correct structure."""
    model_cfg = base_config.model
    assert model_cfg.name == "lstm_autoencoder"
    assert model_cfg.architecture.input_size == 10
    assert model_cfg.architecture.hidden_size == 768
    assert model_cfg.architecture.num_layers == 4
    assert model_cfg.architecture.embedding_dim == 256
    assert model_cfg.architecture.bidirectional is True


def test_training_config_structure(base_config):
    """Test training configuration has correct structure."""
    train_cfg = base_config.training
    assert train_cfg.name == "local_task"
    assert train_cfg.num_epochs == 1000
    assert train_cfg.learning_rate == 0.0015
    assert train_cfg.optimizer.name == "adam"
    assert train_cfg.loss.name == "mse"


def test_paths_config(base_config):
    """Test paths configuration."""
    paths = base_config.paths
    assert "data_dir" in paths
    assert "output_dir" in paths
    assert "checkpoint_dir" in paths
    assert "logs_dir" in paths


def test_mlflow_config(base_config):
    """Test MLflow configuration."""
    mlflow = base_config.mlflow
    assert "tracking_uri" in mlflow
    assert "experiment_name" in mlflow
    assert mlflow.experiment_name == "v2v_anomaly_detection"


def test_baseline_experiment_config(config_dir):
    """Test baseline experiment configuration loads correctly."""
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=["+experiment=baseline"])
        assert cfg.seed == 42
        assert cfg.data.num_vehicles == 20
        assert cfg.model.architecture.hidden_size == 128
        assert cfg.training.num_epochs == 20


def test_config_overrides(config_dir):
    """Test configuration can be overridden via command line."""
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=["seed=123", "data.batch_size=64", "training.num_epochs=50"],
        )
        assert cfg.seed == 123
        assert cfg.data.batch_size == 64
        assert cfg.training.num_epochs == 50


def test_split_ratios_sum_to_one(base_config):
    """Test that data split ratios sum to 1.0."""
    split_ratios = base_config.data.split_ratios
    total = split_ratios.train + split_ratios.val + split_ratios.test
    assert abs(total - 1.0) < 1e-6, f"Split ratios sum to {total}, not 1.0"
