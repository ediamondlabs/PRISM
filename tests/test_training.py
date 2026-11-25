from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from src.models import create_autoencoder_from_config
from src.training.checkpoint import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.trainer import Trainer, create_dataloaders
from src.utils.metrics import (
    AverageMeter,
    MetricsTracker,
    compute_reconstruction_metrics,
    find_best_threshold,
)
from src.utils.reproducibility import set_seed


@pytest.fixture(scope="module")
def config():
    """Fixture to load configuration."""
    config_dir = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")
        return cfg


@pytest.fixture
def sample_data():
    """Create sample training data."""
    train_data = np.random.randn(100, 50, 10).astype(np.float32)
    val_data = np.random.randn(20, 50, 10).astype(np.float32)
    test_data = np.random.randn(20, 50, 10).astype(np.float32)
    return train_data, val_data, test_data


@pytest.fixture
def sample_errors_and_labels():
    """Create sample reconstruction errors and labels."""
    np.random.seed(42)
    errors = np.concatenate([np.random.randn(80) * 0.5 + 0.5, np.random.randn(20) * 0.5 + 2.0])
    labels = np.concatenate([np.zeros(80), np.ones(20)])
    return errors, labels


class TestMetrics:
    """Test metrics utilities."""

    def test_average_meter(self):
        """Test AverageMeter class."""
        meter = AverageMeter()

        meter.update(1.0)
        assert meter.avg == 1.0
        assert meter.count == 1

        meter.update(2.0, n=2)
        assert meter.count == 3
        assert meter.avg == (1.0 + 2.0 * 2) / 3

    def test_average_meter_reset(self):
        """Test AverageMeter reset."""
        meter = AverageMeter()
        meter.update(5.0, n=10)

        meter.reset()

        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0

    def test_metrics_tracker(self):
        """Test MetricsTracker class."""
        tracker = MetricsTracker()

        tracker.update({"loss": 1.0, "f1": 0.5}, epoch=0)
        tracker.update({"loss": 0.8, "f1": 0.6}, epoch=1)

        assert len(tracker.get_metric("loss")) == 2
        assert tracker.get_latest("loss") == 0.8
        assert tracker.get_latest("f1") == 0.6

    def test_metrics_tracker_best(self):
        """Test getting best metric."""
        tracker = MetricsTracker()

        tracker.update({"loss": 1.0}, epoch=0)
        tracker.update({"loss": 0.5}, epoch=1)
        tracker.update({"loss": 0.8}, epoch=2)

        best_val, best_epoch = tracker.get_best("loss", mode="min")
        assert best_val == 0.5
        assert best_epoch == 1

    def test_compute_reconstruction_metrics(self, sample_errors_and_labels):
        """Test reconstruction metrics computation."""
        errors, labels = sample_errors_and_labels

        metrics = compute_reconstruction_metrics(errors, labels, threshold=1.0)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "auc" in metrics
        assert "threshold" in metrics

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_find_best_threshold(self, sample_errors_and_labels):
        """Test threshold optimization."""
        errors, labels = sample_errors_and_labels

        threshold_f1 = find_best_threshold(errors, labels, metric="f1")
        threshold_precision = find_best_threshold(errors, labels, metric="precision")

        assert isinstance(threshold_f1, float)
        assert isinstance(threshold_precision, float)
        assert threshold_f1 > 0
        assert threshold_precision > 0


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_initialization(self):
        """Test early stopping can be initialized."""
        es = EarlyStopping(patience=5, min_delta=0.001, mode="min")

        assert es.patience == 5
        assert es.min_delta == -0.001
        assert es.counter == 0
        assert es.early_stop is False

    def test_early_stopping_improvement(self):
        """Test early stopping with improving scores."""
        es = EarlyStopping(patience=3, mode="min", verbose=False)

        assert es(1.0, epoch=0) is False
        assert es.counter == 0

        assert es(0.8, epoch=1) is False
        assert es.counter == 0

        assert es(0.6, epoch=2) is False
        assert es.counter == 0

    def test_early_stopping_no_improvement(self):
        """Test early stopping with no improvement."""
        es = EarlyStopping(patience=2, mode="min", verbose=False)

        es(1.0, epoch=0)
        es(1.1, epoch=1)
        assert es.counter == 1

        es(1.2, epoch=2)
        assert es.counter == 2
        assert es.early_stop is True

    def test_early_stopping_mode_max(self):
        """Test early stopping in max mode."""
        es = EarlyStopping(patience=2, mode="max", verbose=False)

        es(0.5, epoch=0)
        es(0.6, epoch=1)
        assert es.counter == 0

        es(0.55, epoch=2)
        assert es.counter == 1

    def test_early_stopping_reset(self):
        """Test early stopping reset."""
        es = EarlyStopping(patience=2, mode="min", verbose=False)

        es(1.0, epoch=0)
        es(1.1, epoch=1)

        es.reset()

        assert es.counter == 0
        assert es.best_score is None
        assert es.early_stop is False


class TestCheckpointManager:
    """Test checkpoint management."""

    def test_checkpoint_manager_initialization(self, tmp_path):
        """Test checkpoint manager can be initialized."""
        manager = CheckpointManager(
            checkpoint_dir=tmp_path, save_best=True, save_last=True, monitor="val_loss", mode="min"
        )

        assert manager.checkpoint_dir.exists()
        assert manager.save_best is True
        assert manager.save_last is True

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoints."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        manager = CheckpointManager(checkpoint_dir=tmp_path, save_best=True, save_last=True)

        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=5,
            metrics={"loss": 0.5},
            is_best=True,
        )

        assert checkpoint_path.exists()

        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.Adam(new_model.parameters())

        checkpoint = manager.load_checkpoint(checkpoint_path, new_model, new_optimizer)

        assert checkpoint["epoch"] == 5
        assert "loss" in checkpoint["metrics"]

    def test_check_is_best(self, tmp_path):
        """Test checking if score is best."""
        manager = CheckpointManager(checkpoint_dir=tmp_path, mode="min")

        assert manager.check_is_best(1.0, epoch=0) is True
        assert manager.check_is_best(0.8, epoch=1) is True
        assert manager.check_is_best(0.9, epoch=2) is False

    def test_get_best_checkpoint_path(self, tmp_path):
        """Test getting best checkpoint path."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        manager = CheckpointManager(checkpoint_dir=tmp_path, save_best=True)

        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            epoch=0,
            metrics={"loss": 1.0},
            is_best=True,
        )

        best_path = manager.get_best_checkpoint_path()
        assert best_path is not None
        assert best_path.exists()


class TestDataloaders:
    """Test dataloader creation."""

    def test_create_dataloaders(self, sample_data):
        """Test creating PyTorch dataloaders."""
        train_data, val_data, test_data = sample_data

        train_loader, val_loader, test_loader = create_dataloaders(
            train_data, val_data, test_data, batch_size=32
        )

        assert len(train_loader.dataset) == 100
        assert len(val_loader.dataset) == 20
        assert len(test_loader.dataset) == 20

        batch = next(iter(train_loader))[0]
        assert batch.shape == (32, 50, 10)


class TestTrainer:
    """Test Trainer class."""

    def test_trainer_initialization(self, config, sample_data, tmp_path):
        """Test trainer can be initialized."""
        set_seed(42)

        train_data, val_data, test_data = sample_data
        train_loader, val_loader, _ = create_dataloaders(train_data, val_data, test_data, batch_size=16)

        device = torch.device("cpu")
        model = create_autoencoder_from_config(config, device=device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_trainer_single_epoch(self, config, sample_data, tmp_path):
        """Test training for one epoch."""
        set_seed(42)

        train_data, val_data, test_data = sample_data
        train_loader, val_loader, _ = create_dataloaders(train_data, val_data, test_data, batch_size=16)

        device = torch.device("cpu")
        model = create_autoencoder_from_config(config, device=device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        train_loss = trainer.train_epoch()

        assert isinstance(train_loss, float)
        assert train_loss > 0

    def test_trainer_validation(self, config, sample_data, tmp_path):
        """Test validation."""
        set_seed(42)

        train_data, val_data, test_data = sample_data
        train_loader, val_loader, _ = create_dataloaders(train_data, val_data, test_data, batch_size=16)

        device = torch.device("cpu")
        model = create_autoencoder_from_config(config, device=device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        val_loss = trainer.validate()

        assert isinstance(val_loss, float)
        assert val_loss > 0

    def test_trainer_train_multiple_epochs(self, config, sample_data, tmp_path):
        """Test training for multiple epochs."""
        set_seed(42)

        train_data, val_data, test_data = sample_data
        train_loader, val_loader, _ = create_dataloaders(train_data, val_data, test_data, batch_size=16)

        device = torch.device("cpu")
        model = create_autoencoder_from_config(config, device=device)

        config_copy = config.copy()
        config_copy.training.num_epochs = 3
        config_copy.mlflow.tracking_uri = None

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config_copy,
            device=device,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        history = trainer.train()

        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3

        assert history["train_loss"][0] > history["train_loss"][-1]

    def test_trainer_saves_checkpoints(self, config, sample_data, tmp_path):
        """Test that trainer saves checkpoints."""
        set_seed(42)

        train_data, val_data, test_data = sample_data
        train_loader, val_loader, _ = create_dataloaders(train_data, val_data, test_data, batch_size=16)

        device = torch.device("cpu")
        model = create_autoencoder_from_config(config, device=device)

        config_copy = config.copy()
        config_copy.training.num_epochs = 2
        config_copy.mlflow.tracking_uri = None

        checkpoint_dir = tmp_path / "checkpoints"

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config_copy,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        trainer.train()

        assert (checkpoint_dir / "best_model.pt").exists()
        assert (checkpoint_dir / "last_model.pt").exists()

    def test_trainer_early_stopping(self, config, sample_data, tmp_path):
        """Test early stopping functionality in trainer."""
        set_seed(42)

        train_data, val_data, test_data = sample_data
        train_loader, val_loader, _ = create_dataloaders(train_data, val_data, test_data, batch_size=16)

        device = torch.device("cpu")
        model = create_autoencoder_from_config(config, device=device)

        config_copy = config.copy()
        config_copy.training.num_epochs = 100
        config_copy.training.early_stopping.enabled = True
        config_copy.training.early_stopping.patience = 3
        config_copy.mlflow.tracking_uri = None

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config_copy,
            device=device,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        history = trainer.train()

        assert len(history["train_loss"]) < 100
