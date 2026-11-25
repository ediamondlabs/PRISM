from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.training.checkpoint import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.utils.logging_utils import setup_logger
from src.utils.metrics import AverageMeter, compute_batch_loss

logger = setup_logger(__name__)


class Trainer:
    """Trainer for LSTM Autoencoder with MLflow tracking."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DictConfig,
        device: torch.device,
        checkpoint_dir: Optional[Path] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Hydra configuration
            device: Device to train on
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.num_epochs = config.training.num_epochs
        self.learning_rate = config.training.learning_rate

        self.optimizer = self._create_optimizer()
        self.criterion = self._create_criterion()
        self.scheduler = self._create_scheduler()

        self.early_stopping = self._create_early_stopping()

        if checkpoint_dir is None:
            checkpoint_dir = Path(config.paths.checkpoint_dir)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            save_best=config.training.checkpointing.save_best,
            save_last=config.training.checkpointing.save_last,
            monitor=config.training.checkpointing.monitor,
            mode=config.training.checkpointing.mode,
        )

        self.current_epoch = 0
        self.training_history = {"train_loss": [], "val_loss": []}

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        optimizer_config = self.config.training.optimizer

        if optimizer_config.name.lower() == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config.name}")

    def _create_criterion(self) -> nn.Module:
        """Create loss function from config."""
        loss_config = self.config.training.loss

        if loss_config.name.lower() == "mse":
            return nn.MSELoss(reduction=loss_config.reduction)
        else:
            raise ValueError(f"Unknown loss: {loss_config.name}")

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config."""
        scheduler_config = self.config.training.scheduler

        if scheduler_config.name.lower() == "cosine_annealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.T_max,
                eta_min=scheduler_config.eta_min,
            )
        return None

    def _create_early_stopping(self) -> Optional[EarlyStopping]:
        """Create early stopping from config."""
        es_config = self.config.training.early_stopping

        if es_config.enabled:
            return EarlyStopping(
                patience=es_config.patience,
                min_delta=es_config.min_delta,
                mode=es_config.mode,
                verbose=True,
            )
        return None

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        loss_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            loss = compute_batch_loss(self.model, batch, self.criterion)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            loss_meter.update(loss.item(), batch.size(0))

            if batch_idx % self.config.training.logging.log_interval == 0:
                pbar.set_postfix({"loss": f"{loss_meter.avg:.6f}"})

        return loss_meter.avg

    @torch.no_grad()
    def validate(self) -> float:
        """Validate model.

        Returns:
            Average validation loss
        """
        self.model.eval()
        loss_meter = AverageMeter()

        for batch in self.val_loader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]

            batch = batch.to(self.device)

            loss = compute_batch_loss(self.model, batch, self.criterion)

            loss_meter.update(loss.item(), batch.size(0))

        return loss_meter.avg

    def train(self) -> Dict[str, list]:
        """Train model for multiple epochs.

        Returns:
            Training history dictionary
        """
        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)
        logger.info(f"Model: {self.config.model.name}")
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Batch size: {self.config.data.batch_size}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 80)

        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch

                train_loss = self.train_epoch()

                val_loss = self.validate()

                if self.scheduler is not None:
                    self.scheduler.step()
                    current_lr = self.scheduler.get_last_lr()[0]
                else:
                    current_lr = self.learning_rate

                self.training_history["train_loss"].append(train_loss)
                self.training_history["val_loss"].append(val_loss)

                logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs} - "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}, "
                    f"LR: {current_lr:.6e}"
                )

                metrics = {"train_loss": train_loss, "val_loss": val_loss, "learning_rate": current_lr}

                if self.config.mlflow.tracking_uri:
                    mlflow.log_metrics(metrics, step=epoch)

                is_best = self.checkpoint_manager.check_is_best(val_loss, epoch)

                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=metrics,
                    is_best=is_best,
                )

                if self.early_stopping is not None:
                    if self.early_stopping(val_loss, epoch):
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        logger.info("=" * 80)
        logger.info("Training Complete")
        logger.info(f"Best epoch: {self.checkpoint_manager.best_epoch + 1}")
        logger.info(f"Best val loss: {self.checkpoint_manager.best_score:.6f}")
        logger.info("=" * 80)

        return self.training_history

    def save_model(self, path: Path):
        """Save model state dict.

        Args:
            path: Path to save model
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path):
        """Load model state dict.

        Args:
            path: Path to load model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")


def create_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple:
    """Create PyTorch data loaders from numpy arrays.

    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for CUDA

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_tensor = torch.from_numpy(train_data).float()
    val_tensor = torch.from_numpy(val_data).float()
    test_tensor = torch.from_numpy(test_data).float()

    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    test_dataset = TensorDataset(test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def setup_mlflow(config: DictConfig) -> str:
    """Setup MLflow tracking.

    Args:
        config: Hydra configuration

    Returns:
        Run ID
    """
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    experiment = mlflow.set_experiment(config.mlflow.experiment_name)

    # Generate sequential run name
    if config.mlflow.run_name:
        run_name = config.mlflow.run_name
    else:
        # Count existing runs and create sequential name
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        run_count = len(runs) + 1
        run_name = f"run_{run_count}"

    mlflow.start_run(run_name=run_name)

    mlflow.log_params(
        {
            "model_name": config.model.name,
            "hidden_size": config.model.architecture.hidden_size,
            "num_layers": config.model.architecture.num_layers,
            "embedding_dim": config.model.architecture.embedding_dim,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.data.batch_size,
            "num_epochs": config.training.num_epochs,
            "optimizer": config.training.optimizer.name,
            "seed": config.seed,
        }
    )

    mlflow.log_dict(OmegaConf.to_container(config, resolve=True), "config.yaml")

    run_id = mlflow.active_run().info.run_id
    logger.info(f"MLflow run started: {run_name} (ID: {run_id})")

    return run_id
