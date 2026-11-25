from pathlib import Path
from typing import Dict, Optional

import torch

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class CheckpointManager:
    """Manage model checkpoints during training."""

    def __init__(
        self,
        checkpoint_dir: Path,
        save_best: bool = True,
        save_last: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best: Whether to save best model
            save_last: Whether to save last model
            monitor: Metric to monitor for best model
            mode: "min" or "max" for monitoring metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_best = save_best
        self.save_last = save_last
        self.monitor = monitor
        self.mode = mode

        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """Save checkpoint to disk.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best checkpoint

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        saved_paths = []

        if is_best and self.save_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
            self.best_epoch = epoch
            saved_paths.append(checkpoint_path)

        if self.save_last:
            checkpoint_path = self.checkpoint_dir / "last_model.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved last model to {checkpoint_path}")
            saved_paths.append(checkpoint_path)

        if not saved_paths:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            saved_paths.append(checkpoint_path)

        return saved_paths[0] if saved_paths else None

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict:
        """Load checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into

        Returns:
            Checkpoint dictionary with metrics
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")

        return checkpoint

    def check_is_best(self, current_score: float, epoch: int) -> bool:
        """Check if current score is the best so far.

        Args:
            current_score: Current validation score
            epoch: Current epoch

        Returns:
            True if current score is best, False otherwise
        """
        if self.mode == "min":
            is_best = current_score < self.best_score
        else:
            is_best = current_score > self.best_score

        if is_best:
            self.best_score = current_score
            self.best_epoch = epoch

        return is_best

    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None if not saved
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        return best_path if best_path.exists() else None

    def get_last_checkpoint_path(self) -> Optional[Path]:
        """Get path to last checkpoint.

        Returns:
            Path to last checkpoint or None if not saved
        """
        last_path = self.checkpoint_dir / "last_model.pt"
        return last_path if last_path.exists() else None
