import numpy as np

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: "min" or "max" for monitoring metric
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == "min":
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == "max":
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"Mode {mode} is unknown!")

    def __call__(self, current_score: float, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            current_score: Current validation score
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False

        if self.monitor_op(current_score - self.min_delta, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(
                    f"Validation score improved to {current_score:.6f} at epoch {epoch}"
                )
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping counter: {self.counter} out of {self.patience} "
                    f"(best: {self.best_score:.6f} at epoch {self.best_epoch})"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(
                        f"Early stopping triggered! Best score: {self.best_score:.6f} "
                        f"at epoch {self.best_epoch}"
                    )

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
