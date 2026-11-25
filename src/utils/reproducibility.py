import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_deterministic_mode(enabled: bool = True) -> None:
    """Enable deterministic operations in PyTorch for reproducibility.

    Args:
        enabled: Whether to enable deterministic mode
    """
    if enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _cuda_is_usable() -> bool:
    """Return True only when CUDA is present AND compatible with this PyTorch build."""
    if not torch.cuda.is_available():
        return False
    try:
        # Some environments report CUDA as "available" but ship a PyTorch build
        # without kernels for the GPU's compute capability. This specifically
        # breaks cuDNN RNN (LSTM) execution, so we probe with a minimal LSTM.
        with torch.no_grad():
            lstm = torch.nn.LSTM(
                input_size=4,
                hidden_size=4,
                num_layers=1,
                batch_first=True,
            ).to("cuda")
            x = torch.randn(2, 3, 4, device="cuda")
            _ = lstm(x)
        return True
    except Exception:
        return False


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the appropriate device for computation.

    Args:
        device: Device specification ("cuda", "cpu", "auto", or None)

    Returns:
        PyTorch device object
    """
    if device == "auto" or device is None:
        return torch.device("cuda" if _cuda_is_usable() else "cpu")
    if str(device).startswith("cuda") and not _cuda_is_usable():
        return torch.device("cpu")
    return torch.device(device)
