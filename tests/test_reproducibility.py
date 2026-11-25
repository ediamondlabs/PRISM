import numpy as np
import pytest
import torch

from src.utils.reproducibility import get_device, set_deterministic_mode, set_seed


def test_set_seed_reproducibility():
    """Test that setting the same seed produces identical random values."""
    seed = 42

    set_seed(seed)
    python_rand_1 = [torch.rand(1).item() for _ in range(5)]
    numpy_rand_1 = np.random.rand(5)
    torch_rand_1 = torch.rand(5)

    set_seed(seed)
    python_rand_2 = [torch.rand(1).item() for _ in range(5)]
    numpy_rand_2 = np.random.rand(5)
    torch_rand_2 = torch.rand(5)

    assert python_rand_1 == python_rand_2
    assert np.allclose(numpy_rand_1, numpy_rand_2)
    assert torch.allclose(torch_rand_1, torch_rand_2)


def test_different_seeds_produce_different_values():
    """Test that different seeds produce different random values."""
    set_seed(42)
    values_seed_42 = torch.rand(10)

    set_seed(123)
    values_seed_123 = torch.rand(10)

    assert not torch.allclose(values_seed_42, values_seed_123)


def test_get_device_auto():
    """Test device selection with 'auto' option."""
    device = get_device("auto")
    assert isinstance(device, torch.device)
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_get_device_explicit():
    """Test explicit device selection."""
    cpu_device = get_device("cpu")
    assert cpu_device.type == "cpu"

    if torch.cuda.is_available():
        cuda_device = get_device("cuda")
        assert cuda_device.type == "cuda"


def test_get_device_none():
    """Test device selection with None (should behave like 'auto')."""
    device = get_device(None)
    assert isinstance(device, torch.device)


def test_deterministic_mode():
    """Test that deterministic mode can be set without errors."""
    set_deterministic_mode(True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    set_deterministic_mode(False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_neural_network_reproducibility():
    """Test that neural network initialization is reproducible with same seed."""
    seed = 42

    set_seed(seed)
    model1 = torch.nn.Linear(10, 5)
    weights1 = model1.weight.data.clone()

    set_seed(seed)
    model2 = torch.nn.Linear(10, 5)
    weights2 = model2.weight.data.clone()

    assert torch.allclose(weights1, weights2)


def test_forward_pass_reproducibility():
    """Test that forward passes produce identical results with same seed."""
    seed = 42
    x = torch.randn(8, 10)

    set_seed(seed)
    model1 = torch.nn.Linear(10, 5)
    output1 = model1(x)

    set_seed(seed)
    model2 = torch.nn.Linear(10, 5)
    output2 = model2(x)

    assert torch.allclose(output1, output2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_reproducibility():
    """Test reproducibility on CUDA devices."""
    seed = 42

    set_seed(seed)
    x1 = torch.randn(10, device="cuda")

    set_seed(seed)
    x2 = torch.randn(10, device="cuda")

    assert torch.allclose(x1, x2)
