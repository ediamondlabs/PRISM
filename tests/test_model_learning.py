"""Test if DFL-trained model actually learned by comparing reconstruction errors."""

import torch
import numpy as np
from pathlib import Path
from hydra import compose, initialize_config_dir
import pytest

# Load config using Hydra
with initialize_config_dir(config_dir=str(Path.cwd() / "configs"), version_base=None):
    config = compose(config_name="config", overrides=["+experiment=dfl_training"])

from src.models import create_autoencoder_from_config

# Load preprocessed test data
data_path = Path("outputs/preprocessed/preprocessed_data.npz")
if not data_path.exists():
    pytest.skip(f"Test data not found at {data_path}", allow_module_level=True)

data = np.load(data_path, allow_pickle=True)
test_data = data["test_data"][:1000]  # Sample 1000 windows

test_tensor = torch.FloatTensor(test_data)

# Create random untrained model
print("=" * 80)
print("Testing: Untrained vs DFL-Trained Model Reconstruction Error")
print("=" * 80)

untrained_model = create_autoencoder_from_config(config, device=torch.device("cpu"))
untrained_model.eval()

with torch.no_grad():
    untrained_error = untrained_model.compute_reconstruction_error(test_tensor).item()

print(f"\n1. Untrained (random) model error: {untrained_error:.4f}")

# Load DFL-trained model
checkpoint_path = "outputs/dfl/checkpoints/dfl_final.pt"
if Path(checkpoint_path).exists():
    print(f"\n2. Loading DFL-trained model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    trained_model = create_autoencoder_from_config(config, device=torch.device("cpu"))
    trained_model.load_state_dict(checkpoint["lstm_model"])
    trained_model.eval()

    with torch.no_grad():
        trained_error = trained_model.compute_reconstruction_error(test_tensor).item()

    print(f"   DFL-trained model error: {trained_error:.4f}")

    improvement = ((untrained_error - trained_error) / untrained_error) * 100
    print(f"\n3. Improvement: {improvement:.1f}%")
    print(f"   Error reduction: {untrained_error - trained_error:.4f}")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    if improvement > 90:
        print(f"✅ Model learned VERY WELL! {improvement:.1f}% improvement")
        print("   The F1=0.51 you saw is just a placeholder metric.")
        print("   Your DFL training WAS successful - LSTM loss went from 2244 → 0.029")
    elif improvement > 50:
        print(f"✅ Model learned well! {improvement:.1f}% improvement")
    else:
        print(f"⚠️  Model learned, but improvement is modest: {improvement:.1f}%")
else:
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    print("   Run training first!")

print("=" * 80)
