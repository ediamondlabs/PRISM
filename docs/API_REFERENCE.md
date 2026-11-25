# API Reference

Complete API documentation for the DRL V2V Anomaly Detection system.

## Data Pipeline (`src/data/`)

### `dataloader.V2VDataLoader`

```python
class V2VDataLoader:
    """Data loader for V2V trajectory dataset."""

    def __init__(self, config: DictConfig)
```

**Parameters:**
- `config` (DictConfig): Hydra configuration object containing data pipeline settings

**Attributes:**
- `data_path` (Path): Path to dataset CSV file
- `metadata_path` (Path): Path to metadata JSON file
- `features` (list): List of feature names to extract
- `min_presence_duration` (float): Minimum vehicle presence duration in seconds
- `df` (pd.DataFrame): Loaded dataset
- `metadata` (DatasetMetadata): Validated metadata
- `selected_vehicles` (List[int]): IDs of selected vehicles

#### Methods

##### `load_dataset() -> pd.DataFrame`

Load full dataset from CSV file.

**Returns:**
- `pd.DataFrame`: DataFrame containing all vehicle records

**Raises:**
- `FileNotFoundError`: If dataset file not found

**Example:**
```python
loader = V2VDataLoader(config)
df = loader.load_dataset()
print(f"Loaded {len(df)} records")
```

##### `load_metadata() -> DatasetMetadata`

Load and validate dataset metadata from JSON.

**Returns:**
- `DatasetMetadata`: Validated metadata object with Pydantic

**Raises:**
- `FileNotFoundError`: If metadata file not found
- `ValidationError`: If metadata validation fails

##### `compute_vehicle_statistics() -> Dict[int, VehicleStats]`

Compute statistics for each vehicle in the dataset.

**Returns:**
- `Dict[int, VehicleStats]`: Dictionary mapping vehicle_id to statistics

**Statistics Computed:**
- `num_records`: Number of trajectory records
- `presence_duration`: Duration in seconds (timestamp range)
- `avg_speed`: Average speed in m/s
- `max_speed`: Maximum speed in m/s
- `trajectory_length`: Total distance traveled in meters

**Example:**
```python
stats = loader.compute_vehicle_statistics()
for vehicle_id, vstats in stats.items():
    print(f"Vehicle {vehicle_id}: {vstats.presence_duration:.1f}s")
```

##### `select_vehicles(num_vehicles: int = None) -> List[int]`

Select vehicles based on presence duration criterion.

**Parameters:**
- `num_vehicles` (int, optional): Number of vehicles to select (default: from config)

**Returns:**
- `List[int]`: List of selected vehicle IDs, sorted by presence duration (descending)

**Selection Criteria:**
1. Filter vehicles with `presence_duration > min_presence_duration`
2. Sort by presence duration (descending)
3. Select top N vehicles

**Example:**
```python
selected = loader.select_vehicles(num_vehicles=20)
print(f"Selected vehicles: {selected}")
```

##### `extract_features(vehicle_ids: List[int] = None) -> pd.DataFrame`

Extract specified features for selected vehicles.

**Parameters:**
- `vehicle_ids` (List[int], optional): Vehicle IDs to extract (default: selected vehicles)

**Returns:**
- `pd.DataFrame`: Filtered DataFrame with columns: `vehicle_id`, `timestamp`, `frame`, and feature columns

**Extracted Features:**
- `velocity_x`, `velocity_y`, `speed`
- `accel_x`, `accel_y`, `accel_z`
- `gyro_x`, `gyro_y`, `gyro_z`
- `yaw`

**Example:**
```python
features_df = loader.extract_features()
print(f"Extracted {len(features_df)} records with {len(loader.features)} features")
```

##### `get_data_summary() -> Dict`

Get summary statistics of loaded data.

**Returns:**
- `Dict`: Dictionary with keys:
  - `total_records`: Total number of records
  - `unique_vehicles`: Number of unique vehicles
  - `time_range`: Tuple of (min_timestamp, max_timestamp)
  - `duration_seconds`: Total duration
  - `features`: List of feature names
  - `num_features`: Number of features
  - `selected_vehicles` (optional): Number of selected vehicles
  - `selected_vehicle_ids` (optional): List of selected IDs

---

### `trajectory_window.TrajectoryWindowGenerator`

```python
class TrajectoryWindowGenerator:
    """Generate sliding windows from vehicle trajectories."""

    def __init__(
        self,
        window_size: int = 50,
        stride: int = 10,
        features: list = None,
    )
```

**Parameters:**
- `window_size` (int): Number of timesteps per window
- `stride` (int): Stride for sliding window (overlap = window_size - stride)
- `features` (list): List of feature names to extract

#### Methods

##### `generate_windows(data: pd.DataFrame, vehicle_id_col: str = "vehicle_id") -> Tuple[np.ndarray, np.ndarray]`

Generate sliding windows from trajectory data.

**Parameters:**
- `data` (pd.DataFrame): Trajectory data with vehicle_id and features
- `vehicle_id_col` (str): Column name for vehicle ID

**Returns:**
- Tuple of:
  - `windows` (np.ndarray): Shape (num_windows, window_size, num_features)
  - `vehicle_ids` (np.ndarray): Shape (num_windows,) - vehicle ID for each window

**Window Generation:**
```
For vehicle with 100 timesteps (window_size=50, stride=10):
  Window 0: timesteps [0:50]
  Window 1: timesteps [10:60]
  Window 2: timesteps [20:70]
  ...
  Window 5: timesteps [50:100]
  Total: 6 windows
```

**Example:**
```python
window_gen = TrajectoryWindowGenerator(window_size=50, stride=10, features=features)
windows, vehicle_ids = window_gen.generate_windows(data)
print(f"Generated {len(windows)} windows of shape {windows.shape[1:]}")
```

##### `compute_expected_windows_per_vehicle(data: pd.DataFrame, vehicle_id_col: str = "vehicle_id") -> Dict[int, int]`

Compute expected number of windows for each vehicle.

**Parameters:**
- `data` (pd.DataFrame): Trajectory data
- `vehicle_id_col` (str): Column name for vehicle ID

**Returns:**
- `Dict[int, int]`: Dictionary mapping vehicle_id to expected window count

**Formula:**
```python
if num_timesteps >= window_size:
    num_windows = (num_timesteps - window_size) // stride + 1
else:
    num_windows = 0
```

---

### `trajectory_window.create_preprocessed_dataset`

```python
def create_preprocessed_dataset(
    config: DictConfig,
    data: pd.DataFrame,
    save_dir: Path = None,
) -> Dict[str, np.ndarray]
```

Create complete preprocessed dataset with windows, splits, and normalization.

**Parameters:**
- `config` (DictConfig): Hydra configuration
- `data` (pd.DataFrame): Raw trajectory data
- `save_dir` (Path, optional): Directory to save preprocessed data

**Returns:**
- `Dict[str, np.ndarray]`: Dictionary with keys:
  - `train_data`: Training windows (N_train, 50, 10)
  - `val_data`: Validation windows (N_val, 50, 10)
  - `test_data`: Test windows (N_test, 50, 10)
  - `train_vehicle_ids`: Vehicle IDs for training windows
  - `val_vehicle_ids`: Vehicle IDs for validation windows
  - `test_vehicle_ids`: Vehicle IDs for test windows
  - `config`: Configuration dictionary
  - `normalization`: Normalization parameters (if enabled)

**Pipeline Steps:**
1. Generate sliding windows
2. Create train/val/test splits (stratified by vehicle)
3. Normalize data (fit on train, transform all)
4. Save to .npz file (if save_dir provided)

**Example:**
```python
dataset = create_preprocessed_dataset(
    config=config,
    data=features_df,
    save_dir=Path("outputs/preprocessed")
)

print(f"Train: {dataset['train_data'].shape}")
print(f"Val: {dataset['val_data'].shape}")
print(f"Test: {dataset['test_data'].shape}")
```

---

### `preprocessing.TrajectoryNormalizer`

```python
class TrajectoryNormalizer:
    """Normalizer for trajectory features using StandardScaler."""

    def __init__(self, method: str = "standardize")
```

**Parameters:**
- `method` (str): Normalization method ("standardize" only supported currently)

#### Methods

##### `fit(data: np.ndarray) -> TrajectoryNormalizer`

Fit normalizer to training data.

**Parameters:**
- `data` (np.ndarray): Training data of shape (num_samples, window_size, num_features)

**Returns:**
- `TrajectoryNormalizer`: Self for method chaining

**Example:**
```python
normalizer = TrajectoryNormalizer(method="standardize")
normalizer.fit(train_data)
```

##### `transform(data: np.ndarray) -> np.ndarray`

Transform data using fitted normalizer.

**Parameters:**
- `data` (np.ndarray): Data of shape (num_samples, window_size, num_features)

**Returns:**
- `np.ndarray`: Normalized data with same shape

**Transformation:**
```python
# For each feature:
normalized = (data - mean) / std
```

##### `fit_transform(data: np.ndarray) -> np.ndarray`

Fit normalizer and transform data in one step.

**Parameters:**
- `data` (np.ndarray): Training data

**Returns:**
- `np.ndarray`: Normalized data

##### `get_params() -> Dict`

Get normalizer parameters for saving/loading.

**Returns:**
- `Dict`: Dictionary with keys:
  - `method`: Normalization method
  - `mean`: List of mean values per feature
  - `scale`: List of std values per feature

---

### `preprocessing.create_train_val_test_split`

```python
def create_train_val_test_split(
    windows: np.ndarray,
    vehicle_ids: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
```

Create stratified train/val/test splits ensuring vehicles appear in all splits.

**Parameters:**
- `windows` (np.ndarray): Trajectory windows (num_windows, window_size, num_features)
- `vehicle_ids` (np.ndarray): Vehicle ID for each window
- `train_ratio` (float): Proportion for training set (default: 0.7)
- `val_ratio` (float): Proportion for validation set (default: 0.15)
- `test_ratio` (float): Proportion for test set (default: 0.15)
- `random_state` (int): Random seed for reproducibility

**Returns:**
- Tuple of:
  - `data_splits` (Dict): Keys "train", "val", "test" with window arrays
  - `metadata_splits` (Dict): Keys "train", "val", "test" with vehicle ID arrays

**Splitting Strategy:**
```python
For each vehicle:
    1. Get all windows for that vehicle
    2. Shuffle windows with random_state
    3. Split 70/15/15
    4. Add to respective sets

This ensures:
- All vehicles present in all splits
- No temporal leakage
- Stratified by vehicle
```

**Example:**
```python
data_splits, metadata_splits = create_train_val_test_split(
    windows=windows,
    vehicle_ids=vehicle_ids,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
)

print(f"Train: {len(data_splits['train'])} windows")
print(f"Val: {len(data_splits['val'])} windows")
print(f"Test: {len(data_splits['test'])} windows")
```

---

## Models (`src/models/`)

### `encoder_decoder.LSTMEncoder`

```python
class LSTMEncoder(nn.Module):
    """LSTM encoder that compresses trajectory sequences into embeddings."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        embedding_dim: int = 32,
        dropout: float = 0.0,
    )
```

**Parameters:**
- `input_size` (int): Number of input features
- `hidden_size` (int): LSTM hidden size
- `num_layers` (int): Number of LSTM layers
- `embedding_dim` (int): Dimension of output embedding
- `dropout` (float): Dropout probability (applied between LSTM layers)

#### Methods

##### `forward(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`

Encode input sequence to embedding.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape (batch, seq_len, input_size)

**Returns:**
- Tuple of:
  - `embedding` (torch.Tensor): Shape (batch, embedding_dim)
  - `(hidden, cell)` (Tuple): Hidden and cell states
    - `hidden`: Shape (num_layers, batch, hidden_size)
    - `cell`: Shape (num_layers, batch, hidden_size)

**Example:**
```python
encoder = LSTMEncoder(input_size=10, hidden_size=128, embedding_dim=32)
x = torch.randn(32, 50, 10)  # batch=32, seq_len=50, features=10
embedding, (hidden, cell) = encoder(x)
print(f"Embedding shape: {embedding.shape}")  # (32, 32)
```

---

### `encoder_decoder.LSTMDecoder`

```python
class LSTMDecoder(nn.Module):
    """LSTM decoder that reconstructs trajectory sequences from embeddings."""

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 10,
        seq_len: int = 50,
        dropout: float = 0.0,
    )
```

**Parameters:**
- `embedding_dim` (int): Dimension of input embedding
- `hidden_size` (int): LSTM hidden size
- `num_layers` (int): Number of LSTM layers
- `output_size` (int): Number of output features
- `seq_len` (int): Length of output sequence
- `dropout` (float): Dropout probability

#### Methods

##### `forward(embedding: torch.Tensor, hidden_state: Tuple = None) -> torch.Tensor`

Decode embedding to reconstruct sequence.

**Parameters:**
- `embedding` (torch.Tensor): Input embedding of shape (batch, embedding_dim)
- `hidden_state` (Tuple, optional): Initial (hidden, cell) states

**Returns:**
- `torch.Tensor`: Reconstructed sequence of shape (batch, seq_len, output_size)

**Example:**
```python
decoder = LSTMDecoder(embedding_dim=32, hidden_size=128, output_size=10, seq_len=50)
embedding = torch.randn(32, 32)
reconstruction = decoder(embedding)
print(f"Reconstruction shape: {reconstruction.shape}")  # (32, 50, 10)
```

---

### `autoencoder.LSTMAutoencoder`

```python
class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for trajectory anomaly detection."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        embedding_dim: int = 32,
        seq_len: int = 50,
        dropout: float = 0.0,
    )
```

**Parameters:**
- `input_size` (int): Number of input features
- `hidden_size` (int): LSTM hidden size
- `num_layers` (int): Number of LSTM layers
- `embedding_dim` (int): Dimension of embedding space
- `seq_len` (int): Length of input/output sequences
- `dropout` (float): Dropout probability

#### Methods

##### `forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

Forward pass through autoencoder.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape (batch, seq_len, input_size)

**Returns:**
- Tuple of:
  - `reconstruction` (torch.Tensor): Shape (batch, seq_len, input_size)
  - `embedding` (torch.Tensor): Shape (batch, embedding_dim)

**Example:**
```python
model = LSTMAutoencoder(input_size=10, hidden_size=128, embedding_dim=32, seq_len=50)
x = torch.randn(32, 50, 10)
reconstruction, embedding = model(x)
```

##### `encode(x: torch.Tensor) -> torch.Tensor`

Encode input to embedding space.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape (batch, seq_len, input_size)

**Returns:**
- `torch.Tensor`: Embedding of shape (batch, embedding_dim)

##### `decode(embedding: torch.Tensor) -> torch.Tensor`

Decode embedding to reconstructed sequence.

**Parameters:**
- `embedding` (torch.Tensor): Embedding tensor of shape (batch, embedding_dim)

**Returns:**
- `torch.Tensor`: Reconstructed sequence of shape (batch, seq_len, input_size)

##### `compute_reconstruction_error(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor`

Compute reconstruction error (MSE).

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `reduction` (str): Reduction method
  - `"mean"`: Average over all samples and features
  - `"sum"`: Sum over all samples and features
  - `"none"`: Per-sample error (batch,)

**Returns:**
- `torch.Tensor`: Reconstruction error

**Example:**
```python
error_per_sample = model.compute_reconstruction_error(x, reduction="none")
mean_error = model.compute_reconstruction_error(x, reduction="mean")
```

##### `get_num_parameters() -> int`

Get total number of trainable parameters.

**Returns:**
- `int`: Number of parameters

---

### `autoencoder.create_autoencoder_from_config`

```python
def create_autoencoder_from_config(
    config: DictConfig,
    device: torch.device = None
) -> LSTMAutoencoder
```

Create autoencoder from Hydra configuration.

**Parameters:**
- `config` (DictConfig): Hydra configuration object
- `device` (torch.device, optional): Device to place model on

**Returns:**
- `LSTMAutoencoder`: Initialized autoencoder model

**Example:**
```python
from hydra import compose, initialize
from src.utils.reproducibility import get_device

with initialize(config_path="configs"):
    config = compose(config_name="config")

device = get_device("auto")
model = create_autoencoder_from_config(config, device=device)
```

---

## Utilities (`src/utils/`)

### `reproducibility.set_seed`

```python
def set_seed(seed: int) -> None
```

Set random seed for reproducibility across Python, NumPy, and PyTorch.

**Parameters:**
- `seed` (int): Random seed value

**Sets Seeds For:**
- Python `random` module
- NumPy random state
- PyTorch CPU operations
- PyTorch CUDA operations (single + all devices)
- PYTHONHASHSEED environment variable

**Example:**
```python
from src.utils.reproducibility import set_seed

set_seed(42)
# All random operations now reproducible
```

---

### `reproducibility.get_device`

```python
def get_device(device: Optional[str] = None) -> torch.device
```

Get the appropriate device for computation.

**Parameters:**
- `device` (str, optional): Device specification
  - `"cuda"`: Force GPU
  - `"cpu"`: Force CPU
  - `"auto"` or `None`: Auto-detect (use CUDA if available)

**Returns:**
- `torch.device`: PyTorch device object

**Example:**
```python
device = get_device("auto")
model = model.to(device)
x = x.to(device)
```

---

### `logging_utils.setup_logger`

```python
def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger
```

Set up a logger with console and optional file handlers.

**Parameters:**
- `name` (str): Logger name
- `log_file` (Path, optional): Path to log file
- `level` (int): Logging level (default: INFO)
- `format_string` (str, optional): Custom format string

**Returns:**
- `logging.Logger`: Configured logger instance

**Default Format:**
```
[2025-10-19 13:45:30][module_name][INFO] Log message here
```

**Example:**
```python
from src.utils.logging_utils import setup_logger

logger = setup_logger(
    name="my_module",
    log_file=Path("logs/training.log"),
    level=logging.INFO
)

logger.info("Training started")
logger.warning("Learning rate too high")
logger.error("CUDA out of memory")
```

---

## Data Schemas (`src/data/schemas.py`)

All schemas use Pydantic for validation.

### `VehicleRecord`

Single vehicle sensor record with 23 fields validated for physical constraints.

### `TrajectoryFeatures`

10-feature trajectory window with sequence length validation.

### `DatasetMetadata`

Dataset metadata with sensor specifications.

### `VehicleStats`

Per-vehicle statistics (presence, speed, trajectory length).

---

## Type Hints

All functions use type hints for better IDE support and type checking:

```python
from typing import List, Tuple, Dict, Optional
import torch
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from pathlib import Path
```
