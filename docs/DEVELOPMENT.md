# Development Guide

Internal documentation for developers working on the DRL V2V Anomaly Detection system.

## 📐 Code Architecture

### Design Principles

1. **Modularity:** Each component (data, model, training) is independent
2. **Testability:** Every module has comprehensive unit and integration tests
3. **Configurability:** All hyperparameters managed via Hydra
4. **Reproducibility:** Seed control at all levels
5. **Type Safety:** Type hints throughout the codebase

### Code Style

Following ML Engineering global standards:
- **PEP 8** for Python style
- **PEP 257** for docstrings
- **Type hints** everywhere
- **f-strings** for formatting
- **snake_case** for functions/variables
- **PascalCase** for classes

## 🔍 Component Deep Dive

### 1. Data Pipeline (`src/data/`)

#### `dataloader.py`

**V2VDataLoader class:**
```python
class V2VDataLoader:
    """Handles loading and preprocessing V2V dataset."""

    def load_dataset() -> pd.DataFrame
        # Loads 4M+ records efficiently
        # Returns: Full DataFrame

    def select_vehicles(num_vehicles: int) -> List[int]
        # Filters by presence duration >100s
        # Returns: List of vehicle IDs

    def extract_features(vehicle_ids: List[int]) -> pd.DataFrame
        # Extracts 10 trajectory features
        # Returns: Filtered DataFrame with features
```

**Key Implementation Details:**
- Uses pandas for efficient CSV loading
- Validates with Pydantic schemas
- Computes statistics per vehicle (presence duration, speed, etc.)
- Sorts by timestamp to maintain temporal order

#### `schemas.py`

Pydantic validation ensures data quality:
```python
class VehicleRecord(BaseModel):
    frame: int = Field(gt=0)
    timestamp: float = Field(ge=0.0)
    speed: float = Field(ge=0.0)  # Non-negative
    # ... 20 more fields with validation
```

**Validation catches:**
- Negative speeds (physical impossibility)
- Missing values
- Invalid data types
- Out-of-range sensor readings

#### `trajectory_window.py`

**TrajectoryWindowGenerator:**
```python
def generate_windows(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sliding windows with stride=10

    For vehicle with 100 timesteps:
    - Window 0: [0:50]
    - Window 1: [10:60]
    - Window 2: [20:70]
    - ...
    - Window 5: [50:100]

    Returns: (windows, vehicle_ids)
    - windows: (num_windows, 50, 10)
    - vehicle_ids: (num_windows,)
    """
```

**Performance Optimization:**
- Processes vehicles in parallel (groupby)
- Skips vehicles with <50 timesteps
- Pre-allocates arrays for efficiency

#### `preprocessing.py`

**TrajectoryNormalizer:**
```python
class TrajectoryNormalizer:
    """StandardScaler for trajectory windows."""

    def fit_transform(data):
        # Reshapes (N, 50, 10) → (N*50, 10)
        # Fits StandardScaler
        # Transforms and reshapes back
        # Returns: Normalized (N, 50, 10)
```

**Split Strategy:**
```python
def create_train_val_test_split(...):
    """
    Stratified splitting per vehicle:

    For each vehicle:
        1. Get all windows for that vehicle
        2. Split 70/15/15
        3. Add to respective sets

    Ensures:
    - All vehicles in all splits
    - No temporal leakage
    - Representative distributions
    """
```

### 2. Models (`src/models/`)

#### `encoder_decoder.py`

**LSTMEncoder:**
```python
Architecture:
    Input: (batch, 50, 10)
    ↓
    LSTM (2 layers, hidden=128)
    - Takes last hidden state
    ↓
    Linear (128 → 32)
    ↓
    Output: (batch, 32) embedding
```

**Key Design Choice:**
- Uses last hidden state (not mean of all outputs)
- Batch-first=True for easier handling
- Returns hidden/cell states for decoder

**LSTMDecoder:**
```python
Architecture:
    Input: (batch, 32) embedding
    ↓
    Linear (32 → 128)
    - Repeat 50 times for seq_len
    ↓
    LSTM (2 layers, hidden=128)
    - Processes repeated input
    ↓
    Linear (128 → 10) applied to each timestep
    ↓
    Output: (batch, 50, 10) reconstruction
```

**Alternative Designs Considered:**
1. **Autoregressive decoder** (slower, better for generation)
2. **Attention-based** (more parameters, overkill for this task)
3. **Repeated embedding** ✅ (simple, effective for reconstruction)

#### `autoencoder.py`

**LSTMAutoencoder:**
```python
def forward(x):
    """
    x: (batch, 50, 10)

    Steps:
    1. embedding, hidden_state = encoder(x)
    2. reconstruction = decoder(embedding, hidden_state)

    Returns: (reconstruction, embedding)
    """
```

**Design Patterns:**
- Factory pattern: `create_autoencoder_from_config()`
- Separate `encode()` and `decode()` methods for flexibility
- `compute_reconstruction_error()` for anomaly detection

### 3. Utilities (`src/utils/`)

#### `reproducibility.py`

**Seed Control:**
```python
def set_seed(seed: int):
    """
    Sets seeds for:
    - Python random
    - NumPy random
    - PyTorch CPU
    - PyTorch CUDA (single + all devices)
    - PYTHONHASHSEED env variable
    """
```

**Deterministic Mode:**
```python
def set_deterministic_mode(enabled: bool):
    """
    Controls:
    - torch.backends.cudnn.deterministic
    - torch.backends.cudnn.benchmark

    Trade-off:
    - Deterministic=True: Slower, reproducible
    - Deterministic=False: Faster, non-deterministic
    """
```

#### `logging_utils.py`

**Structured Logging:**
```python
logger = setup_logger(
    name="module_name",
    log_file=Path("logs/train.log"),
    level=logging.INFO
)

# Creates:
# 1. Console handler (stdout)
# 2. File handler (optional)
# Both with consistent formatting
```

## 🧪 Testing Strategy

### Test Organization

```
tests/
├── test_config.py           # Configuration validation
├── test_data_loading.py     # Data pipeline (unit + integration)
├── test_windowing.py         # Window generation and preprocessing
├── test_model.py            # Model architecture
├── test_reproducibility.py  # Seed control
└── test_logging.py          # Logging utilities
```

### Test Types

**1. Unit Tests:**
- Test individual functions in isolation
- Mock external dependencies
- Fast execution (<1s each)

**2. Integration Tests:**
- Test component interactions
- Use real data (small subset)
- Verify end-to-end flows

**3. Property Tests:**
- Test invariants (e.g., "split ratios sum to 1.0")
- Validate shapes, types, ranges
- Check for NaN/Inf values

### Testing Best Practices

```python
# ✅ Good: Tests one thing clearly
def test_encoder_output_shape():
    encoder = LSTMEncoder(input_size=10, hidden_size=128, embedding_dim=32)
    x = torch.randn(8, 50, 10)
    embedding, _ = encoder(x)
    assert embedding.shape == (8, 32)

# ❌ Bad: Tests multiple things
def test_encoder():
    # Creates encoder
    # Tests forward pass
    # Tests backward pass
    # Tests device placement
    # Tests reproducibility
    # ... too much in one test!
```

### Coverage Goals

- **Lines:** >90%
- **Branches:** >80%
- **Critical paths:** 100%

## 🔧 Configuration System

### Hydra Composition

```yaml
# configs/config.yaml
defaults:
  - data: v2v_trajectories    # From configs/data/
  - model: lstm_autoencoder   # From configs/model/
  - training: local_task      # From configs/training/
  - _self_                    # Main config overrides

# This creates a unified config object
```

### Override Priority

1. **Command line** (highest): `python train.py seed=123`
2. **Experiment config:** `+experiment=baseline`
3. **Defaults:** Listed in `config.yaml`
4. **Individual configs:** `data/v2v_trajectories.yaml`

### Adding New Configurations

1. Create YAML in appropriate subdirectory
2. Follow existing structure
3. Add validation in tests
4. Document in README

Example:
```yaml
# configs/model/transformer_autoencoder.yaml
name: transformer_autoencoder

architecture:
  input_size: 10
  d_model: 128
  nhead: 8
  num_layers: 3
  embedding_dim: 32
```

## 🚀 Performance Optimization

### Data Loading

**Current Performance:**
- Load 4M records: ~2-3 seconds
- Window generation (20 vehicles): ~0.5 seconds
- Train/val/test split: ~0.1 seconds

**Optimization Techniques:**
1. **Pandas optimizations:**
   - Read CSV with optimal dtypes
   - Use categorical for vehicle_id
   - Group operations for per-vehicle processing

2. **NumPy vectorization:**
   - Avoid Python loops
   - Use array operations
   - Pre-allocate arrays

3. **Memory efficiency:**
   - Process vehicles in groups
   - Use float32 instead of float64
   - Save as compressed .npz

### Model Inference

**Batch Processing:**
```python
# ✅ Good: Process in batches
for batch in dataloader:
    with torch.no_grad():
        reconstruction, _ = model(batch)

# ❌ Bad: Process one by one
for sample in dataset:
    with torch.no_grad():
        reconstruction, _ = model(sample.unsqueeze(0))
```

**GPU Utilization:**
- Pin memory: `pin_memory=True` in DataLoader
- Non-blocking transfers: `x.to(device, non_blocking=True)`
- Mixed precision (coming): `torch.cuda.amp`

## 🐛 Debugging Tips

### Common Issues

**1. Shape Mismatches**
```python
# Add shape logging
logger.info(f"Input shape: {x.shape}")
reconstruction, embedding = model(x)
logger.info(f"Reconstruction shape: {reconstruction.shape}")
logger.info(f"Embedding shape: {embedding.shape}")
```

**2. NaN Values**
```python
# Check for NaN after each operation
assert not torch.isnan(x).any(), "NaN in input"
reconstruction, _ = model(x)
assert not torch.isnan(reconstruction).any(), "NaN in reconstruction"
```

**3. Gradient Issues**
```python
# Monitor gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        logger.info(f"{name}: grad_norm={grad_norm:.4f}")
```

### Profiling

```python
# Time critical sections
import time

start = time.time()
windows, vehicle_ids = window_gen.generate_windows(data)
elapsed = time.time() - start
logger.info(f"Window generation took {elapsed:.2f}s")
```

## 📊 MLflow Integration (Coming)

### Experiment Tracking

```python
# Log hyperparameters
mlflow.log_params({
    "learning_rate": config.training.learning_rate,
    "batch_size": config.data.batch_size,
    "hidden_size": config.model.architecture.hidden_size,
})

# Log metrics per epoch
mlflow.log_metrics({
    "train_loss": train_loss,
    "val_loss": val_loss,
    "val_f1": f1_score,
}, step=epoch)

# Log model
mlflow.pytorch.log_model(model, "model")

# Log artifacts
mlflow.log_artifact("configs/config.yaml")
mlflow.log_artifact("outputs/plots/loss_curve.png")
```

### Query Experiments

```python
# Find best run
runs = mlflow.search_runs(
    experiment_names=["v2v_anomaly_detection"],
    order_by=["metrics.val_f1 DESC"],
    max_results=1
)

best_run = runs.iloc[0]
best_f1 = best_run["metrics.val_f1"]
```

## 🔄 Git Workflow (Recommended)

### Branch Strategy

- `main` - Stable, tested code
- `dev` - Integration branch
- `feature/X` - New features
- `bugfix/X` - Bug fixes

### Commit Messages

```bash
# Good commit messages
git commit -m "feat: Add LSTM encoder with hidden state extraction"
git commit -m "fix: Handle empty validation set in normalization"
git commit -m "test: Add property tests for window generation"
git commit -m "docs: Update README with installation instructions"

# Types: feat, fix, test, docs, refactor, perf, style
```

## 🎯 Next Steps for Developers

### Short-term (Phase 1 completion)

1. **Training infrastructure:**
   - Implement `Trainer` class
   - Add early stopping logic
   - Integrate MLflow tracking
   - Add checkpoint management

2. **Anomaly detection:**
   - Threshold tuning on validation set
   - F1/Precision/Recall computation
   - Anomaly scoring per window
   - Visualization utilities

3. **Integration:**
   - Main `train.py` script
   - CLI with Hydra
   - End-to-end tests

### Medium-term (Phase 2)

1. **Neighbor encoder:**
   - Per-neighbor feature extraction
   - Permutation-invariant aggregation
   - Set encoding (mean/max/std)

2. **Network simulation:**
   - V2V connectivity graph
   - Link quality computation
   - Dynamic topology handling

### Long-term (Phases 3-4)

1. **RL agent (PPO):**
   - Actor-critic architecture
   - Multi-task decision heads
   - CTDE training setup

2. **Federated learning:**
   - Communication protocol
   - FedAvg aggregation
   - Multiple representations

## 📚 Resources

### Learning Materials

- **PyTorch:** https://pytorch.org/tutorials/
- **Hydra:** https://hydra.cc/docs/intro/
- **MLflow:** https://mlflow.org/docs/latest/index.html
- **Pytest:** https://docs.pytest.org/

### Code References

- **LSTMs:** https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- **Autoencoders:** https://arxiv.org/abs/1312.6114
- **Federated Learning:** https://arxiv.org/abs/1602.05629

## ✅ Code Review Checklist

Before submitting code:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Code follows PEP 8 style
- [ ] Type hints added
- [ ] Docstrings written
- [ ] No print statements (use logger)
- [ ] No hardcoded paths
- [ ] Configuration via Hydra
- [ ] Tests added for new functionality
- [ ] README updated if needed
- [ ] No NaN/Inf checks for numerical code
- [ ] Reproducibility maintained (seed control)

## 💡 Tips for New Contributors

1. **Start with tests:** Read tests to understand component behavior
2. **Use configuration:** Don't hardcode parameters
3. **Log everything:** Use structured logging
4. **Test incrementally:** Write test first, then implementation
5. **Profile before optimizing:** Measure, don't guess
6. **Ask questions:** Check DEVELOPMENT.md and existing code first
