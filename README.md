# PRISM: PPO-driven Representation sharing In Self-organizing Mesh networks

Deep Reinforcement Learning system for intelligent decentralized federated learning in Vehicle-to-Vehicle (V2V) networks with trajectory anomaly detection.

## Architecture

Four-layer system where an RL agent controls federated communication in a V2V mesh:

| Layer | Component | Purpose |
|-------|-----------|---------|
| 1 | Bidirectional LSTM Autoencoder | Trajectory anomaly detection via reconstruction error |
| 2 | DeepSets Neighbor Encoder | Permutation-invariant neighborhood encoding |
| 3 | PPO Actor-Critic (CTDE) | WHO to communicate with, WHAT representation to send |
| 4 | Asynchronous DFL | Staleness-weighted aggregation of weights/embeddings |

**Key design choices:**
- WHO + WHAT heads (no WHEN head — paper uses only two decision heads)
- Representation types: **W** (model weights) and **E** (embeddings) — RL learns the 70/30 ratio
- Pure staleness-weighted averaging: `z̄ = (1/Σw_j) × Σ(w_j × z_j)` where `w_j = exp(-λ × Δt_j)`
- Bidirectional encoder (4 layers, hidden=768, dz=256), unidirectional decoder

## Quick Start

```bash
pip install -r requirements.txt

# Layer 1: Train LSTM autoencoder
python train.py +experiment=baseline

# Layer 3: Train PPO agent
python train_rl.py +experiment=rl_training

# Full DFL training
python train_dfl.py +experiment=dfl_training

# Run tests
python -m pytest tests/ -v
```

## Configuration

All configs managed via [Hydra](https://hydra.cc/) under `configs/`:

```bash
# Override any parameter via CLI
python train.py seed=42 model.architecture.hidden_size=768 training.num_epochs=30

# Use experiment presets
python train.py +experiment=baseline
```

Key config files:
- `configs/model/lstm_autoencoder.yaml` — Architecture (hidden_size, num_layers, bidirectional, embedding_dim)
- `configs/rl/ppo.yaml` — PPO hyperparameters, actor/critic architecture, reward weights
- `configs/simulation/v2v_env.yaml` — V2V environment (num_vehicles, comm_range)

## Project Structure

```
configs/              Hydra YAML configs
src/
  data/               Loading, validation, preprocessing
  models/             LSTM autoencoder, neighbor encoder
  rl/                 PPO actor/critic, rewards, env wrapper
  federated/          Representations, aggregation, DFL protocol
  simulation/         V2V environment, vehicle nodes
  utils/              Reproducibility, logging
tests/                pytest suite
```

## Dataset

SUMO-generated V2V trajectories (Barcelona urban grid):
- 3,999 vehicles, 3,600s duration, 10 Hz sampling
- 10 features: velocity, acceleration, gyroscope, yaw
- Window size: 50 timesteps (5s), stride: 10

## Model Specifications (Paper Table IV)

| Component | Parameter | Value |
|-----------|-----------|-------|
| Encoder LSTM | Layers / Hidden / Bidirectional | 4 / 768 / Yes |
| Embedding | dz | 256 |
| Neighbor features | df | 40 (3+3+1+1+32 projected) |
| Actor | State dim | 106 (10 local + 96 neighbor) |
| Actor | Heads | WHO (top-k=2) + WHAT (W/E) |
| Critic | Type | Centralized (CTDE) |
| PPO | lr_actor / lr_critic | 3e-4 / 1e-3 |
| Reward | r = βi×r_imm + βg×r_global | 0.3 / 0.7 |

## References

- SUMO Traffic Simulator: https://www.eclipse.org/sumo/
- PPO: Schulman et al. (2017)
- DeepSets: Zaheer et al. (2017)
- FedAvg: McMahan et al. (2017)

## License

Research project for academic purposes.
