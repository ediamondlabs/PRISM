# Training Guide — Heuristic Baselines

## Requirements

- Python 3.10+
- CUDA 11.8 compatible GPU (GTX 1080 / 1080 Ti / P100)
- 8 GB+ VRAM, 16 GB+ RAM, 60 GB+ disk

## Quick Start

```bash
# 1. Clone
git clone https://github.com/levgiorg/DRL-v2v-anomaly-detection.git
cd DRL-v2v-anomaly-detection

# 2. Create environment
python -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA 11.8 (must be first)
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run preflight check (auto-assembles dataset + verifies GPU)
python scripts/utils/preflight_check.py

# 6. Run all 6 baselines (4 GPUs)
bash scripts/experiments/run_heuristic_parallel.sh 0 1 2 3
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Episodes per baseline | 3000 |
| Baselines | 6 (weights_only, embeddings_only, fixed_switch, periodic, threshold, fixed_7030) |
| Evaluation interval | Every 100 episodes |

## Results

After completion, results are saved to:

```
outputs/heuristic_<policy>/
    metrics.json          # F1, AUROC, bandwidth, convergence_round
    training_curve.json   # Per-episode metrics
    detection_timing.json # Time-to-detection analysis
    checkpoints/          # Model weights
```

## GPU Options

```bash
# 4 GPUs
bash scripts/experiments/run_heuristic_parallel.sh 0 1 2 3

# 3 GPUs
bash scripts/experiments/run_heuristic_parallel.sh 0 1 2

# Single GPU (sequential)
bash scripts/experiments/run_heuristic_parallel.sh 0
```

## Single Policy Run

```bash
python train_dfl_heuristic.py \
    +experiment=dfl_heuristic_fair \
    heuristic.policy_name=weights_only \
    device=cuda:0 \
    lstm_device=cuda:0
```

## Collect Results

```bash
python - <<'EOF'
import json, pathlib, pandas as pd
rows = []
for p in sorted(pathlib.Path("outputs").glob("heuristic_*/metrics.json")):
    m = json.loads(p.read_text())
    rows.append({
        "policy": m.get("policy"),
        "f1": m.get("f1"),
        "auroc": m.get("auroc"),
        "bandwidth_mb": m.get("total_bandwidth_mb"),
        "convergence": m.get("convergence_round"),
    })
print(pd.DataFrame(rows).to_string(index=False))
EOF
```
