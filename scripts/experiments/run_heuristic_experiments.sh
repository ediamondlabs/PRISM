#!/bin/bash
# Run all heuristic baselines for PRISM paper revision (LCN/MASS 2026).
#
# Addresses R3+R4 (heuristic baselines) and R3+R4 (ablation study).
# All 6 experiments use identical DFL infrastructure as dfl_training.yaml.
#
# Estimated runtime: ~1.5-2 hours per experiment on P100/RTX 4070
# Total: ~9-12 hours for 6 core experiments
#
# Usage:
#   bash scripts/experiments/run_heuristic_experiments.sh          # cuda:0 (default)
#   bash scripts/experiments/run_heuristic_experiments.sh --gpu 1  # cuda:1
#   bash scripts/experiments/run_heuristic_experiments.sh --cpu    # cpu

set -euo pipefail

DEVICE="cuda:0"
if [[ "${1:-}" == "--cpu" ]]; then
    DEVICE="cpu"
elif [[ "${1:-}" == "--gpu" ]]; then
    GPU_ID="${2:-0}"
    DEVICE="cuda:${GPU_ID}"
fi

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_experiment() {
    local policy="$1"
    shift
    local extra_args=("$@")

    echo "========================================"
    echo "[$(date '+%H:%M:%S')] Starting: $policy ${extra_args[*]:-}"
    echo "========================================"

    python train_dfl_heuristic.py \
        +experiment=dfl_heuristic \
        heuristic.policy_name="$policy" \
        run_name="heuristic_${policy}" \
        device="$DEVICE" \
        lstm_device="$DEVICE" \
        "${extra_args[@]}" \
        2>&1 | tee "${LOG_DIR}/heuristic_${policy}_${TIMESTAMP}.log"

    echo "[$(date '+%H:%M:%S')] Completed: $policy"
    echo ""
}

echo "============================================"
echo "  PRISM Heuristic Baselines — Paper Revision"
echo "  Started: $(date)"
echo "  Device: $DEVICE"
echo "============================================"
echo ""

# =============================================
# Ablation baselines (R3, R4: representation ablation)
# =============================================
# Tests contribution of each representation type
run_experiment "weights_only"
run_experiment "embeddings_only"

# =============================================
# Heuristic baselines (R3, R4: fixed strategies)
# =============================================
# FixedSwitch: Weights for rounds 1-1000, Embeddings after.
# 1000 chosen close to PRISM convergence (1200±150 from Table 3).
run_experiment "fixed_switch" heuristic.switch_episode=1000

# Periodic: Alternate W/E every 100 episodes (50/50 split over 3000 rounds).
run_experiment "periodic" heuristic.period=100

# Threshold: Weights when loss>0.20, Embeddings otherwise.
# 0.20 sits between initial loss (0.52) and PRISM's converged loss (0.062).
run_experiment "threshold" heuristic.loss_threshold=0.20

# =============================================
# Ratio ablation (tests if RL timing matters)
# =============================================
# Matches PRISM's emergent 70/30 W/E ratio but randomized (no temporal structure).
run_experiment "fixed_7030"

echo "============================================"
echo "  All 6 experiments completed: $(date)"
echo "  Results in: outputs/heuristic_*/"
echo "============================================"
