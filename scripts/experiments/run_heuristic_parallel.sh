#!/bin/bash
# Parallel heuristic baselines runner — one policy per GPU simultaneously.
#
# Usage:
#   bash scripts/experiments/run_heuristic_parallel.sh           # auto-detect all GPUs
#   bash scripts/experiments/run_heuristic_parallel.sh 0 1 2 3   # use specific GPUs
#
# Examples:
#   4x GTX 1080:  bash run_heuristic_parallel.sh 0 1 2 3   → ~2h (batch1) + ~1h (batch2)
#   3x GTX 1080:  bash run_heuristic_parallel.sh 0 1 2     → ~2h (batch1) + ~2h (batch2)
#
# Each GPU runs exactly ONE policy at a time to avoid VRAM contention (8GB per 1080).
# Experiments that don't fit the first batch are queued automatically.

set -euo pipefail

# ─── GPU selection ────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    N_GPUS=$(nvidia-smi --list-gpus | wc -l)
    GPUS=($(seq 0 $((N_GPUS - 1))))
else
    GPUS=("$@")
fi
N_GPUS=${#GPUS[@]}
echo "Using ${N_GPUS} GPU(s): ${GPUS[*]}"

# ─── Experiments to run ───────────────────────────────────────────────────────
# Format: "policy_name|extra_overrides"
EXPERIMENTS=(
    "weights_only|"
    "embeddings_only|"
    "fixed_switch|heuristic.switch_episode=1000"
    "periodic|heuristic.period=100"
    "threshold|heuristic.loss_threshold=0.20"
    "fixed_7030|"
)

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ─── Parallel launcher ────────────────────────────────────────────────────────
# Assigns each experiment to a GPU slot. When all slots are busy, waits
# for the current batch to complete before launching the next one.

declare -a PIDS=()
declare -a BATCH_POLICIES=()
batch=1

for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r policy overrides <<< "${EXPERIMENTS[$i]}"
    slot=$(( i % N_GPUS ))
    gpu="${GPUS[$slot]}"
    log="${LOG_DIR}/heuristic_${policy}_${TIMESTAMP}.log"

    echo "[$(date '+%H:%M:%S')] Batch ${batch} | GPU ${gpu} ← ${policy}"

    python train_dfl_heuristic.py \
        +experiment=dfl_heuristic_fair \
        heuristic.policy_name="${policy}" \
        run_name="heuristic_${policy}" \
        device="cuda:${gpu}" \
        lstm_device="cuda:${gpu}" \
        data.batch_size=384 \
        data.num_workers=2 \
        federated.lstm_local_epochs=10 \
        ${overrides} \
        > "${log}" 2>&1 &

    PIDS+=($!)
    BATCH_POLICIES+=("${policy}")

    # When all GPU slots are filled OR this is the last experiment → wait for batch
    next=$(( i + 1 ))
    if [ $(( next % N_GPUS )) -eq 0 ] || [ $next -eq ${#EXPERIMENTS[@]} ]; then
        echo "  Waiting for batch ${batch} to finish: [${BATCH_POLICIES[*]}]"
        for pid in "${PIDS[@]}"; do
            wait "$pid" && echo "  ✓ PID $pid done" || echo "  ✗ PID $pid failed"
        done
        PIDS=()
        BATCH_POLICIES=()
        (( batch++ )) || true
        echo ""
    fi
done

echo "============================================"
echo "  All experiments complete: $(date)"
echo "  Results: outputs/heuristic_*/"
echo "  Logs:    ${LOG_DIR}/"
echo "============================================"
