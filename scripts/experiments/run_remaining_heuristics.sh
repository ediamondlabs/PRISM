#!/bin/bash
# Run the 2 remaining heuristic baselines: threshold (GPU 0) + fixed_7030 (GPU 1)
# Each gets its own GPU, both run in parallel.

set -euo pipefail

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "  Remaining heuristic baselines"
echo "  threshold  → GPU 0"
echo "  fixed_7030 → GPU 1"
echo "  Logs: ${LOG_DIR}/"
echo "============================================"

# threshold on GPU 0
python train_dfl_heuristic.py \
    +experiment=dfl_heuristic_fair \
    heuristic.policy_name=threshold \
    heuristic.loss_threshold=0.20 \
    run_name=heuristic_threshold \
    device=cuda:0 \
    lstm_device=cuda:0 \
    data.batch_size=384 \
    data.num_workers=4 \
    federated.lstm_local_epochs=3 \
    > "${LOG_DIR}/heuristic_threshold_${TIMESTAMP}.log" 2>&1 &
PID0=$!
echo "[$(date '+%H:%M:%S')] Started threshold  (PID ${PID0})"

# fixed_7030 on GPU 1
python train_dfl_heuristic.py \
    +experiment=dfl_heuristic_fair \
    heuristic.policy_name=fixed_7030 \
    run_name=heuristic_fixed_7030 \
    device=cuda:1 \
    lstm_device=cuda:1 \
    data.batch_size=384 \
    data.num_workers=4 \
    federated.lstm_local_epochs=3 \
    > "${LOG_DIR}/heuristic_fixed_7030_${TIMESTAMP}.log" 2>&1 &
PID1=$!
echo "[$(date '+%H:%M:%S')] Started fixed_7030 (PID ${PID1})"

echo ""
echo "Both running. Waiting for completion..."
wait "$PID0" && echo "[$(date '+%H:%M:%S')] threshold  done" || echo "[$(date '+%H:%M:%S')] threshold  FAILED"
wait "$PID1" && echo "[$(date '+%H:%M:%S')] fixed_7030 done" || echo "[$(date '+%H:%M:%S')] fixed_7030 FAILED"

echo ""
echo "============================================"
echo "  Done. Results in outputs/heuristic_*/"
echo "============================================"
