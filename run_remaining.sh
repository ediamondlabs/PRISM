#!/bin/bash
# ============================================================
#  PRISM — Remaining Heuristic Baselines (threshold + fixed_7030)
#
#  Handles everything: dependencies, dataset assembly,
#  environment validation, and parallel training.
#
#  Usage:
#    bash run_remaining.sh
#
#  GPUs are auto-detected via nvidia-smi.
#  Results:
#    outputs/heuristic_threshold/
#    outputs/heuristic_fixed_7030/
#    outputs/logs/heuristic_*_<timestamp>.log
#
#  Episodes: 1000 | Local epochs: 3 (~2 days on GTX 1080 Ti)
# ============================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
warn() { echo -e "${YELLOW}  !${NC} $*"; }
fail() { echo -e "${RED}  ✗${NC} $*"; exit 1; }

# ─── Auto-detect GPUs ────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    N=$(nvidia-smi --list-gpus | wc -l)
    GPUS=($(seq 0 $(( N - 1 ))))
else
    fail "nvidia-smi not found — CUDA GPU required"
fi

if [ "${#GPUS[@]}" -lt 2 ]; then
    fail "Need at least 2 GPUs (found ${#GPUS[@]}). threshold→GPU0, fixed_7030→GPU1."
fi

echo ""
echo "============================================================"
echo "  PRISM — Remaining Heuristic Baselines"
echo "  threshold  → GPU ${GPUS[0]}"
echo "  fixed_7030 → GPU ${GPUS[1]}"
echo "  Episodes: 1000 | Local epochs: 3"
echo "  Results: outputs/heuristic_*/"
echo "============================================================"
echo ""

# ─── Step 1: Python version ──────────────────────────────────────────────────
log "Step 1/5  Checking Python version"
PY_MAJOR=$(python -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    fail "Python 3.10+ required, got ${PY_MAJOR}.${PY_MINOR}"
fi
ok "Python ${PY_MAJOR}.${PY_MINOR}"

# ─── Step 2: PyTorch + CUDA ──────────────────────────────────────────────────
log "Step 2/5  PyTorch + CUDA setup"

TORCH_OK=$(python -c "
try:
    import torch
    v = torch.__version__
    print('yes' if v.startswith('2.2') and 'cu118' in v and torch.cuda.is_available() else 'no')
except ImportError:
    print('missing')
" 2>/dev/null)

if [ "$TORCH_OK" = "yes" ]; then
    ok "PyTorch 2.2.0+cu118 with CUDA already installed"
else
    warn "Installing PyTorch 2.2.0+cu118 (required for GTX 1080 Ti)..."
    pip install --quiet \
        torch==2.2.0+cu118 \
        torchvision==0.17.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 \
    || fail "PyTorch install failed — check internet access"
    ok "PyTorch installed"
fi

NUMPY_OK=$(python -c "
import numpy as np
print('yes' if int(np.__version__.split('.')[0]) < 2 else 'no')
" 2>/dev/null || echo "no")

if [ "$NUMPY_OK" != "yes" ]; then
    warn "NumPy 2.x detected — downgrading to 1.x..."
    pip install --quiet "numpy<2" || warn "NumPy pin failed — may see import warnings"
    ok "NumPy pinned to 1.x"
fi

# ─── Step 3: Python dependencies ─────────────────────────────────────────────
log "Step 3/5  Installing Python dependencies"
pip install --quiet -r requirements.txt \
    || fail "pip install requirements.txt failed"
ok "All packages installed"

# ─── Step 4: Dataset assembly ─────────────────────────────────────────────────
log "Step 4/5  Dataset check"

CSV="data/v2v_dataset_20251017_124247.csv"
CHUNKS_DIR="data/chunks"

if [ -f "$CSV" ]; then
    SIZE=$(python -c "import os; print(f'{os.path.getsize(\"$CSV\")/1e9:.2f}')" 2>/dev/null || echo "?")
    ok "Dataset already present (${SIZE} GB)"
elif [ -d "$CHUNKS_DIR" ] && ls "$CHUNKS_DIR"/chunk_*.part &>/dev/null 2>&1; then
    warn "CSV missing — assembling from chunks (takes ~30 seconds)..."
    python scripts/utils/assemble_dataset.py \
        || fail "Dataset assembly failed — check data/chunks/ contents"
    ok "Dataset assembled"
else
    fail "No dataset and no chunks found in ${CHUNKS_DIR}/ — did you run 'git clone'?"
fi

# ─── Step 5: Environment validation ──────────────────────────────────────────
log "Step 5/5  Environment validation"
python scripts/utils/preflight_check.py || {
    echo ""
    warn "Preflight reported issues. Continuing in 10 seconds (Ctrl+C to abort)..."
    sleep 10
}

# ─── Launch both baselines in parallel ───────────────────────────────────────
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "============================================================"
log "Launching threshold + fixed_7030 in parallel"
echo "  Log dir: ${LOG_DIR}/"
echo "============================================================"
echo ""

python train_dfl_heuristic.py \
    +experiment=dfl_heuristic_fair \
    heuristic.policy_name=threshold \
    heuristic.loss_threshold=0.20 \
    run_name=heuristic_threshold \
    device="cuda:${GPUS[0]}" \
    lstm_device="cuda:${GPUS[0]}" \
    data.batch_size=384 \
    data.num_workers=4 \
    federated.lstm_local_epochs=3 \
    rl.training.num_episodes=1000 \
    > "${LOG_DIR}/heuristic_threshold_${TIMESTAMP}.log" 2>&1 &
PID0=$!
echo "[$(date '+%H:%M:%S')] threshold  started → GPU ${GPUS[0]} (PID ${PID0})"

python train_dfl_heuristic.py \
    +experiment=dfl_heuristic_fair \
    heuristic.policy_name=fixed_7030 \
    run_name=heuristic_fixed_7030 \
    device="cuda:${GPUS[1]}" \
    lstm_device="cuda:${GPUS[1]}" \
    data.batch_size=384 \
    data.num_workers=4 \
    federated.lstm_local_epochs=3 \
    rl.training.num_episodes=1000 \
    > "${LOG_DIR}/heuristic_fixed_7030_${TIMESTAMP}.log" 2>&1 &
PID1=$!
echo "[$(date '+%H:%M:%S')] fixed_7030 started → GPU ${GPUS[1]} (PID ${PID1})"

echo ""
echo "Both running. Waiting for completion..."
wait "$PID0" && ok "threshold  done" || warn "threshold  FAILED — check ${LOG_DIR}/heuristic_threshold_${TIMESTAMP}.log"
wait "$PID1" && ok "fixed_7030 done" || warn "fixed_7030 FAILED — check ${LOG_DIR}/heuristic_fixed_7030_${TIMESTAMP}.log"

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo -e "${GREEN}  All done!${NC}"
echo ""
echo "  Results:  outputs/heuristic_threshold/"
echo "            outputs/heuristic_fixed_7030/"
echo "  Logs:     ${LOG_DIR}/heuristic_*_${TIMESTAMP}.log"
echo "============================================================"
