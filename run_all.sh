#!/bin/bash
# ============================================================
#  PRISM Heuristic Baselines — Full Setup & Run
#
#  Handles everything: dependencies, dataset assembly,
#  environment validation, and parallel training.
#
#  Usage:
#    bash run_all.sh
#
#  GPUs are auto-detected via nvidia-smi.
#  Results:
#    outputs/heuristic_<policy>/       <- MLflow metrics & checkpoints
#    outputs/logs/heuristic_<policy>_<timestamp>.log  <- training logs
#
#  Episodes per baseline: 3000 (dfl_heuristic_fair config)
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

# ─── Auto-detect GPUs ─────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    N=$(nvidia-smi --list-gpus | wc -l)
    GPUS=($(seq 0 $(( N - 1 ))))
else
    fail "nvidia-smi not found — CUDA GPU required to run these experiments"
fi

echo ""
echo "============================================================"
echo "  PRISM Heuristic Baselines — Full Setup & Run"
echo "  GPUs: ${GPUS[*]}"
echo "  Episodes per baseline: 3000"
echo "  Results: outputs/heuristic_*/"
echo "============================================================"
echo ""

# ─── Step 1: Python version ───────────────────────────────────────────────────
log "Step 1/5  Checking Python version"
PY_MAJOR=$(python -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    fail "Python 3.10+ required, got ${PY_MAJOR}.${PY_MINOR}"
fi
ok "Python ${PY_MAJOR}.${PY_MINOR}"

# ─── Step 2: PyTorch + CUDA ───────────────────────────────────────────────────
log "Step 2/5  PyTorch + CUDA setup"

TORCH_OK=$(python -c "
import sys
try:
    import torch
    v = torch.__version__
    ok = v.startswith('2.2') and 'cu118' in v and torch.cuda.is_available()
    print('yes' if ok else 'no')
except ImportError:
    print('missing')
" 2>/dev/null)

if [ "$TORCH_OK" = "yes" ]; then
    ok "PyTorch 2.2.0+cu118 with CUDA already installed"
else
    warn "Installing PyTorch 2.2.0+cu118 (required for GTX 1080 / P100)..."
    pip install --quiet \
        torch==2.2.0+cu118 \
        torchvision==0.17.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118 \
    || fail "PyTorch install failed — check internet access"
    ok "PyTorch installed"
fi

# NumPy pin (torch 2.2 is incompatible with numpy 2.x)
NUMPY_OK=$(python -c "
import numpy as np
major = int(np.__version__.split('.')[0])
print('yes' if major < 2 else 'no')
" 2>/dev/null || echo "no")

if [ "$NUMPY_OK" != "yes" ]; then
    warn "NumPy 2.x detected — downgrading to 1.x for torch 2.2 compatibility..."
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
    warn "Preflight reported issues above. Training may still proceed if only WARNINGs."
    echo "Press Ctrl+C within 10 seconds to abort, or wait to continue..."
    sleep 10
}

# ─── Launch experiments ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
log "Launching 6 baselines across ${#GPUS[@]} GPU(s)"
echo "  Policies: weights_only, embeddings_only, fixed_switch,"
echo "            periodic, threshold, fixed_7030"
echo "  Log dir:  outputs/logs/"
echo "  Results:  outputs/heuristic_*/"
echo "============================================================"
echo ""

bash scripts/experiments/run_heuristic_parallel.sh "${GPUS[@]}"

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo -e "${GREEN}  All done!${NC}"
echo ""
echo "  Results are in:  outputs/heuristic_*/"
echo "  Training logs:   outputs/logs/heuristic_*_*.log"
echo ""
echo "  To compare results:"
echo "    python scripts/analysis/compare_all_methods.py"
echo "============================================================"
