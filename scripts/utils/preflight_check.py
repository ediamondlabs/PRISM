"""
Pre-flight check for PRISM heuristic baselines.
Run this BEFORE starting any training to catch all blockers early.

Usage:
    python scripts/utils/preflight_check.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
WARN = "\033[93m  WARN\033[0m"

issues = []


def check(label: str, ok: bool, detail: str = "", fix: str = "") -> bool:
    status = PASS if ok else FAIL
    print(f"{status}  {label}")
    if detail:
        print(f"        {detail}")
    if not ok and fix:
        print(f"        FIX: {fix}")
        issues.append(f"{label}: {fix}")
    return ok


print("=" * 60)
print("  PRISM Pre-flight Check")
print("=" * 60)
print()

# ─── Python ──────────────────────────────────────────────────────────────────
print("[1/6] Python")
major, minor = sys.version_info[:2]
check(
    f"Python version: {major}.{minor}",
    (major, minor) >= (3, 10),
    fix="Install Python 3.10+",
)

# ─── PyTorch + CUDA ──────────────────────────────────────────────────────────
print("\n[2/6] PyTorch + CUDA")
try:
    import torch

    version_ok = torch.__version__.startswith("2.2") and "cu118" in torch.__version__
    check(
        f"PyTorch version: {torch.__version__}",
        version_ok,
        fix="pip install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118",
    )

    cuda_ok = torch.cuda.is_available()
    check(
        "CUDA available",
        cuda_ok,
        fix="Check nvidia-smi output and driver version",
    )

    if cuda_ok:
        n_gpus = torch.cuda.device_count()
        check(f"GPU count: {n_gpus}", n_gpus >= 1)
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            cap = torch.cuda.get_device_capability(i)
            vram_ok = vram >= 7.0
            cap_ok = cap >= (6, 0)
            check(
                f"GPU {i}: {name}  ({vram:.1f} GB, sm_{cap[0]}{cap[1]})",
                vram_ok and cap_ok,
                fix="Need >= 8GB VRAM and sm_60+ (GTX 1080 / P100 or newer)",
            )

        # LSTM smoke test
        try:
            lstm = torch.nn.LSTM(8, 64, 2, batch_first=True).cuda()
            x = torch.randn(32, 50, 8, device="cuda")
            out, _ = lstm(x)
            del lstm, x, out
            torch.cuda.empty_cache()
            check("LSTM kernel on GPU", True)
        except Exception as e:
            check("LSTM kernel on GPU", False, detail=str(e),
                  fix="GPU kernel missing for this compute capability — use PyTorch 2.2.0+cu118")

        # VRAM estimate for full model
        try:
            import torch.nn as nn
            # Rough VRAM estimate: one BiLSTM(768, 4L) + batch(384, 50, 10)
            test_lstm = nn.LSTM(10, 768, 4, batch_first=True, bidirectional=True).cuda()
            test_x = torch.randn(256, 50, 10, device="cuda")  # safe batch=256
            test_out, _ = test_lstm(test_x)
            del test_lstm, test_x, test_out
            torch.cuda.empty_cache()
            check("Full model fits at batch=256 (safe)", True)
        except RuntimeError:
            check(
                "Full model fits at batch=256",
                False,
                fix="Use dfl_heuristic_practical.yaml (smaller model) instead",
            )

except ImportError:
    check("PyTorch importable", False,
          fix="pip install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118")

# ─── Data files ──────────────────────────────────────────────────────────────
print("\n[3/6] Data files")
csv_path = ROOT / "data" / "v2v_dataset_20251017_124247.csv"
chunks_dir = ROOT / "data" / "chunks"

if not csv_path.exists() and chunks_dir.exists() and any(chunks_dir.glob("chunk_*.part")):
    print(f"{WARN}  Dataset CSV missing — assembling from chunks automatically...")
    sys.path.insert(0, str(ROOT))
    from scripts.utils.assemble_dataset import assemble
    assemble()

csv_ok = csv_path.exists()
check(
    f"Dataset CSV ({csv_path.relative_to(ROOT)})",
    csv_ok,
    detail=f"{csv_path.stat().st_size / 1e9:.2f} GB" if csv_ok else "",
    fix="python scripts/utils/assemble_dataset.py",
)

npz_path = ROOT / "outputs" / "preprocessed" / "preprocessed_data.npz"
npz_ok = npz_path.exists()
status = WARN if (csv_ok and not npz_ok) else (PASS if npz_ok else FAIL)
print(f"{status}  Preprocessed data ({npz_path.relative_to(ROOT)})")
if not npz_ok and csv_ok:
    print("        WARN: Will be auto-generated on first run (adds ~5 min startup)")
elif not npz_ok and not csv_ok:
    print("        FAIL: Cannot generate without CSV")

# ─── Python packages ─────────────────────────────────────────────────────────
print("\n[4/6] Required packages")
required = [
    "hydra",
    "omegaconf",
    "sklearn",
    "mlflow",
    "pandas",
    "numpy",
]
for pkg in required:
    try:
        __import__(pkg.replace("-", "_"))
        check(pkg, True)
    except ImportError:
        check(pkg, False, fix=f"pip install {pkg}")

# ─── Config files ────────────────────────────────────────────────────────────
print("\n[5/6] Config files")
for cfg in [
    "configs/experiment/dfl_heuristic_fair.yaml",
    "configs/experiment/dfl_heuristic_practical.yaml",
    "train_dfl_heuristic.py",
    "src/baselines/__init__.py",
]:
    path = ROOT / cfg
    check(cfg, path.exists(), fix=f"git pull (file missing from repo)")

# ─── Scripts ─────────────────────────────────────────────────────────────────
print("\n[6/6] Experiment scripts")
script = ROOT / "scripts" / "experiments" / "run_heuristic_parallel.sh"
check("run_heuristic_parallel.sh", script.exists())
if script.exists():
    import os
    check(
        "run_heuristic_parallel.sh is executable",
        os.access(script, os.X_OK),
        fix="chmod +x scripts/experiments/run_heuristic_parallel.sh",
    )

# ─── Summary ─────────────────────────────────────────────────────────────────
print()
print("=" * 60)
if not issues:
    print("\033[92m  ALL CHECKS PASSED — ready to train\033[0m")
    print()
    print("  Run (4 GPUs):")
    print("    bash scripts/experiments/run_heuristic_parallel.sh 0 1 2 3")
    print()
    print("  Run (3 GPUs):")
    print("    bash scripts/experiments/run_heuristic_parallel.sh 0 1 2")
else:
    print(f"\033[91m  {len(issues)} issue(s) to fix before training:\033[0m")
    for i, msg in enumerate(issues, 1):
        print(f"  {i}. {msg}")
print("=" * 60)
