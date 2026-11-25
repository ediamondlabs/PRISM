"""
Comparison Script: All 4 FL Methods

Compares:
1. DFL (RL-based Decentralized FL) - Our approach
2. CoCo (Baseline from paper)
3. Flower FedAvg (Classic FL)
4. Centralized (Upper Bound)

Generates IEEE-quality plots for paper.

Usage:
    python compare_all_methods.py
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def uniform_filter1d(data, size, mode='nearest'):
    """Moving average filter."""
    data = np.asarray(data, dtype=float)
    kernel = np.ones(size) / size
    pad_width = size // 2
    padded = np.pad(data, pad_width, mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    if len(smoothed) > len(data):
        smoothed = smoothed[:len(data)]
    elif len(smoothed) < len(data):
        smoothed = np.pad(smoothed, (0, len(data) - len(smoothed)), mode='edge')
    return smoothed


# IEEE PUBLICATION SETTINGS
FIG_WIDTH_SINGLE = 3.5
FIG_WIDTH_DOUBLE = 7.16
FIG_HEIGHT_RATIO = 0.75

FIG_SIZE_SINGLE = (FIG_WIDTH_SINGLE, FIG_WIDTH_SINGLE * FIG_HEIGHT_RATIO)
FIG_SIZE_DOUBLE = (FIG_WIDTH_DOUBLE, FIG_WIDTH_DOUBLE * 0.5)

FONT_CONFIG = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
}

COLORS = {
    'dfl': '#1f77b4',         # Blue
    'coco': '#d62728',        # Red
    'flower': '#2ca02c',      # Green
    'centralized': '#9467bd', # Purple (upper bound)
}

LINE_WIDTH = 1.5
DPI = 300


def configure_plot_style():
    """Apply IEEE style."""
    plt.rcParams.update(FONT_CONFIG)
    plt.rcParams['savefig.dpi'] = DPI
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.02
    plt.rcParams['lines.linewidth'] = LINE_WIDTH
    plt.rcParams['mathtext.fontset'] = 'stix'


def load_dfl_metrics() -> Optional[Dict]:
    """Load DFL metrics."""
    base_path = Path("outputs/dfl_metrics/dfl_training")
    episodes_path = base_path / "episodes_summary.json"

    if not episodes_path.exists():
        print(f"⚠ DFL data not found at {episodes_path}")
        return None

    with open(episodes_path, 'r') as f:
        episodes = json.load(f)

    lstm_losses = [ep.get("lstm_loss", np.nan) for ep in episodes]
    bandwidths_mb = [ep.get("comm_cost_mb", 0.0) for ep in episodes]
    rewards = [ep.get("reward", 0.0) for ep in episodes]

    return {
        "method": "DFL",
        "iterations": list(range(len(lstm_losses))),
        "losses": lstm_losses,
        "bandwidths": bandwidths_mb,
        "cumulative_bandwidth": np.cumsum(bandwidths_mb).tolist(),
        "rewards": rewards,
    }


def load_coco_metrics() -> Optional[Dict]:
    """Load CoCo metrics."""
    base_path = Path("outputs/coco_metrics/dfl_coco_training_3000")
    rounds_path = base_path / "rounds_summary.json"

    if not rounds_path.exists():
        print(f"⚠ CoCo data not found at {rounds_path}")
        return None

    with open(rounds_path, 'r') as f:
        rounds = json.load(f)

    lstm_losses = [r.get("lstm_loss", np.nan) for r in rounds]
    bandwidths_mb = [r.get("comm_cost_mb", 0.0) for r in rounds]

    return {
        "method": "CoCo",
        "iterations": list(range(len(lstm_losses))),
        "losses": lstm_losses,
        "bandwidths": bandwidths_mb,
        "cumulative_bandwidth": np.cumsum(bandwidths_mb).tolist(),
    }


def load_flower_metrics() -> Optional[Dict]:
    """Load Flower FL metrics."""
    history_path = Path("outputs/flower_fl/history.json")

    if not history_path.exists():
        print(f"⚠ Flower FL data not found at {history_path}")
        return None

    with open(history_path, 'r') as f:
        history = json.load(f)

    # Flower logs val_loss per round
    rounds = history.get("round", [])
    val_losses = history.get("val_loss", [])

    return {
        "method": "Flower FedAvg",
        "iterations": rounds,
        "losses": val_losses,
        "bandwidths": [0.0] * len(rounds),  # Not tracked separately
        "cumulative_bandwidth": [0.0] * len(rounds),
    }


def load_centralized_metrics() -> Optional[Dict]:
    """Load Centralized metrics."""
    history_path = Path("outputs/centralized/history.json")

    if not history_path.exists():
        print(f"⚠ Centralized data not found at {history_path}")
        return None

    with open(history_path, 'r') as f:
        history = json.load(f)

    val_losses = history.get("val_loss", [])
    epochs = list(range(len(val_losses)))

    return {
        "method": "Centralized",
        "iterations": epochs,
        "losses": val_losses,
        "bandwidths": [0.0] * len(epochs),  # N/A
        "cumulative_bandwidth": [0.0] * len(epochs),
    }


def plot_loss_convergence(methods: List[Dict], output_dir: Path):
    """Plot loss convergence for all methods."""
    configure_plot_style()

    fig, ax = plt.subplots(figsize=FIG_SIZE_DOUBLE)

    for m in methods:
        if m is None:
            continue

        iterations = m["iterations"]
        losses = m["losses"]

        # Remove NaN
        valid_idx = ~np.isnan(losses)
        iters = np.array(iterations)[valid_idx]
        loss = np.array(losses)[valid_idx]

        if len(loss) == 0:
            continue

        # Smooth
        if len(loss) > 100:
            loss = uniform_filter1d(loss, size=50)

        color = COLORS.get(m["method"].lower().replace(" ", "_").replace("fedavg", "flower"), '#000000')
        label = m["method"]
        if m["method"] == "Centralized":
            label += " (Upper Bound)"

        ax.plot(iters, loss, color=color, linewidth=LINE_WIDTH, label=label, alpha=0.9)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Validation Loss (MSE)')
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Save
    for fmt in ['png', 'pdf']:
        output_path = output_dir / f"loss_convergence_all_methods.{fmt}"
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    plt.close()


def plot_loss_convergence_log(methods: List[Dict], output_dir: Path):
    """Plot loss convergence (log scale)."""
    configure_plot_style()

    fig, ax = plt.subplots(figsize=FIG_SIZE_DOUBLE)

    for m in methods:
        if m is None or m["method"] == "Centralized":
            continue  # Skip centralized for log scale (different iteration count)

        iterations = m["iterations"]
        losses = m["losses"]

        valid_idx = ~np.isnan(losses)
        iters = np.array(iterations)[valid_idx]
        loss = np.array(losses)[valid_idx]

        if len(loss) == 0 or len(loss) < 100:
            continue

        loss = uniform_filter1d(loss, size=50)

        color = COLORS.get(m["method"].lower().replace(" ", "_").replace("fedavg", "flower"), '#000000')
        ax.semilogy(iters, loss, color=color, linewidth=LINE_WIDTH, label=m["method"], alpha=0.9)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Validation Loss (MSE, log scale)')
    ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')

    for fmt in ['png', 'pdf']:
        output_path = output_dir / f"loss_convergence_log_all_methods.{fmt}"
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    plt.close()


def plot_final_comparison_bar(methods: List[Dict], output_dir: Path):
    """Bar chart comparing final performance."""
    configure_plot_style()

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE)

    method_names = []
    final_losses = []
    colors = []

    for m in methods:
        if m is None:
            continue

        losses = [l for l in m["losses"] if not np.isnan(l)]
        if len(losses) == 0:
            continue

        final_loss = losses[-1]
        method_names.append(m["method"])
        final_losses.append(final_loss)
        color = COLORS.get(m["method"].lower().replace(" ", "_").replace("fedavg", "flower"), '#000000')
        colors.append(color)

    x = np.arange(len(method_names))
    bars = ax.bar(x, final_losses, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Final Validation Loss (MSE)')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, ha='right')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=6)

    for fmt in ['png', 'pdf']:
        output_path = output_dir / f"final_loss_comparison.{fmt}"
        fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")

    plt.close()


def save_comparison_table(methods: List[Dict], output_dir: Path):
    """Save comparison metrics as JSON."""
    table = {}

    for m in methods:
        if m is None:
            continue

        losses = [l for l in m["losses"] if not np.isnan(l)]
        if len(losses) == 0:
            continue

        table[m["method"]] = {
            "final_loss": float(losses[-1]),
            "min_loss": float(min(losses)),
            "iterations": len(losses),
            "cumulative_bandwidth_mb": float(m["cumulative_bandwidth"][-1]) if m["cumulative_bandwidth"] else 0.0,
        }

    output_path = output_dir / "comparison_table.json"
    with open(output_path, 'w') as f:
        json.dump(table, f, indent=2)
    print(f"✓ Saved: {output_path}")


def main():
    print("=" * 60)
    print("COMPARISON: All 4 FL Methods")
    print("=" * 60)

    # Load all methods
    dfl = load_dfl_metrics()
    coco = load_coco_metrics()
    flower = load_flower_metrics()
    centralized = load_centralized_metrics()

    methods = [dfl, coco, flower, centralized]
    available = [m for m in methods if m is not None]

    print(f"\nFound {len(available)}/4 methods:")
    for m in available:
        print(f"  ✓ {m['method']}: {len(m['iterations'])} iterations")

    if len(available) < 2:
        print("\n⚠ Need at least 2 methods to compare. Run training first.")
        return

    # Create output directory
    output_dir = Path("outputs/comparison/ieee_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating IEEE plots...")
    plot_loss_convergence(methods, output_dir)
    plot_loss_convergence_log(methods, output_dir)
    plot_final_comparison_bar(methods, output_dir)
    save_comparison_table(methods, output_dir)

    print("\n" + "=" * 60)
    print("✅ Comparison complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
