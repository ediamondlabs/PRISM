"""
Evaluation Script: Extract metrics from all 4 methods for IEEE plots.

Generates standardized metrics JSON files compatible with comparison plots.

Usage:
    python evaluate_all_methods.py
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def evaluate_dfl() -> Optional[Dict]:
    """Extract DFL metrics."""
    print("\n" + "=" * 60)
    print("Evaluating: DFL")
    print("=" * 60)

    base_path = Path("outputs/dfl_metrics/dfl_training_3000")
    episodes_path = base_path / "episodes_summary.json"
    summary_path = base_path / "summary_statistics.json"

    if not episodes_path.exists():
        print("⚠ DFL results not found")
        return None

    with open(episodes_path, 'r') as f:
        episodes = json.load(f)

    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    else:
        summary = {}

    lstm_losses = [ep.get("lstm_loss", np.nan) for ep in episodes]
    bandwidths_mb = [ep.get("comm_cost_mb", 0.0) for ep in episodes]
    rewards = [ep.get("reward", 0.0) for ep in episodes]

    # Remove NaN
    valid_losses = [l for l in lstm_losses if not np.isnan(l)]

    metrics = {
        "method": "DFL",
        "num_iterations": len(episodes),
        "final_loss": float(valid_losses[-1]) if valid_losses else None,
        "min_loss": float(min(valid_losses)) if valid_losses else None,
        "mean_loss": float(np.mean(valid_losses)) if valid_losses else None,
        "total_bandwidth_mb": float(sum(bandwidths_mb)),
        "mean_bandwidth_per_iteration": float(np.mean(bandwidths_mb)),
        "final_reward": float(rewards[-1]) if rewards else None,
        "mean_reward": float(np.mean(rewards)) if rewards else None,
    }

    print(f"✓ DFL: {metrics['num_iterations']} episodes")
    print(f"  Final loss: {metrics['final_loss']:.6f}")
    print(f"  Total bandwidth: {metrics['total_bandwidth_mb']:.2f} MB")

    return metrics


def evaluate_coco() -> Optional[Dict]:
    """Extract CoCo metrics."""
    print("\n" + "=" * 60)
    print("Evaluating: CoCo")
    print("=" * 60)

    base_path = Path("outputs/coco_metrics/dfl_coco_training_3000")
    rounds_path = base_path / "rounds_summary.json"

    if not rounds_path.exists():
        print("⚠ CoCo results not found")
        return None

    with open(rounds_path, 'r') as f:
        rounds = json.load(f)

    lstm_losses = [r.get("lstm_loss", np.nan) for r in rounds]
    bandwidths_mb = [r.get("comm_cost_mb", 0.0) for r in rounds]

    valid_losses = [l for l in lstm_losses if not np.isnan(l)]

    metrics = {
        "method": "CoCo",
        "num_iterations": len(rounds),
        "final_loss": float(valid_losses[-1]) if valid_losses else None,
        "min_loss": float(min(valid_losses)) if valid_losses else None,
        "mean_loss": float(np.mean(valid_losses)) if valid_losses else None,
        "total_bandwidth_mb": float(sum(bandwidths_mb)),
        "mean_bandwidth_per_iteration": float(np.mean(bandwidths_mb)),
    }

    print(f"✓ CoCo: {metrics['num_iterations']} rounds")
    print(f"  Final loss: {metrics['final_loss']:.6f}")
    print(f"  Total bandwidth: {metrics['total_bandwidth_mb']:.2f} MB")

    return metrics


def evaluate_flower() -> Optional[Dict]:
    """Extract Flower FL metrics."""
    print("\n" + "=" * 60)
    print("Evaluating: Flower FedAvg")
    print("=" * 60)

    history_path = Path("outputs/flower_fl/history.json")
    results_path = Path("outputs/flower_fl/results.json")

    if not history_path.exists():
        print("⚠ Flower FL results not found")
        return None

    with open(history_path, 'r') as f:
        history = json.load(f)

    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    val_losses = history.get("val_loss", [])
    rounds = history.get("round", [])

    metrics = {
        "method": "Flower FedAvg",
        "num_iterations": len(rounds),
        "final_loss": float(val_losses[-1]) if val_losses else None,
        "min_loss": float(min(val_losses)) if val_losses else None,
        "mean_loss": float(np.mean(val_losses)) if val_losses else None,
        "total_bandwidth_mb": 0.0,  # Not tracked separately
        "local_epochs": 5,
        "num_clients": 5,
    }

    print(f"✓ Flower: {metrics['num_iterations']} rounds")
    print(f"  Final loss: {metrics['final_loss']:.6f}")

    return metrics


def evaluate_centralized() -> Optional[Dict]:
    """Extract Centralized metrics."""
    print("\n" + "=" * 60)
    print("Evaluating: Centralized (Upper Bound)")
    print("=" * 60)

    history_path = Path("outputs/centralized/history.json")
    results_path = Path("outputs/centralized/results.json")

    if not history_path.exists():
        print("⚠ Centralized results not found")
        return None

    with open(history_path, 'r') as f:
        history = json.load(f)

    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    test_losses = history.get("test_loss", [])

    metrics = {
        "method": "Centralized",
        "num_iterations": len(val_losses),
        "final_train_loss": float(train_losses[-1]) if train_losses else None,
        "final_val_loss": float(val_losses[-1]) if val_losses else None,
        "test_loss": float(test_losses[0]) if test_losses else None,
        "min_val_loss": float(min(val_losses)) if val_losses else None,
        "mean_val_loss": float(np.mean(val_losses)) if val_losses else None,
        "convergence_epoch": len(val_losses),
    }

    print(f"✓ Centralized: {metrics['num_iterations']} epochs")
    print(f"  Final val loss: {metrics['final_val_loss']:.6f}")
    print(f"  Test loss: {metrics['test_loss']:.6f}")

    return metrics


def generate_comparison_table(all_metrics: Dict[str, Dict]) -> str:
    """Generate LaTeX table for paper."""
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Performance Comparison of FL Methods}\n"
    latex += "\\label{tab:comparison}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\hline\n"
    latex += "Method & Iterations & Final Loss & Min Loss & Bandwidth (MB) \\\\\n"
    latex += "\\hline\n"

    for method_name in ["DFL", "CoCo", "Flower FedAvg", "Centralized"]:
        if method_name not in all_metrics:
            continue

        m = all_metrics[method_name]
        iters = m.get("num_iterations", "-")
        final = m.get("final_loss") or m.get("final_val_loss", "-")
        min_loss = m.get("min_loss") or m.get("min_val_loss", "-")
        bandwidth = m.get("total_bandwidth_mb", 0.0)

        if isinstance(final, float):
            final = f"{final:.4f}"
        if isinstance(min_loss, float):
            min_loss = f"{min_loss:.4f}"
        if isinstance(bandwidth, float):
            bandwidth = f"{bandwidth:.1f}" if bandwidth > 0 else "N/A"

        display_name = method_name
        if method_name == "Centralized":
            display_name += " (Upper Bound)"

        latex += f"{display_name} & {iters} & {final} & {min_loss} & {bandwidth} \\\\\n"

    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    return latex


def main():
    print("=" * 60)
    print("EVALUATION: All 4 FL Methods")
    print("=" * 60)

    # Evaluate each method
    dfl_metrics = evaluate_dfl()
    coco_metrics = evaluate_coco()
    flower_metrics = evaluate_flower()
    central_metrics = evaluate_centralized()

    # Collect all metrics
    all_metrics = {}
    if dfl_metrics:
        all_metrics["DFL"] = dfl_metrics
    if coco_metrics:
        all_metrics["CoCo"] = coco_metrics
    if flower_metrics:
        all_metrics["Flower FedAvg"] = flower_metrics
    if central_metrics:
        all_metrics["Centralized"] = central_metrics

    # Save consolidated metrics
    output_dir = Path("outputs/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "all_methods_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✓ Saved consolidated metrics: {metrics_path}")

    # Generate LaTeX table
    latex_table = generate_comparison_table(all_metrics)
    latex_path = output_dir / "comparison_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)

    print(f"✓ Saved LaTeX table: {latex_path}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    for method_name, metrics in all_metrics.items():
        print(f"\n{method_name}:")
        final_key = "final_loss" if "final_loss" in metrics else "final_val_loss"
        if final_key in metrics and metrics[final_key] is not None:
            print(f"  Final Loss: {metrics[final_key]:.6f}")
        if "total_bandwidth_mb" in metrics:
            print(f"  Total Bandwidth: {metrics['total_bandwidth_mb']:.2f} MB")

    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
