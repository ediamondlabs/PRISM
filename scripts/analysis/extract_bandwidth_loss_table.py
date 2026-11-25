"""
Extract Reconstruction Loss at Specific Bandwidth Targets

For each method (RL-based DFL, CoCo, CFL), find the reconstruction loss
at cumulative bandwidth closest to 40GB, 80GB, and 120GB.
"""

import json
from pathlib import Path
import numpy as np


def find_nearest_bandwidth_point(cumulative_bw_bytes, target_bw_bytes):
    """Find index where cumulative bandwidth is nearest to target.
    
    Args:
        cumulative_bw_bytes: Array of cumulative bandwidth in bytes
        target_bw_bytes: Target bandwidth in bytes
    
    Returns:
        Tuple of (index, actual_bandwidth)
    """
    cumulative_bw_array = np.array(cumulative_bw_bytes)
    if len(cumulative_bw_array) == 0:
        return None, None
    
    if cumulative_bw_array[-1] < target_bw_bytes:
        return None, None
    
    idx = np.argmin(np.abs(cumulative_bw_array - target_bw_bytes))
    return idx, cumulative_bw_array[idx]


def load_rl_dfl_data():
    """Load RL-based DFL data."""
    episodes_path = Path("outputs/dfl_metrics/dfl_training_3000/episodes_summary.json")
    
    if not episodes_path.exists():
        print(f"Warning: RL-DFL data not found at {episodes_path}")
        return None
    
    with open(episodes_path, 'r') as f:
        episodes = json.load(f)
    
    if not episodes:
        return None
    
    lstm_losses = [ep.get("lstm_loss", np.nan) for ep in episodes]
    bandwidths_mb = [ep.get("comm_cost_mb", 0.0) for ep in episodes]
    cumulative_bandwidth_mb = np.cumsum(bandwidths_mb)
    cumulative_bandwidth_bytes = cumulative_bandwidth_mb * 1e6
    
    return {
        "lstm_loss": lstm_losses,
        "cumulative_bandwidth_bytes": cumulative_bandwidth_bytes.tolist(),
    }


def load_coco_data():
    """Load CoCo data."""
    coco_path = Path("outputs/dfl_coco_5cars/metrics/coco_metrics.npz")
    
    if not coco_path.exists():
        print(f"Warning: CoCo data not found at {coco_path}")
        return None
    
    data = np.load(coco_path)
    avg_loss = data["avg_loss"]
    total_bandwidth_mb = data["total_bandwidth_mb"]
    cumulative_bandwidth_bytes = total_bandwidth_mb * 1e6
    
    return {
        "lstm_loss": avg_loss.tolist(),
        "cumulative_bandwidth_bytes": cumulative_bandwidth_bytes.tolist(),
    }


def load_centralized_data():
    """Load Centralized (CFL) data."""
    history_path = Path("outputs/centralized/history.json")
    
    if not history_path.exists():
        print(f"Warning: Centralized data not found at {history_path}")
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    lstm_loss = history.get("lstm_loss", history.get("val_loss", []))
    cumulative_bandwidth_bytes = [0.0] * len(lstm_loss)
    
    return {
        "lstm_loss": lstm_loss,
        "cumulative_bandwidth_bytes": cumulative_bandwidth_bytes,
    }


def extract_loss_at_bandwidth_targets(method_name, data, targets_bytes):
    """Extract loss values at bandwidth targets.
    
    Args:
        method_name: Name of the method
        data: Dict with 'lstm_loss' and 'cumulative_bandwidth_bytes'
        targets_bytes: List of target bandwidths in bytes
    
    Returns:
        List of dicts with target, actual bandwidth, and loss
    """
    if data is None:
        return []
    
    results = []
    for target in targets_bytes:
        idx, actual_bw = find_nearest_bandwidth_point(
            data["cumulative_bandwidth_bytes"], target
        )
        
        if idx is None:
            results.append({
                "method": method_name,
                "target_gb": target / 1e9,
                "actual_bytes": None,
                "actual_gb": None,
                "loss": None,
                "status": "N/A",
            })
        else:
            results.append({
                "method": method_name,
                "target_gb": target / 1e9,
                "actual_bytes": actual_bw,
                "actual_gb": actual_bw / 1e9,
                "loss": data["lstm_loss"][idx],
                "status": "Found",
            })
    
    return results


def print_results_table(all_results):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("RECONSTRUCTION LOSS AT SPECIFIC BANDWIDTH TARGETS")
    print("=" * 100)
    print()
    print(f"{'Method':<20} | {'Target BW':<15} | {'Actual BW Recorded':<25} | {'Loss Value':<15}")
    print("-" * 100)
    
    for result in all_results:
        method = result["method"]
        target = f"{result['target_gb']:.0f} GB"
        
        if result["status"] == "N/A":
            actual_bw = "N/A"
            loss_str = "N/A"
        else:
            actual_gb = result["actual_gb"]
            actual_bytes = result["actual_bytes"]
            actual_bw = f"{actual_gb:.6f} GB ({actual_bytes:.0f} bytes)"
            loss_str = f"{result['loss']:.8f}"
        
        print(f"{method:<20} | {target:<15} | {actual_bw:<25} | {loss_str:<15}")
    
    print("=" * 100)


def save_results_csv(all_results, output_path):
    """Save results to CSV file."""
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Target_BW_GB", "Actual_BW_GB", "Actual_BW_Bytes", "Loss_Value", "Status"])
        
        for result in all_results:
            writer.writerow([
                result["method"],
                result["target_gb"],
                result["actual_gb"] if result["actual_gb"] is not None else "N/A",
                result["actual_bytes"] if result["actual_bytes"] is not None else "N/A",
                result["loss"] if result["loss"] is not None else "N/A",
                result["status"],
            ])
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main extraction function."""
    targets_gb = [0, 1, 2, 3, 4, 5, 10, 20, 40, 80, 120]
    targets_bytes = [t * 1e9 for t in targets_gb]
    
    print("Loading data...")
    rl_dfl_data = load_rl_dfl_data()
    coco_data = load_coco_data()
    cfl_data = load_centralized_data()
    
    all_results = []
    
    print("\nExtracting RL-based DFL results...")
    rl_results = extract_loss_at_bandwidth_targets("RL-based DFL", rl_dfl_data, targets_bytes)
    all_results.extend(rl_results)
    
    print("Extracting CoCo results...")
    coco_results = extract_loss_at_bandwidth_targets("CoCo", coco_data, targets_bytes)
    all_results.extend(coco_results)
    
    print("Extracting CFL (Centralized) results...")
    cfl_results = extract_loss_at_bandwidth_targets("CFL (Centralized)", cfl_data, targets_bytes)
    all_results.extend(cfl_results)
    
    print_results_table(all_results)
    
    output_csv = Path("comparison/bandwidth_loss_table.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_results_csv(all_results, output_csv)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
