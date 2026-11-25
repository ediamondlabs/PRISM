"""
Compare multiple DFL training runs.

Usage:
    python compare_dfl_runs.py \
        --runs outputs/dfl_metrics/run1 outputs/dfl_metrics/run2 outputs/dfl_metrics/run3 \
        --labels "Baseline" "Method A" "Method B"
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict


class DFLRunComparator:
    """Compare multiple DFL training runs."""

    def __init__(self, run_dirs: List[str], labels: List[str] = None):
        """
        Initialize comparator.

        Args:
            run_dirs: List of run directories
            labels: Labels for each run (optional)
        """
        self.run_dirs = [Path(d) for d in run_dirs]
        self.labels = labels or [f"Run {i+1}" for i in range(len(run_dirs))]
        self.runs_data = []

        print(f"Loading {len(run_dirs)} runs for comparison...")

        for run_dir, label in zip(self.run_dirs, self.labels):
            data = self._load_run(run_dir, label)
            if data:
                self.runs_data.append(data)

        print(f"Successfully loaded {len(self.runs_data)} runs")

    def _load_run(self, run_dir: Path, label: str) -> Dict:
        """Load data from a single run."""
        if not run_dir.exists():
            print(f"Warning: {run_dir} not found, skipping")
            return None

        episodes_file = run_dir / "episodes_summary.json"
        summary_file = run_dir / "summary_statistics.json"

        if not episodes_file.exists():
            print(f"Warning: {episodes_file} not found, skipping")
            return None

        with open(episodes_file, "r") as f:
            episodes = json.load(f)

        summary = {}
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)

        return {
            "label": label,
            "run_dir": run_dir,
            "episodes": episodes,
            "summary": summary,
        }

    def print_comparison_table(self):
        """Print comparison table."""
        print("\n" + "="*80)
        print("DFL TRAINING COMPARISON")
        print("="*80)

        # Header
        header = f"{'Metric':<30} | " + " | ".join([f"{run['label']:>15}" for run in self.runs_data])
        print(header)
        print("-" * len(header))

        # Performance metrics
        print("\nPERFORMANCE:")
        self._print_metric_row("Final F1", [r["summary"].get("performance", {}).get("final_f1", 0) for r in self.runs_data], fmt=".4f")
        self._print_metric_row("Best F1", [r["summary"].get("performance", {}).get("best_f1", 0) for r in self.runs_data], fmt=".4f")
        self._print_metric_row("Avg F1", [r["summary"].get("performance", {}).get("avg_f1", 0) for r in self.runs_data], fmt=".4f")

        # Communication metrics
        print("\nCOMMUNICATION:")
        self._print_metric_row("Final Cost (MB)", [r["summary"].get("communication", {}).get("final_comm_cost", 0) for r in self.runs_data], fmt=".1f")
        self._print_metric_row("Avg Cost (MB)", [r["summary"].get("communication", {}).get("avg_comm_cost", 0) for r in self.runs_data], fmt=".1f")
        self._print_metric_row("Total Cost (MB)", [r["summary"].get("communication", {}).get("total_comm_cost", 0) for r in self.runs_data], fmt=".1f")

        # Action distribution
        print("\nACTION DISTRIBUTION:")
        self._print_metric_row("Embeddings (%)", [r["summary"].get("actions", {}).get("E_pct", 0) for r in self.runs_data], fmt=".1f")
        self._print_metric_row("Prototypes (%)", [r["summary"].get("actions", {}).get("P_pct", 0) for r in self.runs_data], fmt=".1f")
        self._print_metric_row("Weights (%)", [r["summary"].get("actions", {}).get("W_pct", 0) for r in self.runs_data], fmt=".1f")

        # Efficiency
        print("\nEFFICIENCY:")
        self._print_metric_row("F1 per MB", [r["summary"].get("efficiency", {}).get("f1_per_mb", 0) for r in self.runs_data], fmt=".6f")
        self._print_metric_row("Bandwidth Red. (%)", [r["summary"].get("efficiency", {}).get("bandwidth_reduction_pct", 0) for r in self.runs_data], fmt=".1f")

        # Find winner
        f1_scores = [r["summary"].get("performance", {}).get("final_f1", 0) for r in self.runs_data]
        comm_costs = [r["summary"].get("communication", {}).get("final_comm_cost", 1) for r in self.runs_data]
        efficiencies = [f1/comm if comm > 0 else 0 for f1, comm in zip(f1_scores, comm_costs)]

        best_f1_idx = np.argmax(f1_scores)
        best_comm_idx = np.argmin(comm_costs)
        best_eff_idx = np.argmax(efficiencies)

        print("\n" + "="*80)
        print("WINNERS:")
        print(f"  Best F1 Score: {self.runs_data[best_f1_idx]['label']} ({f1_scores[best_f1_idx]:.4f})")
        print(f"  Lowest Comm Cost: {self.runs_data[best_comm_idx]['label']} ({comm_costs[best_comm_idx]:.1f} MB)")
        print(f"  Best Efficiency: {self.runs_data[best_eff_idx]['label']} ({efficiencies[best_eff_idx]:.6f})")
        print("="*80)

    def _print_metric_row(self, name: str, values: List[float], fmt: str = ".2f"):
        """Print a metric comparison row."""
        formatted_values = " | ".join([f"{v:{fmt}}" if v is not None else "N/A" for v in values])
        print(f"{name:<30} | {formatted_values}")

    def plot_comparison(self, save_dir: str = None):
        """Generate comparison plots."""
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path("outputs/dfl_metrics/comparisons")
            save_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: F1 Score Comparison
        plt.figure(figsize=(12, 6))
        for run in self.runs_data:
            episodes = [ep["episode_id"] for ep in run["episodes"]]
            f1_scores = [ep["f1_score"] for ep in run["episodes"]]
            plt.plot(episodes, f1_scores, label=run["label"], alpha=0.7)

        plt.xlabel("Episode")
        plt.ylabel("F1 Score")
        plt.title("F1 Score Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / "comparison_f1.png", dpi=150, bbox_inches='tight')
        print(f"Saved F1 comparison to: {save_dir / 'comparison_f1.png'}")
        plt.close()

        # Plot 2: Communication Cost Comparison
        plt.figure(figsize=(12, 6))
        for run in self.runs_data:
            episodes = [ep["episode_id"] for ep in run["episodes"]]
            comm_costs = [ep["comm_cost_mb"] for ep in run["episodes"]]
            plt.plot(episodes, comm_costs, label=run["label"], alpha=0.7)

        plt.xlabel("Episode")
        plt.ylabel("Communication Cost (MB)")
        plt.title("Communication Cost Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / "comparison_comm_cost.png", dpi=150, bbox_inches='tight')
        print(f"Saved comm cost comparison to: {save_dir / 'comparison_comm_cost.png'}")
        plt.close()

        # Plot 3: Efficiency Scatter (Pareto Front)
        plt.figure(figsize=(10, 8))
        for run in self.runs_data:
            final_ep = run["episodes"][-1]
            f1 = final_ep["f1_score"]
            comm = final_ep["comm_cost_mb"]
            plt.scatter(comm, f1, s=200, label=run["label"], alpha=0.7)
            plt.annotate(run["label"], (comm, f1), xytext=(5, 5),
                        textcoords='offset points', fontsize=10)

        plt.xlabel("Communication Cost (MB)")
        plt.ylabel("F1 Score")
        plt.title("Efficiency Comparison (Pareto Front)\nTop-Left is Best")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / "comparison_pareto.png", dpi=150, bbox_inches='tight')
        print(f"Saved Pareto front to: {save_dir / 'comparison_pareto.png'}")
        plt.close()

        # Plot 4: Action Distribution Comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for i, action_type in enumerate(["E", "P", "W"]):
            for run in self.runs_data:
                episodes = [ep["episode_id"] for ep in run["episodes"]]
                pcts = []
                for ep in run["episodes"]:
                    actions = ep["what_actions"]
                    total = sum(actions.values())
                    pct = 100 * actions[action_type] / total if total > 0 else 0
                    pcts.append(pct)

                axes[i].plot(episodes, pcts, label=run["label"], alpha=0.7)

            axes[i].set_xlabel("Episode")
            axes[i].set_ylabel("Percentage (%)")
            axes[i].set_title(f"{action_type} Action Usage")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "comparison_actions.png", dpi=150, bbox_inches='tight')
        print(f"Saved action comparison to: {save_dir / 'comparison_actions.png'}")
        plt.close()

        print(f"\nAll comparison plots saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare DFL training runs")
    parser.add_argument("--runs", type=str, nargs="+", required=True,
                       help="Paths to run directories")
    parser.add_argument("--labels", type=str, nargs="+",
                       help="Labels for each run (optional)")
    parser.add_argument("--output", type=str,
                       help="Output directory for comparison plots")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        print("Error: Number of labels must match number of runs")
        return

    # Create comparator
    comparator = DFLRunComparator(args.runs, args.labels)

    # Print comparison
    comparator.print_comparison_table()

    # Generate plots
    comparator.plot_comparison(args.output)


if __name__ == "__main__":
    main()
