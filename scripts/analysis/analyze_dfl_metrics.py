"""
Load and analyze DFL metrics from saved JSON files.

Usage:
    python analyze_dfl_metrics.py --run outputs/dfl_metrics/run_20241030_120000
    python analyze_dfl_metrics.py --run outputs/dfl_metrics/run_20241030_120000 --plot
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict


class DFLMetricsAnalyzer:
    """Analyzer for saved DFL metrics."""

    def __init__(self, run_dir: str):
        """
        Initialize analyzer.

        Args:
            run_dir: Directory containing saved metrics
        """
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise ValueError(f"Run directory not found: {run_dir}")

        print(f"Loading metrics from: {self.run_dir}")

        # Load all data
        self.metadata = self._load_json("metadata.json")
        self.episodes = self._load_json("episodes_summary.json")
        self.summary_stats = self._load_json("summary_statistics.json")
        self.aggregation_history = self._load_json("aggregation_history.json")

        print(f"Loaded {len(self.episodes)} episodes")

    def _load_json(self, filename: str) -> Dict:
        """Load JSON file."""
        path = self.run_dir / filename
        if not path.exists():
            print(f"Warning: {filename} not found")
            return {}

        with open(path, "r") as f:
            return json.load(f)

    def print_summary(self):
        """Print training summary."""
        print("\n" + "="*60)
        print("DFL TRAINING ANALYSIS")
        print("="*60)

        if self.metadata:
            print(f"\nRun: {self.metadata.get('run_name')}")
            print(f"Started: {self.metadata.get('start_time')}")
            print(f"Duration: {self.metadata.get('total_duration_seconds', 0) / 3600:.2f} hours")

        if self.summary_stats:
            perf = self.summary_stats.get("performance", {})
            comm = self.summary_stats.get("communication", {})
            actions = self.summary_stats.get("actions", {})
            eff = self.summary_stats.get("efficiency", {})

            print("\nPerformance:")
            print(f"  F1 Score: {perf.get('initial_f1', 0):.4f} → {perf.get('final_f1', 0):.4f}")
            print(f"  Best F1: {perf.get('best_f1', 0):.4f}")
            print(f"  Average F1: {perf.get('avg_f1', 0):.4f}")

            print("\nCommunication:")
            print(f"  Initial Cost: {comm.get('initial_comm_cost', 0):.1f} MB")
            print(f"  Final Cost: {comm.get('final_comm_cost', 0):.1f} MB")
            print(f"  Average Cost: {comm.get('avg_comm_cost', 0):.1f} MB")
            print(f"  Total Cost: {comm.get('total_comm_cost', 0):.1f} MB")

            print("\nAction Distribution:")
            print(f"  Embeddings (E): {actions.get('E_pct', 0):.1f}%")
            print(f"  Prototypes (P): {actions.get('P_pct', 0):.1f}%")
            print(f"  Weights (W): {actions.get('W_pct', 0):.1f}%")

            print("\nEfficiency:")
            print(f"  F1 per MB: {eff.get('f1_per_mb', 0):.6f}")
            print(f"  Bandwidth Reduction: {eff.get('bandwidth_reduction_pct', 0):.1f}%")

        print("="*60)

    def plot_learning_curves(self, save_path: str = None):
        """Plot training curves."""
        if not self.episodes:
            print("No episode data to plot")
            return

        episodes = [ep["episode_id"] for ep in self.episodes]
        rewards = [ep["reward"] for ep in self.episodes]
        f1_scores = [ep["f1_score"] for ep in self.episodes]
        comm_costs = [ep["comm_cost_mb"] for ep in self.episodes]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Reward
        axes[0, 0].plot(episodes, rewards, alpha=0.7)
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].set_title("Episode Reward")
        axes[0, 0].grid(True, alpha=0.3)

        # F1 Score
        axes[0, 1].plot(episodes, f1_scores, alpha=0.7, color='green')
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("F1 Score")
        axes[0, 1].set_title("F1 Score (Detection Rate)")
        axes[0, 1].grid(True, alpha=0.3)

        # Communication Cost
        axes[1, 0].plot(episodes, comm_costs, alpha=0.7, color='red')
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Comm Cost (MB)")
        axes[1, 0].set_title("Communication Cost")
        axes[1, 0].grid(True, alpha=0.3)

        # Efficiency (F1 per MB)
        efficiency = [f1 / comm if comm > 0 else 0 for f1, comm in zip(f1_scores, comm_costs)]
        axes[1, 1].plot(episodes, efficiency, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("F1 per MB")
        axes[1, 1].set_title("Communication Efficiency")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved learning curves to: {save_path}")
        else:
            plt.savefig(self.run_dir / "learning_curves.png", dpi=150, bbox_inches='tight')
            print(f"Saved learning curves to: {self.run_dir / 'learning_curves.png'}")

        plt.close()

    def plot_action_distribution(self, save_path: str = None):
        """Plot action distribution over time."""
        if not self.episodes:
            return

        episodes = [ep["episode_id"] for ep in self.episodes]
        E_pcts = []
        P_pcts = []
        W_pcts = []

        for ep in self.episodes:
            actions = ep["what_actions"]
            total = sum(actions.values())
            if total > 0:
                E_pcts.append(100 * actions["E"] / total)
                P_pcts.append(100 * actions["P"] / total)
                W_pcts.append(100 * actions["W"] / total)
            else:
                E_pcts.append(0)
                P_pcts.append(0)
                W_pcts.append(0)

        plt.figure(figsize=(12, 6))
        plt.plot(episodes, E_pcts, label="Embeddings (E)", alpha=0.7)
        plt.plot(episodes, P_pcts, label="Prototypes (P)", alpha=0.7)
        plt.plot(episodes, W_pcts, label="Weights (W)", alpha=0.7)
        plt.xlabel("Episode")
        plt.ylabel("Percentage (%)")
        plt.title("WHAT Action Distribution Over Training")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.run_dir / "action_distribution.png", dpi=150, bbox_inches='tight')

        print(f"Saved action distribution to: {self.run_dir / 'action_distribution.png'}")
        plt.close()

    def plot_aggregation_stats(self, save_path: str = None):
        """Plot aggregation statistics."""
        if not self.episodes:
            return

        episodes = [ep["episode_id"] for ep in self.episodes]
        num_aggs = [ep["num_aggregations"] for ep in self.episodes]
        avg_neighbors = [ep["avg_neighbors_per_agg"] for ep in self.episodes]

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(episodes, num_aggs, alpha=0.7)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Number of Aggregations")
        axes[0].set_title("Aggregations per Episode")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(episodes, avg_neighbors, alpha=0.7, color='green')
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Avg Neighbors")
        axes[1].set_title("Average Neighbors per Aggregation")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.run_dir / "aggregation_stats.png", dpi=150, bbox_inches='tight')

        print(f"Saved aggregation stats to: {self.run_dir / 'aggregation_stats.png'}")
        plt.close()

    def plot_all(self):
        """Generate all plots."""
        print("\nGenerating all plots...")
        self.plot_learning_curves()
        self.plot_action_distribution()
        self.plot_aggregation_stats()
        print("All plots generated!")

    def export_for_paper(self, output_dir: str = None):
        """Export key metrics in formats useful for papers."""
        if output_dir is None:
            output_dir = self.run_dir / "paper_exports"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Export CSV for plotting in other tools
        import csv

        # Episode metrics CSV
        csv_path = output_dir / "episode_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "episode", "reward", "f1_score", "comm_cost_mb",
                "E_pct", "P_pct", "W_pct", "efficiency"
            ])
            writer.writeheader()

            for ep in self.episodes:
                actions = ep["what_actions"]
                total = sum(actions.values())

                row = {
                    "episode": ep["episode_id"],
                    "reward": ep["reward"],
                    "f1_score": ep["f1_score"],
                    "comm_cost_mb": ep["comm_cost_mb"],
                    "E_pct": 100 * actions["E"] / total if total > 0 else 0,
                    "P_pct": 100 * actions["P"] / total if total > 0 else 0,
                    "W_pct": 100 * actions["W"] / total if total > 0 else 0,
                    "efficiency": ep["f1_score"] / ep["comm_cost_mb"] if ep["comm_cost_mb"] > 0 else 0,
                }
                writer.writerow(row)

        print(f"Exported CSV to: {csv_path}")

        # Export LaTeX table
        if self.summary_stats:
            latex_path = output_dir / "summary_table.tex"
            with open(latex_path, "w") as f:
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\begin{tabular}{lr}\n")
                f.write("\\hline\n")
                f.write("Metric & Value \\\\\n")
                f.write("\\hline\n")

                perf = self.summary_stats.get("performance", {})
                comm = self.summary_stats.get("communication", {})
                actions = self.summary_stats.get("actions", {})
                eff = self.summary_stats.get("efficiency", {})

                f.write(f"Final F1 Score & {perf.get('final_f1', 0):.4f} \\\\\n")
                f.write(f"Final Comm Cost (MB) & {comm.get('final_comm_cost', 0):.1f} \\\\\n")
                f.write(f"Bandwidth Reduction & {eff.get('bandwidth_reduction_pct', 0):.1f}\\% \\\\\n")
                f.write(f"Prototype Usage & {actions.get('P_pct', 0):.1f}\\% \\\\\n")
                f.write(f"Efficiency (F1/MB) & {eff.get('f1_per_mb', 0):.6f} \\\\\n")

                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\caption{DFL Training Results}\n")
                f.write("\\label{tab:dfl_results}\n")
                f.write("\\end{table}\n")

            print(f"Exported LaTeX table to: {latex_path}")

        print(f"All exports saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DFL metrics")
    parser.add_argument("--run", type=str, required=True, help="Path to run directory")
    parser.add_argument("--plot", action="store_true", help="Generate all plots")
    parser.add_argument("--export", action="store_true", help="Export for paper")
    args = parser.parse_args()

    # Load and analyze
    analyzer = DFLMetricsAnalyzer(args.run)
    analyzer.print_summary()

    if args.plot:
        analyzer.plot_all()

    if args.export:
        analyzer.export_for_paper()


if __name__ == "__main__":
    main()
