"""
Analyze Table II: Representation Selection Evolution

This script loads the actual training results and computes the representation
selection distribution (Weights vs Embeddings) across different training phases
to verify the claims in Table II of the paper.

Usage:
    python analyze_table_ii.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


def load_episodes_data(metrics_dir: str = "outputs/dfl_metrics/dfl_training") -> List[Dict]:
    """Load episodes summary data.
    
    Args:
        metrics_dir: Directory containing episodes_summary.json
        
    Returns:
        List of episode dictionaries
    """
    episodes_path = Path(metrics_dir) / "episodes_summary.json"
    
    if not episodes_path.exists():
        raise FileNotFoundError(f"Episodes data not found at {episodes_path}")
    
    with open(episodes_path, 'r') as f:
        episodes = json.load(f)
    
    print(f"✓ Loaded {len(episodes)} episodes from {episodes_path}")
    return episodes


def analyze_phase_distributions(episodes: List[Dict], phase_definitions: Dict[str, Tuple[int, int]]) -> pd.DataFrame:
    """Analyze representation distribution for each training phase.
    
    Args:
        episodes: List of episode data
        phase_definitions: Dict mapping phase name to (start_ep, end_ep)
        
    Returns:
        DataFrame with phase statistics
    """
    results = []
    
    for phase_name, (start_ep, end_ep) in phase_definitions.items():
        # Filter episodes in this phase
        phase_episodes = [ep for ep in episodes if start_ep <= ep['episode_id'] <= end_ep]
        
        if not phase_episodes:
            print(f"⚠️  No episodes found for phase '{phase_name}' ({start_ep}-{end_ep})")
            continue
        
        # Aggregate action counts
        total_e = 0
        total_w = 0
        
        for ep in phase_episodes:
            what_actions = ep.get('what_actions', {})
            total_e += what_actions.get('E', 0)
            total_w += what_actions.get('W', 0)
        
        total_actions = total_e + total_w
        
        if total_actions == 0:
            print(f"⚠️  No actions found for phase '{phase_name}'")
            continue
        
        # Calculate percentages
        e_pct = (total_e / total_actions) * 100
        w_pct = (total_w / total_actions) * 100
        
        results.append({
            'Phase': phase_name,
            'Episodes': f"{start_ep}-{end_ep}",
            'Num_Episodes': len(phase_episodes),
            'Total_E': total_e,
            'Total_W': total_w,
            'E_Percentage': e_pct,
            'W_Percentage': w_pct,
        })
    
    return pd.DataFrame(results)


def get_per_episode_distributions(episodes: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get per-episode E and W percentages.
    
    Args:
        episodes: List of episode data
        
    Returns:
        Tuple of (episode_ids, e_percentages, w_percentages)
    """
    episode_ids = []
    e_percentages = []
    w_percentages = []
    
    for ep in episodes:
        what_actions = ep.get('what_actions', {})
        e_count = what_actions.get('E', 0)
        w_count = what_actions.get('W', 0)
        total = e_count + w_count
        
        if total > 0:
            episode_ids.append(ep['episode_id'])
            e_percentages.append((e_count / total) * 100)
            w_percentages.append((w_count / total) * 100)
    
    return np.array(episode_ids), np.array(e_percentages), np.array(w_percentages)


def plot_evolution_comparison(
    episodes: List[Dict],
    phase_definitions: Dict[str, Tuple[int, int]],
    paper_claims: Dict[str, Dict[str, float]],
    output_dir: str = "outputs/analysis"
) -> None:
    """Plot actual vs paper-claimed representation evolution.
    
    Args:
        episodes: List of episode data
        phase_definitions: Phase boundaries
        paper_claims: Paper's claimed distributions
        output_dir: Where to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get per-episode distributions
    episode_ids, e_pcts, w_pcts = get_per_episode_distributions(episodes)
    
    # Smooth the curves
    window_size = 100
    e_smooth = uniform_filter1d(e_pcts, size=window_size, mode='nearest')
    w_smooth = uniform_filter1d(w_pcts, size=window_size, mode='nearest')
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # ============ PLOT 1: Actual Distribution ============
    ax1.fill_between(episode_ids, 0, w_smooth, 
                     alpha=0.7, color='#e74c3c', label='Weights (W)')
    ax1.fill_between(episode_ids, w_smooth, 100, 
                     alpha=0.7, color='#3498db', label='Embeddings (E)')
    
    # Add phase boundaries
    colors_phase = ['#2ecc71', '#f39c12', '#9b59b6', '#34495e']
    for i, (phase_name, (start, end)) in enumerate(phase_definitions.items()):
        ax1.axvline(start, color=colors_phase[i], linestyle='--', alpha=0.5, linewidth=1.5)
        ax1.axvline(end, color=colors_phase[i], linestyle='--', alpha=0.5, linewidth=1.5)
        # Add phase label
        mid_point = (start + end) / 2
        ax1.text(mid_point, 95, phase_name.replace(' ', '\n'), 
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Training Episode', fontsize=12)
    ax1.set_ylabel('Representation Type (%)', fontsize=12)
    ax1.set_title('ACTUAL: Representation Selection Evolution from Implementation', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ============ PLOT 2: Paper Claims ============
    # Convert paper claims to plottable format
    phase_centers = []
    paper_w = []
    paper_e = []
    
    for phase_name, (start, end) in phase_definitions.items():
        if phase_name in paper_claims:
            phase_centers.append((start + end) / 2)
            paper_w.append(paper_claims[phase_name]['W'])
            paper_e.append(paper_claims[phase_name]['E'])
    
    # Interpolate to create smooth curves for visualization
    episode_range = np.linspace(0, 3000, 3000)
    paper_w_interp = np.interp(episode_range, phase_centers, paper_w)
    paper_e_interp = np.interp(episode_range, phase_centers, paper_e)
    
    ax2.fill_between(episode_range, 0, paper_w_interp, 
                     alpha=0.7, color='#e74c3c', label='Weights (W)')
    ax2.fill_between(episode_range, paper_w_interp, 100, 
                     alpha=0.7, color='#3498db', label='Embeddings (E)')
    
    # Add phase boundaries
    for i, (phase_name, (start, end)) in enumerate(phase_definitions.items()):
        ax2.axvline(start, color=colors_phase[i], linestyle='--', alpha=0.5, linewidth=1.5)
        ax2.axvline(end, color=colors_phase[i], linestyle='--', alpha=0.5, linewidth=1.5)
        mid_point = (start + end) / 2
        ax2.text(mid_point, 95, phase_name.replace(' ', '\n'), 
                ha='center', va='top', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Training Episode', fontsize=12)
    ax2.set_ylabel('Representation Type (%)', fontsize=12)
    ax2.set_title('PAPER CLAIMS: Representation Selection Evolution (Table II)', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save
    for fmt in ['png', 'pdf']:
        save_path = output_path / f"table_ii_comparison.{fmt}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def plot_rolling_statistics(
    episodes: List[Dict],
    window_sizes: List[int] = [50, 100, 500],
    output_dir: str = "outputs/analysis"
) -> None:
    """Plot rolling statistics with different window sizes to see trends.
    
    Args:
        episodes: List of episode data
        window_sizes: Different smoothing window sizes
        output_dir: Where to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    episode_ids, e_pcts, w_pcts = get_per_episode_distributions(episodes)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Plot W percentages with different smoothing
    for i, window in enumerate(window_sizes):
        w_smooth = uniform_filter1d(w_pcts, size=window, mode='nearest')
        ax1.plot(episode_ids, w_smooth, 
                label=f'Window={window}', linewidth=2, color=colors[i], alpha=0.8)
    
    ax1.axhline(np.mean(w_pcts), color='black', linestyle='--', linewidth=2, 
               label=f'Overall Mean: {np.mean(w_pcts):.1f}%', alpha=0.7)
    ax1.set_xlabel('Training Episode', fontsize=12)
    ax1.set_ylabel('Weights (W) Percentage', fontsize=12)
    ax1.set_title('Weight Selection Over Training (Different Smoothing Windows)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Plot E percentages with different smoothing
    for i, window in enumerate(window_sizes):
        e_smooth = uniform_filter1d(e_pcts, size=window, mode='nearest')
        ax2.plot(episode_ids, e_smooth, 
                label=f'Window={window}', linewidth=2, color=colors[i], alpha=0.8)
    
    ax2.axhline(np.mean(e_pcts), color='black', linestyle='--', linewidth=2, 
               label=f'Overall Mean: {np.mean(e_pcts):.1f}%', alpha=0.7)
    ax2.set_xlabel('Training Episode', fontsize=12)
    ax2.set_ylabel('Embeddings (E) Percentage', fontsize=12)
    ax2.set_title('Embedding Selection Over Training (Different Smoothing Windows)', 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    
    for fmt in ['png', 'pdf']:
        save_path = output_path / f"rolling_statistics.{fmt}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def generate_latex_table(df: pd.DataFrame, paper_claims: Dict[str, Dict[str, float]]) -> str:
    """Generate LaTeX table comparing actual vs paper claims.
    
    Args:
        df: DataFrame with actual phase statistics
        paper_claims: Paper's claimed distributions
        
    Returns:
        LaTeX table string
    """
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Representation Selection Evolution: Actual vs Paper Claims}")
    latex.append(r"\label{tab:table_ii_comparison}")
    latex.append(r"\begin{tabular}{|l|c|c|c|c|c|}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Training Phase} & \textbf{Episodes} & \textbf{W (Actual)} & \textbf{E (Actual)} & \textbf{W (Paper)} & \textbf{E (Paper)} \\")
    latex.append(r"\hline")
    
    for _, row in df.iterrows():
        phase = row['Phase']
        episodes = row['Episodes']
        w_actual = row['W_Percentage']
        e_actual = row['E_Percentage']
        
        # Get paper claims
        if phase in paper_claims:
            w_paper = paper_claims[phase]['W']
            e_paper = paper_claims[phase]['E']
        else:
            w_paper = e_paper = 0.0
        
        latex.append(
            f"{phase} & {episodes} & "
            f"{w_actual:.1f}\\% & {e_actual:.1f}\\% & "
            f"{w_paper:.1f}\\% & {e_paper:.1f}\\% \\\\"
        )
    
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)


def main():
    """Main analysis function."""
    print("=" * 80)
    print("TABLE II VERIFICATION: Representation Selection Evolution")
    print("=" * 80)
    print()
    
    # Define training phases as claimed in paper (Table II)
    phase_definitions = {
        'Initial Learning': (0, 499),
        'Active Refinement': (500, 1499),
        'Convergence': (1500, 2499),
        'Maintenance': (2500, 2999),
    }
    
    # Paper's claimed distributions (from Table II)
    paper_claims = {
        'Initial Learning': {'W': 85, 'E': 15},
        'Active Refinement': {'W': 40, 'E': 60},
        'Convergence': {'W': 20, 'E': 80},
        'Maintenance': {'W': 10, 'E': 90},
    }
    
    # Load episodes data
    print("1. Loading training data...")
    episodes = load_episodes_data()
    print()
    
    # Analyze phase distributions
    print("2. Analyzing representation distribution by training phase...")
    phase_stats = analyze_phase_distributions(episodes, phase_definitions)
    print()
    
    # Print comparison table
    print("=" * 80)
    print("ACTUAL vs PAPER COMPARISON")
    print("=" * 80)
    print()
    
    # Create comparison display
    comparison_data = []
    for _, row in phase_stats.iterrows():
        phase = row['Phase']
        episode_range = row['Episodes']  # Renamed to avoid shadowing
        w_actual = row['W_Percentage']
        e_actual = row['E_Percentage']
        
        if phase in paper_claims:
            w_paper = paper_claims[phase]['W']
            e_paper = paper_claims[phase]['E']
            w_diff = w_actual - w_paper
            e_diff = e_actual - e_paper
        else:
            w_paper = e_paper = w_diff = e_diff = 0.0
        
        comparison_data.append({
            'Phase': phase,
            'Episodes': episode_range,
            'W (Actual)': f"{w_actual:.1f}%",
            'W (Paper)': f"{w_paper:.0f}%",
            'W Diff': f"{w_diff:+.1f}%",
            'E (Actual)': f"{e_actual:.1f}%",
            'E (Paper)': f"{e_paper:.0f}%",
            'E Diff': f"{e_diff:+.1f}%",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()
    
    # Statistical summary
    print("=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    print()
    
    _, e_pcts, w_pcts = get_per_episode_distributions(episodes)
    
    # Get per-episode distributions for detailed stats
    episode_ids, e_pcts, w_pcts = get_per_episode_distributions(episodes)
    
    print("Overall Statistics (All 3000 Episodes):")
    print("  Weights (W):")
    print(f"    - Mean: {np.mean(w_pcts):.2f}%")
    print(f"    - Std:  {np.std(w_pcts):.2f}%")
    print(f"    - Min:  {np.min(w_pcts):.2f}%")
    print(f"    - Max:  {np.max(w_pcts):.2f}%")
    print()
    print("  Embeddings (E):")
    print(f"    - Mean: {np.mean(e_pcts):.2f}%")
    print(f"    - Std:  {np.std(e_pcts):.2f}%")
    print(f"    - Min:  {np.min(e_pcts):.2f}%")
    print(f"    - Max:  {np.max(e_pcts):.2f}%")
    print()
    
    # Check for trends
    # Split into early (0-1000) and late (2000-3000) episodes
    early_mask = (e_pcts[:1000] if len(e_pcts) >= 1000 else e_pcts)
    late_mask = (e_pcts[-1000:] if len(e_pcts) >= 1000 else e_pcts)
    
    w_early = np.mean(w_pcts[:1000]) if len(w_pcts) >= 1000 else np.mean(w_pcts)
    w_late = np.mean(w_pcts[-1000:]) if len(w_pcts) >= 1000 else np.mean(w_pcts)
    w_change = w_late - w_early
    
    print("Trend Analysis:")
    print(f"  Early training (Ep 0-999):   W = {w_early:.1f}%")
    print(f"  Late training (Ep 2000-2999): W = {w_late:.1f}%")
    print(f"  Change: {w_change:+.1f}%")
    print()
    
    if abs(w_change) < 10:
        print("  ⚠️  FINDING: Distribution is relatively STABLE over training")
        print("     (Less than 10% change from early to late training)")
    else:
        print("  ✓ FINDING: Distribution shows significant evolution")
    print()
    
    # Key findings
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print()
    
    # Check if paper claims match reality
    max_diff = 0
    for _, row in phase_stats.iterrows():
        phase = row['Phase']
        if phase in paper_claims:
            w_diff = abs(row['W_Percentage'] - paper_claims[phase]['W'])
            max_diff = max(max_diff, w_diff)
    
    if max_diff > 30:
        print(f"🚨 CRITICAL: Paper claims differ from actual by up to {max_diff:.1f}%!")
        print()
        print("   The paper's Table II suggests a dramatic evolution:")
        print("     - Initial: 85% Weights → Maintenance: 10% Weights")
        print("     - Change of 75 percentage points")
        print()
        print("   Actual implementation shows:")
        print(f"     - Relatively stable around {np.mean(w_pcts):.0f}% Weights throughout")
        print(f"     - Total change: {abs(w_change):.0f} percentage points")
    elif max_diff > 10:
        print(f"⚠️  MODERATE: Paper claims differ from actual by up to {max_diff:.1f}%")
    else:
        print("✓ Paper claims closely match actual implementation")
    print()
    
    # Generate visualizations
    print("3. Generating comparison plots...")
    plot_evolution_comparison(episodes, phase_definitions, paper_claims)
    plot_rolling_statistics(episodes)
    print()
    
    # Generate LaTeX table
    print("4. Generating LaTeX table...")
    latex_table = generate_latex_table(phase_stats, paper_claims)
    
    output_dir = Path("outputs/analysis")
    latex_path = output_dir / "table_ii_latex.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"✓ Saved LaTeX table: {latex_path}")
    print()
    
    # Save CSV
    csv_path = output_dir / "table_ii_actual_vs_paper.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV: {csv_path}")
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("Output files:")
    print("  - outputs/analysis/table_ii_comparison.pdf")
    print("  - outputs/analysis/table_ii_comparison.png")
    print("  - outputs/analysis/rolling_statistics.pdf")
    print("  - outputs/analysis/rolling_statistics.png")
    print("  - outputs/analysis/table_ii_latex.tex")
    print("  - outputs/analysis/table_ii_actual_vs_paper.csv")


if __name__ == "__main__":
    main()

