#!/usr/bin/env python3
"""
Plot Average Reward vs Episode for 5, 10, and 15 cars experiments
Creates individual plots and a comparison plot with outlier removal
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def remove_outliers_iqr(data, multiplier=1.5):
    """Remove outliers using Interquartile Range (IQR) method.

    Args:
        data: Array of values
        multiplier: IQR multiplier (1.5 is standard, 3.0 is more conservative)

    Returns:
        Cleaned data with outliers set to NaN
    """
    data = np.array(data, dtype=float)
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)

    # Set outliers to NaN
    cleaned = data.copy()
    cleaned[(data < lower_bound) | (data > upper_bound)] = np.nan

    outliers_count = np.sum(np.isnan(cleaned))

    return cleaned, outliers_count, lower_bound, upper_bound


def smooth_curve(data, window_size=50):
    """Smooth data using moving average, handling NaN values.

    Args:
        data: Input data array
        window_size: Size of smoothing window

    Returns:
        Smoothed data array
    """
    data = np.array(data, dtype=float)
    smoothed = np.zeros_like(data)

    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        window = data[start:end]
        # Use nanmean to ignore NaN values
        smoothed[i] = np.nanmean(window)

    return smoothed


def load_experiment_data(num_cars):
    """Load episodes data for a specific experiment.

    Args:
        num_cars: Number of cars (5, 10, or 15)

    Returns:
        Tuple of (episodes, rewards, episode_ids) or None if not found
    """
    episodes_path = Path(f"outputs/dfl_metrics/dfl_{num_cars}cars/episodes_summary.json")

    if not episodes_path.exists():
        print(f"Warning: {episodes_path} not found!")
        return None

    print(f"Loading {num_cars} cars data from {episodes_path}...")
    with open(episodes_path, 'r') as f:
        episodes = json.load(f)

    episode_ids = [ep['episode_id'] for ep in episodes]
    rewards = np.array([ep['reward'] for ep in episodes])

    return episodes, rewards, episode_ids


def plot_individual(num_cars, episode_ids, rewards_clean, rewards_smooth, outliers_count):
    """Create individual plot for one experiment.

    Args:
        num_cars: Number of cars
        episode_ids: List of episode IDs
        rewards_clean: Cleaned rewards (outliers removed)
        rewards_smooth: Smoothed rewards
        outliers_count: Number of outliers removed
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot raw rewards (with outliers removed as gaps)
    ax.plot(episode_ids, rewards_clean, alpha=0.3, color='#1f77b4',
            label='Raw Rewards (outliers removed)', linewidth=0.8)

    # Plot smoothed rewards
    ax.plot(episode_ids, rewards_smooth, color='#d62728',
            label='Smoothed (window=50)', linewidth=2)

    # Formatting
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title(f'DFL Training ({num_cars} Cars): Average Reward vs Episode',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add statistics text box
    stats_text = (
        f'Vehicles: {num_cars}\n'
        f'Episodes: {len(episode_ids)}\n'
        f'Outliers removed: {outliers_count}\n'
        f'Mean reward: {np.nanmean(rewards_clean):.2f}\n'
        f'Final reward: {rewards_smooth[-1]:.2f}'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5), fontsize=9)

    plt.tight_layout()

    # Save plot
    output_path = Path(f"outputs/dfl_metrics/dfl_{num_cars}cars/reward_vs_episode_clean.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    plt.close()


def plot_comparison(data_dict):
    """Create comparison plot with all experiments.

    Args:
        data_dict: Dictionary with keys as num_cars and values as
                   (episode_ids, rewards_smooth, mean_reward)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    colors = {5: '#1f77b4', 10: '#ff7f0e', 15: '#2ca02c'}

    for num_cars, (episode_ids, rewards_smooth, mean_reward) in data_dict.items():
        ax.plot(episode_ids, rewards_smooth, color=colors[num_cars],
                label=f'{num_cars} Cars (mean: {mean_reward:.2f})',
                linewidth=2, alpha=0.8)

    # Formatting
    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Average Reward (Smoothed)', fontsize=13)
    ax.set_title('DFL Training Comparison: Average Reward vs Episode (All Experiments)',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save plot
    output_path = Path("outputs/reward_comparison_all_experiments.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved: {output_path}")

    plt.close()


def main():
    print("="*80)
    print("DFL Reward Plotting: 5, 10, and 15 Cars Experiments")
    print("="*80)

    comparison_data = {}

    for num_cars in [5, 10, 15]:
        print(f"\n{'='*60}")
        print(f"Processing {num_cars} Cars Experiment")
        print(f"{'='*60}")

        # Load data
        result = load_experiment_data(num_cars)
        if result is None:
            continue

        episodes, rewards, episode_ids = result

        print(f"  Loaded {len(episodes)} episodes")
        print("  Reward stats (with outliers):")
        print(f"    Mean: {np.mean(rewards):.2f}")
        print(f"    Std: {np.std(rewards):.2f}")
        print(f"    Min: {np.min(rewards):.2f}")
        print(f"    Max: {np.max(rewards):.2f}")

        # Remove outliers
        rewards_clean, outliers_count, lower_bound, upper_bound = remove_outliers_iqr(rewards, multiplier=1.5)

        print("  Outlier removal:")
        print(f"    Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"    Outliers removed: {outliers_count}/{len(rewards)}")

        print("  Reward stats (without outliers):")
        print(f"    Mean: {np.nanmean(rewards_clean):.2f}")
        print(f"    Std: {np.nanstd(rewards_clean):.2f}")
        print(f"    Min: {np.nanmin(rewards_clean):.2f}")
        print(f"    Max: {np.nanmax(rewards_clean):.2f}")

        # Smooth rewards
        rewards_smooth = smooth_curve(rewards_clean, window_size=50)

        # Create individual plot
        plot_individual(num_cars, episode_ids, rewards_clean, rewards_smooth, outliers_count)

        # Store for comparison plot
        comparison_data[num_cars] = (episode_ids, rewards_smooth, np.nanmean(rewards_clean))

    # Create comparison plot
    if comparison_data:
        print(f"\n{'='*60}")
        print("Creating Comparison Plot")
        print(f"{'='*60}")
        plot_comparison(comparison_data)

    print("\n" + "="*80)
    print("All plots created successfully!")
    print("="*80)
    print("\nIndividual plots:")
    print("  - outputs/dfl_metrics/dfl_5cars/reward_vs_episode_clean.png")
    print("  - outputs/dfl_metrics/dfl_10cars/reward_vs_episode_clean.png")
    print("  - outputs/dfl_metrics/dfl_15cars/reward_vs_episode_clean.png")
    print("\nComparison plot:")
    print("  - outputs/reward_comparison_all_experiments.png")
    print("="*80)


if __name__ == "__main__":
    main()
