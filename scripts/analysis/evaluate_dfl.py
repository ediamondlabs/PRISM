"""
Comprehensive evaluation script for DFL training.

Usage:
    python evaluate_dfl.py checkpoint=outputs/dfl/checkpoints/dfl_final.pt
"""

import hydra
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from pathlib import Path
from collections import defaultdict

from src.simulation import V2VEnvironment
from src.rl import PPOActor
from src.utils import get_device, set_seed

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    """Evaluate trained DFL model."""

    set_seed(config.seed)
    device = get_device(config.device)

    logger.info("=" * 60)
    logger.info("DFL EVALUATION")
    logger.info("=" * 60)

    # Load checkpoint
    checkpoint_path = config.get('checkpoint', 'outputs/dfl/checkpoints/dfl_final.pt')
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create environment
    logger.info("Creating environment...")
    env = V2VEnvironment(config)

    # Create actors
    logger.info("Creating RL agent...")
    state_dim = 106  # From config
    actor = PPOActor(
        state_dim=state_dim,
        shared_hidden_dim=config.rl.actor.shared_hidden_dim,
        when_hidden_dim=config.rl.actor.when_hidden_dim,
        who_hidden_dim=config.rl.actor.who_hidden_dim,
        what_hidden_dim=config.rl.actor.what_hidden_dim,
        max_neighbors=config.rl.actor.max_neighbors,
        k_neighbors=config.rl.actor.k_neighbors,
        num_repr_types=config.rl.actor.num_repr_types,
        use_when_head=config.get('use_when_head', True),
        dropout=config.rl.actor.dropout,
    ).to(device)

    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()

    logger.info("Running evaluation episodes...")

    # Metrics
    metrics = {
        'episode_rewards': [],
        'episode_f1_scores': [],
        'episode_comm_costs': [],
        'timestep_f1_scores': [],
        'anomaly_scores': [],
        'what_actions': defaultdict(int),  # Count of E, P, W
        'detection_rates': [],
    }

    num_eval_episodes = 10
    repr_types = ['E', 'P', 'W']

    for episode in range(num_eval_episodes):
        logger.info(f"\nEpisode {episode + 1}/{num_eval_episodes}")

        obs = env.reset()
        episode_reward = 0
        episode_comm_cost = 0
        episode_f1_list = []

        for timestep in range(100):
            # Get states (simplified - just use position, velocity, etc.)
            states = {}
            for vid in env.vehicle_ids:
                state = np.concatenate([
                    obs[vid]['position'],
                    obs[vid]['velocity'],
                    [obs[vid]['speed']],
                    [obs[vid]['num_neighbors']],
                    # Add more features as needed
                ])
                # Pad to state_dim
                if len(state) < state_dim:
                    state = np.pad(state, (0, state_dim - len(state)))
                states[vid] = state[:state_dim]

            # Select actions using trained policy
            actions = {}
            with torch.no_grad():
                for vid in env.vehicle_ids:
                    state_tensor = torch.FloatTensor(states[vid]).unsqueeze(0).to(device)

                    # Get neighbor mask (simplified)
                    neighbor_mask = torch.ones(1, config.rl.actor.max_neighbors).bool().to(device)

                    # Sample action
                    action_dist = actor(state_tensor, neighbor_mask)
                    who_action = action_dist['who_dist'].sample()
                    what_action = action_dist['what_dist'].sample()

                    actions[vid] = {
                        'who_action': who_action.cpu().numpy()[0],
                        'what_action': int(what_action.cpu().numpy()[0])
                    }

                    # Track WHAT actions
                    what_idx = int(what_action.cpu().numpy()[0])
                    metrics['what_actions'][repr_types[what_idx]] += 1

            # Step environment
            obs, rewards, done, info = env.step(actions)

            # Collect metrics
            if timestep >= 50:  # Only after window is filled
                # Compute detection rate
                anomaly_scores = [obs[vid].get('anomaly_score', 0.0) for vid in env.vehicle_ids]
                anomaly_scores = [s if s is not None else 0.0 for s in anomaly_scores]
                detection_rate = sum(1 for s in anomaly_scores if s > 0.05) / len(anomaly_scores)

                metrics['detection_rates'].append(detection_rate)
                metrics['anomaly_scores'].extend(anomaly_scores)
                metrics['timestep_f1_scores'].append(detection_rate)
                episode_f1_list.append(detection_rate)

            episode_reward += sum(rewards.values()) if isinstance(rewards, dict) else 0

            # Estimate comm cost (simplified)
            comm_cost = sum(
                0.1 if actions[vid]['what_action'] == 2 else  # P = small
                1.0 if actions[vid]['what_action'] == 0 else  # E = large
                5.0  # W = very large
                for vid in env.vehicle_ids
            )
            episode_comm_cost += comm_cost

            if done:
                break

        # Episode summary
        avg_f1 = np.mean(episode_f1_list) if episode_f1_list else 0.0
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_f1_scores'].append(avg_f1)
        metrics['episode_comm_costs'].append(episode_comm_cost)

        logger.info(f"  Reward: {episode_reward:.2f}")
        logger.info(f"  F1: {avg_f1:.4f}")
        logger.info(f"  Comm Cost: {episode_comm_cost:.2f} MB")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    logger.info("\n1. OVERALL PERFORMANCE:")
    logger.info(f"   Average Reward: {np.mean(metrics['episode_rewards']):.2f} ± {np.std(metrics['episode_rewards']):.2f}")
    logger.info(f"   Average F1: {np.mean(metrics['episode_f1_scores']):.4f} ± {np.std(metrics['episode_f1_scores']):.4f}")
    logger.info(f"   Average Comm Cost: {np.mean(metrics['episode_comm_costs']):.2f} ± {np.std(metrics['episode_comm_costs']):.2f} MB")

    logger.info("\n2. DETECTION PERFORMANCE:")
    logger.info(f"   Average Detection Rate: {np.mean(metrics['detection_rates']):.4f}")
    logger.info(f"   Detection Rate Std: {np.std(metrics['detection_rates']):.4f}")

    logger.info("\n3. ANOMALY SCORE STATISTICS:")
    if metrics['anomaly_scores']:
        logger.info(f"   Mean: {np.mean(metrics['anomaly_scores']):.6f}")
        logger.info(f"   Std: {np.std(metrics['anomaly_scores']):.6f}")
        logger.info(f"   Min: {np.min(metrics['anomaly_scores']):.6f}")
        logger.info(f"   Max: {np.max(metrics['anomaly_scores']):.6f}")
        logger.info(f"   95th percentile: {np.percentile(metrics['anomaly_scores'], 95):.6f}")

    logger.info("\n4. COMMUNICATION STRATEGY:")
    total_actions = sum(metrics['what_actions'].values())
    if total_actions > 0:
        for repr_type, count in metrics['what_actions'].items():
            pct = 100.0 * count / total_actions
            logger.info(f"   {repr_type}: {count}/{total_actions} ({pct:.1f}%)")

    logger.info("\n5. EFFICIENCY METRIC:")
    if np.mean(metrics['episode_comm_costs']) > 0:
        efficiency = np.mean(metrics['episode_f1_scores']) / np.mean(metrics['episode_comm_costs'])
        logger.info(f"   F1 per MB: {efficiency:.6f}")
        logger.info("   (Higher is better - more F1 for less bandwidth)")

    # Create plots
    logger.info("\nGenerating plots...")
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: F1 vs Comm Cost (Efficiency)
    plt.figure(figsize=(8, 6))
    plt.scatter(metrics['episode_comm_costs'], metrics['episode_f1_scores'], s=100, alpha=0.6)
    plt.xlabel('Communication Cost (MB)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Efficiency: F1 vs Communication Cost', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_plot.png', dpi=150)
    logger.info(f"Saved: {output_dir / 'efficiency_plot.png'}")

    # Plot 2: Action Distribution
    plt.figure(figsize=(8, 6))
    actions_list = list(metrics['what_actions'].keys())
    counts_list = [metrics['what_actions'][a] for a in actions_list]
    plt.bar(actions_list, counts_list, alpha=0.7)
    plt.xlabel('Representation Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('WHAT Action Distribution', fontsize=14)
    for i, (action, count) in enumerate(zip(actions_list, counts_list)):
        pct = 100.0 * count / sum(counts_list)
        plt.text(i, count, f'{pct:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(output_dir / 'action_distribution.png', dpi=150)
    logger.info(f"Saved: {output_dir / 'action_distribution.png'}")

    # Plot 3: Anomaly Score Distribution
    plt.figure(figsize=(8, 6))
    plt.hist(metrics['anomaly_scores'], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(0.05, color='red', linestyle='--', linewidth=2, label='Threshold (0.05)')
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Anomaly Score Distribution', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'anomaly_distribution.png', dpi=150)
    logger.info(f"Saved: {output_dir / 'anomaly_distribution.png'}")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
