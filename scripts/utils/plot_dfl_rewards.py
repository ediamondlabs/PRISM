import json
import matplotlib.pyplot as plt
import numpy as np

# Load the episodes summary data
with open('outputs/dfl_metrics/dfl_training/episodes_summary.json', 'r') as f:
    episodes = json.load(f)

# Extract episode IDs and rewards
episode_ids = [ep['episode_id'] for ep in episodes]
rewards = [ep['reward'] for ep in episodes]

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(episode_ids, rewards, linewidth=1.5, alpha=0.7, label='Reward per Episode')

# Add a moving average for better trend visualization
window_size = 50
if len(rewards) >= window_size:
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    moving_avg_episodes = episode_ids[window_size-1:]
    plt.plot(moving_avg_episodes, moving_avg, linewidth=2, color='red',
             label=f'{window_size}-Episode Moving Average')

plt.xlabel('Episode', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.title('DFL Training: Average Reward vs Episode', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

# Save the plot
plt.savefig('outputs/dfl_metrics/dfl_training/reward_vs_episode.png', dpi=300, bbox_inches='tight')
print("Plot saved to: outputs/dfl_metrics/dfl_training/reward_vs_episode.png")
print(f"Total episodes: {len(episode_ids)}")
print(f"Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

plt.show()
