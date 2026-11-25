"""Quick diagnostic to check actual anomaly scores during simulation."""

import hydra
import numpy as np
from omegaconf import DictConfig
from src.simulation import V2VEnvironment
from src.utils import get_device, set_seed

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    set_seed(config.seed)
    device = get_device(config.device)

    # Create environment
    env = V2VEnvironment(config)

    print("=" * 60)
    print("ANOMALY SCORE DIAGNOSTIC")
    print("=" * 60)

    # Run 1 episode and collect all anomaly scores
    obs = env.reset()

    all_scores = []

    for timestep in range(100):
        # Check anomaly scores
        scores = [obs[vid].get('anomaly_score') for vid in env.vehicle_ids]
        non_none_scores = [s for s in scores if s is not None]

        if timestep % 10 == 0:
            print(f"\nTimestep {timestep}:")
            print(f"  Total vehicles: {len(scores)}")
            print(f"  Non-None scores: {len(non_none_scores)}")
            if non_none_scores:
                print(f"  Score stats: min={min(non_none_scores):.6f}, max={max(non_none_scores):.6f}, mean={np.mean(non_none_scores):.6f}")
                print(f"  Num > 0.5: {sum(1 for s in non_none_scores if s > 0.5)}")
                print(f"  Num > 0.1: {sum(1 for s in non_none_scores if s > 0.1)}")
                print(f"  Num > 0.01: {sum(1 for s in non_none_scores if s > 0.01)}")

        all_scores.extend(non_none_scores)

        # Step environment (actions are optional for V2V env)
        obs, rewards, done, info = env.step(actions=None)

        if done:
            break

    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    if all_scores:
        print(f"Total non-None scores collected: {len(all_scores)}")
        print(f"Min: {min(all_scores):.6f}")
        print(f"Max: {max(all_scores):.6f}")
        print(f"Mean: {np.mean(all_scores):.6f}")
        print(f"Median: {np.median(all_scores):.6f}")
        print(f"Std: {np.std(all_scores):.6f}")
        print("\nPercentiles:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  {p}th: {np.percentile(all_scores, p):.6f}")
        print("\nThreshold Analysis:")
        for thresh in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
            count = sum(1 for s in all_scores if s > thresh)
            pct = 100.0 * count / len(all_scores)
            print(f"  > {thresh:.2f}: {count}/{len(all_scores)} ({pct:.1f}%)")
    else:
        print("NO ANOMALY SCORES COLLECTED - ALL WERE NONE!")

if __name__ == "__main__":
    main()
