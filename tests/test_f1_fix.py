"""Test that F1 score is now non-zero with the threshold fix."""

import hydra
from omegaconf import DictConfig
from src.simulation import V2VEnvironment
from src.rl.env_wrapper import RLV2VWrapper
from src.rl.rewards import RewardFunction
from src.utils import get_device, set_seed

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    set_seed(config.seed)
    device = get_device(config.device)

    # Create environment
    env = V2VEnvironment(config)

    # Create reward function
    reward_fn = RewardFunction(
        immediate_weight=config.rl.reward.immediate_weight,
        global_weight=config.rl.reward.global_weight,
        perf_scale=config.rl.reward.perf_scale,
        comm_cost_scale=config.rl.reward.comm_cost_scale,
        staleness_penalty_weight=config.rl.reward.staleness_penalty_weight,
    )

    # Create RL wrapper
    rl_env = RLV2VWrapper(
        env=env,
        reward_fn=reward_fn,
        use_neighbor_encoder=False,
        device=device,
        dfl_protocols=None,
        use_dfl_features=False,
    )

    print("=" * 60)
    print("TESTING F1 SCORE FIX")
    print("=" * 60)

    # Run a few timesteps
    states, global_state = rl_env.reset()

    f1_scores = []

    for timestep in range(100):
        # Random actions
        actions = {vid: {'who_action': [0, 1], 'what_action': 1} for vid in env.vehicle_ids}

        # Step
        states, rewards, done, info = rl_env.step(actions)

        f1 = info.get('f1_score', 0.0)
        f1_scores.append(f1)

        if timestep % 10 == 0:
            print(f"Timestep {timestep}: F1 = {f1:.4f}")

        if done:
            break

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    import numpy as np
    f1_scores = np.array(f1_scores)

    print(f"Total timesteps: {len(f1_scores)}")
    print(f"Non-zero F1 scores: {np.sum(f1_scores > 0)}/{len(f1_scores)}")
    print(f"Mean F1: {np.mean(f1_scores):.4f}")
    print(f"Max F1: {np.max(f1_scores):.4f}")
    print(f"Min F1: {np.min(f1_scores):.4f}")

    # Check first 50 timesteps (should be 0) vs last 50 (should be > 0)
    if len(f1_scores) >= 50:
        first_50 = f1_scores[:50]
        last_50 = f1_scores[50:]
        print("\nFirst 50 timesteps (before window filled):")
        print(f"  Mean F1: {np.mean(first_50):.4f}")
        print(f"  Non-zero: {np.sum(first_50 > 0)}/{len(first_50)}")
        print("\nLast 50 timesteps (after window filled):")
        print(f"  Mean F1: {np.mean(last_50):.4f}")
        print(f"  Non-zero: {np.sum(last_50 > 0)}/{len(last_50)}")

    if np.mean(f1_scores[50:]) > 0:
        print("\n✅ FIX VERIFIED: F1 scores are now non-zero!")
    else:
        print("\n❌ FIX FAILED: F1 scores are still zero!")

if __name__ == "__main__":
    main()
