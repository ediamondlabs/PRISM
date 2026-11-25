"""
Quick debug script to test DFL training for a few episodes and inspect metrics.
"""

import hydra
import logging
import numpy as np
from omegaconf import DictConfig

from src.simulation import V2VEnvironment
from src.utils import get_device, set_seed

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    """Debug DFL training with detailed logging."""

    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    # Create environment
    logger.info("Creating V2V environment...")
    env = V2VEnvironment(config)
    logger.info(f"Environment created with {len(env.vehicle_ids)} vehicles")

    logger.info("Running debug episodes...")

    # Run a few test episodes
    for episode in range(3):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPISODE {episode + 1}")
        logger.info(f"{'='*60}")

        obs = env.reset()

        for timestep in range(10):  # Just 10 timesteps per episode
            logger.info(f"\nTimestep {timestep + 1}")

            # Check observations
            sample_vid = list(obs.keys())[0]
            sample_obs = obs[sample_vid]
            logger.info(f"  Sample vehicle {sample_vid} obs keys: {sample_obs.keys()}")
            logger.info(f"  Sample anomaly_score: {sample_obs.get('anomaly_score', 'MISSING')}")
            logger.info(f"  Sample reconstruction_error: {sample_obs.get('reconstruction_error', 'MISSING')}")

            # Check all vehicles anomaly scores
            anomaly_scores = [obs[vid].get('anomaly_score', 0.0) for vid in env.vehicle_ids]
            logger.info(f"  All anomaly scores (first 5): {anomaly_scores[:5]}")
            logger.info(f"  Anomaly score stats: mean={np.mean(anomaly_scores):.4f}, max={np.max(anomaly_scores):.4f}, min={np.min(anomaly_scores):.4f}")
            logger.info(f"  Num scores > 0.5: {sum(1 for s in anomaly_scores if s > 0.5)}/{len(anomaly_scores)}")

            # Take a step in environment (random actions)
            actions = {vid: env.action_space.sample() for vid in env.vehicle_ids}
            obs, rewards, done, info = env.step(actions)

            logger.info(f"  Episode done: {done}")
            logger.info(f"  Info F1 score: {info.get('f1_score', 'MISSING')}")
            logger.info(f"  Sample reward: {rewards.get(sample_vid, 'MISSING')}")

            if done:
                break

        logger.info(f"\nEpisode {episode + 1} completed")

    logger.info("\nDebug test completed!")


if __name__ == "__main__":
    main()
