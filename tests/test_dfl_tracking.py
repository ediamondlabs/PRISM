"""
Test script to verify DFL tracking integration.

Runs a short training session (5 episodes) to ensure:
1. Tracker initializes correctly
2. All JSON files are created
3. Metrics are being saved properly
4. No errors during tracking

Usage:
    python test_dfl_tracking.py
"""

import json
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_tracking_files(run_dir: Path):
    """Verify that all expected tracking files exist and contain data."""
    logger.info(f"\n{'='*60}")
    logger.info("VERIFYING TRACKING FILES")
    logger.info(f"{'='*60}")

    required_files = [
        "metadata.json",
        "episodes_summary.json",
        "summary_statistics.json",
    ]

    all_good = True

    # Check required files
    for filename in required_files:
        filepath = run_dir / filename
        if not filepath.exists():
            logger.error(f"❌ MISSING: {filename}")
            all_good = False
        else:
            # Check file is not empty
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if not data:
                        logger.error(f"❌ EMPTY: {filename}")
                        all_good = False
                    else:
                        logger.info(f"✅ OK: {filename} ({len(str(data))} bytes)")
            except Exception as e:
                logger.error(f"❌ ERROR reading {filename}: {e}")
                all_good = False

    # Check for vehicles_detail directory
    vehicles_dir = run_dir / "vehicles_detail"
    if not vehicles_dir.exists():
        logger.warning("⚠️  No vehicles_detail directory (may be created later)")
    else:
        vehicle_files = list(vehicles_dir.glob("*.json"))
        logger.info(f"✅ vehicles_detail/: {len(vehicle_files)} files")

    # Check for checkpoints directory
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        logger.warning("⚠️  No checkpoints directory (may be created later)")
    else:
        checkpoint_files = list(checkpoints_dir.glob("*.json"))
        logger.info(f"✅ checkpoints/: {len(checkpoint_files)} files")

    return all_good


def analyze_episode_data(run_dir: Path):
    """Analyze the episodes data to verify metrics."""
    logger.info(f"\n{'='*60}")
    logger.info("ANALYZING EPISODE DATA")
    logger.info(f"{'='*60}")

    episodes_file = run_dir / "episodes_summary.json"
    if not episodes_file.exists():
        logger.error("❌ Cannot analyze: episodes_summary.json not found")
        return False

    with open(episodes_file, 'r') as f:
        episodes = json.load(f)

    logger.info(f"Total episodes tracked: {len(episodes)}")

    if len(episodes) == 0:
        logger.error("❌ No episodes tracked!")
        return False

    # Check first episode
    ep = episodes[0]
    logger.info("\nFirst Episode Data:")
    logger.info(f"  episode_id: {ep.get('episode_id')}")
    logger.info(f"  reward: {ep.get('reward', 0):.2f}")
    logger.info(f"  f1_score: {ep.get('f1_score', 0):.4f}")
    logger.info(f"  comm_cost_mb: {ep.get('comm_cost_mb', 0):.2f}")
    logger.info(f"  what_actions: {ep.get('what_actions', {})}")

    # Check if we have RL stats
    if ep.get('actor_loss') is not None:
        logger.info(f"  actor_loss: {ep.get('actor_loss', 0):.4f}")
        logger.info(f"  critic_loss: {ep.get('critic_loss', 0):.4f}")
        logger.info(f"  entropy: {ep.get('entropy', 0):.4f}")

    # Check last episode
    if len(episodes) > 1:
        ep = episodes[-1]
        logger.info("\nLast Episode Data:")
        logger.info(f"  episode_id: {ep.get('episode_id')}")
        logger.info(f"  reward: {ep.get('reward', 0):.2f}")
        logger.info(f"  f1_score: {ep.get('f1_score', 0):.4f}")
        logger.info(f"  comm_cost_mb: {ep.get('comm_cost_mb', 0):.2f}")

    return True


def analyze_summary_stats(run_dir: Path):
    """Analyze summary statistics."""
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY STATISTICS")
    logger.info(f"{'='*60}")

    summary_file = run_dir / "summary_statistics.json"
    if not summary_file.exists():
        logger.error("❌ summary_statistics.json not found")
        return False

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    # Performance
    perf = summary.get("performance", {})
    logger.info("\nPerformance:")
    logger.info(f"  Initial F1: {perf.get('initial_f1', 0):.4f}")
    logger.info(f"  Final F1: {perf.get('final_f1', 0):.4f}")
    logger.info(f"  Best F1: {perf.get('best_f1', 0):.4f}")

    # Communication
    comm = summary.get("communication", {})
    logger.info("\nCommunication:")
    logger.info(f"  Initial Cost: {comm.get('initial_comm_cost', 0):.1f} MB")
    logger.info(f"  Final Cost: {comm.get('final_comm_cost', 0):.1f} MB")
    logger.info(f"  Total Cost: {comm.get('total_comm_cost', 0):.1f} MB")

    # Actions
    actions = summary.get("actions", {})
    logger.info("\nAction Distribution:")
    logger.info(f"  Embeddings (E): {actions.get('E_pct', 0):.1f}%")
    logger.info(f"  Prototypes (P): {actions.get('P_pct', 0):.1f}%")
    logger.info(f"  Weights (W): {actions.get('W_pct', 0):.1f}%")

    return True


def main():
    """Run test and verify tracking."""
    logger.info("="*60)
    logger.info("DFL TRACKING TEST")
    logger.info("="*60)
    logger.info("\nThis will:")
    logger.info("1. Run 5 episodes of DFL training")
    logger.info("2. Verify all JSON files are created")
    logger.info("3. Check metrics are being tracked correctly")
    logger.info("")

    # Import here to avoid circular imports
    import subprocess

    # Create test config
    test_config = """# Test configuration for tracking verification
+experiment=dfl_training
run_name=tracking_test
rl.training.num_episodes=5
rl.training.update_interval=2
rl.training.log_interval=1
rl.training.save_interval=10
"""

    config_file = Path("test_tracking_config.yaml")
    with open(config_file, 'w') as f:
        f.write(test_config)

    logger.info("Running 5 episodes of training with tracking enabled...")
    logger.info("This will take approximately 2-3 minutes...\n")

    try:
        # Run training
        result = subprocess.run(
            ["python", "train_dfl.py", "--config-name", "test_tracking_config"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )

        if result.returncode != 0:
            logger.error("❌ Training failed with error:")
            logger.error(result.stderr)
            return False

        logger.info("✅ Training completed successfully!\n")

        # Find the run directory
        metrics_dir = Path("outputs/dfl_metrics")
        if not metrics_dir.exists():
            logger.error("❌ Metrics directory not found!")
            return False

        # Find tracking_test run
        run_dirs = list(metrics_dir.glob("tracking_test*"))
        if not run_dirs:
            logger.error("❌ No tracking_test run directory found!")
            return False

        run_dir = run_dirs[0]  # Get most recent
        logger.info(f"Found run directory: {run_dir}\n")

        # Verify files
        files_ok = verify_tracking_files(run_dir)
        if not files_ok:
            logger.error("\n❌ FILE VERIFICATION FAILED!")
            return False

        # Analyze data
        episodes_ok = analyze_episode_data(run_dir)
        if not episodes_ok:
            logger.error("\n❌ EPISODE DATA ANALYSIS FAILED!")
            return False

        summary_ok = analyze_summary_stats(run_dir)
        if not summary_ok:
            logger.error("\n❌ SUMMARY STATS ANALYSIS FAILED!")
            return False

        # All checks passed
        logger.info(f"\n{'='*60}")
        logger.info("✅ ALL TESTS PASSED!")
        logger.info(f"{'='*60}")
        logger.info("\nTracking is working correctly!")
        logger.info(f"Run directory: {run_dir}")
        logger.info("\nYou can now retrain your full model with tracking enabled.")
        logger.info("\nTo analyze results later:")
        logger.info(f"  python analyze_dfl_metrics.py --run {run_dir}")

        return True

    except subprocess.TimeoutExpired:
        logger.error("❌ Training timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if config_file.exists():
            config_file.unlink()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
