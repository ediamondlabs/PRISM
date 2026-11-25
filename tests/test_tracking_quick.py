"""
Quick test for DFL tracking - runs 3 episodes to verify tracking works.

Usage:
    python test_tracking_quick.py
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_tracking_output(run_dir: Path, expected_episodes: int = 3):
    """Verify tracking output."""
    logger.info(f"\n{'='*60}")
    logger.info("VERIFYING TRACKING OUTPUT")
    logger.info(f"{'='*60}")
    logger.info(f"Run directory: {run_dir}\n")

    # Check required files
    required_files = {
        "metadata.json": "Run configuration and metadata",
        "episodes_summary.json": "Episode-level metrics",
        "summary_statistics.json": "Aggregated statistics",
    }

    all_ok = True

    for filename, description in required_files.items():
        filepath = run_dir / filename
        if not filepath.exists():
            logger.error(f"❌ MISSING: {filename} ({description})")
            all_ok = False
            continue

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if not data:
                logger.error(f"❌ EMPTY: {filename}")
                all_ok = False
                continue

            # Show file size and content preview
            file_size = filepath.stat().st_size
            logger.info(f"✅ {filename}: {file_size} bytes")

            # Detailed checks for each file
            if filename == "metadata.json":
                logger.info(f"   - Run: {data.get('run_name', 'N/A')}")
                logger.info(f"   - Total episodes: {data.get('total_episodes', 'N/A')}")
                logger.info(f"   - Duration: {data.get('total_duration_seconds', 0):.1f}s")

            elif filename == "episodes_summary.json":
                logger.info(f"   - Episodes tracked: {len(data)}")
                if len(data) != expected_episodes:
                    logger.warning(f"   ⚠️  Expected {expected_episodes} episodes, got {len(data)}")

                if len(data) > 0:
                    first_ep = data[0]
                    logger.info(f"   - First episode reward: {first_ep.get('reward', 0):.2f}")
                    logger.info(f"   - First episode F1: {first_ep.get('f1_score', 0):.4f}")
                    logger.info(f"   - First episode comm cost: {first_ep.get('comm_cost_mb', 0):.2f} MB")
                    logger.info(f"   - Actions tracked: {first_ep.get('what_actions', {})}")

            elif filename == "summary_statistics.json":
                perf = data.get('performance', {})
                comm = data.get('communication', {})
                actions = data.get('actions', {})

                logger.info(f"   - Final F1: {perf.get('final_f1', 0):.4f}")
                logger.info(f"   - Total comm cost: {comm.get('total_comm_cost', 0):.1f} MB")
                logger.info(f"   - Action dist: E={actions.get('E_pct', 0):.1f}% P={actions.get('P_pct', 0):.1f}% W={actions.get('W_pct', 0):.1f}%")

        except json.JSONDecodeError as e:
            logger.error(f"❌ INVALID JSON in {filename}: {e}")
            all_ok = False
        except Exception as e:
            logger.error(f"❌ ERROR reading {filename}: {e}")
            all_ok = False

    # Check optional directories
    vehicles_dir = run_dir / "vehicles_detail"
    if vehicles_dir.exists():
        vehicle_files = list(vehicles_dir.glob("*.json"))
        logger.info(f"✅ vehicles_detail/: {len(vehicle_files)} files")
    else:
        logger.info("ℹ️  vehicles_detail/: Not created yet (saved periodically)")

    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob("*.json"))
        logger.info(f"✅ checkpoints/: {len(checkpoint_files)} files")
    else:
        logger.info("ℹ️  checkpoints/: Not created yet (saved periodically)")

    logger.info(f"\n{'='*60}")
    if all_ok:
        logger.info("✅ ALL TRACKING FILES VERIFIED!")
        logger.info(f"{'='*60}")
        logger.info("\n🎉 Tracking is working correctly!")
        logger.info("\nNext steps:")
        logger.info("1. Run full training:")
        logger.info("   python train_dfl.py +experiment=dfl_training rl_checkpoint=outputs/rl/checkpoints/final_model.pt run_name=dfl_full")
        logger.info("\n2. Analyze results:")
        logger.info("   python analyze_dfl_metrics.py --run outputs/dfl_metrics/dfl_full")
        logger.info("\n3. Generate plots:")
        logger.info("   python analyze_dfl_metrics.py --run outputs/dfl_metrics/dfl_full --plot")
    else:
        logger.error("❌ TRACKING VERIFICATION FAILED!")
        logger.error(f"{'='*60}")

    return all_ok


if __name__ == "__main__":
    import sys

    logger.info("="*60)
    logger.info("DFL TRACKING QUICK TEST")
    logger.info("="*60)
    logger.info("\nThis script verifies that DFL tracking was integrated correctly.")
    logger.info("\nTo run test training (3 episodes):")
    logger.info("  python train_dfl.py +experiment=dfl_training run_name=tracking_test rl.training.num_episodes=3")
    logger.info("\nThen run this script to verify the output:")
    logger.info("  python test_tracking_quick.py")
    logger.info("")

    # Look for the most recent tracking_test run
    metrics_dir = Path("outputs/dfl_metrics")
    if not metrics_dir.exists():
        logger.error("❌ No metrics directory found!")
        logger.error("   Please run training first with tracking enabled.")
        sys.exit(1)

    # Find tracking_test runs
    run_dirs = sorted(metrics_dir.glob("tracking_test*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        logger.error("❌ No tracking_test run found!")
        logger.error("   Please run: python train_dfl.py +experiment=dfl_training run_name=tracking_test rl.training.num_episodes=3")
        sys.exit(1)

    run_dir = run_dirs[0]
    success = verify_tracking_output(run_dir, expected_episodes=3)

    sys.exit(0 if success else 1)
