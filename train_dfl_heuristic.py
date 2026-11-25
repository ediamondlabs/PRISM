"""
Train DFL with heuristic baselines (no RL).

Replaces the PPO actor's WHO/WHAT decisions with fixed-rule heuristic
policies while keeping the DFL infrastructure identical: same LSTM
autoencoder, same federated protocol, same aggregation, same data.

This script produces directly comparable results to train_dfl.py for
benchmarking whether the RL policy adds genuine value.

Usage:
    python train_dfl_heuristic.py +experiment=dfl_heuristic
    python train_dfl_heuristic.py +experiment=dfl_heuristic heuristic.policy_name=periodic heuristic.period=200
    python train_dfl_heuristic.py +experiment=dfl_heuristic heuristic.policy_name=threshold heuristic.loss_threshold=0.02
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import pandas as pd

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from src.baselines import create_heuristic_policy
from src.data import V2VDataLoader
from src.data.trajectory_window import create_preprocessed_dataset
from src.rl import RewardFunction
from src.rl.env_wrapper import RLV2VWrapper
from src.simulation import V2VEnvironment
from src.utils import get_device, set_seed, setup_logger
from src.utils.dfl_tracker import DFLComprehensiveTracker
from train_dfl import (
    create_dfl_protocols,
    create_vehicle_data_loaders,
    simulate_dfl_communication,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anomaly evaluation helpers (from scripts/analysis/evaluate_anomaly_detection.py)
# Reimplemented here to avoid import path issues with scripts/ directory.
# ---------------------------------------------------------------------------

ANOMALY_TYPES = ["position_spoofing", "speed_manipulation", "sensor_noise"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


def _apply_anomaly(
    window: np.ndarray, atype: str, difficulty: str, rng: np.random.RandomState
) -> np.ndarray:
    """Apply a single anomaly type at given difficulty."""
    w = window.copy()
    seq_len, n_feat = w.shape
    scale = {"easy": 1.0, "medium": 0.45, "hard": 0.18}[difficulty]

    if atype == "position_spoofing":
        start = rng.randint(0, seq_len // 2)
        length = rng.randint(seq_len // 4, seq_len // 2)
        offset = rng.uniform(3.0, 8.0) * rng.choice([-1, 1]) * scale
        w[start : start + length, :3] += offset
    elif atype == "speed_manipulation":
        start = rng.randint(0, seq_len - 10)
        spike = rng.uniform(4.0, 10.0) * rng.choice([-1, 1]) * scale
        w[start : start + 5, 3:6] += spike
        w[start : start + 5, 2] += abs(spike) * 0.5
    elif atype == "sensor_noise":
        noise_std = rng.uniform(3.0, 7.0) * scale
        w[:, 6:] += rng.normal(0, noise_std, (seq_len, n_feat - 6))

    return w


def inject_anomalies(
    data: np.ndarray, anomaly_ratio: float = 0.3, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject synthetic V2V anomalies into trajectory windows."""
    rng = np.random.RandomState(seed)
    n = len(data)
    n_anomaly = int(n * anomaly_ratio)
    anom_idx = rng.choice(n, n_anomaly, replace=False)

    mixed = data.copy()
    labels = np.zeros(n, dtype=np.int32)
    types = np.array(["normal"] * n, dtype=object)

    for i, idx in enumerate(anom_idx):
        atype = ANOMALY_TYPES[i % len(ANOMALY_TYPES)]
        difficulty = DIFFICULTY_LEVELS[i % len(DIFFICULTY_LEVELS)]
        mixed[idx] = _apply_anomaly(mixed[idx], atype, difficulty, rng)
        labels[idx] = 1
        types[idx] = f"{atype}_{difficulty}"

    return mixed, labels, types


@torch.no_grad()
def compute_reconstruction_errors(
    model: nn.Module, data: np.ndarray, batch_size: int = 256, device: str = "cpu"
) -> np.ndarray:
    """Run inference and return per-sample MSE reconstruction errors."""
    model.eval()
    model.to(device)
    errors = []
    for i in range(0, len(data), batch_size):
        batch = torch.tensor(data[i : i + batch_size], dtype=torch.float32).to(device)
        recon, _ = model(batch)
        mse = ((batch - recon) ** 2).mean(dim=(1, 2))
        errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


def evaluate_anomaly_detection(
    errors_mixed: np.ndarray,
    errors_normal: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Compute anomaly detection metrics using Youden's J threshold."""
    try:
        fpr, tpr, thresholds_roc = roc_curve(labels, errors_mixed)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        threshold = float(thresholds_roc[best_idx])
    except ValueError:
        threshold = float(np.percentile(errors_normal, 95))

    predictions = (errors_mixed >= threshold).astype(int)

    try:
        auroc = float(roc_auc_score(labels, errors_mixed))
    except ValueError:
        auroc = 0.5

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auroc": auroc,
        "threshold": threshold,
    }


def prepare_test_data(config: DictConfig) -> dict:
    """Load and prepare test data with injected anomalies for periodic evaluation.

    Uses the same data pipeline as the evaluation script: loads raw CSV,
    creates sliding windows, standardizes, splits, then injects anomalies.
    """
    features = [
        "velocity_x", "velocity_y", "speed",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z", "yaw",
    ]

    csv_path = Path(config.paths.data_dir) / "v2v_dataset_20251017_124247.csv"
    if not csv_path.exists():
        logger.warning(f"Dataset not found at {csv_path}, skipping time-to-detection")
        return {}

    df = pd.read_csv(csv_path)
    vehicle_ids = sorted(df["vehicle_id"].unique())[:config.data.num_vehicles]
    df = df[df["vehicle_id"].isin(vehicle_ids)]

    window_size = config.data.window_size
    stride = config.data.get("stride", 10)

    window_list = []
    for vid in vehicle_ids:
        vdf = df[df["vehicle_id"] == vid].sort_values("timestamp")
        vals = vdf[features].values
        for start in range(0, len(vals) - window_size + 1, stride):
            window_list.append(vals[start : start + window_size])

    windows = np.array(window_list, dtype=np.float64)

    # Train/test split (same as evaluate_anomaly_detection.py)
    n = len(windows)
    idx = np.random.RandomState(42).permutation(n)
    t1, t2 = int(0.7 * n), int(0.85 * n)
    train_idx, test_idx = idx[:t1], idx[t2:]

    scaler = StandardScaler()
    shape = windows.shape
    train_flat = windows[train_idx].reshape(-1, shape[-1])
    scaler.fit(train_flat)
    windows = scaler.transform(windows.reshape(-1, shape[-1])).reshape(shape)

    test_data = windows[test_idx].astype(np.float32)
    train_data = windows[train_idx].astype(np.float32)

    # Inject anomalies into test set
    mixed_data, labels, anom_types = inject_anomalies(test_data, anomaly_ratio=0.3, seed=42)

    logger.info(
        f"Prepared test set: {len(test_data)} samples, "
        f"{labels.sum()} anomalies, {len(labels) - labels.sum()} normal"
    )

    return {
        "test_normal": test_data,
        "test_mixed": mixed_data,
        "labels": labels,
        "anom_types": anom_types,
        "train_data": train_data,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    """Main heuristic DFL training loop."""
    logger = setup_logger(__name__)

    policy_name = config.heuristic.policy_name
    logger.info(f"Starting heuristic DFL training with policy: {policy_name}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Seed and device
    set_seed(config.seed)
    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    # Output directories
    output_dir = Path(config.paths.output_dir) / f"heuristic_{policy_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # MLflow setup
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(f"file://{Path(config.paths.output_dir).absolute()}/mlruns")
        mlflow.set_experiment("heuristic_baselines")
        mlflow.start_run(run_name=f"heuristic_{policy_name}")
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
    else:
        logger.warning("MLflow not available - skipping experiment tracking")

    # Load data
    logger.info("Loading dataset...")
    data_loader = V2VDataLoader(config)
    data_loader.load_dataset()

    # Generate preprocessed data if not already present
    preprocessed_path = Path(config.simulation.data.preprocessed_path)
    npz_file = preprocessed_path / "preprocessed_data.npz"
    if not npz_file.exists():
        logger.info(f"Preprocessed data not found — generating now at {npz_file}")
        create_preprocessed_dataset(config, data_loader.df, save_dir=preprocessed_path)

    # Create V2V environment
    logger.info("Creating V2V environment...")
    env = V2VEnvironment(config)

    # Filter data to environment vehicles
    logger.info(f"Filtering dataset to {len(env.vehicle_ids)} vehicles...")
    full_df = data_loader.df
    env_vehicle_ids = set(env.vehicle_ids)
    data_loader.df = full_df[full_df["vehicle_id"].isin(env_vehicle_ids)]
    data_loader.selected_vehicles = list(env.vehicle_ids)

    logger.info(f"Filtered to {len(data_loader.df)} records from {len(env.vehicle_ids)} vehicles")
    features_df = data_loader.extract_features()

    # Create vehicle-specific data loaders
    logger.info("Creating vehicle data loaders...")
    vehicle_data_loaders = create_vehicle_data_loaders(
        features_df, list(env.vehicle_ids), config
    )

    # Create DFL protocols (identical to train_dfl.py)
    logger.info("Creating DFL protocols...")
    dfl_protocols = create_dfl_protocols(env, vehicle_data_loaders, device, config)

    # Create reward function (kept for logging comparability)
    reward_fn = RewardFunction(
        immediate_weight=config.rl.reward.immediate_weight,
        global_weight=config.rl.reward.global_weight,
        perf_scale=config.rl.reward.perf_scale,
        comm_cost_scale=config.rl.reward.comm_cost_scale,
        staleness_penalty_weight=config.rl.reward.get("staleness_penalty_weight", 0.1),
    )

    # Create RL environment wrapper (for env stepping and reward computation)
    logger.info("Creating RL environment wrapper...")
    rl_env = RLV2VWrapper(
        env=env,
        reward_fn=reward_fn,
        use_neighbor_encoder=config.get("use_neighbor_encoder", False),
        device=device,
        dfl_protocols=dfl_protocols,
        use_dfl_features=config.get("use_dfl_features", True),
    )

    # Create heuristic policy (replaces PPO actor)
    k_neighbors = config.rl.actor.get("k_neighbors", config.rl.k_neighbors)
    max_neighbors = config.simulation.max_neighbors

    policy_kwargs = {
        "k_neighbors": k_neighbors,
        "max_neighbors": max_neighbors,
        "seed": config.seed,
    }

    # Add policy-specific parameters
    if policy_name == "fixed_switch":
        policy_kwargs["switch_episode"] = config.heuristic.switch_episode
    elif policy_name == "periodic":
        policy_kwargs["period"] = config.heuristic.period
    elif policy_name == "threshold":
        policy_kwargs["loss_threshold"] = config.heuristic.loss_threshold
    elif policy_name == "fixed_7030":
        policy_kwargs["weights_ratio"] = config.heuristic.weights_ratio

    policy = create_heuristic_policy(policy_name, **policy_kwargs)
    logger.info(f"Created heuristic policy: {policy.name}")

    # Initialize DFL tracker
    tracker = DFLComprehensiveTracker(
        output_dir=str(output_dir / "metrics"),
        run_name=f"heuristic_{policy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        save_interval=50,
        vehicle_detail_interval=100,
    )
    tracker.set_metadata(
        config=OmegaConf.to_container(config, resolve=True),
        model_info={
            "policy": policy_name,
            "policy_class": policy.__class__.__name__,
            "autoencoder_params": sum(p.numel() for p in env.lstm_model.parameters()),
        },
        dataset_info={
            "num_vehicles": config.data.num_vehicles,
            "dataset_path": config.data.get("dataset_path", "N/A"),
            "window_size": config.data.window_size,
        },
    )
    logger.info(f"DFL tracking initialized: {tracker.output_dir}")

    # Prepare test data for time-to-detection evaluation
    logger.info("Preparing anomaly test data for periodic evaluation...")
    test_data = prepare_test_data(config)
    eval_interval = config.heuristic.get("eval_interval", 100)
    has_test_data = bool(test_data)

    # Track time-to-detection: episode when each anomalous sample is first detected
    detection_history: list[dict] = []
    first_detected: np.ndarray = np.empty(0, dtype=np.int64)
    if has_test_data:
        first_detected = np.full(len(test_data["labels"]), -1, dtype=np.int64)

    # Training loop metrics
    training_curve = []
    cumulative_bandwidth = 0.0
    what_action_counts = {"weights": 0, "embeddings": 0}

    # --- Main training loop ---
    logger.info("Starting heuristic DFL training loop...")
    num_episodes = config.rl.training.get("num_episodes", 1000)
    max_timesteps = config.simulation.max_episode_timesteps

    for episode in range(num_episodes):
        tracker.start_episode(episode)
        episode_start_time = time.time()

        # Reset environment (advances simulation time, updates neighbor graph)
        rl_env.reset()

        episode_reward = 0.0
        episode_comm_cost = 0.0
        episode_train_losses = []
        episode_lstm_losses = []
        episode_what_actions = []

        # === LOCAL TRAINING PHASE (identical to train_dfl.py lines 451-491) ===
        avg_lstm_loss = 0.0
        if config.get("train_layer1", False):
            lstm_epochs = config.federated.get("lstm_local_epochs", 1)
            total_lstm_loss = 0.0
            num_trained = 0

            for vid in env.vehicle_ids:
                loss = dfl_protocols[vid].train_local_model(num_epochs=lstm_epochs)
                if loss > 0:
                    total_lstm_loss += loss
                    num_trained += 1

            avg_lstm_loss = total_lstm_loss / num_trained if num_trained > 0 else 0.0
            episode_lstm_losses.append(avg_lstm_loss)

            if (episode + 1) % config.rl.training.log_interval == 0:
                logger.info(
                    f"Episode {episode+1}: LSTM training complete. "
                    f"Avg loss: {avg_lstm_loss:.6f}"
                )

            # Federated averaging and synchronization
            with torch.no_grad():
                for param_name, param in env.lstm_model.named_parameters():
                    avg_param = torch.zeros_like(param.cpu())
                    for vid in env.vehicle_ids:
                        vehicle_param = dict(
                            dfl_protocols[vid].local_model.named_parameters()
                        )[param_name]
                        avg_param += vehicle_param.cpu()
                    avg_param /= len(env.vehicle_ids)
                    param.copy_(avg_param.to(device))

                for vid in env.vehicle_ids:
                    for local_param, global_param in zip(
                        dfl_protocols[vid].local_model.parameters(),
                        env.lstm_model.parameters(),
                    ):
                        local_param.copy_(global_param.cpu())

        # === COMMUNICATION PHASE (heuristic replaces RL actor) ===
        for timestep in range(max_timesteps):
            actions = {}

            for vid in env.vehicle_ids:
                neighbors_list = sorted(list(env.vehicles[vid].neighbors))
                action = policy.select_action(
                    vehicle_id=vid,
                    neighbors=neighbors_list,
                    episode=episode,
                    timestep=timestep,
                    current_loss=avg_lstm_loss,
                )
                actions[vid] = action
                episode_what_actions.append(action["what_action"])

            # Execute DFL communication (identical to train_dfl.py)
            dfl_info = simulate_dfl_communication(env, actions, dfl_protocols, config)

            # Log vehicle actions and aggregations
            for vid in env.vehicle_ids:
                tracker.log_vehicle_action(
                    vehicle_id=vid,
                    action=actions[vid],
                    communication_success=dfl_info["comm_results"].get(vid, False),
                )

                protocol = dfl_protocols[vid]
                if (
                    hasattr(protocol.aggregator, "received_data")
                    and protocol.aggregator.received_data
                ):
                    num_neighbors = len(
                        protocol.aggregator.received_data.get(vid, {})
                    )
                    tracker.log_aggregation(
                        vid,
                        {
                            "num_neighbors": num_neighbors,
                            "local_weight": getattr(
                                protocol.aggregator, "alpha", 0.7
                            ),
                            "total_alpha": 1.0
                            - getattr(protocol.aggregator, "alpha", 0.7),
                            "num_stale": 0,
                        },
                    )

            # Step environment (computes rewards for logging comparability)
            _, rewards, done, info = rl_env.step(actions)

            # Track metrics
            episode_reward += sum(rewards.values())
            episode_comm_cost += dfl_info["total_data_sent_mb"]
            episode_train_losses.append(info["train_loss"])

            if done:
                break

        # Update bandwidth and action counts
        cumulative_bandwidth += episode_comm_cost
        for wa in episode_what_actions:
            if wa == 1:
                what_action_counts["weights"] += 1
            else:
                what_action_counts["embeddings"] += 1

        total_actions = what_action_counts["weights"] + what_action_counts["embeddings"]
        weights_ratio = (
            what_action_counts["weights"] / total_actions if total_actions > 0 else 0.0
        )

        # End episode tracking
        avg_lstm_loss_ep = np.mean(episode_lstm_losses) if episode_lstm_losses else None
        avg_train_loss = (
            np.mean(episode_train_losses) if episode_train_losses else 0.0
        )
        tracker.end_episode(
            final_reward=episode_reward,
            final_train_loss=avg_train_loss,
            comm_cost_mb=episode_comm_cost,
            rl_stats=None,
            duration_seconds=time.time() - episode_start_time,
            lstm_loss=avg_lstm_loss_ep,
        )

        # Training curve entry
        training_curve.append(
            {
                "episode": episode,
                "avg_loss": float(avg_train_loss),
                "lstm_loss": float(avg_lstm_loss) if avg_lstm_loss > 0 else None,
                "bandwidth_cumulative_mb": float(cumulative_bandwidth),
                "episode_bandwidth_mb": float(episode_comm_cost),
                "weights_ratio": float(weights_ratio),
                "episode_reward": float(episode_reward),
            }
        )

        # MLflow logging
        if MLFLOW_AVAILABLE and (episode + 1) % config.rl.training.log_interval == 0:
            metrics_dict = {
                "episode_reward": episode_reward,
                "episode_comm_cost_mb": episode_comm_cost,
                "episode_train_loss": avg_train_loss,
                "weights_ratio": weights_ratio,
                "cumulative_bandwidth_mb": cumulative_bandwidth,
            }
            if avg_lstm_loss_ep is not None:
                metrics_dict["lstm_loss"] = float(avg_lstm_loss_ep)
            mlflow.log_metrics(metrics_dict, step=episode)

        # Logging
        if (episode + 1) % config.rl.training.log_interval == 0:
            log_msg = (
                f"Episode {episode+1}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Comm Cost: {episode_comm_cost:.2f} MB | "
                f"W/E Ratio: {weights_ratio:.2f}"
            )
            if avg_lstm_loss_ep is not None:
                log_msg += f" | LSTM Loss: {float(avg_lstm_loss_ep):.4f}"
            logger.info(log_msg)

        # === TIME-TO-DETECTION EVALUATION (every eval_interval episodes) ===
        if has_test_data and (episode + 1) % eval_interval == 0:
            logger.info(f"Episode {episode+1}: Running anomaly detection evaluation...")

            eval_device = str(device)
            errors_mixed = compute_reconstruction_errors(
                env.lstm_model, test_data["test_mixed"], device=eval_device
            )
            errors_normal = compute_reconstruction_errors(
                env.lstm_model, test_data["test_normal"], device=eval_device
            )

            # Find optimal threshold from normal data errors
            try:
                fpr, tpr, thresholds_roc = roc_curve(
                    test_data["labels"], errors_mixed
                )
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                threshold = float(thresholds_roc[best_idx])
            except ValueError:
                threshold = float(np.percentile(errors_normal, 95))

            detected = errors_mixed >= threshold
            anomaly_mask = test_data["labels"] == 1

            # Update first-detected for anomalous samples only (avoids
            # false-positive contamination of normal-sample slots)
            newly_detected = detected & anomaly_mask & (first_detected == -1)
            first_detected[newly_detected] = episode + 1

            # Compute detection metrics for anomalous samples
            anomalous_detected = first_detected[anomaly_mask]
            n_detected = (anomalous_detected > 0).sum()
            n_anomalies = anomaly_mask.sum()
            detection_rate = n_detected / n_anomalies if n_anomalies > 0 else 0.0

            delays = anomalous_detected[anomalous_detected > 0]
            mean_delay = float(np.mean(delays)) if len(delays) > 0 else float("inf")
            median_delay = float(np.median(delays)) if len(delays) > 0 else float("inf")

            eval_metrics = evaluate_anomaly_detection(
                errors_mixed, errors_normal, test_data["labels"]
            )

            detection_snapshot = {
                "episode": episode + 1,
                "detection_rate": float(detection_rate),
                "n_detected": int(n_detected),
                "n_anomalies": int(n_anomalies),
                "mean_first_detection_episode": mean_delay,
                "median_first_detection_episode": median_delay,
                "threshold": threshold,
                **eval_metrics,
            }
            detection_history.append(detection_snapshot)

            logger.info(
                f"  Detection rate: {detection_rate:.3f} ({n_detected}/{n_anomalies}) | "
                f"F1: {eval_metrics['f1']:.4f} | AUROC: {eval_metrics['auroc']:.4f} | "
                f"Mean delay: {mean_delay:.0f} episodes"
            )

            if MLFLOW_AVAILABLE:
                mlflow.log_metrics(
                    {
                        "detection_rate": detection_rate,
                        "detection_f1": eval_metrics["f1"],
                        "detection_auroc": eval_metrics["auroc"],
                    },
                    step=episode,
                )

        # Save checkpoint
        if (episode + 1) % config.rl.training.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"heuristic_checkpoint_{episode+1}.pt"
            torch.save(
                {
                    "episode": episode,
                    "policy": policy_name,
                    "global_model_state_dict": env.lstm_model.state_dict(),
                    "vehicle_models": {
                        vid: dfl_protocols[vid].local_model.state_dict()
                        for vid in env.vehicle_ids
                    },
                    "config": OmegaConf.to_container(config),
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # --- Post-training ---

    # Final anomaly evaluation
    final_eval = {}
    if has_test_data:
        logger.info("Running final anomaly detection evaluation...")
        eval_device = str(device)
        errors_mixed = compute_reconstruction_errors(
            env.lstm_model, test_data["test_mixed"], device=eval_device
        )
        errors_normal = compute_reconstruction_errors(
            env.lstm_model, test_data["test_normal"], device=eval_device
        )
        final_eval = evaluate_anomaly_detection(
            errors_mixed, errors_normal, test_data["labels"]
        )
        logger.info(
            f"Final metrics: F1={final_eval['f1']:.4f} | "
            f"AUROC={final_eval['auroc']:.4f} | "
            f"Precision={final_eval['precision']:.4f} | "
            f"Recall={final_eval['recall']:.4f}"
        )

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "heuristic_final.pt"
    torch.save(
        {
            "episode": num_episodes,
            "policy": policy_name,
            "global_model_state_dict": env.lstm_model.state_dict(),
            "vehicle_models": {
                vid: dfl_protocols[vid].local_model.state_dict()
                for vid in env.vehicle_ids
            },
            "config": OmegaConf.to_container(config),
        },
        final_checkpoint_path,
    )
    logger.info(f"Saved final checkpoint: {final_checkpoint_path}")

    # Save metrics.json
    final_lstm_loss = training_curve[-1]["lstm_loss"] if training_curve else None
    metrics = {
        "policy": policy_name,
        "policy_class": policy.__class__.__name__,
        "num_episodes": num_episodes,
        "final_train_loss": float(training_curve[-1]["avg_loss"]) if training_curve else 0.0,
        "final_lstm_loss": float(final_lstm_loss) if final_lstm_loss is not None else None,
        "total_bandwidth_mb": float(cumulative_bandwidth),
        "weights_ratio": float(weights_ratio),
        "what_action_counts": what_action_counts,
        **final_eval,
    }

    # Add convergence round (first episode where LSTM loss < 0.02)
    convergence_round = None
    for entry in training_curve:
        if entry["lstm_loss"] is not None and entry["lstm_loss"] < 0.02:
            convergence_round = entry["episode"]
            break
    metrics["convergence_round"] = convergence_round

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics: {metrics_path}")

    # Save training_curve.json
    curve_path = output_dir / "training_curve.json"
    with open(curve_path, "w") as f:
        json.dump(training_curve, f, indent=2)
    logger.info(f"Saved training curve: {curve_path}")

    # Save detection_timing.json
    if has_test_data:
        anomaly_mask = test_data["labels"] == 1
        anomalous_first_detected = first_detected[anomaly_mask]
        detection_results = []
        anomaly_indices = np.where(anomaly_mask)[0]

        for i, sample_idx in enumerate(anomaly_indices):
            det_episode = int(anomalous_first_detected[i])
            detection_results.append(
                {
                    "sample_idx": int(sample_idx),
                    "anomaly_type": str(test_data["anom_types"][sample_idx]),
                    "detected": det_episode > 0,
                    "first_detection_episode": det_episode if det_episode > 0 else None,
                }
            )

        delays = [
            r["first_detection_episode"]
            for r in detection_results
            if r["detected"]
        ]

        detection_timing = {
            "policy": policy_name,
            "eval_interval": eval_interval,
            "total_anomalies": int(anomaly_mask.sum()),
            "total_detected": len(delays),
            "detection_rate": len(delays) / int(anomaly_mask.sum()) if anomaly_mask.sum() > 0 else 0.0,
            "mean_detection_delay_episodes": float(np.mean(delays)) if delays else None,
            "median_detection_delay_episodes": float(np.median(delays)) if delays else None,
            "p95_detection_delay_episodes": float(np.percentile(delays, 95)) if delays else None,
            "detection_history": detection_history,
            "per_sample_results": detection_results,
        }

        timing_path = output_dir / "detection_timing.json"
        with open(timing_path, "w") as f:
            json.dump(detection_timing, f, indent=2)
        logger.info(f"Saved detection timing: {timing_path}")

    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(config))
    logger.info(f"Saved config: {config_path}")

    # Finalize tracking
    logger.info("Finalizing DFL metrics tracking...")
    tracker.finalize()
    logger.info(f"All DFL metrics saved to: {tracker.output_dir}")

    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(str(final_checkpoint_path))
            mlflow.log_artifact(str(metrics_path))
        except Exception as e:
            logger.warning(f"Failed to log artifact to MLflow: {e}")
        finally:
            mlflow.end_run()

    logger.info(f"Heuristic DFL training complete! Policy: {policy_name}")


if __name__ == "__main__":
    main()
