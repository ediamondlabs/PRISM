"""
Train DRL-based V2V system with Decentralized Federated Learning (DFL).

This script trains the complete 4-layer system:
- Layer 1: LSTM Autoencoder (pre-trained, frozen)
- Layer 2: Neighbor Encoder (optional, trainable)
- Layer 3: RL Agent (PPO with WHEN/WHO/WHAT decisions)
- Layer 4: DFL (asynchronous aggregation with staleness weighting)

Usage:
    python train_dfl.py                          # Default config
    python train_dfl.py +experiment=dfl_training # Use experiment config
    python train_dfl.py device=cuda              # Override device
"""

import logging
from pathlib import Path
from typing import Dict, List
import time
from datetime import datetime

import hydra
import numpy as np

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger = None  # Will be set up later
import torch
from omegaconf import DictConfig, OmegaConf

from src.data import V2VDataLoader
from src.federated import AsynchronousAggregator, DFLProtocol, RepresentationExtractor
from src.models import create_autoencoder_from_config
from src.rl import PPO, PPOActor, CentralizedCritic, MultiAgentRolloutBuffer, RewardFunction
from src.rl.env_wrapper import RLV2VWrapper
from src.simulation import V2VEnvironment
from src.utils import get_device, set_seed, setup_logger
from src.utils.dfl_tracker import DFLComprehensiveTracker

logger = logging.getLogger(__name__)


def create_dfl_protocols(
    env: V2VEnvironment,
    data_loaders: Dict[int, torch.utils.data.DataLoader],
    device: torch.device,
    config: DictConfig,
) -> Dict[int, DFLProtocol]:
    """Create DFL protocol for each vehicle.

    Args:
        env: V2V environment
        data_loaders: Data loaders per vehicle
        device: Torch device
        config: Hydra configuration

    Returns:
        Dictionary mapping vehicle_id to DFLProtocol
    """
    dfl_protocols = {}

    train_model = config.get("train_layer1", False)

    # Get training device for LSTMs (separate from RL device)
    lstm_device = get_device(config.get("lstm_device", "auto"))

    for vehicle_id in env.vehicle_ids:
        # Create local model (copy of autoencoder)
        # Keep models on CPU to save GPU memory
        # They will be moved to training_device one at a time during training
        local_model = create_autoencoder_from_config(config, device=torch.device("cpu"))

        if train_model:
            # Copy state from env model (which is fresh/untrained)
            local_model.load_state_dict(env.lstm_model.cpu().state_dict())
            env.lstm_model.to(device)  # Move back to original device
        else:
            # Copy state from pretrained env model
            local_model.load_state_dict(env.lstm_model.cpu().state_dict())
            env.lstm_model.to(device)  # Move back to original device

        # Create representation extractor
        repr_extractor = RepresentationExtractor(
            embedding_dim=config.model.architecture.embedding_dim,
            num_prototypes=config.federated.num_prototypes,
            device=device,
        )

        # Create aggregator
        aggregator = AsynchronousAggregator(
            alpha=config.federated.alpha,
            staleness_decay=config.federated.staleness_decay,
            max_staleness_seconds=config.federated.max_staleness_seconds,
        )

        # Create DFL protocol
        model_lr = config.federated.get("lstm_lr", 0.001)

        dfl_protocol = DFLProtocol(
            vehicle_id=vehicle_id,
            local_model=local_model,
            data_loader=data_loaders[vehicle_id],
            representation_extractor=repr_extractor,
            aggregator=aggregator,
            device=device,
            train_model=train_model,
            model_lr=model_lr,
            training_device=lstm_device,
        )

        dfl_protocols[vehicle_id] = dfl_protocol

    logger.info(f"Created DFL protocols for {len(dfl_protocols)} vehicles")
    return dfl_protocols


def create_vehicle_data_loaders(
    features_df, vehicle_ids: List[int], config: DictConfig
) -> Dict[int, torch.utils.data.DataLoader]:
    """Create data loaders for each vehicle (private data).

    Args:
        features_df: Full features dataframe
        vehicle_ids: List of vehicle IDs
        config: Hydra configuration

    Returns:
        Dictionary mapping vehicle_id to DataLoader
    """
    data_loaders = {}
    skipped_vehicles = []

    for vehicle_id in vehicle_ids:
        # Get vehicle-specific data
        vehicle_data = features_df[features_df["vehicle_id"] == vehicle_id]

        # Check if vehicle has sufficient data
        num_timesteps = len(vehicle_data)
        window_size = config.data.window_size

        if num_timesteps < window_size:
            logger.warning(
                f"Vehicle {vehicle_id} has only {num_timesteps} timesteps, "
                f"less than window_size={window_size}. Creating empty data loader."
            )
            # Create empty data loader (shuffle=False to avoid RandomSampler error)
            empty_tensor = torch.FloatTensor(0, window_size, len(config.data.features))
            dataset = torch.utils.data.TensorDataset(empty_tensor)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=config.data.batch_size, shuffle=False
            )
            data_loaders[vehicle_id] = loader
            skipped_vehicles.append(vehicle_id)
            continue

        # Create windows for this vehicle
        from src.data.trajectory_window import create_sliding_windows

        windows = create_sliding_windows(
            vehicle_data[config.data.features].values,
            window_size=config.data.window_size,
            stride=config.data.stride,
        )

        # Convert to tensors
        windows_tensor = torch.FloatTensor(windows)

        # Create dataset and loader
        dataset = torch.utils.data.TensorDataset(windows_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=config.data.batch_size, shuffle=True
        )

        data_loaders[vehicle_id] = loader

    valid_vehicles = len(data_loaders) - len(skipped_vehicles)
    logger.info(f"Created data loaders for {len(data_loaders)} vehicles ({valid_vehicles} valid, {len(skipped_vehicles)} skipped)")
    if skipped_vehicles:
        logger.warning(f"Skipped vehicles with insufficient data: {skipped_vehicles}")

    return data_loaders


def simulate_dfl_communication(
    env: V2VEnvironment,
    actions: Dict[int, Dict],
    dfl_protocols: Dict[int, DFLProtocol],
    config: DictConfig = None,
) -> Dict:
    """Simulate DFL communication between vehicles.

    Args:
        env: V2V environment
        actions: RL actions per vehicle
        dfl_protocols: DFL protocols per vehicle
        config: Configuration (for lstm_local_epochs)

    Returns:
        Communication info dictionary
    """
    comm_results = {}
    total_data_sent = 0.0

    # NOTE: LSTM training happens once per episode (outside this function)
    # This function only handles communication (WHO/WHAT decisions)
    # WHEN removed: all vehicles communicate every timestep (synchronized)

    # Step 1: Vehicles send representations (all vehicles communicate)
    messages = {}  # Store messages to deliver

    for vid in env.vehicle_ids:
        action = actions.get(vid, {})
        
        # Execute communication (no WHEN check - always communicate)
        who_action = action.get("who_action", [])  # List of neighbor indices
        what_action = action.get("what_action", 0)  # 0=E, 1=W (default: E, was 0=W, 1=E, 2=P)

        # Map WHAT action to representation type (removed Prototypes)
        repr_types = ["E", "W"]  # 0=Embeddings, 1=Weights (removed P)
        repr_type = repr_types[what_action]

        # Get actual neighbor IDs from indices
        vehicle = env.vehicles[vid]
        neighbors_list = sorted(list(vehicle.neighbors))

        # Filter who_action to valid neighbor indices
        valid_who_action = [
            neighbors_list[idx] for idx in who_action if idx < len(neighbors_list)
        ]

        # Execute communication
        success, comm_info = dfl_protocols[vid].execute_communication_action(
            who_action=valid_who_action,
            what_action=repr_type,
            neighbors=vehicle.neighbors,
        )

        comm_results[vid] = success

        if success:
            # Store messages for delivery
            for recipient_id in comm_info["recipients"]:
                if recipient_id not in messages:
                    messages[recipient_id] = []
                messages[recipient_id].append((vid, comm_info["data"]))

            total_data_sent += comm_info["size_mb"] * len(comm_info["recipients"])

    # Step 2: Deliver messages
    for recipient_id, incoming_messages in messages.items():
        for sender_id, repr_package in incoming_messages:
            dfl_protocols[recipient_id].receive_representation(sender_id, repr_package)

    # Step 3: Aggregate (if pending updates)
    for vid in env.vehicle_ids:
        if dfl_protocols[vid].has_pending_updates():
            dfl_protocols[vid].aggregate_received_data()

    return {
        "comm_results": comm_results,
        "total_data_sent_mb": total_data_sent,
        "num_messages": sum(len(msgs) for msgs in messages.values()),
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    """Main DFL training loop."""
    # Setup
    logger = setup_logger(__name__)
    logger.info("Starting DFL training")
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Set seed
    set_seed(config.seed)
    device = get_device(config.device)
    logger.info(f"Using device: {device}")

    # Create output directories
    output_dir = Path(config.paths.output_dir) / "dfl"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # MLflow setup (optional)
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(f"file://{Path(config.paths.output_dir).absolute()}/mlruns")
        mlflow.set_experiment("dfl_training")
        mlflow.start_run(run_name=config.get("run_name", "dfl_training"))
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
    else:
        logger.warning("MLflow not available - skipping experiment tracking")

    # Load data
    logger.info("Loading dataset...")
    data_loader = V2VDataLoader(config)
    data_loader.load_dataset()

    # Create V2V environment first to get vehicle IDs from preprocessed data
    logger.info("Creating V2V environment...")
    env = V2VEnvironment(config)

    # Filter data_loader to only include vehicles from the environment
    # This ensures both use the same vehicle set
    logger.info(f"Filtering dataset to {len(env.vehicle_ids)} vehicles from preprocessed data...")
    full_df = data_loader.df
    env_vehicle_ids = set(env.vehicle_ids)

    # Filter to only vehicles in env
    data_loader.df = full_df[full_df["vehicle_id"].isin(env_vehicle_ids)]
    data_loader.selected_vehicles = list(env.vehicle_ids)  # Fixed: use selected_vehicles not selected_vehicle_ids

    logger.info(f"Filtered to {len(data_loader.df)} records from {len(env.vehicle_ids)} vehicles")
    features_df = data_loader.extract_features()

    # Create vehicle-specific data loaders
    logger.info("Creating vehicle data loaders...")
    vehicle_data_loaders = create_vehicle_data_loaders(
        features_df, list(env.vehicle_ids), config
    )

    # Create DFL protocols
    logger.info("Creating DFL protocols...")
    dfl_protocols = create_dfl_protocols(env, vehicle_data_loaders, device, config)

    # Create reward function
    reward_fn = RewardFunction(
            immediate_weight=config.rl.reward.immediate_weight,
            global_weight=config.rl.reward.global_weight,
            perf_scale=config.rl.reward.perf_scale,
            comm_cost_scale=config.rl.reward.comm_cost_scale,
            staleness_penalty_weight=config.rl.reward.get("staleness_penalty_weight", 0.1),
    )

    # Create RL environment wrapper
    logger.info("Creating RL environment wrapper...")
    rl_env = RLV2VWrapper(
            env=env,
            reward_fn=reward_fn,
            use_neighbor_encoder=config.get("use_neighbor_encoder", False),
            device=device,
            dfl_protocols=dfl_protocols,
            use_dfl_features=config.get("use_dfl_features", True),  # Use config value
    )

    # Create RL agent
    logger.info("Creating RL agent...")
    # State dim: 106 (base) or 110 (with DFL features)
    state_dim = config.rl.actor.get("state_dim", 110 if config.get("use_dfl_features", True) else 106)

    actor = PPOActor(
            state_dim=state_dim,
            shared_hidden_dim=config.rl.actor.shared_hidden_dim,
            when_hidden_dim=config.rl.actor.get("when_hidden_dim", 128),
            who_hidden_dim=config.rl.actor.who_hidden_dim,
            what_hidden_dim=config.rl.actor.what_hidden_dim,
            max_neighbors=config.simulation.max_neighbors,
            k_neighbors=config.rl.actor.get("k_neighbors", config.rl.k_neighbors),
            num_repr_types=config.rl.actor.get("num_repr_types", 2),
            dropout=config.rl.actor.dropout,
            use_when_head=config.get("use_when_head", True),
    ).to(device)

    critic = CentralizedCritic(
            state_dim=state_dim,
            num_vehicles=config.data.num_vehicles,
            hidden_dim=config.rl.critic.hidden_dim,
            num_layers=config.rl.critic.num_layers,
            dropout=config.rl.critic.dropout,
    ).to(device)

    ppo = PPO(
            actor=actor,
            critic=critic,
            lr_actor=config.rl.training.learning_rate,
            lr_critic=config.rl.training.learning_rate,
            gamma=config.rl.training.gamma,
            gae_lambda=config.rl.training.gae_lambda,
            clip_epsilon=config.rl.training.clip_epsilon,
            value_loss_coef=config.rl.training.value_loss_coef,
            entropy_coef=config.rl.training.entropy_coef,
            max_grad_norm=config.rl.training.max_grad_norm,
            device=device,
    )

    # Load RL checkpoint if provided
    if config.get("rl_checkpoint"):
            logger.info(f"Loading RL checkpoint: {config.rl_checkpoint}")
            checkpoint = torch.load(config.rl_checkpoint, map_location=device)
            actor.load_state_dict(checkpoint["actor_state_dict"])
            critic.load_state_dict(checkpoint["critic_state_dict"])
            logger.info("RL checkpoint loaded successfully")

    # Create rollout buffer (use actual vehicle IDs from environment)
    buffer = MultiAgentRolloutBuffer(
            num_vehicles=config.data.num_vehicles,
            buffer_size=config.rl.training.buffer_size,
            state_dim=state_dim,
            max_neighbors=config.simulation.max_neighbors,
            k_neighbors=config.rl.k_neighbors,
            vehicle_ids=list(env.vehicle_ids),  # Pass actual vehicle IDs
            device=device,  # Ensure buffer tensors are on the same device as the model
    )

    # Initialize comprehensive DFL tracker
    tracker = DFLComprehensiveTracker(
        output_dir="outputs/dfl_metrics",
        run_name=config.get("run_name", f"dfl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        save_interval=50,
        vehicle_detail_interval=100,
    )

    # Set metadata
    tracker.set_metadata(
        config=OmegaConf.to_container(config, resolve=True),
        model_info={
            "actor_params": sum(p.numel() for p in actor.parameters()),
            "critic_params": sum(p.numel() for p in critic.parameters()),
            "autoencoder_params": sum(p.numel() for p in env.lstm_model.parameters()),
        },
        dataset_info={
            "num_vehicles": config.data.num_vehicles,
            "dataset_path": config.data.get("dataset_path", "N/A"),
            "window_size": config.data.window_size,
        },
    )
    logger.info(f"DFL tracking initialized: {tracker.output_dir}")

    # Training loop
    logger.info("Starting DFL training loop...")
    num_episodes = config.rl.training.get("num_episodes", 1000)
    max_timesteps = config.simulation.max_episode_timesteps

    for episode in range(num_episodes):
            # Start tracking this episode
            tracker.start_episode(episode)
            episode_start_time = time.time()

            # Reset environment
            states, global_state = rl_env.reset()
            neighbor_masks = rl_env.get_neighbor_masks()

            episode_reward = 0.0
            episode_comm_cost = 0.0
            episode_train_losses = []
            episode_lstm_losses = []

            # LOCAL TRAINING PHASE: Train all LSTMs once per episode (DFL round)
            # This matches CoCo's approach: local training → communication optimization
            if config.get("train_layer1", False):
                logger.info(f"Episode {episode+1}: Training {len(env.vehicle_ids)} local LSTMs on GPU {config.get('lstm_device', 'cuda:0')}...")
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
                logger.info(f"Episode {episode+1}: LSTM training complete. Avg loss: {avg_lstm_loss:.6f}")

                # This ensures anomaly detection uses trained weights AND models don't diverge
                logger.info(f"Episode {episode+1}: Federated averaging and synchronization...")
                with torch.no_grad():
                    # Step 1: Average all trained model parameters into global model
                    for param_name, param in env.lstm_model.named_parameters():
                        # Collect parameters from all vehicles (all on CPU)
                        avg_param = torch.zeros_like(param.cpu())
                        for vid in env.vehicle_ids:
                            vehicle_param = dict(dfl_protocols[vid].local_model.named_parameters())[param_name]
                            avg_param += vehicle_param.cpu()
                        avg_param /= len(env.vehicle_ids)

                        # Update global model
                        param.copy_(avg_param.to(device))

                    # Step 2: Synchronize all local models back to global average
                    # This prevents model divergence (critical for federated learning!)
                    for vid in env.vehicle_ids:
                        for local_param, global_param in zip(
                            dfl_protocols[vid].local_model.parameters(),
                            env.lstm_model.parameters()
                        ):
                            local_param.copy_(global_param.cpu())

            # COMMUNICATION PHASE: RL decides WHEN/WHO/WHAT over 100 timesteps
            # Episode loop
            for timestep in range(max_timesteps):
                # Select actions
                actions = {}
                values = {}
                log_probs = {}

                for vid in env.vehicle_ids:
                    state_t = torch.from_numpy(states[vid]).unsqueeze(0).to(device)
                    mask_t = torch.from_numpy(neighbor_masks[vid]).unsqueeze(0).to(device)
                    global_t = torch.from_numpy(global_state).unsqueeze(0).to(device)

                    with torch.no_grad():
                        action_dict = actor.get_action(state_t, mask_t)
                        value = ppo.get_value(state_t, global_t)

                    # Convert to numpy (WHEN removed)
                    actions[vid] = {
                        "who_action": action_dict["who_action"].cpu().numpy()[0],
                        "what_action": int(action_dict["what_action"].cpu().numpy()[0]),
                    }

                    # Store log probs and values for later (WHEN removed)
                    log_probs[vid] = (
                        action_dict["who_log_prob"]
                        + action_dict["what_log_prob"]
                    )
                    values[vid] = value

                # Execute DFL communication
                dfl_info = simulate_dfl_communication(env, actions, dfl_protocols, config)

                # Log vehicle actions and aggregations
                for vid in env.vehicle_ids:
                    # Log vehicle action
                    tracker.log_vehicle_action(
                        vehicle_id=vid,
                        action=actions[vid],
                        communication_success=dfl_info["comm_results"].get(vid, False),
                    )

                    # Log aggregation if it happened
                    protocol = dfl_protocols[vid]
                    if hasattr(protocol.aggregator, 'received_data') and protocol.aggregator.received_data:
                        # Get aggregation stats
                        num_neighbors = len(protocol.aggregator.received_data.get(vid, {}))
                        tracker.log_aggregation(vid, {
                            'num_neighbors': num_neighbors,
                            'local_weight': protocol.aggregator.alpha if hasattr(protocol.aggregator, 'alpha') else 0.7,
                            'total_alpha': 1.0 - protocol.aggregator.alpha if hasattr(protocol.aggregator, 'alpha') else 0.3,
                            'num_stale': 0,  # Can be computed from staleness weights if needed
                        })

                # Step environment (get rewards)
                next_states, rewards, done, info = rl_env.step(actions)

                # Now add transitions to buffer with actual rewards
                for vid in env.vehicle_ids:
                    buffer.add(
                        vid,
                        states[vid],
                        actions[vid]["who_action"],
                        actions[vid]["what_action"],
                        float(log_probs[vid].cpu().numpy()[0]),
                        rewards[vid],  # Actual reward from environment
                        float(values[vid].cpu().numpy()[0, 0]),
                        done,
                        global_state,
                        neighbor_masks[vid],
                    )

                # Update state
                states = next_states
                global_state = info["global_state"]
                neighbor_masks = rl_env.get_neighbor_masks()

                # Track metrics
                episode_reward += sum(rewards.values())
                episode_comm_cost += dfl_info["total_data_sent_mb"]
                episode_train_losses.append(info["train_loss"])

                if done:
                    break

            # PPO update
            rl_stats = None
            if (episode + 1) % config.rl.training.update_interval == 0:
                data = buffer.get_all(gamma=ppo.gamma, gae_lambda=ppo.gae_lambda)
                stats = ppo.update(
                    states=data["states"],
                    who_actions=data["who_actions"],
                    what_actions=data["what_actions"],
                    old_log_probs=data["log_probs"],
                    advantages=data["advantages"],
                    returns=data["returns"],
                    global_states=data["global_states"],
                    neighbor_masks=data["neighbor_masks"],
                    num_epochs=config.rl.training.ppo_epochs,
                    batch_size=config.rl.training.batch_size,
                )
                rl_stats = stats

                # Log metrics
                if MLFLOW_AVAILABLE:
                    metrics_dict = {
                        "episode_reward": episode_reward,
                        "episode_comm_cost_mb": episode_comm_cost,
                        "episode_train_loss": np.mean(episode_train_losses) if episode_train_losses else 0.0,
                        "actor_loss": stats["actor_loss"],
                        "critic_loss": stats["critic_loss"],
                        "entropy": stats["entropy"],
                    }
                    # Add LSTM loss if training from scratch
                    if len(episode_lstm_losses) > 0:
                        metrics_dict["lstm_loss"] = np.mean(episode_lstm_losses)

                    mlflow.log_metrics(metrics_dict, step=episode)

                # Clear buffer
                buffer.clear()

            # End episode tracking
            avg_lstm_loss = np.mean(episode_lstm_losses) if episode_lstm_losses else None
            avg_train_loss = np.mean(episode_train_losses) if episode_train_losses else 0.0
            tracker.end_episode(
                final_reward=episode_reward,
                final_train_loss=avg_train_loss,
                comm_cost_mb=episode_comm_cost,
                rl_stats=rl_stats,
                duration_seconds=time.time() - episode_start_time,
                lstm_loss=avg_lstm_loss,
            )

            # Logging
            if (episode + 1) % config.rl.training.log_interval == 0:
                log_msg = (
                    f"Episode {episode+1}/{num_episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Comm Cost: {episode_comm_cost:.2f} MB"
                )
                if avg_lstm_loss is not None:
                    log_msg += f" | LSTM Loss: {avg_lstm_loss:.4f}"
                logger.info(log_msg)

            # Save checkpoint
            if (episode + 1) % config.rl.training.save_interval == 0:
                checkpoint_path = checkpoint_dir / f"dfl_checkpoint_{episode+1}.pt"
                torch.save(
                    {
                        "episode": episode,
                        "actor_state_dict": actor.state_dict(),
                        "critic_state_dict": critic.state_dict(),
                        "config": OmegaConf.to_container(config),
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_checkpoint_path = checkpoint_dir / "dfl_final.pt"
    torch.save(
            {
                "episode": num_episodes,
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "config": OmegaConf.to_container(config),
            },
            final_checkpoint_path,
    )
    logger.info(f"Saved final checkpoint: {final_checkpoint_path}")

    # Finalize tracking
    logger.info("Finalizing DFL metrics tracking...")
    tracker.finalize()
    logger.info(f"All DFL metrics saved to: {tracker.output_dir}")

    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(str(final_checkpoint_path))
        except Exception as e:
            logger.warning(f"Failed to log artifact to MLflow: {e}")
        finally:
            mlflow.end_run()

    logger.info("DFL training complete!")


if __name__ == "__main__":
    main()
