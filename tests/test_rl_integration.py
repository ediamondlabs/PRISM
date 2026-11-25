"""
Integration tests for RL training components.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.rl import (
    CentralizedCritic,
    MultiAgentRolloutBuffer,
    PPO,
    PPOActor,
    RewardFunction,
    RLV2VWrapper,
)
from src.simulation import V2VEnvironment


@pytest.fixture
def rl_config():
    """Create RL training configuration."""
    config_dict = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "window_size": 50,
            "features": [
                "velocity_x",
                "velocity_y",
                "speed",
                "accel_x",
                "accel_y",
                "accel_z",
                "gyro_x",
                "gyro_y",
                "gyro_z",
                "yaw",
            ],
        },
        "model": {
            "name": "lstm_autoencoder",
            "architecture": {
                "input_size": 10,
                "hidden_size": 32,
                "num_layers": 1,
                "embedding_dim": 16,
            },
            "neighbor_encoder": {
                "architecture": {
                    "neighbor_feature_dim": 40,
                    "phi_hidden_dim": 64,
                    "phi_output_dim": 32,
                    "rho_hidden_dim": 64,
                    "rho_output_dim": 32,
                    "dropout": 0.0,
                }
            },
        },
        "simulation": {
            "num_vehicles": 3,
            "comm_range_m": 100.0,
            "time_step_duration_s": 0.1,
            "max_episode_timesteps": 10,
            "layer1_checkpoint": "nonexistent.pt",
            "use_neighbor_encoder": False,
            "data": {
                "use_preprocessed": True,
                "preprocessed_path": "outputs/preprocessed",
            },
        },
        "rl": {
            "actor": {
                "state_dim": 106,
                "shared_hidden_dim": 64,
                "who_hidden_dim": 32,
                "what_hidden_dim": 32,
                "max_neighbors": 20,
                "k_neighbors": 2,
                "num_repr_types": 3,
                "dropout": 0.0,
            },
            "critic": {
                "type": "centralized",
                "state_dim": 106,
                "num_vehicles": 3,
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "ppo": {
                "lr_actor": 3e-4,
                "lr_critic": 1e-3,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "entropy_coef": 0.01,
                "value_loss_coef": 0.5,
                "max_grad_norm": 0.5,
            },
            "reward": {
                "immediate_weight": 0.3,
                "global_weight": 0.7,
                "perf_scale": 100.0,
            },
        },
    }
    return OmegaConf.create(config_dict)


# ==================== Reward Function Tests ====================


def test_reward_function():
    """Test reward function computation."""
    reward_fn = RewardFunction(
        immediate_weight=0.3,
        global_weight=0.7,
        perf_scale=100.0,
    )

    # Test immediate reward
    immediate = reward_fn.compute_immediate_reward(
        vehicle_id=0,
        what_action=0,
        communication_success=True,
        anomaly_detected=True,
        neighbor_stability=0.8,
    )
    assert immediate > 0
    assert isinstance(immediate, float)

    # Test global reward (initialize first, then improve)
    reward_fn.compute_global_reward(current_train_loss=1.0)  # init
    global_reward = reward_fn.compute_global_reward(current_train_loss=0.8, comm_cost_mb=1.0)
    assert isinstance(global_reward, float)

    # Test total reward
    result = reward_fn.compute_total_reward(
        vehicle_id=0,
        what_action=1,
        communication_success=True,
        anomaly_detected=False,
        neighbor_stability=0.5,
        current_train_loss=0.75,
        comm_cost_mb=0.5,
    )
    assert "immediate" in result
    assert "global" in result
    assert "total" in result


# ==================== Environment Wrapper Tests ====================


@pytest.mark.skipif(
    not pytest.importorskip("src").simulation.__file__, reason="Simulation not available"
)
def test_rl_wrapper_initialization(rl_config):
    """Test RLV2VWrapper initialization."""
    # Skip if preprocessed data not available
    import pathlib

    if not pathlib.Path("outputs/preprocessed/preprocessed_data.npz").exists():
        pytest.skip("Preprocessed data not available")

    env = V2VEnvironment(rl_config)
    reward_fn = RewardFunction()

    wrapper = RLV2VWrapper(
        env=env, reward_fn=reward_fn, use_neighbor_encoder=False, device=torch.device("cpu")
    )

    assert wrapper.num_vehicles == env.num_vehicles
    assert wrapper.state_dim == 106


@pytest.mark.skipif(
    not pytest.importorskip("src").simulation.__file__, reason="Simulation not available"
)
def test_rl_wrapper_reset(rl_config):
    """Test environment reset."""
    import pathlib

    if not pathlib.Path("outputs/preprocessed/preprocessed_data.npz").exists():
        pytest.skip("Preprocessed data not available")

    env = V2VEnvironment(rl_config)
    reward_fn = RewardFunction()
    wrapper = RLV2VWrapper(env=env, reward_fn=reward_fn)

    states, global_state = wrapper.reset()

    assert isinstance(states, dict)
    assert len(states) == env.num_vehicles

    # Check state dimensions
    for vid, state in states.items():
        assert state.shape == (106,)
        assert state.dtype == np.float32

    # Check global state
    assert global_state.shape == (env.num_vehicles, 106)


@pytest.mark.skipif(
    not pytest.importorskip("src").simulation.__file__, reason="Simulation not available"
)
def test_rl_wrapper_step(rl_config):
    """Test environment step."""
    import pathlib

    if not pathlib.Path("outputs/preprocessed/preprocessed_data.npz").exists():
        pytest.skip("Preprocessed data not available")

    env = V2VEnvironment(rl_config)
    reward_fn = RewardFunction()
    wrapper = RLV2VWrapper(env=env, reward_fn=reward_fn)

    states, global_state = wrapper.reset()

    # Create dummy actions
    actions = {}
    for vid in env.vehicle_ids:
        actions[vid] = {
            "who_action": np.array([0, 1], dtype=np.int64),
            "what_action": 1,
        }

    # Step
    next_states, rewards, done, info = wrapper.step(actions)

    assert isinstance(next_states, dict)
    assert isinstance(rewards, dict)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    # Check rewards
    for vid in env.vehicle_ids:
        assert vid in rewards
        assert isinstance(rewards[vid], float)


# ==================== Integration Tests ====================


@pytest.mark.skipif(
    not pytest.importorskip("src").simulation.__file__, reason="Simulation not available"
)
def test_full_rollout(rl_config):
    """Test full rollout collection."""
    import pathlib

    if not pathlib.Path("outputs/preprocessed/preprocessed_data.npz").exists():
        pytest.skip("Preprocessed data not available")

    device = torch.device("cpu")

    # Create environment
    env = V2VEnvironment(rl_config)
    reward_fn = RewardFunction()
    wrapper = RLV2VWrapper(env=env, reward_fn=reward_fn, device=device)

    # Create agent
    actor = PPOActor(
        state_dim=106,
        shared_hidden_dim=64,
        who_hidden_dim=32,
        what_hidden_dim=32,
        max_neighbors=20,
        k_neighbors=2,
        num_repr_types=3,
        dropout=0.0,
    ).to(device)

    critic = CentralizedCritic(
        state_dim=106, num_vehicles=3, hidden_dim=128, num_layers=2, dropout=0.0
    ).to(device)

    ppo = PPO(actor=actor, critic=critic, device=device)

    # Create buffer
    buffer = MultiAgentRolloutBuffer(
        buffer_size=20,
        state_dim=106,
        num_vehicles=3,
        max_neighbors=20,
        k_neighbors=2,
        device=device,
        vehicle_ids=list(env.vehicle_ids),
    )

    # Collect rollout
    states, global_state = wrapper.reset()

    for step in range(10):
        # Get actions from all vehicles
        actions = {}
        log_probs = {}
        values = {}

        neighbor_masks = wrapper.get_neighbor_masks()

        for vid in env.vehicle_ids:
            state_tensor = torch.from_numpy(states[vid]).unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(neighbor_masks[vid]).unsqueeze(0).to(device)
            global_state_tensor = torch.from_numpy(global_state).unsqueeze(0).to(device)

            with torch.no_grad():
                action_dict = actor.get_action(state_tensor, mask_tensor)
                value = ppo.get_value(state_tensor, global_state_tensor)

            actions[vid] = {
                "who_action": action_dict["who_action"].cpu().numpy()[0],
                "what_action": int(action_dict["what_action"].cpu().numpy()[0]),
            }

            log_probs[vid] = float(
                (action_dict["who_log_prob"] + action_dict["what_log_prob"]).cpu().numpy()[0]
            )
            values[vid] = float(value.cpu().numpy()[0, 0])

        # Step environment
        next_states, rewards, done, info = wrapper.step(actions)

        # Add to buffer
        for vid in env.vehicle_ids:
            buffer.add(
                vehicle_id=vid,
                state=states[vid],
                who_action=actions[vid]["who_action"],
                what_action=actions[vid]["what_action"],
                log_prob=log_probs[vid],
                reward=rewards[vid],
                value=values[vid],
                done=done,
                global_state=global_state,
                neighbor_mask=neighbor_masks[vid],
            )

        states = next_states
        global_state = info["global_state"]

        if done:
            break

    # Check buffer
    assert len(buffer) > 0

    # Get data
    data = buffer.get_all()

    assert "states" in data
    assert "advantages" in data
    assert "returns" in data


@pytest.mark.skipif(
    not pytest.importorskip("src").simulation.__file__, reason="Simulation not available"
)
def test_ppo_update_with_rollout(rl_config):
    """Test PPO update with collected rollout."""
    import pathlib

    if not pathlib.Path("outputs/preprocessed/preprocessed_data.npz").exists():
        pytest.skip("Preprocessed data not available")

    device = torch.device("cpu")

    # Create simple environment and collect rollout (simplified version)
    env = V2VEnvironment(rl_config)
    reward_fn = RewardFunction()
    wrapper = RLV2VWrapper(env=env, reward_fn=reward_fn)

    actor = PPOActor(
        state_dim=106,
        who_hidden_dim=32,
        what_hidden_dim=32,
        max_neighbors=20,
        k_neighbors=2,
    ).to(device)

    critic = CentralizedCritic(state_dim=106, num_vehicles=3, hidden_dim=64).to(device)

    ppo = PPO(actor=actor, critic=critic, device=device)

    buffer = MultiAgentRolloutBuffer(
        buffer_size=20,
        state_dim=106,
        num_vehicles=3,
        max_neighbors=20,
        k_neighbors=2,
        device=device,
        vehicle_ids=list(env.vehicle_ids),
    )

    # Quick rollout
    states, global_state = wrapper.reset()
    neighbor_masks = wrapper.get_neighbor_masks()

    for _ in range(5):
        actions = {}
        for vid in env.vehicle_ids:
            state_t = torch.from_numpy(states[vid]).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(neighbor_masks[vid]).unsqueeze(0).to(device)
            global_t = torch.from_numpy(global_state).unsqueeze(0).to(device)

            with torch.no_grad():
                act = actor.get_action(state_t, mask_t)
                val = ppo.get_value(state_t, global_t)

            actions[vid] = {
                "who_action": act["who_action"].cpu().numpy()[0],
                "what_action": int(act["what_action"].cpu().numpy()[0]),
            }

            buffer.add(
                vid,
                states[vid],
                actions[vid]["who_action"],
                actions[vid]["what_action"],
                float((act["who_log_prob"] + act["what_log_prob"]).cpu().numpy()[0]),
                0.1,
                float(val.cpu().numpy()[0, 0]),
                False,
                global_state,
                neighbor_masks[vid],
            )

        states, _, _, info = wrapper.step(actions)
        global_state = info["global_state"]
        neighbor_masks = wrapper.get_neighbor_masks()

    # Update
    data = buffer.get_all()
    stats = ppo.update(
        states=data["states"],
        who_actions=data["who_actions"],
        what_actions=data["what_actions"],
        old_log_probs=data["log_probs"],
        advantages=data["advantages"],
        returns=data["returns"],
        global_states=data["global_states"],
        neighbor_masks=data["neighbor_masks"],
        num_epochs=2,
        batch_size=8,
    )

    assert "actor_loss" in stats
    assert "critic_loss" in stats
    assert isinstance(stats["actor_loss"], float)
