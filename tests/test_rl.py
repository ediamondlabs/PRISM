"""
Tests for RL components (Actor, Critic, PPO, Buffer).
"""

import numpy as np
import pytest
import torch

from src.rl import (
    CentralizedCritic,
    DecentralizedCritic,
    MultiAgentRolloutBuffer,
    PPO,
    PPOActor,
    RolloutBuffer,
)


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def actor(device):
    """Create PPO Actor for testing."""
    return PPOActor(
        state_dim=106,
        shared_hidden_dim=128,
        who_hidden_dim=64,
        what_hidden_dim=64,
        max_neighbors=20,
        k_neighbors=2,
        num_repr_types=3,
        dropout=0.0,  # Disable dropout for testing
        use_when_head=False,  # Disable WHEN head for backward compatibility
    ).to(device)


@pytest.fixture
def centralized_critic(device):
    """Create Centralized Critic for testing."""
    return CentralizedCritic(
        state_dim=106,
        num_vehicles=20,
        hidden_dim=256,
        num_layers=3,
        dropout=0.0,
    ).to(device)


@pytest.fixture
def decentralized_critic(device):
    """Create Decentralized Critic for testing."""
    return DecentralizedCritic(
        state_dim=106,
        hidden_dim=128,
        num_layers=2,
        dropout=0.0,
    ).to(device)


@pytest.fixture
def sample_state(device):
    """Create sample state."""
    return torch.randn(8, 106, device=device)


@pytest.fixture
def sample_neighbor_mask(device):
    """Create sample neighbor mask."""
    mask = torch.zeros(8, 20, dtype=torch.bool, device=device)
    # Each sample has different number of neighbors
    for i in range(8):
        num_neighbors = np.random.randint(1, 15)
        mask[i, :num_neighbors] = True
    return mask


# ==================== Actor Tests ====================


def test_actor_initialization(actor):
    """Test Actor initialization."""
    assert actor.state_dim == 106
    assert actor.max_neighbors == 20
    assert actor.k_neighbors == 2
    assert actor.num_repr_types == 3
    assert actor.get_num_parameters() > 0


def test_actor_forward(actor, sample_state, sample_neighbor_mask, device):
    """Test Actor forward pass."""
    actor.eval()

    with torch.no_grad():
        who_logits, what_logits = actor(sample_state, sample_neighbor_mask)

    assert who_logits.shape == (8, 20)
    assert what_logits.shape == (8, 3)
    assert not torch.isnan(who_logits).any()
    assert not torch.isnan(what_logits).any()


def test_actor_get_action_deterministic(actor, sample_state, sample_neighbor_mask):
    """Test deterministic action selection."""
    actor.eval()

    with torch.no_grad():
        action_dict = actor.get_action(sample_state, sample_neighbor_mask, deterministic=True)

    assert action_dict["who_action"].shape == (8, 2)
    assert action_dict["what_action"].shape == (8,)
    assert action_dict["who_log_prob"].shape == (8,)
    assert action_dict["what_log_prob"].shape == (8,)


def test_actor_get_action_stochastic(actor, sample_state, sample_neighbor_mask):
    """Test stochastic action selection."""
    actor.eval()

    with torch.no_grad():
        action_dict = actor.get_action(sample_state, sample_neighbor_mask, deterministic=False)

    assert action_dict["who_action"].shape == (8, 2)
    assert action_dict["what_action"].shape == (8,)
    assert not torch.isnan(action_dict["who_log_prob"]).any()
    assert not torch.isnan(action_dict["what_log_prob"]).any()


def test_actor_evaluate_actions(actor, sample_state, sample_neighbor_mask):
    """Test action evaluation."""
    actor.eval()

    # Get actions
    with torch.no_grad():
        action_dict = actor.get_action(sample_state, sample_neighbor_mask)

    # Evaluate them
    eval_results = actor.evaluate_actions(
        sample_state,
        action_dict["who_action"],
        action_dict["what_action"],
        sample_neighbor_mask,
    )

    assert "log_prob" in eval_results
    assert "entropy" in eval_results
    assert eval_results["log_prob"].shape == (8,)
    assert eval_results["entropy"].shape == (8,)


def test_actor_masking(actor, device):
    """Test that masking properly excludes invalid neighbors."""
    state = torch.randn(1, 106, device=device)
    mask = torch.zeros(1, 20, dtype=torch.bool, device=device)
    mask[0, :5] = True  # Only first 5 neighbors valid

    actor.eval()
    with torch.no_grad():
        action_dict = actor.get_action(state, mask, deterministic=True)

    # Selected neighbors should be within valid range
    who_action = action_dict["who_action"][0]
    assert torch.all(who_action < 5)


# ==================== Critic Tests ====================


def test_centralized_critic_initialization(centralized_critic):
    """Test Centralized Critic initialization."""
    assert centralized_critic.state_dim == 106
    assert centralized_critic.num_vehicles == 20
    assert centralized_critic.global_state_dim == 106 * 20
    assert centralized_critic.get_num_parameters() > 0


def test_centralized_critic_forward(centralized_critic, device):
    """Test Centralized Critic forward pass."""
    global_state = torch.randn(8, 20, 106, device=device)

    centralized_critic.eval()
    with torch.no_grad():
        value = centralized_critic(global_state)

    assert value.shape == (8, 1)
    assert not torch.isnan(value).any()


def test_centralized_critic_flattened_input(centralized_critic, device):
    """Test Centralized Critic with flattened input."""
    global_state = torch.randn(8, 20 * 106, device=device)

    centralized_critic.eval()
    with torch.no_grad():
        value = centralized_critic(global_state)

    assert value.shape == (8, 1)


def test_decentralized_critic_initialization(decentralized_critic):
    """Test Decentralized Critic initialization."""
    assert decentralized_critic.state_dim == 106
    assert decentralized_critic.get_num_parameters() > 0


def test_decentralized_critic_forward(decentralized_critic, sample_state):
    """Test Decentralized Critic forward pass."""
    decentralized_critic.eval()
    with torch.no_grad():
        value = decentralized_critic(sample_state)

    assert value.shape == (8, 1)
    assert not torch.isnan(value).any()


# ==================== Buffer Tests ====================


def test_rollout_buffer_initialization(device):
    """Test RolloutBuffer initialization."""
    buffer = RolloutBuffer(
        buffer_size=100,
        state_dim=106,
        num_vehicles=20,
        max_neighbors=20,
        k_neighbors=2,
        device=device,
    )

    assert buffer.buffer_size == 100
    assert buffer.size == 0
    assert buffer.ptr == 0


def test_rollout_buffer_add(device):
    """Test adding data to buffer."""
    buffer = RolloutBuffer(
        buffer_size=100,
        state_dim=106,
        num_vehicles=20,
        max_neighbors=20,
        k_neighbors=2,
        device=device,
    )

    state = np.random.randn(106).astype(np.float32)
    who_action = np.array([0, 1], dtype=np.int64)
    what_action = 2
    global_state = np.random.randn(20, 106).astype(np.float32)
    neighbor_mask = np.zeros(20, dtype=bool)
    neighbor_mask[:5] = True

    buffer.add(
        state=state,
        who_action=who_action,
        what_action=what_action,
        log_prob=-1.5,
        reward=0.5,
        value=0.3,
        done=False,
        global_state=global_state,
        neighbor_mask=neighbor_mask,
    )

    assert buffer.size == 1
    assert buffer.ptr == 1


def test_rollout_buffer_get(device):
    """Test getting data from buffer with GAE computation."""
    buffer = RolloutBuffer(
        buffer_size=100,
        state_dim=106,
        num_vehicles=20,
        max_neighbors=20,
        k_neighbors=2,
        device=device,
    )

    # Add some transitions
    for i in range(10):
        state = np.random.randn(106).astype(np.float32)
        who_action = np.array([0, 1], dtype=np.int64)
        what_action = i % 3
        global_state = np.random.randn(20, 106).astype(np.float32)

        buffer.add(
            state=state,
            who_action=who_action,
            what_action=what_action,
            log_prob=-1.5,
            reward=0.5,
            value=0.3,
            done=(i == 9),
            global_state=global_state,
        )

    data = buffer.get(gamma=0.99, gae_lambda=0.95)

    assert data["states"].shape == (10, 106)
    assert data["advantages"].shape == (10,)
    assert data["returns"].shape == (10,)
    assert not torch.isnan(data["advantages"]).any()


def test_multi_agent_buffer(device):
    """Test MultiAgentRolloutBuffer."""
    buffer = MultiAgentRolloutBuffer(
        buffer_size=50,
        state_dim=106,
        num_vehicles=3,  # Use 3 for testing
        max_neighbors=20,
        k_neighbors=2,
        device=device,
    )

    # Add data for each vehicle
    for vid in range(3):
        for t in range(5):
            state = np.random.randn(106).astype(np.float32)
            who_action = np.array([0, 1], dtype=np.int64)

            buffer.add(
                vehicle_id=vid,
                state=state,
                who_action=who_action,
                what_action=t % 3,
                log_prob=-1.5,
                reward=0.5,
                value=0.3,
                done=(t == 4),
            )

    assert len(buffer) == 15  # 3 vehicles * 5 transitions

    # Get all data
    data = buffer.get_all(gamma=0.99, gae_lambda=0.95)
    assert data["states"].shape == (15, 106)


# ==================== PPO Tests ====================


def test_ppo_initialization(actor, centralized_critic, device):
    """Test PPO initialization."""
    ppo = PPO(
        actor=actor,
        critic=centralized_critic,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
    )

    assert ppo.actor is not None
    assert ppo.critic is not None
    assert ppo.gamma == 0.99


def test_ppo_compute_gae(actor, centralized_critic, device):
    """Test GAE computation."""
    ppo = PPO(
        actor=actor,
        critic=centralized_critic,
        device=device,
    )

    batch_size = 4
    seq_len = 10

    rewards = torch.randn(batch_size, seq_len, device=device)
    values = torch.randn(batch_size, seq_len, device=device)
    dones = torch.zeros(batch_size, seq_len, device=device)
    dones[:, -1] = 1  # Last step is terminal
    next_values = torch.randn(batch_size, seq_len, device=device)

    advantages, returns = ppo.compute_gae(rewards, values, dones, next_values)

    assert advantages.shape == (batch_size, seq_len)
    assert returns.shape == (batch_size, seq_len)
    assert not torch.isnan(advantages).any()
    assert not torch.isnan(returns).any()


def test_ppo_update(actor, centralized_critic, device):
    """Test PPO update."""
    ppo = PPO(
        actor=actor,
        critic=centralized_critic,
        device=device,
    )

    # Create dummy data
    num_samples = 32
    states = torch.randn(num_samples, 106, device=device)
    who_actions = torch.randint(0, 20, (num_samples, 2), device=device)
    what_actions = torch.randint(0, 3, (num_samples,), device=device)
    old_log_probs = torch.randn(num_samples, device=device)
    advantages = torch.randn(num_samples, device=device)
    returns = torch.randn(num_samples, device=device)
    global_states = torch.randn(num_samples, 20, 106, device=device)

    stats = ppo.update(
        states=states,
        who_actions=who_actions,
        what_actions=what_actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
        global_states=global_states,
        num_epochs=2,
        batch_size=16,
    )

    assert "actor_loss" in stats
    assert "critic_loss" in stats
    assert "entropy" in stats
    assert isinstance(stats["actor_loss"], float)


def test_ppo_get_value(actor, centralized_critic, device):
    """Test getting state value."""
    ppo = PPO(
        actor=actor,
        critic=centralized_critic,
        device=device,
    )

    state = torch.randn(8, 106, device=device)
    global_state = torch.randn(8, 20, 106, device=device)

    value = ppo.get_value(state, global_state)

    assert value.shape == (8, 1)
    assert not torch.isnan(value).any()


def test_ppo_save_load(actor, centralized_critic, device, tmp_path):
    """Test saving and loading PPO."""
    ppo = PPO(
        actor=actor,
        critic=centralized_critic,
        device=device,
    )

    # Save
    save_path = tmp_path / "ppo_checkpoint.pt"
    ppo.save(str(save_path))

    # Load
    ppo.load(str(save_path))

    # Check that model works after loading
    state = torch.randn(1, 106, device=device)
    global_state = torch.randn(1, 20, 106, device=device)
    value = ppo.get_value(state, global_state)
    assert value.shape == (1, 1)


# ==================== Integration Tests ====================


def test_end_to_end_training_step(actor, centralized_critic, device):
    """Test complete training step."""
    ppo = PPO(
        actor=actor,
        critic=centralized_critic,
        device=device,
    )

    buffer = RolloutBuffer(
        buffer_size=50,
        state_dim=106,
        num_vehicles=20,
        max_neighbors=20,
        k_neighbors=2,
        device=device,
    )

    # Collect rollout
    for t in range(20):
        state = torch.randn(1, 106, device=device)
        neighbor_mask = torch.ones(1, 20, dtype=torch.bool, device=device)
        global_state = torch.randn(1, 20, 106, device=device)

        # Get action
        with torch.no_grad():
            action_dict = actor.get_action(state, neighbor_mask)
            value = ppo.get_value(state, global_state)

        # Add to buffer
        buffer.add(
            state=state.cpu().numpy()[0],
            who_action=action_dict["who_action"].cpu().numpy()[0],
            what_action=int(action_dict["what_action"].cpu().numpy()[0]),
            log_prob=float(action_dict["who_log_prob"].cpu().numpy()[0] + action_dict["what_log_prob"].cpu().numpy()[0]),
            reward=np.random.rand(),
            value=float(value.cpu().numpy()[0, 0]),
            done=(t == 19),
            global_state=global_state.cpu().numpy()[0],
            neighbor_mask=neighbor_mask.cpu().numpy()[0],
        )

    # Get data and update
    data = buffer.get()
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
        batch_size=10,
    )

    assert stats["actor_loss"] is not None
    assert stats["critic_loss"] is not None


def test_gradient_flow(actor, device):
    """Test that gradients flow properly through actor."""
    actor.train()

    state = torch.randn(4, 106, device=device, requires_grad=True)
    neighbor_mask = torch.ones(4, 20, dtype=torch.bool, device=device)

    who_logits, what_logits = actor(state, neighbor_mask)
    loss = who_logits.sum() + what_logits.sum()
    loss.backward()

    # Check gradients exist
    for name, param in actor.named_parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
