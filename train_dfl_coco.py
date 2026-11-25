"""
CoCo Baseline Training Script
Implements CoCo algorithm from "Accelerating Decentralized Federated Learning in Heterogeneous Edge Computing"

Completely separate from RL-based DFL implementation (train_dfl.py).
Runs Algorithm 1 (Worker) and Algorithm 2 (Coordinator) for comparison.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.federated.coco_algorithm import CoCo
from src.federated.coco_coordinator import CoCoCoordinator
from src.federated.coco_worker import CoCoWorker
from src.models import create_autoencoder_from_config


def create_worker_dataloaders(
    train_data: np.ndarray,
    num_vehicles: int,
    batch_size: int,
) -> List[DataLoader]:
    """Split training data among workers and create DataLoaders.

    Args:
        train_data: Full training dataset (N, seq_len, features)
        num_vehicles: Number of workers
        batch_size: Batch size for training

    Returns:
        List of DataLoaders, one per worker
    """
    N = len(train_data)
    samples_per_worker = N // num_vehicles

    dataloaders = []
    for i in range(num_vehicles):
        start_idx = i * samples_per_worker
        end_idx = (i + 1) * samples_per_worker if i < num_vehicles - 1 else N

        worker_data = train_data[start_idx:end_idx]
        dataset = TensorDataset(torch.FloatTensor(worker_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        dataloaders.append(dataloader)

    return dataloaders


def estimate_worker_resources(
    worker_id: int,
    num_vehicles: int,
    base_bandwidth: float = 1e7,  # 10 MB/s
    base_comp_time: float = 1.0,  # 1 second
) -> Dict[str, float]:
    """Estimate heterogeneous resources for worker (simplified).

    Args:
        worker_id: Worker ID
        num_vehicles: Total number of workers
        base_bandwidth: Base bandwidth
        base_comp_time: Base computation time

    Returns:
        Dict with bandwidth_in, bandwidth_out, comp_time
    """
    # Add heterogeneity: random variation ±30%
    np.random.seed(worker_id)
    variation = 0.7 + 0.6 * np.random.random()

    return {
        "bandwidth_in": base_bandwidth * variation,
        "bandwidth_out": base_bandwidth * variation,
        "comp_time": base_comp_time / variation,
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training loop for CoCo baseline."""

    print("=" * 80)
    print("CoCo Baseline Training")
    print("Paper: Accelerating Decentralized Federated Learning in Heterogeneous Edge Computing")
    print("=" * 80)

    # Check if coco config exists, if not provide defaults
    if not hasattr(cfg, 'coco') or cfg.coco is None:
        print("Warning: No 'coco' config found. Using defaults from paper.")
        cfg.coco = OmegaConf.create({
            'beta_1': 0.5,
            'beta_2': 0.1,
            'tau': 50,
            'base_topology': 'complete',
            'training': {
                'num_rounds': 300,
                'batch_size': 32
            }
        })

    # Set defaults for missing configs
    if not hasattr(cfg, 'federated') or cfg.federated is None:
        cfg.federated = OmegaConf.create({'lstm_lr': 0.001})
    elif not hasattr(cfg.federated, 'lstm_lr'):
        cfg.federated.lstm_lr = 0.001

    if not hasattr(cfg, 'device'):
        cfg.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    if not hasattr(cfg, 'lstm_device'):
        cfg.lstm_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Configuration:\n{OmegaConf.to_yaml(cfg.coco)}")
    print("=" * 80)

    # Extract config
    num_vehicles = cfg.data.num_vehicles
    num_rounds = cfg.coco.training.num_rounds
    tau = cfg.coco.tau
    batch_size = cfg.coco.training.batch_size
    beta_1 = cfg.coco.beta_1
    beta_2 = cfg.coco.beta_2
    base_topology = cfg.coco.base_topology
    lstm_lr = cfg.federated.lstm_lr

    # Devices
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    lstm_device = torch.device(cfg.lstm_device if torch.cuda.is_available() else "cpu")

    print(f"\nDevices: Main={device}, LSTM={lstm_device}")
    print(f"Workers: {num_vehicles}, Rounds: {num_rounds}, τ={tau}")

    # Load preprocessed data
    data_path = Path("outputs/preprocessed/preprocessed_data.npz")
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        print("Run preprocessing first!")
        return

    data = np.load(data_path, allow_pickle=True)
    train_data = data["train_data"]
    test_data = data["test_data"][:1000]  # Sample for evaluation

    print(f"\nData loaded: train={train_data.shape}, test={test_data.shape}")

    # Create worker DataLoaders
    print("\nCreating worker DataLoaders...")
    worker_dataloaders = create_worker_dataloaders(train_data, num_vehicles, batch_size)

    # Initialize workers
    print("\nInitializing workers...")
    workers: List[CoCoWorker] = []
    coco_optimizer = CoCo(num_vehicles=num_vehicles, beta_1=beta_1, beta_2=beta_2)

    # GPU allocation: Split workers across 2 GPUs
    # For 5 workers: Workers 0-1 → GPU 0 (2 workers), Workers 2-4 → GPU 1 (3 workers)
    # For 20 workers: Workers 0-9 → GPU 0 (10 workers), Workers 10-19 → GPU 1 (10 workers)
    gpu_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Calculate split point (floor division to put more on GPU 1 for 5 workers, balanced for 20)
    gpu_0_workers = num_vehicles // 2  # 2 for 5 vehicles, 10 for 20 vehicles

    print(f"GPU allocation: Workers 0-{gpu_0_workers-1} on {gpu_0}, Workers {gpu_0_workers}-{num_vehicles-1} on {gpu_1}")

    for i in range(num_vehicles):
        # Assign GPU based on worker ID
        worker_device = gpu_0 if i < gpu_0_workers else gpu_1

        # Create model for worker on assigned GPU
        model = create_autoencoder_from_config(cfg, device=worker_device)

        # Create worker
        worker = CoCoWorker(
            worker_id=i,
            model=model,
            data_loader=worker_dataloaders[i],
            device=worker_device,
            tau=tau,
            learning_rate=lstm_lr,
            coco_optimizer=coco_optimizer,
        )

        # Estimate resources
        resources = estimate_worker_resources(i, num_vehicles)
        worker.update_bandwidth_and_comp_time(
            resources["bandwidth_in"],
            resources["bandwidth_out"],
            resources["comp_time"],
        )

        workers.append(worker)

    gpu_0_count = sum(1 for w in workers if w.device == gpu_0)
    gpu_1_count = sum(1 for w in workers if w.device == gpu_1)
    print(f"Created {len(workers)} workers ({gpu_0_count} on GPU0, {gpu_1_count} on GPU1)")

    # Calculate actual model size for CoCo LP solver (CRITICAL FIX)
    # Bug was: default model_size_bytes=1MB, actual model is ~143MB!
    # This caused LP solver to choose wrong compression ratios
    sample_model = workers[0].model
    model_params = sum(p.numel() for p in sample_model.parameters())
    model_size_bytes = model_params * 4  # float32 = 4 bytes per parameter
    print(f"\nModel size: {model_params:,} parameters = {model_size_bytes / 1e6:.2f} MB")

    # Initialize coordinator
    print("\nInitializing coordinator...")
    coordinator = CoCoCoordinator(
        num_vehicles=num_vehicles,
        base_topology=base_topology,
        beta_1=beta_1,
        beta_2=beta_2,
        initial_D_max=0.1,
        model_size_bytes=model_size_bytes,  # CRITICAL: Pass actual model size!
    )

    print(f"Coordinator created with base topology: {base_topology}")

    # Output directories (unique per num_vehicles to avoid conflicts)
    output_dir = Path(f"outputs/dfl_coco_{num_vehicles}cars")
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Metrics tracking
    metrics = {
        "round": [],
        "avg_loss": [],
        "total_bandwidth_mb": [],
        "round_time_sec": [],
        "test_error": [],
    }

    # Test data tensor (will be moved to each worker's device during evaluation)
    test_tensor = torch.FloatTensor(test_data)

    print("\n" + "=" * 80)
    print("Starting CoCo Training")
    print("=" * 80)

    # Training loop: Algorithm 1 + Algorithm 2
    training_start_time = time.time()

    # Track gradient norms from previous round (for Equation 28)
    worker_gradient_norms = {i: 0.0 for i in range(num_vehicles)}

    for round_idx in range(1, num_rounds + 1):
        round_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Round {round_idx}/{num_rounds}")
        print(f"{'='*60}")

        # Step 1: Workers compute consensus distances (Algorithm 1 Line 2)
        print("Step 1: Computing consensus distances...")
        worker_models = {i: workers[i].get_model_state() for i in range(num_vehicles)}
        worker_statuses = []

        for worker in workers:
            # Get current neighbors (initially all neighbors for complete topology)
            if round_idx == 1:
                # First round: use base topology
                A_base = coordinator.get_current_topology()
                neighbors = np.where(A_base[worker.worker_id, :] == 1)[0].tolist()
            else:
                neighbors = worker.current_neighbors

            # Compute consensus distances with neighbors
            neighbor_models = {j: worker_models[j] for j in neighbors if j in worker_models}
            consensus_distances = worker.compute_consensus_distances_line2(neighbor_models)

            # Prepare status (Algorithm 1 Line 3) with gradient norm from previous round
            status = worker.prepare_status_for_coordinator_line3(
                consensus_distances,
                gradient_norm=worker_gradient_norms[worker.worker_id]
            )
            worker_statuses.append(status)

        # Step 2: Coordinator runs CoCo (Algorithm 2)
        print("Step 2: Coordinator optimizing topology and compression...")
        assignments = coordinator.execute_round(worker_statuses)

        # Step 3: Workers receive assignments (Algorithm 1 Line 4)
        print("Step 3: Workers receiving topology and ratios...")
        for worker in workers:
            assignment = assignments[worker.worker_id]
            worker.receive_from_coordinator_line4(
                assignment["neighbors"],
                assignment["compression_ratio"]
            )

        # Step 4: Workers perform local training (Algorithm 1 Line 5)
        # Workers distributed across 2 GPUs for memory distribution
        print(f"Step 4: Workers training locally (τ={tau} steps)...")
        worker_losses = []
        for worker in workers:
            # BUG FIX #1: Unpack both loss and gradient_norm
            loss, gradient_norm = worker.local_update_line5_eq5()
            worker_losses.append(loss)
            # Store gradient norm for next round's Equation 28
            worker_gradient_norms[worker.worker_id] = gradient_norm

        avg_loss = np.mean(worker_losses)
        print(f"   Average worker loss: {avg_loss:.6f}")

        # Step 5: Workers compress models (Algorithm 1 Line 6)
        print("Step 5: Workers compressing models...")
        compressed_models = {}
        for worker in workers:
            compressed, compressed_size = worker.compress_model_line6()
            compressed_models[worker.worker_id] = compressed

            # Track bandwidth: compressed_size × num_neighbors (each worker sends to all neighbors)
            num_neighbors = len(worker.current_neighbors)
            worker.total_bytes_sent += compressed_size * num_neighbors

        # Step 6: Workers communicate (Algorithm 1 Line 7)
        print("Step 6: Workers communicating with neighbors...")
        for worker in workers:
            # Receive models from neighbors
            neighbor_models = {
                j: compressed_models[j]
                for j in worker.current_neighbors
                if j in compressed_models
            }
            received = worker.communicate_with_neighbors_line7(
                compressed_models[worker.worker_id],
                neighbor_models
            )

            # Step 7: Aggregate (Algorithm 1 Line 8)
            worker.aggregate_models_line8_eq6(received)

        # Evaluate on test set
        print("Step 7: Evaluating on test set...")
        test_errors = []
        for worker in workers:
            worker.model.eval()
            # Move test tensor to worker's device
            test_data_device = test_tensor.to(worker.device)
            with torch.no_grad():
                test_error = worker.model.compute_reconstruction_error(test_data_device).item()
            test_errors.append(test_error)

        avg_test_error = np.mean(test_errors)

        # Track bandwidth (only count SENT bytes to avoid double-counting)
        total_bandwidth = sum(w.get_communication_stats()["total_bytes_sent"] for w in workers)
        total_bandwidth_mb = total_bandwidth / (1024 ** 2)

        round_time = time.time() - round_start_time

        # Log metrics
        metrics["round"].append(round_idx)
        metrics["avg_loss"].append(avg_loss)
        metrics["total_bandwidth_mb"].append(total_bandwidth_mb)
        metrics["round_time_sec"].append(round_time)
        metrics["test_error"].append(avg_test_error)

        print(f"\n{'='*60}")
        print(f"Round {round_idx} Summary:")
        print(f"  Avg Loss: {avg_loss:.6f}")
        print(f"  Avg Test Error: {avg_test_error:.6f}")
        print(f"  Total Bandwidth: {total_bandwidth_mb:.2f} MB")
        print(f"  Round Time: {round_time:.2f}s")
        print(f"{'='*60}")

        # Save checkpoint every 50 rounds
        if round_idx % 50 == 0 or round_idx == num_rounds:
            checkpoint_path = checkpoint_dir / f"coco_round_{round_idx}.pt"
            torch.save({
                "round": round_idx,
                "worker_models": [w.get_model_state() for w in workers],
                "metrics": metrics,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    training_time = time.time() - training_start_time

    print("\n" + "=" * 80)
    print("CoCo Training Completed")
    print("=" * 80)
    print(f"Total time: {training_time/3600:.2f} hours")
    print(f"Final avg loss: {metrics['avg_loss'][-1]:.6f}")
    print(f"Final test error: {metrics['test_error'][-1]:.6f}")
    print(f"Total bandwidth: {metrics['total_bandwidth_mb'][-1]:.2f} MB")

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / "coco_final.pt"
    torch.save({
        "round": num_rounds,
        "worker_models": [w.get_model_state() for w in workers],
        "metrics": metrics,
        "config": OmegaConf.to_container(cfg),
    }, final_checkpoint_path)
    print(f"\nFinal checkpoint: {final_checkpoint_path}")

    # Save metrics as NPZ
    metrics_path = metrics_dir / "coco_metrics.npz"
    np.savez(
        metrics_path,
        round=np.array(metrics["round"]),
        avg_loss=np.array(metrics["avg_loss"]),
        total_bandwidth_mb=np.array(metrics["total_bandwidth_mb"]),
        round_time_sec=np.array(metrics["round_time_sec"]),
        test_error=np.array(metrics["test_error"]),
    )
    print(f"Metrics saved: {metrics_path}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
