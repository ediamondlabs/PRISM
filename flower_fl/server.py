"""
Flower FL Server with FedAvg aggregation.
Uses shared config for fair comparison with DFL, CoCo, and Centralized.

Usage:
    python flower_fl/server.py
    python flower_fl/server.py --rounds 50
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
import yaml


def load_shared_config() -> dict:
    """Load shared configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "comparison" / "shared_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {"federated": {"num_clients": 5, "num_rounds": 100}}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics weighted by number of samples."""
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}

    weighted_loss = sum(
        num_samples * m.get("val_loss", 0.0)
        for num_samples, m in metrics
    ) / total_samples

    return {"val_loss": weighted_loss}


class LoggingFedAvg(FedAvg):
    """FedAvg with logging for comparison."""

    def __init__(self, *args, output_dir: Path = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir or Path("outputs/flower_fl")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            "round": [],
            "train_loss": [],  # Training loss (averaged across clients)
            "val_loss": [],
            "lstm_loss": [],  # Same as val_loss for compatibility
            "round_time_sec": [],
            "cumulative_bandwidth_mb": [],
            "bandwidth_per_round_mb": [],
        }

        import time
        self.round_start_time = None
        self.training_start_time = time.time()
        self.cumulative_bandwidth = 0.0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        """Track time, bandwidth, and training loss during training aggregation."""
        import time

        # Start timing this round
        if self.round_start_time is None:
            self.round_start_time = time.time()

        # Estimate bandwidth: model size × number of clients × 2 (upload + download)
        if results:
            # Get model size from first result (parameters)
            params = results[0][1].parameters
            model_size_mb = sum(len(tensor) for tensor in params.tensors) / (1024 * 1024)

            # Bandwidth = clients upload models + server broadcasts aggregated model
            num_clients = len(results)
            round_bandwidth_mb = model_size_mb * num_clients * 2  # Upload + download

            self.cumulative_bandwidth += round_bandwidth_mb
            self.history["bandwidth_per_round_mb"].append(round_bandwidth_mb)
            self.history["cumulative_bandwidth_mb"].append(self.cumulative_bandwidth)

            # Extract training loss from client metrics
            train_losses = []
            for _, fit_res in results:
                if fit_res.metrics and "train_loss" in fit_res.metrics:
                    train_losses.append(fit_res.metrics["train_loss"])

            # Compute weighted average (simple average for now)
            if train_losses:
                avg_train_loss = sum(train_losses) / len(train_losses)
                self.history["train_loss"].append(avg_train_loss)
                print(f"Round {server_round}: Aggregated train_loss = {avg_train_loss:.6f}")

        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        import time

        if not results:
            return None

        # Weighted average
        total_samples = sum(r.num_examples for _, r in results)
        weighted_loss = sum(
            r.num_examples * r.loss for _, r in results
        ) / total_samples

        # Record round time
        if self.round_start_time is not None:
            round_time = time.time() - self.round_start_time
            self.history["round_time_sec"].append(round_time)
            self.round_start_time = time.time()  # Reset for next round

        self.history["round"].append(server_round)
        self.history["val_loss"].append(weighted_loss)
        self.history["lstm_loss"].append(weighted_loss)  # Same as val_loss

        print(f"Round {server_round}: Aggregated val_loss = {weighted_loss:.6f}")

        # Save history after each round
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f)

        return super().aggregate_evaluate(server_round, results, failures)


def main():
    parser = argparse.ArgumentParser(description="Flower FL Server (FedAvg)")
    parser.add_argument("--rounds", type=int, default=3000, help="Number of FL rounds")
    parser.add_argument("--min-clients", type=int, default=5, help="Minimum clients")
    parser.add_argument("--output-dir", type=str, default="outputs/flower_fl")
    args = parser.parse_args()

    config = load_shared_config()
    num_rounds = args.rounds or config["federated"]["num_rounds"]
    min_clients = args.min_clients or config["federated"]["num_clients"]
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("FLOWER FL SERVER (FedAvg)")
    print(f"Rounds: {num_rounds}, Min clients: {min_clients}")
    print("Using shared config for fair comparison")
    print("=" * 60)

    strategy = LoggingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        output_dir=output_dir,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8983",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Save final results
    import time
    total_training_time = time.time() - strategy.training_start_time

    results = {
        "method": "Flower FL (FedAvg)",
        "num_rounds": num_rounds,
        "num_clients": min_clients,
        "final_val_loss": strategy.history["val_loss"][-1] if strategy.history["val_loss"] else None,
        "final_lstm_loss": strategy.history["val_loss"][-1] if strategy.history["val_loss"] else None,
        "total_time_sec": total_training_time,
        "total_bandwidth_mb": strategy.cumulative_bandwidth,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
