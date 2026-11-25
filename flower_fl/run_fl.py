"""
Run Flower FL experiment with FedAvg.
Launches server + 5 clients automatically.

Usage:
    python flower_fl/run_fl.py
"""

import subprocess
import time
import sys
from pathlib import Path


def main():
    num_clients = 5
    num_rounds = 3000  # Number of FL rounds
    base_dir = Path(__file__).parent

    print(f"Starting Flower FL with {num_clients} clients, {num_rounds} rounds")
    print("Local epochs per round: 5 (matching DFL)")
    print("=" * 50)

    # Start server in background
    print("Starting server...")
    server_proc = subprocess.Popen(
        [sys.executable, str(base_dir / "server.py"), "--rounds", str(num_rounds)],
    )
    time.sleep(5)  # Wait for server startup

    # Start clients sequentially to avoid GPU memory issues (matching DFL approach)
    # Clients 0,1,2 on GPU 0, Clients 3,4 on GPU 1
    print("Starting clients sequentially (avoids GPU OOM)...")
    client_procs = []
    for i in range(num_clients):
        # GPU assignment: first 3 clients on GPU 0, last 2 on GPU 1
        gpu_id = 0 if i < 3 else 1
        print(f"Starting client {i} on cuda:{gpu_id}...")
        proc = subprocess.Popen(
            [
                sys.executable,
                str(base_dir / "client.py"),
                "--client-id", str(i),
                "--num-clients", str(num_clients),
                "--device", f"cuda:{gpu_id}",
            ],
        )
        client_procs.append(proc)
        time.sleep(2)  # Stagger client starts

    print("\nAll processes started. Waiting for completion...")
    print("=" * 50)

    # Wait for server to complete (no timeout for long training runs)
    try:
        server_proc.wait()  # No timeout - let it run until completion
        print("\nServer completed!")
    except KeyboardInterrupt:
        print("\nInterrupted - killing processes")
        server_proc.kill()

    # Cleanup clients
    for i, proc in enumerate(client_procs):
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("\nFL training complete!")


if __name__ == "__main__":
    main()
