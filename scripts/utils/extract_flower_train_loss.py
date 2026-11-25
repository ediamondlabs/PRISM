#!/usr/bin/env python3
"""
Extract training loss from Flower FL logs and update history.json
"""

import json
import re
from pathlib import Path
import numpy as np

def extract_train_losses_from_log(log_path: str) -> list:
    """Extract per-client training losses from log file.

    Log structure:
    - Client X train loss (5 epochs avg): Y
    - Client X val loss: Z

    Appears sequentially: all Client 0 rounds, then all Client 1 rounds, etc.
    """

    all_client_losses = {}  # {client_id: [loss1, loss2, ...]}

    with open(log_path, 'r') as f:
        for line in f:
            # Extract train loss: "Client X train loss (5 epochs avg): Y"
            match = re.search(r'Client (\d+) train loss \(5 epochs avg\): ([\d.]+)', line)
            if match:
                client_id = int(match.group(1))
                train_loss = float(match.group(2))

                if client_id not in all_client_losses:
                    all_client_losses[client_id] = []

                all_client_losses[client_id].append(train_loss)

    # Verify all clients have same number of rounds
    num_rounds_per_client = {cid: len(losses) for cid, losses in all_client_losses.items()}
    print(f"  Rounds per client: {num_rounds_per_client}")

    if len(set(num_rounds_per_client.values())) > 1:
        print("  WARNING: Clients have different number of rounds!")

    # Get expected number of rounds
    max_rounds = max(num_rounds_per_client.values())

    # Compute average train loss per round across all clients
    train_losses = []
    for round_idx in range(max_rounds):
        round_losses = []
        for client_id in sorted(all_client_losses.keys()):
            if round_idx < len(all_client_losses[client_id]):
                round_losses.append(all_client_losses[client_id][round_idx])

        if round_losses:
            avg_loss = np.mean(round_losses)
            train_losses.append(float(avg_loss))
        else:
            train_losses.append(None)

    return train_losses


def main():
    log_path = Path('logs/flower_training_3000.log')
    history_path = Path('outputs/flower_fl/history.json')

    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return

    if not history_path.exists():
        print(f"Error: History file not found: {history_path}")
        return

    print("Extracting training losses from log file...")
    train_losses = extract_train_losses_from_log(log_path)

    print(f"Extracted {len(train_losses)} training loss values")

    # Load existing history
    with open(history_path, 'r') as f:
        history = json.load(f)

    # Verify length matches
    expected_rounds = len(history['round'])
    if len(train_losses) != expected_rounds:
        print(f"Warning: Extracted {len(train_losses)} losses, but history has {expected_rounds} rounds")
        # Pad or truncate to match
        if len(train_losses) < expected_rounds:
            train_losses.extend([None] * (expected_rounds - len(train_losses)))
        else:
            train_losses = train_losses[:expected_rounds]

    # Add train_loss field
    history['train_loss'] = train_losses

    # Save updated history
    backup_path = history_path.with_suffix('.json.backup')
    print(f"Creating backup: {backup_path}")
    with open(backup_path, 'w') as f:
        json.dump(json.load(open(history_path)), f)

    print("Updating history.json with train_loss field...")
    with open(history_path, 'w') as f:
        json.dump(history, f)

    # Print statistics
    valid_losses = [l for l in train_losses if l is not None]
    if valid_losses:
        print("\nStatistics:")
        print(f"  Valid entries: {len(valid_losses)}/{len(train_losses)}")
        print(f"  First train_loss: {valid_losses[0]:.6f}")
        print(f"  Last train_loss: {valid_losses[-1]:.6f}")
        print(f"  Min train_loss: {min(valid_losses):.6f}")
        print(f"  Max train_loss: {max(valid_losses):.6f}")

    print("\n✅ Done! history.json updated with train_loss field")


if __name__ == "__main__":
    main()
