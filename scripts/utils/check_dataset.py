"""Check if dataset has ground truth anomaly labels."""

import pandas as pd
from pathlib import Path
import glob

print("=" * 60)
print("CHECKING DATASET FOR GROUND TRUTH LABELS")
print("=" * 60)

# Find data files
data_dir = Path("data/v2v")
if not data_dir.exists():
    data_dir = Path("data")

csv_files = glob.glob(str(data_dir / "*.csv"))

if not csv_files:
    print("\n❌ No CSV files found in data directory!")
    print(f"Searched in: {data_dir}")
    exit(1)

print(f"\nFound {len(csv_files)} CSV files")
print(f"Checking first file: {csv_files[0]}")

# Load first file
df = pd.read_csv(csv_files[0], nrows=100)

print("\nColumns in dataset:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Check for anomaly-related columns
anomaly_columns = [col for col in df.columns if any(
    keyword in col.lower()
    for keyword in ['anomaly', 'attack', 'label', 'malicious', 'normal', 'anomalous']
)]

print("\n" + "=" * 60)
if anomaly_columns:
    print("✅ FOUND POTENTIAL ANOMALY LABELS:")
    for col in anomaly_columns:
        print(f"\n  Column: {col}")
        print(f"  Unique values: {df[col].unique()}")
        print("  Value counts:")
        print(df[col].value_counts())
    print("\n✅ You can use these for ground truth evaluation!")
else:
    print("❌ NO ANOMALY LABELS FOUND")
    print("\nYour dataset does NOT have ground truth labels!")
    print("\nOptions:")
    print("  1. Find a labeled V2V dataset (e.g., VeReMi, OMNET++)")
    print("  2. Manually label anomalies in your dataset")
    print("  3. Use synthetic anomaly injection:")
    print("     - Inject random noise in some trajectories")
    print("     - Modify speed/acceleration patterns")
    print("     - Simulate GPS spoofing attacks")
    print("\nWithout labels, you can only evaluate:")
    print("  - Communication efficiency (bandwidth usage)")
    print("  - RL convergence (reward trends)")
    print("  - Relative performance vs baselines")

print("=" * 60)
