#!/usr/bin/env python3
"""
Verify that the system is ready for RL and DFL training.

This script checks:
1. LSTM Autoencoder checkpoint exists
2. Configurations are consistent (50 vehicles)
3. Required dependencies are available
4. GPU is available (if configured)
"""

import sys
from pathlib import Path

import torch
import yaml


def check_checkpoint():
    """Check if LSTM Autoencoder checkpoint exists."""
    checkpoint_path = Path("outputs/checkpoints/best_model.pt")
    
    if not checkpoint_path.exists():
        print("❌ LSTM Autoencoder checkpoint not found!")
        print(f"   Expected: {checkpoint_path}")
        return False
    
    # Try to load it
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"✅ LSTM Autoencoder checkpoint found: {checkpoint_path}")
        print(f"   Size: {checkpoint_path.stat().st_size / 1024**2:.1f} MB")
        
        # Check what's in the checkpoint
        if isinstance(checkpoint, dict):
            print(f"   Keys: {list(checkpoint.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False


def check_config_consistency():
    """Check that configurations are consistent."""
    print("\n📋 Checking configuration consistency...")
    
    configs = {
        "data": "configs/data/v2v_trajectories.yaml",
        "rl_training": "configs/experiment/rl_training.yaml",
        "dfl_training": "configs/experiment/dfl_training.yaml",
    }
    
    num_vehicles = {}
    all_good = True
    
    for name, path in configs.items():
        config_path = Path(path)
        if not config_path.exists():
            print(f"❌ Config not found: {path}")
            all_good = False
            continue
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Extract num_vehicles from different locations
        if name == "data":
            num_veh = config.get("num_vehicles")
        elif name == "rl_training":
            num_veh = config.get("data", {}).get("num_vehicles")
        elif name == "dfl_training":
            num_veh = config.get("data", {}).get("num_vehicles")
        
        num_vehicles[name] = num_veh
        print(f"   {name}: num_vehicles = {num_veh}")
    
    # Check consistency
    unique_values = set(num_vehicles.values())
    if len(unique_values) == 1:
        print(f"✅ All configurations use {list(unique_values)[0]} vehicles")
    else:
        print(f"❌ Inconsistent num_vehicles across configs: {num_vehicles}")
        all_good = False
    
    return all_good


def check_gpu():
    """Check GPU availability."""
    print("\n🖥️  Checking GPU availability...")
    
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU available: {device}")
        print(f"   Memory: {memory:.1f} GB")
        return True
    else:
        print("⚠️  GPU not available, will use CPU (training will be slower)")
        return False


def check_dependencies():
    """Check required dependencies."""
    print("\n📦 Checking dependencies...")
    
    required = [
        "torch",
        "hydra",
        "omegaconf",
        "mlflow",
        "numpy",
        "pandas",
        "tqdm",
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install " + " ".join(missing))
        return False
    
    return True


def check_data():
    """Check if dataset exists."""
    print("\n📊 Checking dataset...")
    
    data_path = Path("data/v2v_dataset_20251017_124247.csv")
    metadata_path = Path("data/metadata_20251017_124247.json")
    
    if not data_path.exists():
        print(f"❌ Dataset not found: {data_path}")
        return False
    
    print(f"✅ Dataset found: {data_path}")
    print(f"   Size: {data_path.stat().st_size / 1024**2:.1f} MB")
    
    if metadata_path.exists():
        print(f"✅ Metadata found: {metadata_path}")
    
    return True


def main():
    """Run all checks."""
    print("=" * 60)
    print("DRL V2V Anomaly Detection - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Checkpoint", check_checkpoint),
        ("Configurations", check_config_consistency),
        ("Dependencies", check_dependencies),
        ("Dataset", check_data),
        ("GPU", check_gpu),
    ]
    
    results = {}
    for name, check_fn in checks:
        results[name] = check_fn()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    if all_passed:
        print("\n🎉 All checks passed! Ready to train RL and DFL.")
        print("\n📝 Next steps:")
        print("   1. Train RL agent: python train_rl.py")
        print("   2. Monitor training: cd outputs/mlruns && mlflow ui")
        print("   3. Train DFL system: python train_dfl.py +experiment=dfl_training")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
