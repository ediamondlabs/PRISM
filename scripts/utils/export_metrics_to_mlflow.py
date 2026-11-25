"""Export metrics from old MLflow format to a clean new instance."""
import mlflow
import os

# Set up the new MLflow tracking URI
mlflow.set_tracking_uri("file:///Users/georgelevis/Projects/DRL v2v anomaly detection/clean_mlruns")

# Read metrics from the old runs
old_runs_base = "/Users/georgelevis/Projects/DRL v2v anomaly detection/outputs/mlruns/333775136591270520"

run_ids = [
    "1e5a637dfe9542e4aa47bff56b3e6730",
    "5fb98b1b5ac641a194d391da11bdfad2"
]

# Create experiment
mlflow.set_experiment("dfl_training")

for i, run_id in enumerate(run_ids):
    print(f"Exporting run {i+1}/{len(run_ids)}: {run_id}")
    
    run_dir = os.path.join(old_runs_base, run_id)
    metrics_dir = os.path.join(run_dir, "metrics")
    
    if not os.path.exists(metrics_dir):
        print("  Metrics directory not found, skipping")
        continue
    
    # Start a new MLflow run
    with mlflow.start_run(run_name=f"dfl_training_run_{i+1}"):
        # Log basic params
        mlflow.log_param("original_run_id", run_id)
        mlflow.log_param("device", "mps")
        mlflow.log_param("seed", 42)
        
        # Read and log all metrics
        for metric_file in os.listdir(metrics_dir):
            metric_path = os.path.join(metrics_dir, metric_file)
            if os.path.isfile(metric_path):
                print(f"  Processing metric: {metric_file}")
                with open(metric_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            timestamp, value, step = parts
                            mlflow.log_metric(metric_file, float(value), step=int(step))
        
        print("  ✓ Run exported successfully")

print("\n✅ All metrics exported to clean_mlruns/")
print("Run: mlflow ui --backend-store-uri file:///Users/georgelevis/Projects/DRL\\ v2v\\ anomaly\\ detection/clean_mlruns --port 5001")





