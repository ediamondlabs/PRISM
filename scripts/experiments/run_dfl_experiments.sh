#!/bin/bash
# Run DFL training experiments with different numbers of vehicles (5, 10, 15)
# Each experiment saves to a different folder

# Create logs directory
mkdir -p logs

# Experiment 1: 5 cars (GPU 0 - both RL and LSTM)
echo "Starting DFL training with 5 cars on GPU 0..."
nohup python train_dfl.py \
    +experiment=dfl_training \
    run_name=dfl_5cars \
    data.num_vehicles=5 \
    simulation.num_vehicles=5 \
    rl.critic.num_vehicles=5 \
    rl.actor.max_neighbors=5 \
    rl.training.buffer_size=1500 \
    device=cuda:0 \
    lstm_device=cuda:0 \
    > logs/dfl_training_5cars_3000.log 2>&1 &

echo "5 cars training started on GPU 0 (PID: $!)"
PID_5CARS=$!

# Wait a bit before starting the next one to avoid resource conflicts
sleep 10

# Experiment 2: 10 cars (GPU 0 - both RL and LSTM)
echo "Starting DFL training with 10 cars on GPU 0..."
nohup python train_dfl.py \
    +experiment=dfl_training \
    run_name=dfl_10cars \
    data.num_vehicles=10 \
    simulation.num_vehicles=10 \
    rl.critic.num_vehicles=10 \
    rl.actor.max_neighbors=10 \
    rl.training.buffer_size=3000 \
    device=cuda:0 \
    lstm_device=cuda:0 \
    > logs/dfl_training_10cars_3000.log 2>&1 &

echo "10 cars training started on GPU 0 (PID: $!)"
PID_10CARS=$!

# Wait a bit before starting the next one
sleep 10

# Experiment 3: 15 cars (GPU 1 - both RL and LSTM)
echo "Starting DFL training with 15 cars on GPU 1..."
nohup python train_dfl.py \
    +experiment=dfl_training \
    run_name=dfl_15cars \
    data.num_vehicles=15 \
    simulation.num_vehicles=15 \
    rl.critic.num_vehicles=15 \
    rl.actor.max_neighbors=15 \
    rl.training.buffer_size=4500 \
    device=cuda:1 \
    lstm_device=cuda:1 \
    > logs/dfl_training_15cars_3000.log 2>&1 &

echo "15 cars training started on GPU 1 (PID: $!)"
PID_15CARS=$!

# Save PIDs to a file for easy monitoring
echo "$PID_5CARS" > logs/dfl_pids.txt
echo "$PID_10CARS" >> logs/dfl_pids.txt
echo "$PID_15CARS" >> logs/dfl_pids.txt

echo ""
echo "========================================"
echo "All 3 DFL experiments started!"
echo "========================================"
echo "5 cars  - PID: $PID_5CARS - GPU 0 - Log: logs/dfl_training_5cars_3000.log"
echo "10 cars - PID: $PID_10CARS - GPU 0 - Log: logs/dfl_training_10cars_3000.log"
echo "15 cars - PID: $PID_15CARS - GPU 1 - Log: logs/dfl_training_15cars_3000.log"
echo ""
echo "Monitor logs with:"
echo "  tail -f logs/dfl_training_5cars_3000.log"
echo "  tail -f logs/dfl_training_10cars_3000.log"
echo "  tail -f logs/dfl_training_15cars_3000.log"
echo ""
echo "Check status with:"
echo "  ps -p $PID_5CARS,$PID_10CARS,$PID_15CARS"
echo ""
echo "Results will be saved to:"
echo "  outputs/dfl_metrics/dfl_5cars/"
echo "  outputs/dfl_metrics/dfl_10cars/"
echo "  outputs/dfl_metrics/dfl_15cars/"
echo "========================================"
