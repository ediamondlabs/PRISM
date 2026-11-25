#!/bin/bash
# Monitor DFL training experiments

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "DFL Experiments Monitor"
echo "========================================"

# Check if PIDs file exists
if [ -f logs/dfl_pids.txt ]; then
    PIDS=($(cat logs/dfl_pids.txt))

    echo ""
    echo "Process Status:"
    echo "---------------"

    # 5 cars
    if ps -p ${PIDS[0]} > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} 5 cars  (PID: ${PIDS[0]}) - RUNNING"
    else
        echo -e "${RED}✗${NC} 5 cars  (PID: ${PIDS[0]}) - STOPPED"
    fi

    # 10 cars
    if ps -p ${PIDS[1]} > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} 10 cars (PID: ${PIDS[1]}) - RUNNING"
    else
        echo -e "${RED}✗${NC} 10 cars (PID: ${PIDS[1]}) - STOPPED"
    fi

    # 15 cars
    if ps -p ${PIDS[2]} > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} 15 cars (PID: ${PIDS[2]}) - RUNNING"
    else
        echo -e "${RED}✗${NC} 15 cars (PID: ${PIDS[2]}) - STOPPED"
    fi
else
    echo -e "${YELLOW}No PID file found. Experiments may not be running.${NC}"
fi

echo ""
echo "Latest Progress:"
echo "----------------"

# Show last few lines from each log
for log in logs/dfl_training_*cars_3000.log; do
    if [ -f "$log" ]; then
        cars=$(echo $log | grep -oP '\d+(?=cars)')
        echo ""
        echo -e "${YELLOW}${cars} cars:${NC}"
        # Get last line that contains "Episode" or "F1"
        tail -n 50 "$log" | grep -E "Episode|F1:" | tail -n 1 || echo "No progress yet"
    fi
done

echo ""
echo "========================================"
echo "GPU Usage:"
echo "----------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "GPU %s: %s%% | Memory: %s/%s MB\n", $1, $3, $4, $5}'

echo ""
echo "========================================"
echo "Commands:"
echo "  View logs:    tail -f logs/dfl_training_<N>cars_3000.log"
echo "  Kill process: kill <PID>"
echo "  Kill all:     cat logs/dfl_pids.txt | xargs kill"
echo "========================================"
