#!/bin/bash
# Stop all running DFL experiments

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "========================================"
echo "Stop DFL Experiments"
echo "========================================"

if [ -f logs/dfl_pids.txt ]; then
    PIDS=($(cat logs/dfl_pids.txt))

    echo "Stopping processes..."
    echo ""

    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        case $i in
            0) name="5 cars" ;;
            1) name="10 cars" ;;
            2) name="15 cars" ;;
        esac

        if ps -p $pid > /dev/null 2>&1; then
            kill $pid
            echo -e "${GREEN}✓${NC} Stopped $name (PID: $pid)"
        else
            echo -e "${RED}✗${NC} $name (PID: $pid) - Not running"
        fi
    done

    echo ""
    echo "All experiments stopped!"
else
    echo -e "${RED}No PID file found. No experiments to stop.${NC}"
fi

echo "========================================"
