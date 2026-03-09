#!/bin/bash
# Launch Mind2Web curriculum learning experiments
# Run AFTER eval/mind2web/prepare_data.py has completed.
#
# Usage:
#   cd /workspace/ace_leo
#   source .env
#   bash run_mind2web_curriculum.sh

set -e
cd /workspace/ace_leo
source .env 2>/dev/null || true

mkdir -p results/mind2web_curriculum
LOG_DIR="results/mind2web_curriculum"

echo "========================================"
echo "Mind2Web Curriculum Learning Experiments"
echo "========================================"

SELECTORS=(random phased easy_first thompson stratified ucb_cat)

for sel in "${SELECTORS[@]}"; do
    echo "Launching: $sel"
    nohup python -m eval.mind2web.run_curriculum \
        --selector "$sel" \
        --budget 200 \
        --eval_every 25 \
        > "${LOG_DIR}/run_${sel}.log" 2>&1 &
    echo "  PID: $! → ${LOG_DIR}/run_${sel}.log"
    sleep 2
done

echo ""
echo "All experiments launched. Monitor with:"
echo "  tail -f results/mind2web_curriculum/run_*.log"
