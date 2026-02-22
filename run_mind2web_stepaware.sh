#!/bin/bash
# ============================================================
# Experiment: Step-Position-Aware Clustering for Mind2Web
# Tests: cluster{k}_stepaware10 for k=10,15,20 (pos_weight=1.0)
# Each run: training + automatic test eval
# ============================================================
set -e

cd "$(dirname "$0")"

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
COMMON_EVAL_ARGS="--mode eval_only"

echo "============================================================"
echo " Mind2Web: Step-Aware Cluster (pos_weight=1.0)"
echo " k = 10, 15, 20"
echo "============================================================"

run_experiment() {
    local TASK_NAME=$1
    local K=$2
    local SAVE_DIR=$3

    echo ""
    echo "------------------------------------------------------------"
    echo " Training: ${TASK_NAME} (k=${K})"
    echo "------------------------------------------------------------"
    $PYTHON -m eval.mind2web.run \
        --task_name "$TASK_NAME" \
        --mode offline --skip_initial_test --eval_steps "$K" \
        --save_path "$SAVE_DIR"

    LATEST_RUN_DIR=$(ls -td "$SAVE_DIR"/ace_run_* | head -1)
    if [ -z "$LATEST_RUN_DIR" ]; then
        echo "Error: Could not find latest run directory in $SAVE_DIR"
        exit 1
    fi

    PLAYBOOK_PATH="${LATEST_RUN_DIR}/best_playbook.txt"
    if [ ! -f "$PLAYBOOK_PATH" ]; then
        echo "Error: Playbook not found at $PLAYBOOK_PATH"
        exit 1
    fi

    echo ""
    echo " Test eval: ${TASK_NAME}"
    $PYTHON -m eval.mind2web.run \
        --task_name "$TASK_NAME" \
        $COMMON_EVAL_ARGS \
        --initial_playbook_path "$PLAYBOOK_PATH" \
        --save_path "${SAVE_DIR}_test"

    echo " Done: ${TASK_NAME}"
}

BASE_DIR="results/mind2web_stepaware"

run_experiment "mind2web_cluster10_stepaware10" 10 "${BASE_DIR}/cluster10_stepaware10"
run_experiment "mind2web_cluster15_stepaware10" 15 "${BASE_DIR}/cluster15_stepaware10"
run_experiment "mind2web_cluster20_stepaware10" 20 "${BASE_DIR}/cluster20_stepaware10"

echo ""
echo "============================================================"
echo " All done! Results: ${BASE_DIR}/"
echo "============================================================"
