#!/bin/bash
set -e
cd "$(dirname "$0")"
PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
COMMON_EVAL_ARGS="--mode eval_only"

echo "============================================================"
echo " Mind2Web: Step-Aware Cluster (pos_weight=1.0)"
echo " k = 30, 50"
echo "============================================================"

run_experiment() {
    local K=$1
    local POS_WEIGHT_STR=$2
    local TASK_NAME="mind2web_cluster${K}_stepaware${POS_WEIGHT_STR}"
    local SAVE_DIR="results/mind2web_stepaware/${TASK_NAME}"
    local COMMON_TRAIN_ARGS="--mode offline --skip_initial_test --eval_steps ${K}"

    echo ""
    echo "------------------------------------------------------------"
    echo " Training: ${TASK_NAME} (k=${K})"
    echo "------------------------------------------------------------"
    $PYTHON -m eval.mind2web.run \
        --task_name "$TASK_NAME" \
        $COMMON_TRAIN_ARGS \
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
    echo " Test eval: ${TASK_NAME} (playbook from ${LATEST_RUN_DIR})"
    $PYTHON -m eval.mind2web.run \
        --task_name "$TASK_NAME" \
        $COMMON_EVAL_ARGS \
        --initial_playbook_path "$PLAYBOOK_PATH" \
        --save_path "${SAVE_DIR}_test"
}

run_experiment 30 "10"
run_experiment 50 "10"

echo ""
echo "============================================================"
echo " All step-aware k=30,50 experiments done!"
echo "============================================================"
