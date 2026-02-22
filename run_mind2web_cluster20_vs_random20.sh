#!/bin/bash
# ============================================================
# Experiment: Cluster-20 vs Random-20 for Mind2Web
# 4 random seeds (seed=0..3) vs cluster20
# ============================================================
set -e

cd "$(dirname "$0")"

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
COMMON_TRAIN_ARGS="--mode offline --skip_initial_test --eval_steps 20"
COMMON_EVAL_ARGS="--mode eval_only"

echo "============================================================"
echo " Mind2Web: Cluster-20 vs Random-20 (4 seeds)"
echo "============================================================"

run_experiment() {
    local TASK_NAME=$1
    local SAVE_DIR=$2

    echo ""
    echo "------------------------------------------------------------"
    echo " Training: ${TASK_NAME}"
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
    echo " Test eval: ${TASK_NAME}"
    $PYTHON -m eval.mind2web.run \
        --task_name "$TASK_NAME" \
        $COMMON_EVAL_ARGS \
        --initial_playbook_path "$PLAYBOOK_PATH" \
        --save_path "${SAVE_DIR}_test"

    echo " Done: ${TASK_NAME}"
}

BASE_DIR="results/mind2web_cluster20_vs_random20"

# ---------- Cluster-20 (baseline) ----------
run_experiment "mind2web_cluster20" "${BASE_DIR}/cluster20"

# ---------- 4 × Random-20 ----------
for SEED in 0 1 2 3; do
    run_experiment "mind2web_random20_seed${SEED}" "${BASE_DIR}/random20_seed${SEED}"
done

echo ""
echo "============================================================"
echo " All experiments done!"
echo " Results: ${BASE_DIR}/"
echo "============================================================"
