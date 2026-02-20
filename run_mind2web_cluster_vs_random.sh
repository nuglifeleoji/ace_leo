#!/bin/bash
# ============================================================
# Experiment: Cluster vs Random sampling for Mind2Web (10 samples)
# 5 random seeds (seed=0..4) vs cluster10
# Each experiment: offline training → eval_only on test set
# ============================================================
set -e

cd "$(dirname "$0")"

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
TRAIN_ARGS="--mode offline --skip_initial_test --eval_steps 10"
TEST_ARGS="--mode eval_only --test_workers 20"

# ── Helper: find the latest run dir and run eval_only on test set ──
run_test_eval() {
    local TASK_NAME="$1"
    local SAVE_PATH="$2"

    # Find the most recently created run directory
    local RUN_DIR
    RUN_DIR=$(ls -td "${SAVE_PATH}"/ace_run_* 2>/dev/null | head -1)

    if [ -z "$RUN_DIR" ]; then
        echo "  [WARNING] No run directory found in ${SAVE_PATH}, skipping test eval."
        return
    fi

    local PLAYBOOK="${RUN_DIR}/best_playbook.txt"
    if [ ! -f "$PLAYBOOK" ]; then
        echo "  [WARNING] best_playbook.txt not found in ${RUN_DIR}, skipping test eval."
        return
    fi

    echo ""
    echo "  → Running test eval for ${TASK_NAME}"
    echo "    Playbook: ${PLAYBOOK}"
    $PYTHON -m eval.mind2web.run \
        --task_name "$TASK_NAME" \
        $TEST_ARGS \
        --initial_playbook_path "$PLAYBOOK" \
        --save_path "${SAVE_PATH}_test"
}

echo "============================================================"
echo " Mind2Web: Cluster-10 vs Random-10 (5 seeds)"
echo " [offline train] + [eval_only test] for each experiment"
echo "============================================================"

# ---------- Cluster-10 ----------
echo ""
echo "[1/6] Training mind2web_cluster10 ..."
$PYTHON -m eval.mind2web.run \
    --task_name mind2web_cluster10 \
    $TRAIN_ARGS \
    --save_path results/mind2web_cluster10_vs_random/cluster10
run_test_eval "mind2web_cluster10" "results/mind2web_cluster10_vs_random/cluster10"

# ---------- 5 × Random-10 ----------
for SEED in 0 1 2 3 4; do
    TASK="mind2web_random10_seed${SEED}"
    SAVE="results/mind2web_cluster10_vs_random/random10_seed${SEED}"
    N=$((SEED + 2))
    echo ""
    echo "[${N}/6] Training ${TASK} ..."
    $PYTHON -m eval.mind2web.run \
        --task_name "$TASK" \
        $TRAIN_ARGS \
        --save_path "$SAVE"
    run_test_eval "$TASK" "$SAVE"
done

echo ""
echo "============================================================"
echo " All experiments done!"
echo " Results in: results/mind2web_cluster10_vs_random/"
echo "============================================================"

# ---------- Summary ----------
echo ""
echo "============================================================"
echo " SUMMARY (best_validation_accuracy from each run)"
echo "============================================================"
for DIR in \
    results/mind2web_cluster10_vs_random/cluster10 \
    results/mind2web_cluster10_vs_random/random10_seed0 \
    results/mind2web_cluster10_vs_random/random10_seed1 \
    results/mind2web_cluster10_vs_random/random10_seed2 \
    results/mind2web_cluster10_vs_random/random10_seed3 \
    results/mind2web_cluster10_vs_random/random10_seed4; do
    RUN_DIR=$(ls -td "${DIR}"/ace_run_* 2>/dev/null | head -1)
    if [ -n "$RUN_DIR" ] && [ -f "${RUN_DIR}/final_results.json" ]; then
        VAL_ACC=$(python3 -c "import json; d=json.load(open('${RUN_DIR}/final_results.json')); print(d.get('best_validation_accuracy', d.get('training_results', {}).get('best_validation_accuracy', 'N/A')))" 2>/dev/null || echo "N/A")
        echo "  $(basename $DIR): val_acc=${VAL_ACC}"
    fi
    # Also check test result
    TEST_RUN=$(ls -td "${DIR}_test"/ace_run_* 2>/dev/null | head -1)
    if [ -n "$TEST_RUN" ] && [ -f "${TEST_RUN}/final_results.json" ]; then
        TEST_ACC=$(python3 -c "import json; d=json.load(open('${TEST_RUN}/final_results.json')); print(d.get('accuracy', 'N/A'))" 2>/dev/null || echo "N/A")
        echo "    → test_acc=${TEST_ACC}"
    fi
done
echo "============================================================"
