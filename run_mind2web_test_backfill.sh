#!/bin/bash
# Run test-set eval_only for already-completed offline experiments
set -e
cd "$(dirname "$0")"

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
TEST_ARGS="--mode eval_only --test_workers 20"
BASE="results/mind2web_cluster10_vs_random"

run_test() {
    local TASK="$1"
    local PLAYBOOK="$2"
    local SAVE="$3"
    echo ""
    echo "=========================================="
    echo " Test eval: ${TASK}"
    echo "=========================================="
    $PYTHON -m eval.mind2web.run \
        --task_name "$TASK" \
        $TEST_ARGS \
        --initial_playbook_path "$PLAYBOOK" \
        --save_path "$SAVE"
}

run_test "mind2web_cluster10" \
    "${BASE}/cluster10/ace_run_20260219_135139_mind2web_cluster10_offline/best_playbook.txt" \
    "${BASE}/cluster10_test"

run_test "mind2web_random10_seed0" \
    "${BASE}/random10_seed0/ace_run_20260219_142051_mind2web_random10_seed0_offline/best_playbook.txt" \
    "${BASE}/random10_seed0_test"

run_test "mind2web_random10_seed1" \
    "${BASE}/random10_seed1/ace_run_20260219_144516_mind2web_random10_seed1_offline/best_playbook.txt" \
    "${BASE}/random10_seed1_test"

run_test "mind2web_random10_seed2" \
    "${BASE}/random10_seed2/ace_run_20260219_161112_mind2web_random10_seed2_offline/best_playbook.txt" \
    "${BASE}/random10_seed2_test"

echo ""
echo "=========================================="
echo " Backfill test evals done!"
echo "=========================================="
