#!/bin/bash
# Test evaluation for lesson cluster k=15,20,30,40

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
BASE=results/mind2web_lesson_cluster

run_test() {
    local NAME=$1
    local PLAYBOOK=$2
    echo ""
    echo "============================================================"
    echo " Test eval: $NAME"
    echo "============================================================"
    PYTHONUNBUFFERED=1 $PYTHON -u -m eval.mind2web.run \
        --task_name $NAME \
        --mode eval_only \
        --initial_playbook_path "$PLAYBOOK" \
        --save_path $BASE/${NAME}_test \
        2>&1 | tee $BASE/${NAME}_test.log
    ACC=$(grep -oE "Accuracy: [0-9]+\.[0-9]+" $BASE/${NAME}_test.log | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    echo ">>> $NAME  test_acc=$ACC"
}

run_test "mind2web_cluster15_lesson" \
    "$BASE/mind2web_cluster15_lesson/ace_run_20260224_080649_mind2web_cluster15_lesson_offline/best_playbook.txt"

run_test "mind2web_cluster20_lesson" \
    "$BASE/mind2web_cluster20_lesson/ace_run_20260224_110136_mind2web_cluster20_lesson_offline/best_playbook.txt"

run_test "mind2web_cluster30_lesson" \
    "$BASE/mind2web_cluster30_lesson/ace_run_20260224_114740_mind2web_cluster30_lesson_offline/best_playbook.txt"

run_test "mind2web_cluster40_lesson" \
    "$BASE/mind2web_cluster40_lesson/ace_run_20260224_133646_mind2web_cluster40_lesson_offline/best_playbook.txt"

echo ""
echo "All lesson cluster test evals done."
