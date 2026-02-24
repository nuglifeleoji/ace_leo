#!/bin/bash
# Run random20 seed 4-7 and random30 seed 0-7, with auto test eval after each

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
BASE=results/mind2web_random_more

mkdir -p $BASE

run_one() {
    local NAME=$1
    local K=$2

    echo ""
    echo "============================================================"
    echo " Training: $NAME"
    echo "============================================================"

    mkdir -p $BASE/$NAME
    PYTHONUNBUFFERED=1 $PYTHON -u -m eval.mind2web.run \
        --task_name $NAME \
        --mode offline \
        --skip_initial_test \
        --eval_steps $K \
        --save_path $BASE/$NAME \
        2>&1 | tee $BASE/${NAME}_train.log

    # Get best playbook path
    BEST_RUN=$(ls -td $BASE/$NAME/ace_run_*_offline 2>/dev/null | head -1)
    if [ -z "$BEST_RUN" ]; then
        echo "ERROR: No offline run found for $NAME"
        return
    fi
    PLAYBOOK=$BEST_RUN/best_playbook.txt

    echo ""
    echo "============================================================"
    echo " Test eval: $NAME"
    echo "============================================================"

    mkdir -p $BASE/${NAME}_test
    PYTHONUNBUFFERED=1 $PYTHON -u -m eval.mind2web.run \
        --task_name $NAME \
        --mode eval_only \
        --initial_playbook_path $PLAYBOOK \
        --save_path $BASE/${NAME}_test \
        2>&1 | tee $BASE/${NAME}_test.log

    # Print final test accuracy
    RESULT=$(find $BASE/${NAME}_test -name "final_results.json" | head -1)
    if [ -n "$RESULT" ]; then
        echo ">>> $NAME TEST RESULT:"
        cat $RESULT
    fi
}

# random20 seed 4-7
for SEED in 4 5 6 7; do
    run_one "mind2web_random20_seed${SEED}" 20
done

# random30 seed 0-7
for SEED in 0 1 2 3 4 5 6 7; do
    run_one "mind2web_random30_seed${SEED}" 30
done

echo ""
echo "============================================================"
echo " ALL DONE - Summary"
echo "============================================================"
for NAME in mind2web_random20_seed4 mind2web_random20_seed5 mind2web_random20_seed6 mind2web_random20_seed7 \
            mind2web_random30_seed0 mind2web_random30_seed1 mind2web_random30_seed2 mind2web_random30_seed3 \
            mind2web_random30_seed4 mind2web_random30_seed5 mind2web_random30_seed6 mind2web_random30_seed7; do
    RESULT=$(find $BASE/${NAME}_test -name "final_results.json" 2>/dev/null | head -1)
    if [ -n "$RESULT" ]; then
        ACC=$(cat $RESULT | grep accuracy | grep -oE '[0-9]+\.[0-9]+')
        echo "$NAME: test=$ACC"
    else
        echo "$NAME: no result"
    fi
done
