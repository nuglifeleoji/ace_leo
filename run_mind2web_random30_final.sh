#!/usr/bin/env bash
# Run random30 seeds 0-7: train + val + test
# Waits for current experiments (k10 retrain + few-shot) to finish first.
# Uses nohup — survives lid close.

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
BASE=results/mind2web_random_more

cd /Users/leo/Desktop/ace
mkdir -p $BASE

# ── Wait for k=10 retrain and few-shot eval to finish ─────────────────────────
echo "⏳ Waiting for k=10 retrain and few-shot eval to finish..."
while pgrep -f "run_mind2web_lesson_k10_retrain\|run_mind2web_few_shot\|cluster10_lesson" > /dev/null 2>&1; do
    sleep 60
done
echo "✅ All previous experiments done. Starting random30."

# ── Helper: train + test one seed ─────────────────────────────────────────────
run_one() {
    local NAME=$1

    echo ""
    echo "============================================================"
    echo " Training: $NAME"
    echo "============================================================"
    mkdir -p $BASE/$NAME

    PYTHONUNBUFFERED=1 $PYTHON -u -m eval.mind2web.run \
        --task_name $NAME \
        --mode offline \
        --skip_initial_test \
        --eval_steps 30 \
        --save_path $BASE/$NAME \
        2>&1 | tee $BASE/${NAME}_train.log

    # Find best playbook
    PLAYBOOK=$(find $BASE/$NAME -name "best_playbook.txt" | sort | tail -1)
    if [ -z "$PLAYBOOK" ]; then
        echo "ERROR: No best_playbook.txt found for $NAME — skipping test."
        return
    fi

    echo ""
    echo "============================================================"
    echo " Test eval: $NAME"
    echo "============================================================"
    mkdir -p $BASE/${NAME}_test

    PYTHONUNBUFFERED=1 $PYTHON -u -m eval.mind2web.run \
        --task_name $NAME \
        --mode eval_only \
        --initial_playbook_path "$PLAYBOOK" \
        --save_path $BASE/${NAME}_test \
        2>&1 | tee $BASE/${NAME}_test.log

    RESULT=$(find $BASE/${NAME}_test -name "final_results.json" | head -1)
    if [ -n "$RESULT" ]; then
        ACC=$(python3 -c "import json,sys; d=json.load(open('$RESULT')); print(d.get('test_results',{}).get('accuracy','N/A'))" 2>/dev/null)
        echo ">>> $NAME TEST ACC: $ACC"
    fi
}

# ── Run seeds 0-7 ─────────────────────────────────────────────────────────────
for SEED in 0 1 2 3 4 5 6 7; do
    run_one "mind2web_random30_seed${SEED}"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " ALL DONE — random30 Summary"
echo "============================================================"
for SEED in 0 1 2 3 4 5 6 7; do
    NAME="mind2web_random30_seed${SEED}"
    RESULT=$(find $BASE/${NAME}_test -name "final_results.json" 2>/dev/null | head -1)
    if [ -n "$RESULT" ]; then
        ACC=$(python3 -c "import json; d=json.load(open('$RESULT')); print(d.get('test_results',{}).get('accuracy','N/A'))" 2>/dev/null)
        echo "  $NAME: $ACC"
    else
        echo "  $NAME: no result"
    fi
done
