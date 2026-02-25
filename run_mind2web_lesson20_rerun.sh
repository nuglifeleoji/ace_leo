#!/usr/bin/env bash
# Rerun lesson cluster k=20: reselect (seed=123) + retrain + val + test
# Queues after current few-shot + random30 processes.
# nohup-safe — survives lid close.

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
RESULTS=results/mind2web_lesson_cluster
TASK=mind2web_cluster20_lesson_rerun

cd /Users/leo/Desktop/ace
mkdir -p $RESULTS

# ── Wait for few-shot and random30 to finish ──────────────────────────────────
echo "⏳ Waiting for few-shot eval and random30 to finish..."
while pgrep -f "few_shot_eval\|run_mind2web_random30_final" > /dev/null 2>&1; do
    sleep 60
done
echo "✅ Previous experiments done. Starting k=20 lesson rerun @ $(date)"

# ── Step 1: Re-select k=20 with fresh seed (seed=123) ────────────────────────
echo ""
echo "=== Re-selecting k=20 lesson cluster (seed=123, suffix=_rerun) ==="
$PYTHON -m eval.mind2web.cluster_lesson_select \
    --clusters 20 \
    --seed 123 \
    --suffix _rerun \
    2>&1 | tee $RESULTS/cluster20_reselect.log

# ── Step 2: Train ─────────────────────────────────────────────────────────────
echo ""
echo "=== Training: $TASK ==="
mkdir -p $RESULTS/$TASK
$PYTHON -u -m eval.mind2web.run \
    --task_name $TASK \
    --mode offline \
    --skip_initial_test \
    --eval_steps 20 \
    --save_path $RESULTS/$TASK \
    2>&1 | tee $RESULTS/${TASK}_train.log

# ── Step 3: Test eval ─────────────────────────────────────────────────────────
echo ""
echo "=== Test eval: $TASK ==="
PLAYBOOK=$(find $RESULTS/$TASK -name "best_playbook.txt" | sort | tail -1)
if [ -z "$PLAYBOOK" ]; then
    echo "❌ best_playbook.txt not found, skipping test."
else
    echo "Using playbook: $PLAYBOOK"
    mkdir -p $RESULTS/${TASK}_test
    $PYTHON -u -m eval.mind2web.run \
        --task_name $TASK \
        --mode eval_only \
        --initial_playbook_path "$PLAYBOOK" \
        --save_path $RESULTS/${TASK}_test \
        2>&1 | tee $RESULTS/${TASK}_test.log
    echo ""
    echo "🎉 k=20 lesson rerun complete!"
    grep -E "Final Accuracy|best_test_accuracy" $RESULTS/${TASK}_test.log | tail -3
fi
