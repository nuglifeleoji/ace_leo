#!/usr/bin/env bash
# Retrain k=10 lesson cluster with correct --eval_steps 10
# Waits for k=50 test to finish first
set -e

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
RESULTS=results/mind2web_lesson_cluster

cd /Users/leo/Desktop/ace

# ── Wait for k=50 test to finish ─────────────────────────────────────────────
echo "⏳ Waiting for k=50 test eval to finish..."
while pgrep -f "cluster50_lesson.*eval_only\|run_mind2web_lesson_extra" > /dev/null 2>&1; do
    sleep 30
done
sleep 10  # extra buffer
echo "✅ k=50 done. Starting k=10 retrain."

# ── Train k=10 (with correct eval_steps=10) ──────────────────────────────────
echo ""
echo "=== Training: mind2web_cluster10_lesson (eval_steps=10) ==="
rm -rf $RESULTS/mind2web_cluster10_lesson  # clean old broken run
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster10_lesson \
    --mode offline --skip_initial_test --eval_steps 10 \
    --save_path $RESULTS/mind2web_cluster10_lesson \
    2>&1 | tee $RESULTS/mind2web_cluster10_lesson_train2.log

# ── Test k=10 ────────────────────────────────────────────────────────────────
echo ""
echo "=== Test eval: mind2web_cluster10_lesson ==="
PLAYBOOK=$(find $RESULTS/mind2web_cluster10_lesson -name "best_playbook.txt" | sort | tail -1)
echo "Using playbook: $PLAYBOOK"
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster10_lesson \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK" \
    --save_path $RESULTS/mind2web_cluster10_lesson_test2 \
    2>&1 | tee $RESULTS/mind2web_cluster10_lesson_test2.log

echo ""
echo "🎉 k=10 retrain and test done!"
