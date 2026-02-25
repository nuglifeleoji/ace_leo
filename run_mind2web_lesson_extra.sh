#!/usr/bin/env bash
# Run lesson cluster k=10 (retrain) and k=50 (new), val + test
# Waits for k=40 test eval to finish first, then runs.
set -e

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
RESULTS=results/mind2web_lesson_cluster
LOG_DIR=$RESULTS

cd /Users/leo/Desktop/ace
mkdir -p $RESULTS

# ── Wait for k=40 test to finish ──────────────────────────────────────────────
echo "⏳ Waiting for k=40 test eval to finish..."
while pgrep -f "cluster40_lesson.*eval_only" > /dev/null 2>&1 || \
      pgrep -f "run_mind2web_lesson_test" > /dev/null 2>&1; do
    sleep 30
done
echo "✅ k=40 test done. Starting k=10 retrain and k=50."

# ── Step 1: Generate k=50 lesson cluster data ─────────────────────────────────
echo ""
echo "=== Generating k=50 lesson cluster selection ==="
$PYTHON -m eval.mind2web.cluster_lesson_select --clusters 50 \
    2>&1 | tee $LOG_DIR/cluster50_select.log

# ── Step 2: Train k=10 ────────────────────────────────────────────────────────
echo ""
echo "=== Training: mind2web_cluster10_lesson ==="
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster10_lesson \
    --mode offline --skip_initial_test --eval_steps 40 \
    --save_path $RESULTS/mind2web_cluster10_lesson \
    2>&1 | tee $LOG_DIR/mind2web_cluster10_lesson_train.log

# ── Step 3: Test k=10 ─────────────────────────────────────────────────────────
echo ""
echo "=== Test eval: mind2web_cluster10_lesson ==="
PLAYBOOK_10=$(find $RESULTS/mind2web_cluster10_lesson -name "best_playbook.txt" | sort | tail -1)
echo "Using playbook: $PLAYBOOK_10"
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster10_lesson \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK_10" \
    --save_path $RESULTS/mind2web_cluster10_lesson_test \
    2>&1 | tee $LOG_DIR/mind2web_cluster10_lesson_test.log

# ── Step 4: Train k=50 ────────────────────────────────────────────────────────
echo ""
echo "=== Training: mind2web_cluster50_lesson ==="
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster50_lesson \
    --mode offline --skip_initial_test --eval_steps 40 \
    --save_path $RESULTS/mind2web_cluster50_lesson \
    2>&1 | tee $LOG_DIR/mind2web_cluster50_lesson_train.log

# ── Step 5: Test k=50 ─────────────────────────────────────────────────────────
echo ""
echo "=== Test eval: mind2web_cluster50_lesson ==="
PLAYBOOK_50=$(find $RESULTS/mind2web_cluster50_lesson -name "best_playbook.txt" | sort | tail -1)
echo "Using playbook: $PLAYBOOK_50"
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster50_lesson \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK_50" \
    --save_path $RESULTS/mind2web_cluster50_lesson_test \
    2>&1 | tee $LOG_DIR/mind2web_cluster50_lesson_test.log

echo ""
echo "🎉 All done! k=10 and k=50 lesson cluster experiments complete."
