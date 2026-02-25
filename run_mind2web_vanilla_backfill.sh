#!/usr/bin/env bash
# Backfill vanilla cluster experiments:
#   cluster30: test eval only (playbook exists)
#   cluster40: generate data + train + test eval
#   cluster50: test eval only (playbook exists)
# nohup-safe — survives lid close.

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
RESULTS=/Users/leo/Desktop/ace/results

cd /Users/leo/Desktop/ace
echo "Starting vanilla cluster backfill @ $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# cluster30 — test eval only
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== cluster30: test eval ==="
PLAYBOOK30=$RESULTS/mind2web_cluster30/ace_run_20260213_174925_mind2web_cluster30_offline/best_playbook.txt
mkdir -p $RESULTS/mind2web_cluster30_test
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster30 \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK30" \
    --save_path $RESULTS/mind2web_cluster30_test \
    2>&1 | tee $RESULTS/mind2web_cluster30_test.log
echo "cluster30 test done: $(grep 'Final Accuracy' $RESULTS/mind2web_cluster30_test.log | tail -1)"

# ─────────────────────────────────────────────────────────────────────────────
# cluster40 — generate data + train + test eval
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== cluster40: generate data ==="
$PYTHON -m eval.mind2web.cluster_train --clusters 40 \
    2>&1 | tee $RESULTS/cluster40_generate.log

echo ""
echo "=== cluster40: train ==="
mkdir -p $RESULTS/mind2web_cluster40
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster40 \
    --mode offline \
    --skip_initial_test \
    --eval_steps 40 \
    --save_path $RESULTS/mind2web_cluster40 \
    2>&1 | tee $RESULTS/mind2web_cluster40_train.log

echo ""
echo "=== cluster40: test eval ==="
PLAYBOOK40=$(find $RESULTS/mind2web_cluster40 -name "best_playbook.txt" | sort | tail -1)
mkdir -p $RESULTS/mind2web_cluster40_test
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster40 \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK40" \
    --save_path $RESULTS/mind2web_cluster40_test \
    2>&1 | tee $RESULTS/mind2web_cluster40_test.log
echo "cluster40 test done: $(grep 'Final Accuracy' $RESULTS/mind2web_cluster40_test.log | tail -1)"

# ─────────────────────────────────────────────────────────────────────────────
# cluster50 — test eval only
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== cluster50: test eval ==="
PLAYBOOK50=$RESULTS/mind2web_cluster50/ace_run_20260213_192524_mind2web_cluster50_offline/best_playbook.txt
mkdir -p $RESULTS/mind2web_cluster50_test
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster50 \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK50" \
    --save_path $RESULTS/mind2web_cluster50_test \
    2>&1 | tee $RESULTS/mind2web_cluster50_test.log
echo "cluster50 test done: $(grep 'Final Accuracy' $RESULTS/mind2web_cluster50_test.log | tail -1)"

echo ""
echo "🎉 All vanilla cluster backfill complete @ $(date)"
