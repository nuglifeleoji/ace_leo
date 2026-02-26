#!/usr/bin/env bash
# Backfill vanilla cluster experiments:
#   cluster40: generate data + train + test eval  (Together API)
#   cluster50: test eval only (playbook exists)   (Together API)
# cluster30 is already done (test=0.350).
# nohup-safe — survives lid close.

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
RESULTS=/Users/leo/Desktop/ace/results
API=together
MODEL=deepseek-ai/DeepSeek-V3

cd /Users/leo/Desktop/ace
echo "Starting vanilla cluster backfill (Together API) @ $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# cluster40 — generate data + train + test eval
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== cluster40: generate data ==="
$PYTHON -m eval.mind2web.cluster_train --clusters 40 \
    2>&1 | tee $RESULTS/cluster40_generate.log

echo ""
echo "=== cluster40: train ==="
rm -rf $RESULTS/mind2web_cluster40
mkdir -p $RESULTS/mind2web_cluster40
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster40 \
    --mode offline \
    --skip_initial_test \
    --eval_steps 40 \
    --api_provider $API \
    --generator_model $MODEL \
    --reflector_model $MODEL \
    --curator_model $MODEL \
    --save_path $RESULTS/mind2web_cluster40 \
    2>&1 | tee $RESULTS/mind2web_cluster40_train.log

echo ""
echo "=== cluster40: test eval ==="
PLAYBOOK40=$(find $RESULTS/mind2web_cluster40 -name "best_playbook.txt" | sort | tail -1)
rm -rf $RESULTS/mind2web_cluster40_test
mkdir -p $RESULTS/mind2web_cluster40_test
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster40 \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK40" \
    --api_provider $API \
    --generator_model $MODEL \
    --reflector_model $MODEL \
    --curator_model $MODEL \
    --save_path $RESULTS/mind2web_cluster40_test \
    2>&1 | tee $RESULTS/mind2web_cluster40_test.log
echo "cluster40 test done: $(grep 'Final Accuracy' $RESULTS/mind2web_cluster40_test.log | tail -1)"

# ─────────────────────────────────────────────────────────────────────────────
# cluster50 — test eval only
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== cluster50: test eval ==="
PLAYBOOK50=$RESULTS/mind2web_cluster50/ace_run_20260213_192524_mind2web_cluster50_offline/best_playbook.txt
rm -rf $RESULTS/mind2web_cluster50_test
mkdir -p $RESULTS/mind2web_cluster50_test
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_cluster50 \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK50" \
    --api_provider $API \
    --generator_model $MODEL \
    --reflector_model $MODEL \
    --curator_model $MODEL \
    --save_path $RESULTS/mind2web_cluster50_test \
    2>&1 | tee $RESULTS/mind2web_cluster50_test.log
echo "cluster50 test done: $(grep 'Final Accuracy' $RESULTS/mind2web_cluster50_test.log | tail -1)"

echo ""
echo "🎉 All vanilla cluster backfill complete @ $(date)"
