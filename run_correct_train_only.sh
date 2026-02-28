#!/usr/bin/env bash
# Train ACE on correct100 and incorrect100 (data + config already ready)
PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
RESULTS=/Users/leo/Desktop/ace/results
API=together
MODEL=deepseek-ai/DeepSeek-V3
cd /Users/leo/Desktop/ace

echo "=== correct100 vs incorrect100 training @ $(date) ==="

# ── correct100 ────────────────────────────────────────────────────────────────
echo ""
echo "=== mind2web_correct100: train ==="
rm -rf $RESULTS/mind2web_correct100
mkdir -p $RESULTS/mind2web_correct100
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_correct100 \
    --mode offline \
    --skip_initial_test \
    --eval_steps 20 \
    --api_provider $API \
    --generator_model $MODEL \
    --reflector_model $MODEL \
    --curator_model $MODEL \
    --save_path $RESULTS/mind2web_correct100 \
    2>&1 | tee $RESULTS/mind2web_correct100_train.log
echo "correct100 train done @ $(date)"
grep "best_validation_accuracy" $RESULTS/mind2web_correct100_train.log | tail -1

echo ""
echo "=== mind2web_correct100: test eval ==="
PLAYBOOK_C=$(find $RESULTS/mind2web_correct100 -name "best_playbook.txt" | sort | tail -1)
rm -rf $RESULTS/mind2web_correct100_test
mkdir -p $RESULTS/mind2web_correct100_test
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_correct100 \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK_C" \
    --api_provider $API \
    --generator_model $MODEL \
    --reflector_model $MODEL \
    --curator_model $MODEL \
    --save_path $RESULTS/mind2web_correct100_test \
    2>&1 | tee $RESULTS/mind2web_correct100_test.log
echo "correct100 test done: $(grep 'Final Accuracy' $RESULTS/mind2web_correct100_test.log | tail -1)"

# ── incorrect100 ──────────────────────────────────────────────────────────────
echo ""
echo "=== mind2web_incorrect100: train ==="
rm -rf $RESULTS/mind2web_incorrect100
mkdir -p $RESULTS/mind2web_incorrect100
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_incorrect100 \
    --mode offline \
    --skip_initial_test \
    --eval_steps 20 \
    --api_provider $API \
    --generator_model $MODEL \
    --reflector_model $MODEL \
    --curator_model $MODEL \
    --save_path $RESULTS/mind2web_incorrect100 \
    2>&1 | tee $RESULTS/mind2web_incorrect100_train.log
echo "incorrect100 train done @ $(date)"
grep "best_validation_accuracy" $RESULTS/mind2web_incorrect100_train.log | tail -1

echo ""
echo "=== mind2web_incorrect100: test eval ==="
PLAYBOOK_IC=$(find $RESULTS/mind2web_incorrect100 -name "best_playbook.txt" | sort | tail -1)
rm -rf $RESULTS/mind2web_incorrect100_test
mkdir -p $RESULTS/mind2web_incorrect100_test
$PYTHON -u -m eval.mind2web.run \
    --task_name mind2web_incorrect100 \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK_IC" \
    --api_provider $API \
    --generator_model $MODEL \
    --reflector_model $MODEL \
    --curator_model $MODEL \
    --save_path $RESULTS/mind2web_incorrect100_test \
    2>&1 | tee $RESULTS/mind2web_incorrect100_test.log
echo "incorrect100 test done: $(grep 'Final Accuracy' $RESULTS/mind2web_incorrect100_test.log | tail -1)"

echo ""
echo "════════════════ SUMMARY ════════════════"
echo "correct100  : $(grep 'Final Accuracy' $RESULTS/mind2web_correct100_test.log | tail -1)"
echo "incorrect100: $(grep 'Final Accuracy' $RESULTS/mind2web_incorrect100_test.log | tail -1)"
echo "Done @ $(date)"
