#!/usr/bin/env bash
# Experiment: Does ACE learn better from examples the base LLM got right vs wrong?
#
# Steps:
#   1. Sample 300 training examples, run DeepSeek-V3 zero-shot inference
#   2. Split into correct100 / incorrect100
#   3. Train ACE on each subset + test eval
#
# Uses Together AI (DeepSeek-V3).  nohup-safe — survives lid close.

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
RESULTS=/Users/leo/Desktop/ace/results
API=together
MODEL=deepseek-ai/DeepSeek-V3

cd /Users/leo/Desktop/ace
echo "=== correct vs incorrect experiment @ $(date) ==="

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Sample 300, run DeepSeek-V3 inference, split & register
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 1: Sample 500 & evaluate with DeepSeek-V3 ==="
$PYTHON -u -m eval.mind2web.sample300_split \
    --sample_size 500 \
    --select_n 100 \
    --seed 42 \
    --workers 20 \
    2>&1 | tee $RESULTS/sample500_split.log
echo "Split done @ $(date)"

# Read actual task names from sample_config.json (N may vary)
CORRECT_TASK=$($PYTHON -c "
import json, re
cfg = json.load(open('eval/mind2web/data/sample_config.json'))
names = [k for k in cfg if re.match(r'mind2web_correct\d+$', k)]
print(sorted(names)[-1] if names else 'mind2web_correct100')
")
INCORRECT_TASK=$($PYTHON -c "
import json, re
cfg = json.load(open('eval/mind2web/data/sample_config.json'))
names = [k for k in cfg if re.match(r'mind2web_incorrect\d+$', k)]
print(sorted(names)[-1] if names else 'mind2web_incorrect100')
")
echo "Correct task   : $CORRECT_TASK"
echo "Incorrect task : $INCORRECT_TASK"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Train ACE on correct100
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 2a: ${CORRECT_TASK} — train ==="
rm -rf $RESULTS/${CORRECT_TASK}
mkdir -p $RESULTS/${CORRECT_TASK}
$PYTHON -u -m eval.mind2web.run \
    --task_name ${CORRECT_TASK} \
    --mode offline \
    --skip_initial_test \
    --eval_steps 20 \
    --api_provider $API \
    --generator_model $MODEL \
    --reflector_model $MODEL \
    --curator_model $MODEL \
    --save_path $RESULTS/${CORRECT_TASK} \
    2>&1 | tee $RESULTS/${CORRECT_TASK}_train.log
echo "${CORRECT_TASK} train done @ $(date)"
grep "best_validation_accuracy" $RESULTS/${CORRECT_TASK}_train.log | tail -1

echo ""
echo "=== Step 2b: ${CORRECT_TASK} — test eval ==="
PLAYBOOK_C=$(find $RESULTS/${CORRECT_TASK} -name "best_playbook.txt" | sort | tail -1)
rm -rf $RESULTS/${CORRECT_TASK}_test
mkdir -p $RESULTS/${CORRECT_TASK}_test
$PYTHON -u -m eval.mind2web.run \
    --task_name ${CORRECT_TASK} \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK_C" \
    --api_provider $API \
    --generator_model $MODEL \
    --reflector_model $MODEL \
    --curator_model $MODEL \
    --save_path $RESULTS/${CORRECT_TASK}_test \
    2>&1 | tee $RESULTS/${CORRECT_TASK}_test.log
echo "${CORRECT_TASK} test done: $(grep 'Final Accuracy' $RESULTS/${CORRECT_TASK}_test.log | tail -1)"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Train ACE on incorrect100
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 3a: ${INCORRECT_TASK} — train ==="
rm -rf $RESULTS/${INCORRECT_TASK}
mkdir -p $RESULTS/${INCORRECT_TASK}
$PYTHON -u -m eval.mind2web.run \
    --task_name ${INCORRECT_TASK} \
    --mode offline \
    --skip_initial_test \
    --eval_steps 20 \
    --api_provider $API \
    --generator_model $MODEL \
    --reflector_model $MODEL \
    --curator_model $MODEL \
    --save_path $RESULTS/${INCORRECT_TASK} \
    2>&1 | tee $RESULTS/${INCORRECT_TASK}_train.log
echo "${INCORRECT_TASK} train done @ $(date)"
grep "best_validation_accuracy" $RESULTS/${INCORRECT_TASK}_train.log | tail -1

echo ""
echo "=== Step 3b: ${INCORRECT_TASK} — test eval ==="
PLAYBOOK_IC=$(find $RESULTS/${INCORRECT_TASK} -name "best_playbook.txt" | sort | tail -1)
rm -rf $RESULTS/${INCORRECT_TASK}_test
mkdir -p $RESULTS/${INCORRECT_TASK}_test
$PYTHON -u -m eval.mind2web.run \
    --task_name ${INCORRECT_TASK} \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK_IC" \
    --api_provider $API \
    --generator_model $MODEL \
    --reflector_model $MODEL \
    --curator_model $MODEL \
    --save_path $RESULTS/${INCORRECT_TASK}_test \
    2>&1 | tee $RESULTS/${INCORRECT_TASK}_test.log
echo "${INCORRECT_TASK} test done: $(grep 'Final Accuracy' $RESULTS/${INCORRECT_TASK}_test.log | tail -1)"

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "════════════════════════════════════════════════════"
echo "${CORRECT_TASK}   test: $(grep 'Final Accuracy' $RESULTS/${CORRECT_TASK}_test.log | tail -1)"
echo "${INCORRECT_TASK} test: $(grep 'Final Accuracy' $RESULTS/${INCORRECT_TASK}_test.log | tail -1)"
echo ""
echo "Done @ $(date)"
