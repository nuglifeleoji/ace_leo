#!/usr/bin/env bash
# =============================================================================
# Resume script for finer_random (K=50 test + K=80 full)
# and finer_lesson (K=80 test only).
# Both crashed due to missing sample_config.json.
# =============================================================================
set -euo pipefail

PYTHON=/workspace/miniconda3/envs/ace311/bin/python
RESULTS=/workspace/ace_leo/results
API=together
MODEL=deepseek-ai/DeepSeek-V3.1

cd /workspace/ace_leo

echo "=================================================================="
echo "  FiNer Resume — random(K=50 test, K=80 full) + lesson(K=80 test)"
echo "  Date: $(date)"
echo "=================================================================="

# ── finer_random K=50: test eval only (training already done) ─────────────
echo ""
echo "── random K=50: test eval ──"
PLAYBOOK_50=$(find "$RESULTS/finer_random_50" -name "best_playbook.txt" | sort | tail -1)
echo "  Playbook: $PLAYBOOK_50"
rm -rf "$RESULTS/finer_random_50_test" && mkdir -p "$RESULTS/finer_random_50_test"
$PYTHON -u -m eval.finance.run \
    --task_name finer \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK_50" \
    --api_provider "$API" \
    --generator_model "$MODEL" \
    --reflector_model "$MODEL" \
    --curator_model   "$MODEL" \
    --save_path "$RESULTS/finer_random_50_test" \
    2>&1 | tee "$RESULTS/finer_random/finer_random50_test.log"
echo "  random K=50 test done: $(grep 'Final Accuracy' "$RESULTS/finer_random/finer_random50_test.log" | tail -1)"

# ── finer_random K=80: full train + test ─────────────────────────────────
echo ""
echo "── random K=80: training ──"
rm -rf "$RESULTS/finer_random_80" && mkdir -p "$RESULTS/finer_random_80"
$PYTHON -u -m eval.finance.run \
    --task_name finer_random80 \
    --mode offline \
    --eval_steps 80 \
    --api_provider "$API" \
    --generator_model "$MODEL" \
    --reflector_model "$MODEL" \
    --curator_model   "$MODEL" \
    --save_path "$RESULTS/finer_random_80" \
    2>&1 | tee "$RESULTS/finer_random/finer_random80_train.log"
echo "  random K=80 training done @ $(date)"

PLAYBOOK_80=$(find "$RESULTS/finer_random_80" -name "best_playbook.txt" | sort | tail -1)
if [ -z "$PLAYBOOK_80" ]; then
    echo "  [WARN] No best_playbook.txt for K=80"
else
    echo "── random K=80: test eval ──"
    rm -rf "$RESULTS/finer_random_80_test" && mkdir -p "$RESULTS/finer_random_80_test"
    $PYTHON -u -m eval.finance.run \
        --task_name finer \
        --mode eval_only \
        --initial_playbook_path "$PLAYBOOK_80" \
        --api_provider "$API" \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model   "$MODEL" \
        --save_path "$RESULTS/finer_random_80_test" \
        2>&1 | tee "$RESULTS/finer_random/finer_random80_test.log"
    echo "  random K=80 test done: $(grep 'Final Accuracy' "$RESULTS/finer_random/finer_random80_test.log" | tail -1)"
fi

# ── finer_lesson K=80: test eval only (training already done) ─────────────
echo ""
echo "── lesson K=80: test eval ──"
PLAYBOOK_L80=$(find "$RESULTS/finer_lesson_80" -name "best_playbook.txt" | sort | tail -1)
echo "  Playbook: $PLAYBOOK_L80"
rm -rf "$RESULTS/finer_lesson_80_test" && mkdir -p "$RESULTS/finer_lesson_80_test"
$PYTHON -u -m eval.finance.run \
    --task_name finer \
    --mode eval_only \
    --initial_playbook_path "$PLAYBOOK_L80" \
    --api_provider "$API" \
    --generator_model "$MODEL" \
    --reflector_model "$MODEL" \
    --curator_model   "$MODEL" \
    --save_path "$RESULTS/finer_lesson_80_test" \
    2>&1 | tee "$RESULTS/finer_lesson_main.log"
echo "  lesson K=80 test done: $(grep 'Final Accuracy' "$RESULTS/finer_lesson_main.log" | tail -1)"

echo ""
echo "=================================================================="
echo "  SUMMARY"
echo "=================================================================="
echo "  finer_random results:"
for K in 5 10 20 30 40 50 80; do
    LOG="$RESULTS/finer_random/finer_random${K}_test.log"
    if [ -f "$LOG" ]; then
        ACC=$(grep "Final Accuracy" "$LOG" | tail -1 || echo "N/A")
        echo "    random${K}: $ACC"
    fi
done
echo ""
echo "  finer_lesson results:"
for K in 5 10 20 30 40 50 80; do
    LOG="$RESULTS/finer_lesson/finer_cluster${K}_lesson_test.log"
    if [ -f "$LOG" ]; then
        ACC=$(grep "Final Accuracy" "$LOG" | tail -1 || echo "N/A")
        echo "    lesson${K}: $ACC"
    fi
done
echo "  Completed @ $(date)"
