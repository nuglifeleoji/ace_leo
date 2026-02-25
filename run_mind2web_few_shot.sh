#!/usr/bin/env bash
# Few-shot ICL baseline for Mind2Web (0-shot, 5-shot, 10-shot)
# Waits for k=10 lesson retrain to finish, then runs.
set -e

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
cd /Users/leo/Desktop/ace
mkdir -p results/mind2web_few_shot

# ── Wait for k=10 retrain to finish ──────────────────────────────────────────
echo "⏳ Waiting for k=10 lesson retrain to finish..."
while pgrep -f "run_mind2web_lesson_k10_retrain\|cluster10_lesson.*offline" > /dev/null 2>&1; do
    sleep 60
done
echo "✅ k=10 retrain done. Starting few-shot eval."

# ── Run few-shot evaluation: 0, 5, 10 shot × 3 seeds ─────────────────────────
echo ""
echo "=== Mind2Web Few-Shot ICL Evaluation ==="
echo "Shots: 0, 5, 10 | Seeds: 3 | Model: DeepSeek-V3.1"
$PYTHON -u -m eval.mind2web.few_shot_eval \
    --shots 0 5 10 \
    --seeds 3 \
    --model DeepSeek-V3.1 \
    --api_provider sambanova \
    --max_workers 20 \
    --save_path results/mind2web_few_shot \
    2>&1 | tee results/mind2web_few_shot/few_shot_run.log

echo ""
echo "🎉 Few-shot evaluation complete! Results in results/mind2web_few_shot/"
