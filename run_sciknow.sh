#!/usr/bin/env bash
# =============================================================================
# SciKnowEval Chemistry L3 MCQ – ACE Experiments
#   1. Baseline LLM  (eval_only, no playbook)
#   2. ACE Full Train (offline, all 1000 train samples)
#
# max_tokens=2048 to avoid response truncation on chemistry questions.
# =============================================================================
set -euo pipefail

PYTHON=/workspace/miniconda3/envs/ace311/bin/python
API=together
MODEL=deepseek-ai/DeepSeek-V3.1
TASK=sciknow_chem_l3

cd /workspace/ace_leo
mkdir -p results/sciknow_baseline results/sciknow_full_train

echo "=================================================================="
echo "  SciKnowEval Chemistry L3 MCQ"
  echo "  Model: $MODEL  API: $API  max_tokens: 4096"
echo "  Date: $(date)"
echo "=================================================================="

# ── 1. Baseline LLM ────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Baseline LLM (no playbook) ──"
$PYTHON -u -m eval.sciknow.run \
    --task_name   "$TASK" \
    --mode        eval_only \
    --api_provider "$API" \
    --generator_model "$MODEL" \
    --reflector_model "$MODEL" \
    --curator_model   "$MODEL" \
    --max_tokens  4096 \
    --test_workers 20 \
    --save_path   results/sciknow_baseline \
    2>&1 | tee results/sciknow_baseline.log

echo "Baseline done @ $(date)"
grep "Final Accuracy" results/sciknow_baseline.log | tail -1

# ── 2. ACE Full Train ───────────────────────────────────────────────────────
echo ""
echo "── Step 2: ACE Full Train ──"
$PYTHON -u -m eval.sciknow.run \
    --task_name   "$TASK" \
    --mode        offline \
    --api_provider "$API" \
    --generator_model "$MODEL" \
    --reflector_model "$MODEL" \
    --curator_model   "$MODEL" \
    --max_tokens  4096 \
    --test_workers 20 \
    --save_path   results/sciknow_full_train \
    2>&1 | tee results/sciknow_full_train.log

echo "Full Train done @ $(date)"
grep "Final Accuracy" results/sciknow_full_train.log | tail -1

echo ""
echo "=================================================================="
echo "  DONE  $(date)"
echo "=================================================================="
