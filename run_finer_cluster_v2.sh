#!/usr/bin/env bash
# =============================================================================
# FiNer Curriculum Selection Pipeline — V2 (with GT labels in embeddings)
#
# Difference from v1: embeddings now include ground-truth XBRL tags so that
# K-means clusters by tagging-pattern diversity, not just text similarity.
#
# Steps:
#   Step 1: Re-generate embeddings (sentences + GT labels)
#   Step 2: K-means clustering → subsets of size 5,10,20,30,40,50,80,100
#   Step 3: ACE offline training for each cluster subset
#   Step 4: Baseline — ACE training on full 1000-sample train set
#
# Results saved to  results/finer_cluster_v2/
#
# Usage:
#   bash run_finer_cluster_v2.sh                  # run everything
#   bash run_finer_cluster_v2.sh --skip_embed     # skip Step 1
#   bash run_finer_cluster_v2.sh --only_eval      # only eval existing playbooks
#
# nohup-safe:
#   nohup bash run_finer_cluster_v2.sh > results/finer_cluster_v2_main.log 2>&1 &
# =============================================================================

set -euo pipefail

PYTHON=/workspace/miniconda3/envs/ace311/bin/python
RESULTS=/workspace/ace_leo/results
API=together
MODEL=deepseek-ai/DeepSeek-V3.1

SKIP_EMBED=false
ONLY_EVAL=false

for arg in "$@"; do
    case $arg in
        --skip_embed) SKIP_EMBED=true ;;
        --only_eval)  ONLY_EVAL=true  ;;
    esac
done

cd /workspace/ace_leo
mkdir -p "$RESULTS/finer_cluster_v2"

echo "=================================================================="
echo "  FiNer Curriculum Selection Pipeline — V2"
echo "  Date: $(date)"
echo "  Model: $MODEL  API: $API"
echo "  Embedding: sentences + GT labels"
echo "=================================================================="

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Re-generate semantic embeddings (with GT labels)
# ──────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_EMBED" = false ] && [ "$ONLY_EVAL" = false ]; then
    echo ""
    echo "=== Step 1: Generating semantic embeddings (sentences + GT labels) ==="
    $PYTHON -m eval.finance.embed_train --force \
        2>&1 | tee "$RESULTS/finer_cluster_v2/embed.log"
    echo "Embeddings done @ $(date)"
else
    echo "=== Step 1: Skipping embedding generation ==="
fi

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: K-means clustering  (k = 5, 10, 20, 30, 40, 50, 80, 100)
# ──────────────────────────────────────────────────────────────────────────────
if [ "$ONLY_EVAL" = false ]; then
    echo ""
    echo "=== Step 2: K-means clustering (k=5,10,20,30,40,50,80,100) ==="
    $PYTHON -m eval.finance.cluster_train \
        --clusters 5 10 20 30 40 50 80 100 \
        2>&1 | tee "$RESULTS/finer_cluster_v2/cluster.log"
    echo "Clustering done @ $(date)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Helper: train + test-eval for one cluster size
# ──────────────────────────────────────────────────────────────────────────────
run_cluster() {
    local K=$1
    local TASK="finer_cluster${K}"
    local TRAIN_DIR="$RESULTS/finer_cluster_v2_${K}"
    local TEST_DIR="$RESULTS/finer_cluster_v2_${K}_test"

    echo ""
    echo "─────────────────────────────────────────────"
    echo "  Cluster K=${K}: training"
    echo "─────────────────────────────────────────────"

    rm -rf "$TRAIN_DIR"
    mkdir -p "$TRAIN_DIR"

    $PYTHON -u -m eval.finance.run \
        --task_name "$TASK" \
        --mode offline \
        --eval_steps "$K" \
        --api_provider "$API" \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model   "$MODEL" \
        --save_path "$TRAIN_DIR" \
        2>&1 | tee "$RESULTS/finer_cluster_v2/${TASK}_train.log"

    echo "  K=${K} training done @ $(date)"

    local PLAYBOOK
    PLAYBOOK=$(find "$TRAIN_DIR" -name "best_playbook.txt" 2>/dev/null | sort | tail -1 || true)
    if [ -z "$PLAYBOOK" ]; then
        echo "  [WARN] No best_playbook.txt found for K=${K}, skipping test eval"
        return
    fi

    echo ""
    echo "  Cluster K=${K}: test evaluation (playbook: $PLAYBOOK)"

    rm -rf "$TEST_DIR"
    mkdir -p "$TEST_DIR"

    $PYTHON -u -m eval.finance.run \
        --task_name "finer" \
        --mode eval_only \
        --initial_playbook_path "$PLAYBOOK" \
        --api_provider "$API" \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model   "$MODEL" \
        --save_path "$TEST_DIR" \
        2>&1 | tee "$RESULTS/finer_cluster_v2/${TASK}_test.log"

    echo "  K=${K} test done: $(grep 'Final Accuracy\|accuracy' \
        "$RESULTS/finer_cluster_v2/${TASK}_test.log" | tail -1)"
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper: eval only
# ──────────────────────────────────────────────────────────────────────────────
eval_cluster() {
    local K=$1
    local TASK="finer_cluster${K}"
    local TRAIN_DIR="$RESULTS/finer_cluster_v2_${K}"
    local TEST_DIR="$RESULTS/finer_cluster_v2_${K}_test"

    local PLAYBOOK
    PLAYBOOK=$(find "$TRAIN_DIR" -name "best_playbook.txt" 2>/dev/null | sort | tail -1 || true)
    if [ -z "$PLAYBOOK" ]; then
        echo "  [WARN] No playbook for K=${K} at $TRAIN_DIR — run training first"
        return
    fi

    echo ""
    echo "  Cluster K=${K}: test evaluation (playbook: $PLAYBOOK)"
    rm -rf "$TEST_DIR"
    mkdir -p "$TEST_DIR"

    $PYTHON -u -m eval.finance.run \
        --task_name "finer" \
        --mode eval_only \
        --initial_playbook_path "$PLAYBOOK" \
        --api_provider "$API" \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model   "$MODEL" \
        --save_path "$TEST_DIR" \
        2>&1 | tee "$RESULTS/finer_cluster_v2/${TASK}_test.log"

    echo "  K=${K} test done: $(grep 'Final Accuracy\|accuracy' \
        "$RESULTS/finer_cluster_v2/${TASK}_test.log" | tail -1)"
}

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Train + eval for each cluster size
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 3: ACE training for each cluster subset ==="

for K in 5 10 20 30 40 50 80 100; do
    if [ "$ONLY_EVAL" = true ]; then
        eval_cluster $K
    else
        run_cluster $K
    fi
done

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Baseline — full 1000-sample training
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 4: Baseline — full FiNer training (1000 samples) ==="

BASELINE_DIR="$RESULTS/finer_v2_baseline"
BASELINE_TEST_DIR="$RESULTS/finer_v2_baseline_test"

if [ "$ONLY_EVAL" = false ]; then
    rm -rf "$BASELINE_DIR"
    mkdir -p "$BASELINE_DIR"

    $PYTHON -u -m eval.finance.run \
        --task_name finer \
        --mode offline \
        --eval_steps 100 \
        --api_provider "$API" \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model   "$MODEL" \
        --save_path "$BASELINE_DIR" \
        2>&1 | tee "$RESULTS/finer_cluster_v2/finer_baseline_train.log"

    echo "Baseline training done @ $(date)"
fi

BASELINE_PLAYBOOK=$(find "$BASELINE_DIR" -name "best_playbook.txt" 2>/dev/null | sort | tail -1 || true)
if [ -n "$BASELINE_PLAYBOOK" ]; then
    echo "Baseline test evaluation (playbook: $BASELINE_PLAYBOOK)"
    rm -rf "$BASELINE_TEST_DIR"
    mkdir -p "$BASELINE_TEST_DIR"

    $PYTHON -u -m eval.finance.run \
        --task_name finer \
        --mode eval_only \
        --initial_playbook_path "$BASELINE_PLAYBOOK" \
        --api_provider "$API" \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model   "$MODEL" \
        --save_path "$BASELINE_TEST_DIR" \
        2>&1 | tee "$RESULTS/finer_cluster_v2/finer_baseline_test.log"

    echo "Baseline test done: $(grep 'Final Accuracy\|accuracy' \
        "$RESULTS/finer_cluster_v2/finer_baseline_test.log" | tail -1)"
else
    echo "[WARN] No baseline playbook found at $BASELINE_DIR"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  SUMMARY — FiNer Cluster V2 Experiment Results"
echo "=================================================================="
echo "  Test accuracy by training subset size:"
echo ""
for K in 5 10 20 30 40 50 80 100; do
    LOG="$RESULTS/finer_cluster_v2/finer_cluster${K}_test.log"
    if [ -f "$LOG" ]; then
        ACC=$(grep -E "Final Accuracy|accuracy" "$LOG" | tail -1 || echo "N/A")
        echo "    cluster${K}  : $ACC"
    else
        echo "    cluster${K}  : (not run)"
    fi
done

BLOG="$RESULTS/finer_cluster_v2/finer_baseline_test.log"
if [ -f "$BLOG" ]; then
    BACC=$(grep -E "Final Accuracy|accuracy" "$BLOG" | tail -1 || echo "N/A")
    echo "    baseline (1000): $BACC"
else
    echo "    baseline (1000): (not run)"
fi

echo ""
echo "  Completed @ $(date)"
echo "=================================================================="
