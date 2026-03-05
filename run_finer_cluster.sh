#!/usr/bin/env bash
# =============================================================================
# FiNer Curriculum Selection Pipeline
#
# Runs the full cluster-based curriculum selection experiment on FiNer:
#   Step 1: Generate semantic embeddings for 1000 FiNer training samples
#   Step 2: K-means clustering → select subsets of size 5,10,20,30,40,50,80
#   Step 3: ACE offline training for each cluster subset
#   Step 4: Baseline — ACE training on full 1000-sample train set
#
# Results are saved to  results/finer_cluster/
#
# Usage:
#   bash run_finer_cluster.sh                  # run everything
#   bash run_finer_cluster.sh --skip_embed     # skip Step 1 (embeddings exist)
#   bash run_finer_cluster.sh --only_eval      # only eval existing playbooks
#
# nohup-safe — pipe to a log file:
#   nohup bash run_finer_cluster.sh > results/finer_cluster_main.log 2>&1 &
# =============================================================================

set -euo pipefail

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
RESULTS=/Users/leo/Desktop/ace/results
API=sambanova
MODEL=DeepSeek-V3.1

SKIP_EMBED=false
ONLY_EVAL=false

for arg in "$@"; do
    case $arg in
        --skip_embed) SKIP_EMBED=true ;;
        --only_eval)  ONLY_EVAL=true  ;;
    esac
done

cd /Users/leo/Desktop/ace
mkdir -p "$RESULTS/finer_cluster"

echo "=================================================================="
echo "  FiNer Curriculum Selection Pipeline"
echo "  Date: $(date)"
echo "  Model: $MODEL  API: $API"
echo "=================================================================="

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Generate semantic embeddings
# ──────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_EMBED" = false ] && [ "$ONLY_EVAL" = false ]; then
    echo ""
    echo "=== Step 1: Generating semantic embeddings ==="
    $PYTHON -m eval.finance.embed_train \
        2>&1 | tee "$RESULTS/finer_cluster/embed.log"
    echo "Embeddings done @ $(date)"
else
    echo "=== Step 1: Skipping embedding generation ==="
fi

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: K-means clustering  (k = 5, 10, 20, 30, 40, 50, 80)
# ──────────────────────────────────────────────────────────────────────────────
if [ "$ONLY_EVAL" = false ]; then
    echo ""
    echo "=== Step 2: K-means clustering (k=5,10,20,30,40,50,80) ==="
    $PYTHON -m eval.finance.cluster_train \
        --clusters 5 10 20 30 40 50 80 \
        --visualize \
        2>&1 | tee "$RESULTS/finer_cluster/cluster.log"
    echo "Clustering done @ $(date)"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Helper: train + test-eval for one cluster size
# ──────────────────────────────────────────────────────────────────────────────
run_cluster() {
    local K=$1
    local TASK="finer_cluster${K}"
    local TRAIN_DIR="$RESULTS/finer_cluster${K}"
    local TEST_DIR="$RESULTS/finer_cluster${K}_test"

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
        2>&1 | tee "$RESULTS/finer_cluster/${TASK}_train.log"

    echo "  K=${K} training done @ $(date)"

    # Find best playbook
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
        2>&1 | tee "$RESULTS/finer_cluster/${TASK}_test.log"

    echo "  K=${K} test done: $(grep 'Final Accuracy\|accuracy' \
        "$RESULTS/finer_cluster/${TASK}_test.log" | tail -1)"
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper: eval only (uses existing playbook)
# ──────────────────────────────────────────────────────────────────────────────
eval_cluster() {
    local K=$1
    local TASK="finer_cluster${K}"
    local TRAIN_DIR="$RESULTS/finer_cluster${K}"
    local TEST_DIR="$RESULTS/finer_cluster${K}_test"

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
        2>&1 | tee "$RESULTS/finer_cluster/${TASK}_test.log"

    echo "  K=${K} test done: $(grep 'Final Accuracy\|accuracy' \
        "$RESULTS/finer_cluster/${TASK}_test.log" | tail -1)"
}

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Train + eval for each cluster size
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 3: ACE training for each cluster subset ==="

for K in 5 10 20 30 40 50 80; do
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

BASELINE_DIR="$RESULTS/finer_baseline"
BASELINE_TEST_DIR="$RESULTS/finer_baseline_test"

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
        2>&1 | tee "$RESULTS/finer_cluster/finer_baseline_train.log"

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
        2>&1 | tee "$RESULTS/finer_cluster/finer_baseline_test.log"

    echo "Baseline test done: $(grep 'Final Accuracy\|accuracy' \
        "$RESULTS/finer_cluster/finer_baseline_test.log" | tail -1)"
else
    echo "[WARN] No baseline playbook found at $BASELINE_DIR"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  SUMMARY — FiNer Cluster Experiment Results"
echo "=================================================================="
echo "  Test accuracy by training subset size:"
echo ""
for K in 5 10 20 30 40 50 80; do
    LOG="$RESULTS/finer_cluster/finer_cluster${K}_test.log"
    if [ -f "$LOG" ]; then
        ACC=$(grep -E "Final Accuracy|accuracy" "$LOG" | tail -1 || echo "N/A")
        echo "    cluster${K}  : $ACC"
    else
        echo "    cluster${K}  : (not run)"
    fi
done

BLOG="$RESULTS/finer_cluster/finer_baseline_test.log"
if [ -f "$BLOG" ]; then
    BACC=$(grep -E "Final Accuracy|accuracy" "$BLOG" | tail -1 || echo "N/A")
    echo "    baseline (1000): $BACC"
else
    echo "    baseline (1000): (not run)"
fi

echo ""
echo "  Completed @ $(date)"
echo "=================================================================="
