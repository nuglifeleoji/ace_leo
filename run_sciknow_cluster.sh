#!/usr/bin/env bash
# =============================================================================
#  SciKnowEval Chemistry L3 — Curriculum Selection Experiments
#  Methods: Semantic Clustering, Lesson-based Clustering, Random Sampling
#  K values: 5, 10, 20, 30, 40, 50, 80
# =============================================================================
set -euo pipefail

PYTHON=/workspace/miniconda3/envs/ace311/bin/python
RESULTS=/workspace/ace_leo/results
API=together
MODEL=deepseek-ai/DeepSeek-V3.1
K_VALUES=(5 10 20 30 40 50 80)

cd /workspace/ace_leo
# Load API keys from .env if not already set
if [ -f .env ]; then
    set -a; source .env; set +a
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Helper: run ACE for a single K subset, then evaluate on test set
# ---------------------------------------------------------------------------
run_cluster() {
    local task_name=$1   # e.g. sciknow_chem_l3_cluster5
    local k=$2
    local save_train=$3
    local save_test=$4

    log "── Train  $task_name  (K=$k) ──"
    $PYTHON -m eval.sciknow.run \
        --task_name      "$task_name" \
        --mode           offline \
        --api_provider   $API \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model   "$MODEL" \
        --max_tokens      8192 \
        --eval_steps      "$k" \
        --test_workers    20 \
        --save_path       "$save_train"

    log "── Test   $task_name  (K=$k) ──"
    # Use best_playbook.txt saved by ACE after training
    PLAYBOOK=$(find "$save_train" -name "best_playbook.txt" | head -1 || true)
    if [ -z "$PLAYBOOK" ]; then
        PLAYBOOK=$(find "$save_train" -name "final_playbook.txt" | head -1 || true)
    fi

    PB_ARG=""
    if [ -n "$PLAYBOOK" ]; then
        log "  Using playbook: $PLAYBOOK"
        PB_ARG="--initial_playbook_path $PLAYBOOK"
    else
        log "  WARNING: no playbook found, running eval without playbook"
    fi

    $PYTHON -m eval.sciknow.run \
        --task_name      "$task_name" \
        --mode           eval_only \
        --api_provider   $API \
        --generator_model "$MODEL" \
        --reflector_model "$MODEL" \
        --curator_model   "$MODEL" \
        --max_tokens      8192 \
        --test_workers    20 \
        --save_path       "$save_test" \
        $PB_ARG
}

# ===========================================================================
# STEP 1: Generate semantic embeddings (skip if already done)
# ===========================================================================
EMBED_FILE=./eval/sciknow/data/sciknow_embeddings.npy
if [ ! -f "$EMBED_FILE" ]; then
    log "Generating semantic embeddings ..."
    $PYTHON -m eval.sciknow.embed_train
else
    log "Semantic embeddings already exist — skipping"
fi

# ===========================================================================
# STEP 2: Semantic clustering → select K subsets
# ===========================================================================
log "Running K-means clustering (semantic) ..."
$PYTHON -m eval.sciknow.cluster_select

# ===========================================================================
# STEP 3: Random sampling baseline
# ===========================================================================
log "Generating random subsets ..."
$PYTHON -m eval.sciknow.random_sample

# ===========================================================================
# STEP 4: Generate lessons + lesson embeddings (skip if already done)
# ===========================================================================
LESSON_FILE=./eval/sciknow/data/sciknow_lessons.jsonl
if [ ! -f "$LESSON_FILE" ]; then
    log "Generating lessons via Qwen ..."
    $PYTHON -m eval.sciknow.lesson_generate
else
    log "Lessons already exist — skipping lesson generation"
fi

LESSON_EMBED=./eval/sciknow/data/sciknow_lesson_embeddings.npy
if [ ! -f "$LESSON_EMBED" ]; then
    log "Generating lesson embeddings ..."
    $PYTHON -m eval.sciknow.lesson_generate
else
    log "Lesson embeddings already exist — skipping"
fi

# ===========================================================================
# STEP 5: Lesson-based clustering → select K subsets
# ===========================================================================
log "Running K-means clustering (lesson-based) ..."
$PYTHON -m eval.sciknow.cluster_lesson_select

# ===========================================================================
# STEP 6: Run ACE for each method × K
# ===========================================================================
echo ""
log "============================================================"
log "  Starting ACE training for all K values"
log "============================================================"

# --- Semantic ---
for K in "${K_VALUES[@]}"; do
    run_cluster \
        "sciknow_chem_l3_cluster${K}" \
        "$K" \
        "$RESULTS/sciknow_cluster_${K}" \
        "$RESULTS/sciknow_cluster_${K}_test"
done

# --- Random ---
for K in "${K_VALUES[@]}"; do
    run_cluster \
        "sciknow_chem_l3_random${K}" \
        "$K" \
        "$RESULTS/sciknow_random_${K}" \
        "$RESULTS/sciknow_random_${K}_test"
done

# --- Lesson ---
for K in "${K_VALUES[@]}"; do
    run_cluster \
        "sciknow_chem_l3_lesson${K}" \
        "$K" \
        "$RESULTS/sciknow_lesson_${K}" \
        "$RESULTS/sciknow_lesson_${K}_test"
done

log "============================================================"
log "  ALL EXPERIMENTS COMPLETE"
log "============================================================"
