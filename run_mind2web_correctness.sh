#!/usr/bin/env bash
# Correctness-augmented lesson clustering: k=10,15,20
# Step 1: cluster_correctness_select
# Step 2: ACE training (val)
# Step 3: test eval
# nohup-safe — survives lid close.

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
RESULTS=/Users/leo/Desktop/ace/results
API=together
MODEL=deepseek-ai/DeepSeek-V3

cd /Users/leo/Desktop/ace
echo "Starting correctness clustering experiments @ $(date)"

# ── Step 1: Generate datasets ─────────────────────────────────────────────────
echo ""
echo "=== Generating correctness-augmented clusters k=10,15,20 ==="
$PYTHON -m eval.mind2web.cluster_correctness_select --clusters 10 15 20 \
    2>&1 | tee $RESULTS/correctness_select.log
echo "Selection done @ $(date)"

# ── Helper function ───────────────────────────────────────────────────────────
run_one() {
    local k=$1
    local name="mind2web_cluster${k}_correctness"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  k=${k}: train + test eval"
    echo "════════════════════════════════════════════════════"

    # Train
    rm -rf $RESULTS/${name}
    mkdir -p $RESULTS/${name}
    $PYTHON -u -m eval.mind2web.run \
        --task_name ${name} \
        --mode offline \
        --skip_initial_test \
        --eval_steps ${k} \
        --api_provider $API \
        --generator_model $MODEL \
        --reflector_model $MODEL \
        --curator_model $MODEL \
        --save_path $RESULTS/${name} \
        2>&1 | tee $RESULTS/${name}_train.log
    echo "k=${k} train done @ $(date)"
    grep "best_validation_accuracy" $RESULTS/${name}_train.log | tail -1

    # Test eval
    PLAYBOOK=$(find $RESULTS/${name} -name "best_playbook.txt" | sort | tail -1)
    if [ -z "$PLAYBOOK" ]; then
        echo "ERROR: no best_playbook.txt found for k=${k}, skipping test"
        return
    fi
    rm -rf $RESULTS/${name}_test
    mkdir -p $RESULTS/${name}_test
    $PYTHON -u -m eval.mind2web.run \
        --task_name ${name} \
        --mode eval_only \
        --initial_playbook_path "$PLAYBOOK" \
        --api_provider $API \
        --generator_model $MODEL \
        --reflector_model $MODEL \
        --curator_model $MODEL \
        --save_path $RESULTS/${name}_test \
        2>&1 | tee $RESULTS/${name}_test.log
    echo "k=${k} test done: $(grep 'Final Accuracy' $RESULTS/${name}_test.log | tail -1)"
}

# ── Step 2+3: Train and test each k ──────────────────────────────────────────
run_one 10
run_one 15
run_one 20

echo ""
echo "🎉 All correctness experiments complete @ $(date)"
