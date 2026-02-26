#!/usr/bin/env bash
# Run random30 seeds 0,3,4,5,6,7 (seeds 1&2 already done)
# Uses Together AI + DeepSeek-V3
# nohup-safe — survives lid close.

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
RESULTS=/Users/leo/Desktop/ace/results
API=together
MODEL=deepseek-ai/DeepSeek-V3

cd /Users/leo/Desktop/ace
echo "Starting random30 experiments (Together AI) @ $(date)"

run_one() {
    local seed=$1
    local name="mind2web_random30_seed${seed}"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  random30 seed=${seed}: train + test eval"
    echo "════════════════════════════════════════════════════"

    # Train
    rm -rf $RESULTS/${name}
    mkdir -p $RESULTS/${name}
    $PYTHON -u -m eval.mind2web.run \
        --task_name ${name} \
        --mode offline \
        --skip_initial_test \
        --eval_steps 30 \
        --api_provider $API \
        --generator_model $MODEL \
        --reflector_model $MODEL \
        --curator_model $MODEL \
        --save_path $RESULTS/${name} \
        2>&1 | tee $RESULTS/${name}_train.log
    echo "seed=${seed} train done @ $(date)"
    grep "best_validation_accuracy" $RESULTS/${name}_train.log | tail -1

    # Test eval
    PLAYBOOK=$(find $RESULTS/${name} -name "best_playbook.txt" | sort | tail -1)
    if [ -z "$PLAYBOOK" ]; then
        echo "ERROR: no best_playbook.txt for seed=${seed}, skipping test"
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
    echo "seed=${seed} test done: $(grep 'Final Accuracy' $RESULTS/${name}_test.log | tail -1)"
}

run_one 0
run_one 3
run_one 4
run_one 5
run_one 6
run_one 7

echo ""
echo "🎉 All random30 experiments complete @ $(date)"
