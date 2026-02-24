#!/bin/bash
# Lesson-based cluster experiment: k=10, 15, 20, 30
# Generates subsets → trains ACE → checks validation accuracy
# Runs serially; nohup-safe (survives lid close)

PYTHON=/opt/miniconda3/envs/ace-leo/bin/python
BASE=results/mind2web_lesson_cluster
mkdir -p $BASE

# ── Step 1: Cluster & select ─────────────────────────────────────────────────
echo "============================================================"
echo " Step 1: Lesson-based K-means (k=10,15,20,30)"
echo "============================================================"
PYTHONUNBUFFERED=1 $PYTHON -u -m eval.mind2web.cluster_lesson_select \
    --clusters 10 15 20 30 \
    2>&1 | tee $BASE/cluster_select.log

if [ $? -ne 0 ]; then
    echo "ERROR: clustering failed"; exit 1
fi

# ── Step 2: Train + val for each k ───────────────────────────────────────────
run_one() {
    local NAME=$1
    local K=$2

    echo ""
    echo "============================================================"
    echo " Training: $NAME  (k=$K)"
    echo "============================================================"
    mkdir -p $BASE/$NAME
    PYTHONUNBUFFERED=1 $PYTHON -u -m eval.mind2web.run \
        --task_name  $NAME \
        --mode       offline \
        --skip_initial_test \
        --eval_steps $K \
        --save_path  $BASE/$NAME \
        2>&1 | tee $BASE/${NAME}_train.log

    # Extract best val accuracy from log
    VAL=$(grep -oP "val.*?accuracy[:\s]+\K[0-9]+\.[0-9]+" $BASE/${NAME}_train.log 2>/dev/null | tail -1)
    echo ">>> $NAME  val_acc=$VAL"
}

run_one "mind2web_cluster10_lesson" 10
run_one "mind2web_cluster15_lesson" 15
run_one "mind2web_cluster20_lesson" 20
run_one "mind2web_cluster30_lesson" 30

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Validation Summary — Lesson Cluster"
echo "============================================================"
for K in 10 15 20 30; do
    NAME="mind2web_cluster${K}_lesson"
    LOG=$BASE/${NAME}_train.log
    if [ -f "$LOG" ]; then
        BEST_VAL=$(grep -oP "Best val.*?[0-9]+\.[0-9]+" $LOG 2>/dev/null | grep -oP "[0-9]+\.[0-9]+" | tail -1)
        # fallback: last accuracy line
        if [ -z "$BEST_VAL" ]; then
            BEST_VAL=$($PYTHON -c "
import re, sys
lines = open('$LOG').read()
accs = re.findall(r'accuracy[\":\s]+([0-9]+\.[0-9]+)', lines)
print(max(float(a) for a in accs) if accs else 'N/A')
" 2>/dev/null)
        fi
        echo "  cluster${K}_lesson : val=$BEST_VAL"
    else
        echo "  cluster${K}_lesson : no log"
    fi
done
echo ""
echo "DONE"
