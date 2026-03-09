#!/usr/bin/env bash
# =============================================================================
# Phased vs Random: 相同预算 K 下的学习曲线对比
#
# 实验设计：
#   - 方法 1: phased (phase_split=0.50，即前 K/2 步选 easy，后 K/2 步选 hard)
#   - 方法 2: random (uniform random selection)
#   - 预算 K ∈ {10, 20, 30, 50, 80, 100}  (200 已完成，直接复用)
#   - 在 Finer 和 Mind2Web 上都跑
#
# 已有结果（budget=200, 无需重跑）：
#   Finer:    phased_50 → 0.7761,  random → 0.7602
#   Mind2Web: 全部崩溃，无结果
#
# Phase_split=0.50: Finer 上最优，用作代表值
# =============================================================================
set -euo pipefail

cd /workspace/ace_leo
source .env 2>/dev/null || true

PYTHON=/workspace/miniconda3/envs/ace311/bin/python
FINER_LOG=results/finer_curriculum
M2W_LOG=results/mind2web_curriculum
mkdir -p "$FINER_LOG" "$M2W_LOG"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# 启动单个 Finer run
# ---------------------------------------------------------------------------
finer() {
    local label=$1  # 日志文件名后缀
    local extra=$2  # 额外参数
    local logfile="$FINER_LOG/run_${label}.log"
    log "  finer  $label"
    nohup $PYTHON -m eval.finance.run_curriculum \
        $extra \
        > "$logfile" 2>&1 &
    echo "    PID $! → $logfile"
    sleep 1
}

# ---------------------------------------------------------------------------
# 启动单个 Mind2Web run
# ---------------------------------------------------------------------------
m2w() {
    local label=$1
    local extra=$2
    local logfile="$M2W_LOG/run_${label}.log"
    log "  mind2web  $label"
    nohup $PYTHON -m eval.mind2web.run_curriculum \
        $extra \
        > "$logfile" 2>&1 &
    echo "    PID $! → $logfile"
    sleep 1
}

# K 值列表（budget=200 已有结果，跳过）
K_VALUES=(10 20 30 50 80 100)
# 最优 phase_split（Finer 验证）
SPLIT=0.50

log "========================================================================"
log "  Phased vs Random @ K = ${K_VALUES[*]} + 200(已有)"
log "  phase_split = $SPLIT"
log "========================================================================"
echo ""

# ==========================================================================
# FINER
# ==========================================================================
log "── FINER ────────────────────────────────────────────────────────────"
for K in "${K_VALUES[@]}"; do
    finer "phased_k${K}" "--selector phased --phase_split $SPLIT --budget $K --eval_every $((K/2 < 5 ? 5 : K/2)) --seed 42"
    finer "random_k${K}" "--selector random  --budget $K --eval_every $((K/2 < 5 ? 5 : K/2)) --seed 42"
done

echo ""

# ==========================================================================
# MIND2WEB
# ==========================================================================
log "── MIND2WEB ─────────────────────────────────────────────────────────"
for K in "${K_VALUES[@]}"; do
    m2w "phased_k${K}" "--selector phased --phase_split $SPLIT --budget $K --eval_every $((K/2 < 5 ? 5 : K/2)) --seed 42"
    m2w "random_k${K}" "--selector random  --budget $K --eval_every $((K/2 < 5 ? 5 : K/2)) --seed 42"
done

# 也补一个 mind2web budget=200 (之前全崩了，作为终点数据)
m2w "phased_k200" "--selector phased --phase_split $SPLIT --budget 200 --eval_every 25 --seed 42"
m2w "random_k200" "--selector random  --budget 200 --eval_every 25 --seed 42"

echo ""
log "========================================================================"
log "  已启动 $((${#K_VALUES[@]} * 4 + 2)) 个 runs"
log ""
log "  监控: bash run_watch_phased_vs_random.sh"
log "========================================================================"
