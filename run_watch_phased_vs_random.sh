#!/usr/bin/env bash
# =============================================================================
# 监控 Phased vs Random 学习曲线实验进度
# 用法：bash run_watch_phased_vs_random.sh
# =============================================================================
cd /workspace/ace_leo

K_VALUES=(10 20 30 50 80 100 200)
SPLIT=0.50

# 从 log 提取摘要
show() {
    local log=$1
    local label=$2
    if [ ! -f "$log" ]; then
        printf "    ⏳ %-22s  (未启动)\n" "$label"
        return
    fi
    local cnt final base
    cnt=$(grep -c "Running Curator" "$log" 2>/dev/null | tr -d '\n' || echo 0)
    final=$(grep "test-final"    "$log" 2>/dev/null | tail -1 | grep -oP '(?<=acc=)[0-9.]+' || true)
    base=$(grep  "test-baseline" "$log" 2>/dev/null | tail -1 | grep -oP '(?<=acc=)[0-9.]+' || true)
    vals=$(grep  "val-step"      "$log" 2>/dev/null | awk -F'acc=' '{print $2}' | awk '{printf "%.4f ", $1}')
    budget=$(grep -oP 'budget=\K[0-9]+' "$log" 2>/dev/null | head -1 || echo "?")
    credit=$(grep -c "credit_limit"      "$log" 2>/dev/null || echo 0)
    ioerr=$( grep -c "Input/output error" "$log" 2>/dev/null || echo 0)

    if [ -n "$final" ] && [ -n "$base" ]; then
        local delta
        delta=$(python3 -c "print(f'{float(\"$final\")-float(\"$base\"):+.4f}')" 2>/dev/null)
        printf "    ✅ %-22s  base=%.4f → %.4f  Δ=%s\n" "$label" "$base" "$final" "$delta"
    elif [ "$credit" -gt "0" ]; then
        printf "    ❌ %-22s  CREDIT_LIMIT  (%d/%s steps)\n" "$label" "$cnt" "$budget"
    elif [ "$ioerr" -gt "0" ]; then
        printf "    ❌ %-22s  IO_ERROR      (%d/%s steps)\n" "$label" "$cnt" "$budget"
    else
        printf "    🟢 %-22s  (%d/%s steps)  vals=[%s]\n" "$label" "$cnt" "$budget" "$vals"
    fi
}

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  Phased (split=${SPLIT}) vs Random — 学习曲线对比  $(date '+%H:%M:%S')"
echo "════════════════════════════════════════════════════════════════════"

echo ""
echo "  ┌─────┬────────────────────────────────────────────────────────┐"
echo "  │     │  FINER Finance                                          │"
echo "  └─────┴────────────────────────────────────────────────────────┘"
for K in "${K_VALUES[@]}"; do
    if [ "$K" -eq "200" ]; then
        # 已完成的历史结果
        printf "    📋 %-22s  final=0.7761 (已有结果)\n" "phased_k200"
        printf "    📋 %-22s  final=0.7602 (已有结果)\n" "random_k200"
    else
        show "results/finer_curriculum/run_phased_k${K}.log" "phased_k${K}"
        show "results/finer_curriculum/run_random_k${K}.log"  "random_k${K}"
    fi
done

echo ""
echo "  ┌─────┬────────────────────────────────────────────────────────┐"
echo "  │     │  Mind2Web                                               │"
echo "  └─────┴────────────────────────────────────────────────────────┘"
for K in "${K_VALUES[@]}"; do
    show "results/mind2web_curriculum/run_phased_k${K}.log" "phased_k${K}"
    show "results/mind2web_curriculum/run_random_k${K}.log"  "random_k${K}"
done

echo ""
echo "  提示: watch -n 60 'bash run_watch_phased_vs_random.sh'"
echo ""
