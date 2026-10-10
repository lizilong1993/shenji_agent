#!/bin/bash
# ============================================================
# shenji_agent 远程诊断与同步脚本 (Phase 1 修复)
# 使用方式: 在 Git Bash 中运行
#   bash scripts/remote_ops.sh diagnose   # 诊断远程状态
#   bash scripts/remote_ops.sh sync       # 同步文件到本地
#   bash scripts/remote_ops.sh upload     # 上传修复后的 policy.py
#   bash scripts/remote_ops.sh smoke      # 远程运行 smoke test
# ============================================================

set -euo pipefail

REMOTE_HOST="lizilong@146.56.220.99"
REMOTE_PORT="21431"
REMOTE_ROOT="/remote-home/lizilong/shenji_agent"
LOCAL_ROOT="D:/WHUSAI/Papers/shenji_agent"
SSH_CMD="ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no ${REMOTE_HOST}"
SCP_CMD="scp -P ${REMOTE_PORT} -o StrictHostKeyChecking=no"

# ----------------------------------------------------------
# 1. diagnose: 检查远程服务器状态
# ----------------------------------------------------------
diagnose() {
    echo "========== B1: 检查远程权重文件 =========="
    $SSH_CMD << 'REMOTE_DIAG'
echo "--- ai/weights/ 目录内容 ---"
ls -la ~/shenji_agent/ai/weights/
echo ""
echo "--- model_manifest.json ---"
cat ~/shenji_agent/ai/weights/model_manifest.json
echo ""
echo "--- 检查 .pt 文件完整性 ---"
for f in intent_lstm.pt situation_gcn.pt tactical_model.pt; do
    if [ -f ~/shenji_agent/ai/weights/$f ]; then
        echo "[OK] $f exists ($(du -h ~/shenji_agent/ai/weights/$f | cut -f1))"
    else
        echo "[MISSING] $f not found"
    fi
done

echo ""
echo "========== B2: 检查 consecutive_windows =========="
echo "--- 最近的 league 实验 ---"
ls -td ~/shenji_agent/experiments/shenji_agent/league_* 2>/dev/null | head -3
echo ""
echo "--- 最近实验的 league_state.json ---"
LATEST=$(ls -td ~/shenji_agent/experiments/shenji_agent/league_* 2>/dev/null | head -1)
if [ -n "$LATEST" ] && [ -f "$LATEST/league_state.json" ]; then
    cat "$LATEST/league_state.json"
    echo ""
fi

echo ""
echo "--- consecutive_windows 在所有 league 中的分布 ---"
grep -r "consecutive_windows" ~/shenji_agent/experiments/shenji_agent/league_*/league_state.json 2>/dev/null | tail -5 || echo "No league_state.json files found"

echo ""
echo "========== B3: 检查 cost_data 覆盖 =========="
echo "--- 地图 cost 文件列表 ---"
ls -la ~/shenji_agent/land_wargame_sdk/Data/Data/maps/map_212/*cost* 2>/dev/null | head -10
echo ""
ls ~/shenji_agent/land_wargame_sdk/Data/Data/maps/ 2>/dev/null

echo ""
echo "========== B4: 检查远程 training/ 目录 =========="
echo "--- training/ Python 文件 ---"
ls -la ~/shenji_agent/training/*.py 2>/dev/null | head -20
echo ""
echo "--- scripts/ Python 文件 ---"
ls -la ~/shenji_agent/scripts/*.py 2>/dev/null | head -20
echo ""
echo "--- tests/ Python 文件 ---"
ls -la ~/shenji_agent/tests/*.py 2>/dev/null | head -20

echo ""
echo "========== GPU 状态 =========="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
REMOTE_DIAG
}

# ----------------------------------------------------------
# 2. sync: 从远程同步缺失文件到本地
# ----------------------------------------------------------
sync_files() {
    echo "========== B4: 同步 training/ 脚本 =========="
    $SCP_CMD -r "${REMOTE_HOST}:${REMOTE_ROOT}/training/"*.py "${LOCAL_ROOT}/training/" 2>/dev/null || echo "No .py files in training/ or scp failed"

    echo "========== B4: 同步 scripts/ 脚本 =========="
    $SCP_CMD -r "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"*.py "${LOCAL_ROOT}/scripts/" 2>/dev/null || echo "No .py files in scripts/ or scp failed"

    echo "========== B4: 同步 tests/ 脚本 =========="
    $SCP_CMD -r "${REMOTE_HOST}:${REMOTE_ROOT}/tests/"*.py "${LOCAL_ROOT}/tests/" 2>/dev/null || echo "No .py files in tests/ or scp failed"
    $SCP_CMD -r "${REMOTE_HOST}:${REMOTE_ROOT}/tests/conftest.py" "${LOCAL_ROOT}/tests/" 2>/dev/null || true

    echo "========== B1: 同步模型权重 =========="
    echo "注意: 权重文件较大, 请确认网络稳定"
    for f in intent_lstm.pt situation_gcn.pt tactical_model.pt; do
        echo "同步 $f ..."
        $SCP_CMD "${REMOTE_HOST}:${REMOTE_ROOT}/ai/weights/${f}" "${LOCAL_ROOT}/ai/weights/" 2>/dev/null || echo "[SKIP] $f not found on remote or scp failed"
    done

    echo ""
    echo "========== 同步完成 =========="
    echo "请检查以下目录:"
    echo "  training/: $(ls ${LOCAL_ROOT}/training/*.py 2>/dev/null | wc -l) Python files"
    echo "  scripts/:  $(ls ${LOCAL_ROOT}/scripts/*.py 2>/dev/null | wc -l) Python files"
    echo "  tests/:    $(ls ${LOCAL_ROOT}/tests/*.py 2>/dev/null | wc -l) Python files"
    echo "  weights/:  $(ls ${LOCAL_ROOT}/ai/weights/*.pt 2>/dev/null | wc -l) .pt files"
}

# ----------------------------------------------------------
# 3. upload: 上传修复后的 policy.py 到远程
# ----------------------------------------------------------
upload_fix() {
    echo "========== 上传修复后的 policy.py =========="
    $SCP_CMD "${LOCAL_ROOT}/ai/policy.py" "${REMOTE_HOST}:${REMOTE_ROOT}/ai/policy.py"
    echo "[OK] policy.py uploaded"

    echo ""
    echo "========== 验证远程文件 =========="
    $SSH_CMD "head -5 ~/shenji_agent/ai/policy.py && echo '...' && grep -n 'tactical model file' ~/shenji_agent/ai/policy.py"
    echo ""
    echo "[OK] Upload verified"
}

# ----------------------------------------------------------
# 4. smoke: 在远程运行 smoke test 验证修复
# ----------------------------------------------------------
smoke_test() {
    echo "========== 远程 Smoke Test =========="
    $SSH_CMD << 'REMOTE_SMOKE'
cd ~/shenji_agent
export PYTHONPATH=/remote-home/lizilong/.local/lib/python3.10/site-packages

echo "--- 检查 GPU 可用性 ---"
/remote-home/lizilong/venvs/shenji/bin/python3.10 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem // 1024**3}GB)')
"

echo ""
echo "--- 检查模型加载 ---"
/remote-home/lizilong/venvs/shenji/bin/python3.10 -c "
import sys, os
sys.path.insert(0, '.')
from ai.policy import LearnedPolicy
import logging
logging.basicConfig(level=logging.INFO)

policy = LearnedPolicy()
policy.configure(faction=1, artifact_dir='ai/weights', policy_profile='challenger')
print(f'Intent model loaded: {policy.intent_model is not None}')
print(f'Situation model loaded: {policy.situation_model is not None}')
print(f'Tactical model loaded: {policy.tactical_model is not None}')
print(f'Normalizer loaded: {bool(policy._feature_normalizer)}')
print(f'Device: {policy.device}')
print(f'Loaded from: {policy.loaded_from}')
"

echo ""
echo "--- 运行 smoke test ---"
/remote-home/lizilong/venvs/shenji/bin/python3.10 -m training.selfplay \
    --config configs/selfplay_smoke_learned_engagement.yaml \
    --candidate ai/weights \
    --incumbent ai/weights_incumbent_rule \
    2>&1 | tail -50
REMOTE_SMOKE
}

# ----------------------------------------------------------
# 5. check_consecutive: 检查 consecutive_windows 详情
# ----------------------------------------------------------
check_consecutive() {
    echo "========== B2: consecutive_windows 详情 =========="
    $SSH_CMD << 'REMOTE_CHECK'
cd ~/shenji_agent
export PYTHONPATH=/remote-home/lizilong/.local/lib/python3.10/site-packages

echo "--- 搜索 consecutive_windows 在 selfplay.py 中的定义 ---"
grep -n "consecutive_windows" training/selfplay.py 2>/dev/null || echo "selfplay.py not found or no match"

echo ""
echo "--- 搜索 promotion 判定逻辑 ---"
grep -n -A5 "def.*promotion\|def.*gate\|def.*accept\|ACCEPTED\|REJECTED" training/selfplay.py 2>/dev/null | head -40

echo ""
echo "--- 所有 league_state.json 中的 consecutive_windows ---"
for f in $(find experiments/shenji_agent/league_*/league_state.json 2>/dev/null | sort); do
    cw=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('consecutive_windows', 'N/A'))" 2>/dev/null)
    dir=$(dirname $f)
    echo "  $(basename $dir): consecutive_windows=$cw"
done

echo ""
echo "--- 最近3个 league 的 promotion.json ---"
for dir in $(ls -td experiments/shenji_agent/league_* 2>/dev/null | head -3); do
    if [ -f "$dir/promotion.json" ]; then
        echo "=== $(basename $dir) ==="
        cat "$dir/promotion.json"
        echo ""
    fi
done
REMOTE_CHECK
}

# ----------------------------------------------------------
# Main dispatch
# ----------------------------------------------------------
case "${1:-help}" in
    diagnose)
        diagnose
        ;;
    sync)
        sync_files
        ;;
    upload)
        upload_fix
        ;;
    smoke)
        smoke_test
        ;;
    check_consecutive|consecutive|b2)
        check_consecutive
        ;;
    help|*)
        echo "Usage: bash scripts/remote_ops.sh <command>"
        echo ""
        echo "Commands:"
        echo "  diagnose          诊断远程服务器状态 (B1/B2/B3/B4)"
        echo "  sync              同步远程文件到本地 (B4)"
        echo "  upload            上传修复后的 policy.py 到远程 (B1)"
        echo "  smoke             远程运行 smoke test 验证修复"
        echo "  check_consecutive 检查 consecutive_windows 详情 (B2)"
        ;;
esac
