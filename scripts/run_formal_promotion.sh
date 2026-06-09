#!/bin/bash
# ============================================
# Shenji Agent - 一键启动 100 场 Promotion Gate
# ============================================
# 用法: 在远程服务器上执行
#   bash /remote-home/lizilong/shenji_agent/scripts/run_formal_promotion.sh
#
# 预计耗时: 3-6 小时
# 输出目录: experiments/shenji_agent/league_<timestamp>/
# ============================================

set -e
cd /remote-home/lizilong/shenji_agent

echo "=========================================="
echo "  Shenji Agent - Formal Promotion Gate"
echo "  100 paired matches x 400 steps"
echo "  Started: $(date)"
echo "=========================================="

# Pre-flight checks
echo ""
echo "[1/4] Pre-flight checks..."

# GPU
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
fi

# Git
CURRENT_SHA=$(git rev-parse HEAD)
echo "  HEAD: $CURRENT_SHA"

# Python
PYTHONPATH="/remote-home/lizilong/.local/lib/python3.10/site-packages:."
export PYTHONPATH

# Verify key files exist
for f in ai/weights/model_manifest.json ai/weights/intention_lstm.pth ai/weights/situation_gcn.pth ai/weights/tactical_model.pt configs/selfplay_sota_stop.yaml; do
    if [ ! -f "$f" ]; then
        echo "  ERROR: Missing $f"
        exit 1
    fi
done
echo "  All required files present."

# Run
echo ""
echo "[2/4] Starting promotion gate (100 matches x 2 legs x 400 steps)..."
echo "  ETA: 3-6 hours"
echo ""

LOGFILE="/tmp/promotion_gate_$(date +%Y%m%d_%H%M%S).log"
python3.10 -m training.selfplay \
    --config configs/selfplay_sota_stop.yaml \
    --candidate ai/weights \
    --incumbent ai/weights_incumbent_rule \
    2>&1 | tee "$LOGFILE"

# Results
echo ""
echo "=========================================="
echo "[3/4] Collecting results..."
echo "=========================================="

LATEST_DIR=$(ls -td experiments/shenji_agent/league_*/ | head -1)
echo "  Output: $LATEST_DIR"

if [ -f "${LATEST_DIR}summary.md" ]; then
    cat "${LATEST_DIR}summary.md"
fi

echo ""
echo "--- Promotion Decision ---"
if [ -f "${LATEST_DIR}promotion.json" ]; then
    python3.10 << PYEOF
import json
d = json.load(open("${LATEST_DIR}promotion.json"))
status = "ACCEPTED" if d.get("accepted") else "REJECTED"
print("  Status: %s" % status)
if d.get("failures"):
    for f in d["failures"]:
        print("  FAIL: %s" % f)
print("  Learned coverage: %.4f" % d.get("learned_path_coverage", 0))
print("  Move coverage: %.4f" % d.get("move_path_coverage", 0))
print("  Shoot coverage: %.4f" % d.get("shoot_path_coverage", 0))
print("  Elo delta lower: %.2f" % d.get("elo_delta_lower_bound", 0))
print("  Max exploiter WR: %.4f" % d.get("max_exploiter_win_rate", 0))
PYEOF
fi

echo ""
echo "--- Rating ---"
if [ -f "${LATEST_DIR}rating.json" ]; then
    python3.10 << PYEOF
import json
d = json.load(open("${LATEST_DIR}rating.json"))
ov = d.get("overall", {})
ci = ov.get("ci95", [0, 0])
print("  Overall: %.4f (CI95: [%.4f, %.4f])" % (ov.get("point_estimate", 0), ci[0], ci[1]))
for bkt, s in d.get("buckets", {}).items():
    bci = s.get("ci95", [0, 0])
    print("  %s: %.4f (CI95: [%.4f, %.4f])" % (bkt, s.get("point_estimate", 0), bci[0], bci[1]))
PYEOF
fi

echo ""
echo "--- Exploitability ---"
if [ -f "${LATEST_DIR}exploitability.json" ]; then
    python3.10 << PYEOF
import json
d = json.load(open("${LATEST_DIR}exploitability.json"))
for p, r in d.items():
    print("  %s: WR=%.3f matches=%d" % (p, r.get("win_rate_vs_candidate", 0), r.get("matches", 0)))
PYEOF
fi

echo ""
echo "=========================================="
echo "  Completed: $(date)"
echo "  Log: $LOGFILE"
echo "  Results: $LATEST_DIR"
echo "=========================================="
