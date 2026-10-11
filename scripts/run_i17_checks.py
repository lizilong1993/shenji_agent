#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`I17` 判据 2 + 3 的**可运行入口 + 样例计算**。

## 判据原文

> 2) **规则冲突检测可运行并输出清单**；
> 3) **认知对齐度给出公式与样例计算**。

## ★ 样例数据的性质（必须写明）

本脚本用的样例是**合成长度固定的最小序列**，**不是真实评测结果**。
它的用途是：① 让公式**可验证**（给出可复算的数值）；② 让 CLI **可运行**。

**它不构成 `I17` 判据 1 要求的"至少 1 轮专家评分结果"** —— 那需要真实评分者。

## 用法

    python3 scripts/run_i17_checks.py                 # 跑内置样例
    python3 scripts/run_i17_checks.py --actions a.json  # 对给定动作清单跑冲突检测
    python3 scripts/run_i17_checks.py --json          # 机器可读输出
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.alignment import cognitive_alignment  # noqa: E402
from training.expert_rating import rule_conflicts    # noqa: E402

# ── 内置样例（**合成**，固定内容以便复算；非真实评测）
SAMPLE_ACTIONS: List[Dict[str, Any]] = [
    {"kind": "move", "actor": 1, "target": "H101"},
    {"kind": "occupy", "actor": 1, "target": "C7"},
    {"kind": "attack", "actor": 2, "target": 9},
    {"kind": "hold_fire", "actor": 2, "target": 9},      # ← 与上一条冲突
    {"kind": "occupy", "actor": 3, "target": "C8"},
    {"kind": "withdraw", "actor": 4, "target": "C8"},    # ← **不同主体**，不算冲突
]
SAMPLE_SYSTEM_INTENTS = ["ASSAULT", "HOLD", "RECON", "FIRE_SUPPORT"]
SAMPLE_HUMAN_INTENTS = ["ASSAULT", "HOLD", "RECON", "FIRE_SUPPORT"]
SAMPLE_CONFIDENCES = [0.9, 0.8, 0.7, 0.6]
SAMPLE_HUMAN_ACTIONS = ["move", "occupy", "attack", "hold_fire", "occupy", "withdraw"]
SAMPLE_SYSTEM_ACTIONS_FOR_ALIGN = ["move", "occupy", "attack", "hold_fire", "occupy"]


def _demo_conflicts() -> Dict[str, Any]:
    hits = rule_conflicts(SAMPLE_ACTIONS)
    return {
        "n_actions": len(SAMPLE_ACTIONS),
        "n_conflicts": len(hits),
        "conflicts": hits,
        "note": ("同主体同目标才判冲突；`actor=4` 的 withdraw 与 `actor=3` 的 occupy "
                 "目标相同但主体不同 → **不判冲突**"),
    }


def _demo_alignment() -> Dict[str, Any]:
    r = cognitive_alignment(
        SAMPLE_SYSTEM_ACTIONS_FOR_ALIGN, SAMPLE_HUMAN_ACTIONS,
        SAMPLE_SYSTEM_INTENTS, SAMPLE_HUMAN_INTENTS,
        confidences=SAMPLE_CONFIDENCES,
    )
    # 手算复核（写进输出，便于人对照）
    import difflib  # noqa: F401  （仅说明 LCS 口径；实现用自己的 DP）
    lcs = 5          # ["move","occupy","attack","hold_fire","occupy"] 是 6 项的子序列
    explicit = lcs / max(len(SAMPLE_SYSTEM_ACTIONS_FOR_ALIGN), len(SAMPLE_HUMAN_ACTIONS))
    implicit = 1.0   # 四位意图全中，权重和为 3.0 → 3.0/3.0
    r["worked_example"] = {
        "explicit_hand_calc": "LCS=5, max(len)=6 → 5/6 = %.6f" % explicit,
        "implicit_hand_calc": "四位全中，Σw=3.0 → 3.0/3.0 = 1.0",
        "combined_hand_calc": "0.5·%.6f + 0.5·1.0 = %.6f" % (explicit, 0.5 * explicit + 0.5),
    }
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="run_i17_checks",
        description="I17: runnable rule-conflict detection and cognitive-alignment worked example")
    ap.add_argument("--actions", help="动作清单 JSON（列表，元素含 kind/actor/target）")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    args = ap.parse_args(argv)

    conflicts = None
    if args.actions:
        p = Path(args.actions)
        if not p.is_file():
            print("  ! 动作文件不存在: %s" % p)
            return 2
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print("  ! 动作文件不是合法 JSON: %s" % e)
            return 2
        acts = data.get("actions") if isinstance(data, dict) else data
        if not isinstance(acts, list):
            print("  ! 动作清单应为列表（或 {'actions': [...]}）")
            return 2
        conflicts = {"n_actions": len(acts), "n_conflicts": len(rule_conflicts(acts)),
                     "conflicts": rule_conflicts(acts), "source": str(p)}
    else:
        conflicts = _demo_conflicts()
        conflicts["source"] = "builtin_synthetic_sample"

    alignment = _demo_alignment()
    out = {
        "rule_conflict_detection": conflicts,
        "cognitive_alignment": alignment,
        "sample_data_note": ("内置样例为**合成**最小序列，用于验证公式与 CLI 可运行；"
                             "**不构成 I17 判据 1 要求的真实专家评分结果**"),
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    print("I17 可运行检查")
    print()
    print("【1】规则冲突检测（判据 2）")
    print("  动作数: %d；冲突数: %d" % (conflicts["n_actions"], conflicts["n_conflicts"]))
    for c in conflicts["conflicts"]:
        print("     %-22s actor=%s target=%s  (%d,%d)"
              % (c["rule"], c["actor"], c["target"], c["index_a"], c["index_b"]))
    if not conflicts["conflicts"]:
        print("     （无冲突）")
    print("  说明: %s" % conflicts.get("note", ""))
    print()
    print("【2】认知对齐度（判据 3）")
    print("  显性行动吻合度 = %s" % alignment["explicit_action_agreement"])
    print("  隐性意图匹配度 = %s" % alignment["implicit_intent_match"])
    print("  认知对齐度     = %s" % alignment["value"])
    we = alignment["worked_example"]
    print("  手算复核:")
    for k in ("explicit_hand_calc", "implicit_hand_calc", "combined_hand_calc"):
        print("     %-22s %s" % (k, we[k]))
    print("  权重: %s（%s）" % (alignment["weights_declared"], alignment["note"]))
    print()
    print("  ⚠ %s" % out["sample_data_note"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
