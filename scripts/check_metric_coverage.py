#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''**指标 × 实现 × 测试** 对应关系核对。

## 为什么需要

§106–§122 交付了 6 个新模块，覆盖 31007 6.2.3/6.2.4 的全部点名指标。
**但"交付了模块"不等于"指标都被覆盖"**：
某指标可能没有实现、实现了但没测试、或测试文件其实不存在。

**本脚本把这张对应表变成可核对的**，而不是靠我逐项回忆。

## 核对四件事（每指标一行）

| 列 | 判定 |
| --- | --- |
| **指标** | 来自 31007 点名的清单（本脚本硬编码，来源见各条注释） |
| **实现** | 该模块文件是否存在 **且** 导出期望的函数名 |
| **测试** | 测试文件是否存在 **且** 用例数 > 0 |
| **判据** | 归属的任务项 ID（用于回溯到主清单） |

## ⚠ 边界

**"存在且有用例"不等于"正确"** —— 正确性由各自的测试与实测负责。
本表只防**"以为覆盖了、其实没有"**。
'''
from __future__ import annotations

import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path("/remote-home/lizilong/shenji_agent")

# (指标名, 标准条款, 实现文件, 期望导出, 测试文件, 归属任务项)
METRICS = [
    ("三层评估框架 + 报告模板", "P3 步骤1/6", "training/eval_report.py",
     ["build_report", "render_markdown"], "tests/test_p3_eval_report.py", "P3"),
    ("蒙特卡洛 + 置信区间", "31007 6.1.2", "training/monte_carlo.py",
     ["aggregate", "aggregate_by_case", "overall_summary", "t_critical"],
     "tests/test_i16_monte_carlo_ci.py", "I16"),
    ("专家评分聚合", "31007 6.2.3", "training/expert_rating.py",
     ["weighted_total", "icc_2_1", "rule_conflicts", "aggregate_expert_scores"],
     "tests/test_p3_expert_rating.py", "I17"),
    ("Kendall's W 一致性", "I17 步骤1 点名", "training/kendall_w.py",
     ["kendall_w", "compare_w_and_icc"], "tests/test_i17_kendall_w.py", "I17"),
    ("认知对齐度", "31007 6.2.4", "training/alignment.py",
     ["action_agreement", "intent_match", "cognitive_alignment", "aggregate_alignment"],
     "tests/test_p3_cognitive_alignment.py", "I17"),
    ("跨域泛化", "31007 6.2.4", "training/robustness.py",
     ["cross_domain_generalisation"], "tests/test_i17_robustness.py", "I17"),
    ("抗扰容错阈值", "31007 6.2.4", "training/robustness.py",
     ["disturbance_threshold"], "tests/test_i17_robustness.py", "I17"),
    ("策略鲁棒性", "31007 6.2.3", "training/robustness.py",
     ["policy_robustness"], "tests/test_i17_robustness.py", "I17"),
    ("决策可解释性", "31007 6.2.4", "ai/explain.py",
     ["explain_decision"], "tests/test_i17_explainability_safety.py", "I17"),
    ("决策安全约束", "31007 6.2.4", "ai/explain.py",
     ["check_safety", "safety_summary"], "tests/test_i17_explainability_safety.py", "I17"),
    ("规则冲突检测（可运行）", "31007 6.2.3", "scripts/run_i17_checks.py",
     ["main"], "tests/test_i17_runnable_checks.py", "I17"),
    ("专家评分协议（文档）", "31007 6.2.3", "docs/expert_rating_protocol.md",
     None, "tests/test_p3_expert_protocol_doc.py", "I17"),
]

_fails = []


def check(cond, label, detail=""):
    if not cond:
        _fails.append(label)
    print("  %s %-44s %s" % ("✓" if cond else "✗", label, detail))


def main() -> int:
    print("  ══════════ 指标 × 实现 × 测试 ══════════")
    print()
    hdr = "  %-26s %-16s %-8s %-8s" % ("指标", "条款", "实现", "测试")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, clause, impl, exports, test, item in METRICS:
        ip = REPO / impl
        tp = REPO / test
        ok_i = ip.is_file()
        ok_t = tp.is_file()
        n_tests = 0
        if ok_t:
            n_tests = len(re.findall(r"(?m)^def test_", tp.read_text(encoding="utf-8", errors="ignore")))
        print("  %-26s %-16s %-8s %-8s" % (name, clause,
                                           "✓" if ok_i else "✗",
                                           "%d 例" % n_tests if ok_t else "✗"))
        _fails.extend([] if ok_i else ["%s: 实现缺失 %s" % (name, impl)])
        _fails.extend([] if ok_t and n_tests else ["%s: 测试缺失或空 %s" % (name, test)])

        # 导出检查
        if exports and ok_i and ip.suffix == ".py":
            src = ip.read_text(encoding="utf-8", errors="ignore")
            miss = [e for e in exports if not re.search(r"(?m)^def %s\(" % re.escape(e), src)]
            if miss:
                _fails.append("%s: %s 未导出 %s" % (name, impl, miss))

    print()
    print("  ══════════ 任务项回溯 ══════════")
    items = sorted({m[5] for m in METRICS})
    for it in items:
        n = sum(1 for m in METRICS if m[5] == it)
        print("     %-8s %d 项指标" % (it, n))

    print()
    print("  ══════════ 汇总 ══════════")
    print("  指标数: %d；涉及任务项: %s" % (len(METRICS), " ".join(items)))
    if _fails:
        print("  **%d 处缺口，需处理**：" % len(_fails))
        for f in _fails:
            print("     - %s" % f)
        return 1
    print("  **全部有实现、有测试、有导出** ✓")
    print("  ⚠ 本表只防「以为覆盖了、其实没有」；**正确性由各自测试负责**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
