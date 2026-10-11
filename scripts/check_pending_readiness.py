#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''**待批准项的"就绪度"核对**：每项声称"实现侧已尽"，那是**事实还是说法**？

## 为什么需要

`待批准` 这一定性意味着「**实现侧已尽，余需裁定/浏览器/评审**」。
**但那是我在备注里写的** —— 若某件的实现其实不存在，"已尽"就是虚报。

**本脚本把该定性变成可核对的**：对每个 `待批准` 项，检查其声称交付的
**实现文件与测试文件是否真实存在**。

## ⚠ 边界（重要）

**本脚本不判断"实现是否正确"**，也不判断"是否真的还需要外部条件"。
它只回答一个问题：**该文件在不在**。
故 `OK` 只表示"声称的产物存在"，**不表示该项可以批准**。
'''
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path("/remote-home/lizilong/shenji_agent")
GIT = "/remote-home/lizilong/.local/bin/git"

# 待批准项 → 其声称交付的产物（**从各节【进展】抄录，非我现编**）
PENDING_ARTIFACTS = {
    "I04": ["ai/adapters/agent_api.py", "docs/data_dictionary_mapping.md",
            "scripts/gen_multimodal_sample.py", "tests/test_i04_sample_script.py"],
    "I05": ["docs/five_dim_mapping.md", "tests/test_i05_five_dim_mapping.py"],
    "I06": ["scripts/diagnose_model_quality.py", "docs/intent_move_bias_diagnosis_20261010.md",
            "tests/test_i06_parallel_distribution_gate.py"],
    "I07": ["training/checkpoint_writer.py", "training/supervised.py",
            "scripts/verify_checkpoint.py", "tests/test_i07_checkpoint_policy.py"],
    "I08": ["situation3d/web/src/eventFeed.js", "situation3d/web/src/dataHealth.js",
            "tests/test_i08_data_health_offline.py", "tests/test_i08_status_panels_events.py"],
    "I09": ["situation3d/web/src/theme.css", "situation3d/web/src/symbology.js",
            "docs/visual_encoding_table.md", "tests/test_i09_symbology_render.py"],
    "I15": ["training/replay.py", "tests/test_i15_end_to_end_replay.py"],
    "D1": ["Shenji/configs/collection_plan_v1.json", "docs/eval_intensity_spec.md"],
    "C3": ["docs/engine_attribution_review_20261010.md"],
    "P1": ["training/eval_validity.py", "configs/eval_official_demo.yaml"],
    "P0-2": ["docs/deploy_situ155.md", "scripts/verify_db_native.py",
             "tests/test_p02_native_postgres.py"],
}

_fails = []


def main() -> int:
    r = subprocess.run([GIT, "ls-files"], cwd=REPO, capture_output=True, text=True)
    tracked = set((r.stdout or "").split("\n"))

    print("  ══════════ 待批准 %d 项：声称产物是否存在 ══════════" % len(PENDING_ARTIFACTS))
    print()
    for item in sorted(PENDING_ARTIFACTS):
        arts = PENDING_ARTIFACTS[item]
        found, miss, untracked = 0, [], []
        for a in arts:
            p = REPO / a
            if p.is_file():
                found += 1
                if a not in tracked:
                    untracked.append(a)
            else:
                miss.append(a)
        flag = "✓" if not miss and not untracked else "✗"
        print("  %s 【%-5s】%d/%d 存在%s" % (
            flag, item, found, len(arts),
            ("；**未追踪**: %s" % untracked) if untracked else ""))
        for m in miss:
            print("        **缺失**: %s" % m)
        _fails.extend(["%s: 缺 %s" % (item, m) for m in miss])
        _fails.extend(["%s: 未追踪 %s" % (item, u) for u in untracked])

    print()
    print("  ══════════ 结论 ══════════")
    if _fails:
        print("  **%d 处不符**（「待批准=实现侧已尽」这一说法在下列项上不成立）：" % len(_fails))
        for f in _fails:
            print("     - %s" % f)
        return 1
    print("  **全部待批准项声称的产物都存在且已提交** ✓")
    print("  ⚠ 本表**不判断实现是否正确**，也不判断「是否真的还需要外部条件」；")
    print("     `OK` 只表示「声称的产物存在」，**不等于该项可以批准**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
