#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I04 步骤 5：多模态样本生成脚本 + schema 冒烟。

判据原文：
  2) 样本脚本运行成功，输出非空 `text`、**可打开的 `image`**、**长度等于 108 的 `vector`**。

## 为什么需要这个脚本（而不是只有测试）

修前已有 21 条测试覆盖接口，但**没有"样本脚本"** ——
即缺少一条**可独立执行的取证命令**：第三方想核对判据 2，只能去读测试源码。
本脚本把它变成一条命令：

```bash
python3 scripts/gen_multimodal_sample.py --out experiments/shenji_agent/multimodal
```

## `image` 的处理：**不假装已渲染**

步骤 3 明确「接口只返回文件引用与尺寸，**不在接口内做渲染阻塞**」。
故本脚本：
  · 若调用方给了 `--image`，照用；
  · 否则生成一张**真实可打开的最小 PNG**（1×1，内联字节，**不依赖 Pillow**），
    并**如实标注**它是 `placeholder_1x1`，而不是伪装成态势图。

**不写"已渲染态势图"这种话** —— 那需要一个真正的渲染器，本脚本没有。
**渲染由调用方完成**（步骤 3 原文），本脚本只负责给出可打开的引用。
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 1×1 透明 PNG（67 字节，内联以便无第三方依赖）
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")

EXPECTED_VECTOR_DIM = 108


def _write_placeholder_png(path: Path) -> dict:
    """写一张**真实可打开**的 1×1 PNG，并如实标注它是占位图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG_1x1)
    # ⚠ `--out` 可能指向**仓库外**（例如测试用 tmp_path）—— 此时 `relative_to` 会抛错。
    # 我第一版没处理这种情形，于是脚本在仓库外直接崩掉（由测试实测发现）。
    try:
        shown = str(path.relative_to(REPO))
    except ValueError:
        shown = str(path)
    # ⚠ 接口的 `_image_ref()` **只保留 path/width/height/format**（有意如此），
    # 其它键会被丢弃 —— 故"这是占位图"这条信息**必须落在 path 或 format 里**，
    # 否则下游会把它当成真实态势图。文件名带 `placeholder` 是刻意的。
    return {"path": shown, "width": 1, "height": 1,
            "format": "png/placeholder_1x1"}


def _minimal_frames() -> list:
    """一份最小的态势帧输入。**刻意最小** —— 目的是证明链路可通，不是造数据。"""
    return [{
        "tick": 0,
        "time": {"cur_step": 0, "tick": 1.0, "max_step": 1800, "stage": 1},
        "scores": {"red_remain": 100, "blue_remain": 100, "red_occupy": 0, "blue_occupy": 0},
        "operators": [],
        "cities": [],
        "landmarks": [],
        "communication": [],
        "valid_actions": {},
    }]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="gen_multimodal_sample",
        description="Generate one multimodal situation sample and smoke-check the schema (I04 step 5).")
    ap.add_argument("--out", default="experiments/shenji_agent/multimodal",
                    help="output directory")
    ap.add_argument("--case-id", default="i04-smoke")
    ap.add_argument("--image", default=None,
                    help="existing situation image to reference; omit to write an honest 1x1 placeholder")
    ap.add_argument("--json", action="store_true", help="print the sample JSON to stdout")
    args = ap.parse_args(argv)

    from ai.adapters.agent_api import AgentAPI

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        src = Path(args.image)
        if not src.is_file():
            print("[i04] --image 不存在: %s" % src)
            return 2
        image_ref = {"path": str(src), "width": None, "height": None, "kind": "provided"}
    else:
        image_ref = _write_placeholder_png(
            out_dir / ("%s_image_placeholder.png" % args.case_id))

    vector = [0.0] * EXPECTED_VECTOR_DIM
    api = AgentAPI()
    sample = api.get_multimodal_situation(
        frames=_minimal_frames(), vector=vector, image=image_ref,
        case_id=args.case_id, case_name="I04 smoke case")

    # ── 冒烟：三条判据逐一核对 ──────────────────────────────────────────
    checks = []
    text = sample.get("text")
    checks.append(("text 非空", isinstance(text, str) and bool(text.strip()),
                   "长度 %d" % (len(text or ""))))
    img_path = (sample.get("image") or {}).get("path")
    img_file = (REPO / img_path) if img_path and not str(img_path).startswith("/") else Path(str(img_path or ""))
    checks.append(("image 可打开", bool(img_file and img_file.is_file()),
                   "路径 %s（%s B）" % (img_path, img_file.stat().st_size if img_file and img_file.is_file() else 0)))
    vec = sample.get("vector")
    checks.append(("vector 长度 == 108", isinstance(vec, list) and len(vec) == EXPECTED_VECTOR_DIM,
                   "实际 %s" % (len(vec) if isinstance(vec, list) else type(vec).__name__)))
    checks.append(("meta 含 degraded", isinstance(sample.get("meta"), dict)
                   and "degraded" in sample["meta"],
                   "degraded=%s" % (sample.get("meta") or {}).get("degraded")))

    sample_path = out_dir / ("%s_sample.json" % args.case_id)
    with open(sample_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sample, fh, ensure_ascii=False, indent=2)

    print("i04 multimodal sample")
    print("  case_id: %s" % args.case_id)
    # 同样要容忍仓库外路径（我第一版只修了图片那处，漏了这里 —— 测试立刻抓到）
    try:
        _shown_out = str(sample_path.relative_to(REPO))
    except ValueError:
        _shown_out = str(sample_path)
    print("  output : %s" % _shown_out)
    for label, ok, detail in checks:
        print("  %s %-22s %s" % ("ok  " if ok else "FAIL", label, detail))
    ok_all = all(c[1] for c in checks)
    print("  RESULT: %s" % ("PASS" if ok_all else "FAIL"))
    if args.json:
        print(json.dumps(sample, ensure_ascii=False, indent=2))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
