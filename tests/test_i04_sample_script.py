"""I04 步骤 5：多模态**样本脚本**可独立执行，并使判据 2 成为一条可复跑的命令。

## 修前的缺口

`I04` 已有 21 条测试覆盖接口（`test_i04_multimodal_situation.py` 15 条、
`test_i04_feature_dimension_mapping.py` 6 条），但**没有"样本脚本"** ——
即缺少一条**可独立执行的取证命令**：第三方想核对判据 2，只能去读测试源码。

步骤 5 原文：「新增**多模态样本生成脚本**与 schema 冒烟测试，
确认至少 1 个案例**三件套齐全**且可被策略侧读取。」

## 判据 2 的原文与实现

> 样本脚本运行成功，输出非空 `text`、**可打开的 `image`**、**长度等于 108 的 `vector`**。

脚本 `scripts/gen_multimodal_sample.py` 逐条自检并打印 `PASS/FAIL`，退出码与之对应。

## `image` 的处理：**不假装已渲染**

步骤 3 明确「接口只返回文件引用与尺寸，**不在接口内做渲染阻塞**」。
故无 `--image` 时脚本生成一张**真实可打开的最小 PNG**（1×1，内联字节，不依赖 Pillow），
并**如实标注** `kind = placeholder_1x1` —— **不写"已渲染态势图"**（那需要真正的渲染器）。
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gen_multimodal_sample.py"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _run(tmp_path: Path, *extra):
    if not SCRIPT.is_file():
        pytest.skip("gen_multimodal_sample.py not present")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path)] + list(extra),
        cwd=REPO, capture_output=True, text=True)


# ────────────── 脚本可执行且自检通过

def test_sample_script_runs_and_self_checks_pass(tmp_path: Path) -> None:
    """**判据 2 的可复跑形式**：脚本退出码 0，且四条自检全 ok。"""
    r = _run(tmp_path)
    assert r.returncode == 0, (r.stdout or "") + (r.stderr or "")[-600:]
    out = r.stdout or ""
    assert "RESULT: PASS" in out, out[-600:]
    for label in ("text 非空", "image 可打开", "vector 长度 == 108"):
        assert "ok   " + label in out, "自检缺 %s" % label


def test_sample_writes_the_three_modalities(tmp_path: Path) -> None:
    """输出样本 JSON 必须含 `text`/`image`/`vector`/`meta` 四键。"""
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout
    samples = list(tmp_path.glob("*_sample.json"))
    assert samples, "未产出样本 JSON"
    d = json.loads(samples[0].read_text(encoding="utf-8"))
    assert set(d) >= {"text", "image", "vector", "meta"}, sorted(d)
    assert isinstance(d["text"], str) and d["text"].strip(), "text 为空"
    assert len(d["vector"]) == 108, "vector 长度 %s" % len(d["vector"])
    assert isinstance(d["meta"], dict) and "degraded" in d["meta"]


def test_placeholder_image_is_a_real_openable_png(tmp_path: Path) -> None:
    """**"可打开"要真的可打开**：校验 PNG 魔数并读出 IHDR 尺寸。

    只断言"文件存在"是不够的 —— 一个 0 字节文件也"存在"。
    """
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout
    pngs = list(tmp_path.glob("*.png"))
    assert pngs, "未产出 image 文件"
    b = pngs[0].read_bytes()
    assert b[:8] == PNG_MAGIC, "不是合法 PNG（魔数不符）"
    assert len(b) >= 24, "PNG 过短，无 IHDR"
    w, h = struct.unpack(">II", b[16:24])
    assert w >= 1 and h >= 1, "IHDR 尺寸非法: %dx%d" % (w, h)


def test_placeholder_is_labelled_not_pretended_to_be_rendered(tmp_path: Path) -> None:
    """**不假装已渲染**：占位图必须自带 `kind` 说明，不得伪装成态势图。

    步骤 3 要求渲染由调用方完成；若脚本把 1×1 图当"态势图"返回，
    下游会以为拿到了真实渲染结果。
    """
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout
    d = json.loads(next(tmp_path.glob("*_sample.json")).read_text(encoding="utf-8"))
    img = d.get("image") or {}
    # ⚠ 接口的 `_image_ref()` **只保留 path/width/height/format**（有意设计），
    # 故"占位"标识必须体现在**路径或 format** 上。我第一版断言 `kind`/`note`，
    # 而它们**会被接口正常丢弃** —— 那是我的断言不符接口契约，不是接口的缺陷。
    assert "placeholder" in str(img.get("path", "")), "路径未标明占位: %s" % img
    assert "placeholder" in str(img.get("format", "")), "format 未标明占位: %s" % img


def test_provided_image_is_used_when_given(tmp_path: Path) -> None:
    """`--image` 给定真实文件时应引用它，而不是仍写占位图。"""
    real = tmp_path / "real_situation.png"
    real.write_bytes(PNG_MAGIC + b"\x00" * 40)
    r = _run(tmp_path, "--image", str(real))
    assert r.returncode == 0, r.stdout
    d = json.loads(next(tmp_path.glob("*_sample.json")).read_text(encoding="utf-8"))
    assert (d.get("image") or {}).get("path") == str(real), "未引用给定的 image"
    # 给定的图不该被标成占位
    assert "placeholder" not in str((d.get("image") or {}).get("path"))


def test_missing_image_argument_fails_loudly(tmp_path: Path) -> None:
    """**不静默降级**：`--image` 指向不存在的文件时应报错退出，而不是悄悄用占位图。"""
    r = _run(tmp_path, "--image", str(tmp_path / "nope.png"))
    assert r.returncode != 0, "不存在的 --image 竟被视为成功"


def test_script_documents_the_placeholder_decision() -> None:
    """脚本必须写明**为何用占位图**（否则后人会以为它是偷懒）。"""
    if not SCRIPT.is_file():
        pytest.skip("script not present")
    s = SCRIPT.read_text(encoding="utf-8")
    assert "不假装已渲染" in s or "不假装" in s
    assert "渲染由调用方完成" in s or "渲染是调用方" in s


def test_script_exit_code_matches_its_own_checks() -> None:
    """**自检与退出码必须一致** —— 否则 CI 会拿到"打印 PASS 但返回 1"这类矛盾。"""
    s = SCRIPT.read_text(encoding="utf-8")
    assert "return 0 if ok_all else 1" in s, "退出码未与自检结果绑定"
