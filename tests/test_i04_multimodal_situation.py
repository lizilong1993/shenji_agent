"""I04 步骤 1/2/3/5：多模态态势接口 `get_multimodal_situation()`。

判据 2 要求「样本脚本运行成功，输出非空 `text`、可打开的 `image`、
长度等于 108 的 `vector`」—— 本文件把它做成**可重复的测试**，而不是跑一次脚本。

用**真实复盘帧**驱动（`replays/replay_1770632023.2594852.zip`），
比合成帧更有说服力：文本模态真的走了 P2 的 `build_brief()`。
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ai.adapters import AgentAPI  # noqa: E402
from ai.adapters.agent_api import (EXPECTED_VECTOR_DIM,  # noqa: E402
                                  MULTIMODAL_SCHEMA_VERSION)

REAL = REPO / "replays" / "replay_1770632023.2594852.zip"


def _real_frames(n: int = 40):
    if not REAL.is_file():
        pytest.skip("real replay artifact not present")
    from training.replay import frame_entries

    out = []
    with zipfile.ZipFile(REAL) as z:
        for _, name in frame_entries(z):
            out.append(json.loads(z.read(name).decode("utf-8")))
            if len(out) >= n:
                break
    return out


# ─────────────────────── ① 四字段契约（步骤 1）

def test_returns_exactly_the_four_contracted_fields() -> None:
    got = AgentAPI().get_multimodal_situation(_real_frames(), image="x.png")
    assert set(got) == {"text", "image", "vector", "meta"}, sorted(got)


def test_meta_carries_source_time_and_version() -> None:
    """步骤 1 要求 `meta` 含**来源、时间、版本**。"""
    meta = AgentAPI().get_multimodal_situation(_real_frames()).get("meta")
    for key in ("source", "generated_at", "schema_version"):
        assert meta.get(key), "meta 缺 %s" % key
    assert meta["schema_version"] == MULTIMODAL_SCHEMA_VERSION


# ─────────────────────── ② 三件套齐全（判据 2）

def test_text_modality_is_non_empty_and_reuses_p2_brief() -> None:
    """步骤 2：文本**复用 P2 `brief.py`**，不另写叙述逻辑。

    证据：产出的文本带 `render_text()` 的特征抬头（`TS简报 —— `）。
    """
    got = AgentAPI().get_multimodal_situation(
        _real_frames(), case_id="smoke", case_name="冒烟")
    text = got["text"]
    assert text.strip(), "text 为空"
    assert "TS简报" in text, "文本不是 P2 render_text 的产物，可能另写了一套叙述"
    assert got["meta"].get("brief_headline"), "未记录 brief 摘要"


def test_image_modality_returns_reference_and_size_without_rendering() -> None:
    """步骤 3：**只返回引用与尺寸**，接口内不做渲染阻塞。"""
    got = AgentAPI().get_multimodal_situation(
        _real_frames(), image={"path": "out/x.png", "width": 800, "height": 600, "format": "png"})
    img = got["image"]
    assert img["path"] == "out/x.png"
    assert img["width"] == 800 and img["height"] == 600
    # 断言**没有**在接口里真去渲染：路径不存在却仍返回引用
    assert not Path("out/x.png").exists()


def test_vector_length_is_108_from_real_observation() -> None:
    """判据 2：`vector` 长度等于 **108**，且由 P3 `ai.features` 生成。"""
    frames = _real_frames()
    got = AgentAPI().get_multimodal_situation(frames, observation=frames[-1])
    assert len(got["vector"]) == EXPECTED_VECTOR_DIM == 108
    assert got["meta"]["vector_feature_dim"] == 108
    assert "P3" in got["meta"]["vector_source"]


def test_all_three_modalities_available_in_the_full_case() -> None:
    frames = _real_frames()
    got = AgentAPI().get_multimodal_situation(
        frames, observation=frames[-1], image="out/s.png", case_id="c1")
    assert got["meta"]["modalities_available"] == ["text", "image", "vector"]
    assert got["meta"]["degraded"] is False
    assert got["text"].strip() and got["image"] and len(got["vector"]) == 108


# ─────────────────────── ③ 降级：分模态、不隐式替代（回滚约定）

def test_each_modality_fails_independently_and_is_marked_degraded() -> None:
    """**回滚约定的核心**：任一模态失败**置空并标 `degraded=true`**，其余不受影响。"""
    frames = _real_frames()
    got = AgentAPI().get_multimodal_situation(frames, case_id="c")   # 不给 image/vector
    assert got["text"].strip(), "文本不该因为别的模态缺失而失败"
    assert got["image"] == {}
    assert got["vector"] == []
    assert got["meta"]["degraded"] is True
    reasons = " ".join(got["meta"].get("degraded_reasons") or [])
    assert "image" in reasons and "vector" in reasons


def test_text_failure_does_not_break_other_modalities() -> None:
    """文本失败时，向量与图像仍应可用（分模态隔离）。"""
    frames = _real_frames()
    got = AgentAPI().get_multimodal_situation(
        None, observation=frames[-1], image="out/s.png")     # frames=None → 文本失败
    assert got["text"] == ""
    assert len(got["vector"]) == 108
    assert got["image"]["path"] == "out/s.png"
    assert got["meta"]["degraded"] is True


def test_wrong_length_vector_is_rejected_not_truncated_or_padded() -> None:
    """**禁止隐式替代**：长度不符必须拒绝，**不截断也不补齐**。

    静默截断/补齐会改变语义，正是 `I04` 回滚要求点名的"隐式替代"。
    """
    frames = _real_frames()
    for bad_len in (50, 107, 109, 216):
        got = AgentAPI().get_multimodal_situation(frames, vector=[1.0] * bad_len)
        assert got["vector"] == [], "长度 %d 的向量被接受了" % bad_len
        assert any("不截断也不补齐" in r for r in got["meta"]["degraded_reasons"])


def test_non_numeric_vector_is_rejected() -> None:
    got = AgentAPI().get_multimodal_situation(_real_frames(), vector=["a"] * 108)
    assert got["vector"] == []
    assert any("数值序列" in r for r in got["meta"]["degraded_reasons"])


def test_bad_image_reference_is_rejected() -> None:
    for bad in ({"no_path": 1}, 12345, []):
        got = AgentAPI().get_multimodal_situation(_real_frames(), image=bad)
        assert got["image"] == {}
        assert got["meta"]["degraded"] is True


# ─────────────────────── ④ Q10：两套 schema **不合并**

def test_p2_schema_is_registered_separately_and_not_merged() -> None:
    """`Q10` 未关闭：接口**分别**登记两套 108 维，**不合并、不假装同义**。

    见 `docs/data_dictionary_mapping.md`（实测证明两套语义不同、且 P3 有 50 维空洞）。
    """
    frames = _real_frames()
    got = AgentAPI().get_multimodal_situation(frames, observation=frames[-1])
    p2 = got["meta"].get("p2_vector_schema")
    assert isinstance(p2, dict), "未登记 P2 schema"
    assert p2["schema_version"] == "ifv2"
    assert p2["feature_dim"] == 108
    assert "语义不同" in p2["note"], "未声明两套语义不同"
    # 返回值里**只有一个** vector —— 不得把两套拼成一个
    assert isinstance(got["vector"], list) and len(got["vector"]) == 108


# ─────────────────────── ⑤ 接口健壮性

def test_no_arguments_at_all_does_not_raise() -> None:
    got = AgentAPI().get_multimodal_situation()
    assert set(got) == {"text", "image", "vector", "meta"}
    assert got["meta"]["degraded"] is True


def test_forwarded_counter_includes_multimodal() -> None:
    """可审计：新接口也要出现在转发计数里。"""
    api = AgentAPI()
    api.get_multimodal_situation()
    assert api.forwarded.get("get_multimodal_situation") == 1


def test_return_is_json_serialisable() -> None:
    """样本要能落盘，故整体必须可 JSON 序列化。"""
    got = AgentAPI().get_multimodal_situation(_real_frames(), image="x.png")
    blob = json.dumps(got, ensure_ascii=False)
    assert len(blob) > 50
