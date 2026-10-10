"""I05 步骤 1/2：五维映射表 + 31007 附录B 逐项对照。

判据 1 要求「映射表覆盖五维全部字段与表B.1 的 17 项、表B.2 的 10 项，
**无空白或"待定"项**」；判据 3 要求「陆战/空战差异逐项有说明文字」。

本文件把这两条变成**可执行的核对**：
  - 行数必须与标准原文一致（17 / 10），且每项都有判定；
  - 判定只允许 等价 / 近似 / N/A 三档，**不允许留空或写"待定"**；
  - 每个 N/A 必须带原因文字（不是只写 N/A）；
  - 文档必须显式说明域差异（空战 vs 陆战）。

原文在 `.scratch/std/s31007.txt`（表 B.1 `:917`、表 B.2 `:972`）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DOC = REPO / "docs" / "five_dim_mapping.md"
STD = REPO.parent.parent / ".scratch" / "std" / "s31007.txt"

B1_START, B1_END = "## 3. 表 B.1 逐项映射", "### 表 B.1 统计"
B2_START, B2_END = "## 4. 表 B.2 逐项映射", "### 表 B.2 统计"


def _doc() -> str:
    if not DOC.is_file():
        pytest.skip("five_dim_mapping.md not present")
    return DOC.read_text(encoding="utf-8")


def _table_rows(text: str, start: str, end: str):
    seg = text[text.find(start):text.find(end)]
    assert seg, "找不到章节 %s" % start
    return [l for l in seg.split("\n") if re.match(r"^\|\s*\d+\s*\|", l)]


# ─────────────────────── 判据 1：覆盖 17 + 10 且无空白/待定

def test_table_b1_covers_all_17_items() -> None:
    rows = _table_rows(_doc(), B1_START, B1_END)
    assert len(rows) == 17, "表 B.1 应逐项覆盖 17 项，实际 %d" % len(rows)
    nums = [int(re.match(r"^\|\s*(\d+)", r).group(1)) for r in rows]
    assert nums == list(range(1, 18)), "编号必须 1..17 连续，实际 %s" % nums


def test_table_b2_covers_all_10_items() -> None:
    rows = _table_rows(_doc(), B2_START, B2_END)
    assert len(rows) == 10, "表 B.2 应逐项覆盖 10 项，实际 %d" % len(rows)
    nums = [int(re.match(r"^\|\s*(\d+)", r).group(1)) for r in rows]
    assert nums == list(range(1, 11)), "编号必须 1..10 连续，实际 %s" % nums


@pytest.mark.parametrize("section,start,end,n", [
    ("B.1", B1_START, B1_END, 17),
    ("B.2", B2_START, B2_END, 10),
])
def test_every_item_has_a_verdict_and_no_blank_or_pending(section, start, end, n) -> None:
    """**判据 1 的硬要求**：无空白、无"待定"。"""
    rows = _table_rows(_doc(), start, end)
    assert len(rows) == n
    verdicts = {"等价": 0, "近似": 0, "N/A": 0}
    for r in rows:
        cells = [c.strip() for c in r.strip().strip("|").split("|")]
        # 判定列必须含三档之一（用粗体标注）
        hit = [k for k in verdicts if ("**%s**" % k) in r]
        assert hit, "%s 第 %s 行没有判定（等价/近似/N-A）" % (section, cells[0])
        assert len(hit) == 1, "%s 第 %s 行判定不唯一: %s" % (section, cells[0], hit)
        verdicts[hit[0]] += 1
        # 最后一列（P3 对应 / N-A 原因）必须非空且够长 —— 防止"只写 N/A 不写原因"
        last = cells[-1]
        assert last and last not in ("—", "-", "N/A", "无"), \
            "%s 第 %s 行末列空白或只写了 N/A，必须写明原因" % (section, cells[0])
        if hit[0] == "N/A":
            assert len(last) >= 12, \
                "%s 第 %s 行为 N/A 但原因过短（%d 字）: %r" % (section, cells[0], len(last), last)
    assert verdicts["等价"] + verdicts["近似"] + verdicts["N/A"] == n


def test_no_pending_wording_anywhere_in_the_two_tables() -> None:
    """不许出现「待定 / 待补 / TBD / ？？？」这类占位。"""
    text = _doc()
    seg = text[text.find(B1_START):text.find(B2_END)]
    for bad in ("待定", "待补", "TBD", "？？？", "TODO"):
        assert bad not in seg, "对照表里出现占位词 %r —— 判据 1 要求无待定项" % bad


# ─────────────────────── 判据 3：差异逐项有说明

def test_doc_states_the_air_vs_land_domain_difference() -> None:
    """判据 3：陆战/空战差异**逐项有说明文字**。

    至少要显式点明：附录B 面向空战、P3 面向陆战，且这是"域不同"而非"实现不足"。
    """
    d = _doc()
    for token in ("空战", "陆战", "域"):
        assert token in d, "未说明域差异（缺 %r）" % token
    assert ("N/A" in d) or ("N／A" in d)
    # 必须点出几个典型的空战专有项在 P3 不存在
    for item in ("ship_dic", "submarine_dic", "sensor_dic", "weather"):
        assert item in d, "未逐项说明 %s" % item


def test_doc_records_the_three_verdict_levels_as_the_scale() -> None:
    """`I05`【风险与回滚】要求结论分「等价映射 / 近似映射 / N/A」三档。"""
    d = _doc()
    for lvl in ("等价", "近似", "N/A"):
        assert lvl in d, "缺少档位 %s" % lvl


# ─────────────────────── 与标准原文对得上（防自造）

def test_standard_text_actually_contains_both_tables() -> None:
    """**防止自造**：标准原文里确实有表 B.1 / B.2，且项数与文档一致。"""
    if not STD.is_file():
        pytest.skip("31007 source text not present in this checkout")
    t = STD.read_text(encoding="utf-8", errors="replace")
    assert "表 B.1" in t and "表 B.2" in t
    assert "附录B 空战算法与模型接口数据格式规范" in t.replace(" ", "")

    # 表 B.1 的 17 个参数名必须与原文一致
    b1_names = ["side_dic", "aircraft_dic", "ship_dic", "submarine_dic", "sensor_dic",
                "contact_dic", "doctrine_dic", "mssnpatrol_dic", "mssnstrike_dic",
                "loadout_dic", "mount_dic", "magazine_dic", "weapon_dic", "waypoint_dic",
                "weather", "response_dic", "logged_messages"]
    assert len(b1_names) == 17
    missing = [n for n in b1_names if n not in t]
    assert missing == [], "原文里找不到这些参数名: %s" % missing

    # 表 B.2 的 10 个动作名
    b2_names = ["add_mission", "set_mission_status", "set_mission_doctrine", "add_area",
                "add_wayline", "assign_area_to_mission", "assign_wayline_to_mission",
                "assign_target_to_mission", "assign_unit_to_mission", "set_unit_status"]
    assert len(b2_names) == 10
    missing2 = [n for n in b2_names if n not in t]
    assert missing2 == [], "原文里找不到这些动作名: %s" % missing2


def test_doc_lists_the_same_parameter_names_as_the_standard() -> None:
    """文档表格里的参数名必须**逐个**来自原文（否则就是自造映射）。"""
    d = _doc()
    b1 = ["side_dic", "aircraft_dic", "ship_dic", "submarine_dic", "sensor_dic",
          "contact_dic", "doctrine_dic", "mssnpatrol_dic", "mssnstrike_dic",
          "loadout_dic", "mount_dic", "magazine_dic", "weapon_dic", "waypoint_dic",
          "weather", "response_dic", "logged_messages"]
    b2 = ["add_mission", "set_mission_status", "set_mission_doctrine", "add_area",
          "add_wayline", "assign_area_to_mission", "assign_wayline_to_mission",
          "assign_target_to_mission", "assign_unit_to_mission", "set_unit_status"]
    for n in b1 + b2:
        assert n in d, "文档缺标准项 %s" % n


# ─────────────────────── P3 侧字段必须真实（不能编）

def test_p3_fields_cited_in_the_doc_really_exist() -> None:
    """文档点名的 P3 字段必须真的在 P3 里 —— 否则映射是编的。"""
    import ai.features as F

    d = _doc()
    # 帧顶层键：用真实复盘帧核对
    real = REPO / "replays" / "replay_1770632023.2594852.zip"
    if not real.is_file():
        pytest.skip("real replay not present")
    import json
    import zipfile

    from training.replay import frame_entries

    with zipfile.ZipFile(real) as z:
        for _, name in frame_entries(z):
            fr = json.loads(z.read(name).decode("utf-8"))
            if fr.get("operators"):
                break
    frame_keys = set(fr)
    op_keys = set()
    for o in fr.get("operators") or []:
        if isinstance(o, dict):
            op_keys |= set(o)

    # 文档里点名的几个关键 P3 字段
    for key in ("cities", "operators", "time", "valid_actions", "jm_points",
                "landmarks", "scores", "communication", "role_and_grouping_info"):
        assert key in frame_keys, "帧里没有 %s（文档却引用了）" % key
        assert key in d, "文档未提及帧键 %s" % key
    for key in ("cur_hex", "color", "speed", "move_path", "see_enemy_bop_ids",
                "carry_weapon_ids", "remain_bullet_nums", "blood"):
        assert key in op_keys, "算子字段 %s 在真实帧里不存在" % key
    # 确认 P3 **确实没有** weather（文档判 N/A 的依据）
    assert "weather" not in frame_keys, "帧里出现了 weather，文档的 N/A 判定需重写"
    assert "weather" not in op_keys


def test_action_types_cited_in_doc_exist_in_action_dictionary() -> None:
    """文档引用的动作类型必须在本仓动作字典里（步骤 3 的 `ACTION_SPECS`）。"""
    from ai.action_safety import ACTION_SPECS

    d = _doc()
    for t in (1, 2, 6, 11, 12, 17):
        assert t in ACTION_SPECS, "动作类型 %d 不在 ACTION_SPECS 里" % t
    # 文档应提到平台动作字典这一依据
    assert "valid_actions" in d
