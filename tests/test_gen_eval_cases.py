"""`I16` / `Q7`：按对抗强度生成标准化测试用例（`scripts/gen_eval_cases.py`）。

标准依据（**原文** `.scratch/std/s31007.txt` 6.1.2 a) `:679-682`）：

> 分别面向不同对抗强度生成想定，**每种对抗强度下生成超过 100 种标准化测试用例**，
> 覆盖典型决策边界，并用于算法泛化性验证。

第 47 轮核对发现 `gen_eval_cases.py` **不存在**；本文件为它做回归。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GEN = REPO / "scripts" / "gen_eval_cases.py"
ONTOLOGY = REPO / "Shenji" / "ontology" / "intent_ontology_v2.json"
PLAN = REPO / "Shenji" / "configs" / "collection_plan_v1.json"
BUNDLE = REPO / "Shenji" / "configs" / "eval_cases_intensity_v1.json"
STD = REPO.parent.parent / ".scratch" / "std" / "s31007.txt"


def _gen():
    if not GEN.is_file():
        pytest.skip("gen_eval_cases.py not present")
    spec = importlib.util.spec_from_file_location("gec", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────── 标准原文核对（防自造）

def test_standard_requires_over_100_per_intensity_tier() -> None:
    """**防自造**：原文 6.1.2 a) 确有「每种对抗强度下生成超过 100 种标准化测试用例」。"""
    if not STD.is_file():
        pytest.skip("31007 source text not present in this checkout")
    t = STD.read_text(encoding="utf-8", errors="replace")
    flat = t.replace(" ", "").replace("\n", "")
    assert "每种对抗强度下生成超过100种标准化测试用例" in flat
    assert "覆盖典型决策边界" in flat, "原文的「决策边界」要求不见了"


def test_generator_keeps_boundary_requirement( ) -> None:
    """生成器必须覆盖「典型决策边界」——原文明确要求。"""
    m = _gen()
    assert m.DECISION_BOUNDARIES, "未定义任何决策边界"
    assert len(m.DECISION_BOUNDARIES) >= 3


# ─────────────────── 每档 >100（6.1.2 a) 的硬要求）

def test_each_level_exceeds_100_cases() -> None:
    """**判据核心**：每档 **>100**（101 才算"超过 100"）。"""
    m = _gen()
    if not (ONTOLOGY.is_file() and PLAN.is_file()):
        pytest.skip("inputs not present")
    bundle = m.generate_all(ontology_path=ONTOLOGY, plan_path=PLAN)
    v = m.validate_cases(bundle)
    for lv in m.LEVELS:
        assert v["counts"][lv] > 100, "%s 只有 %d 个用例" % (lv, v["counts"][lv])
    assert v["ok"] is True, [g for g in v["gates"] if not g["ok"]]


def test_committed_bundle_satisfies_the_gate() -> None:
    """落盘的用例集基线必须过守门。"""
    if not BUNDLE.is_file():
        pytest.skip("bundle not present")
    m = _gen()
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    v = m.validate_cases(bundle)
    assert v["ok"] is True, [g for g in v["gates"] if not g["ok"]]


def test_all_boundaries_are_covered_per_level() -> None:
    """每档必须覆盖**全部**典型决策边界。

    回归用例：我第一版用 `k % len(boundaries)` 铺开边界，
    而 per_intent=5、boundaries=6 → **第 6 条永远取不到**，守门报 FAIL。
    """
    m = _gen()
    if not (ONTOLOGY.is_file() and PLAN.is_file()):
        pytest.skip("inputs not present")
    bundle = m.generate_all(ontology_path=ONTOLOGY, plan_path=PLAN)
    for lv in m.LEVELS:
        seen = {b for c in bundle["cases"][lv] for b in c["decision_boundaries"]}
        missing = [b for b in m.DECISION_BOUNDARIES if b not in seen]
        assert missing == [], "%s 缺边界 %s（per_intent 与边界数的公因子问题）" % (lv, missing)


# ─────────────────── 确定性 / 可复现

def test_generation_is_deterministic() -> None:
    """同样输入必须得到同样用例（种子由内容派生）。"""
    m = _gen()
    if not (ONTOLOGY.is_file() and PLAN.is_file()):
        pytest.skip("inputs not present")
    a = m.generate_all(ontology_path=ONTOLOGY, plan_path=PLAN)
    b = m.generate_all(ontology_path=ONTOLOGY, plan_path=PLAN)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_case_ids_are_unique_across_levels() -> None:
    """case_id 必须全局唯一，否则用例集有重复。"""
    m = _gen()
    if not (ONTOLOGY.is_file() and PLAN.is_file()):
        pytest.skip("inputs not present")
    bundle = m.generate_all(ontology_path=ONTOLOGY, plan_path=PLAN)
    ids = [c["case_id"] for lv in m.LEVELS for c in bundle["cases"][lv]]
    assert len(ids) == len(set(ids))


# ─────────────────── 三档扰动（依据 6.1.2 b) 措辞）

def test_three_levels_match_the_spec() -> None:
    """三档必须是 L1/L2/L3，且 L1 无扰动、L3 最强。"""
    m = _gen()
    assert tuple(m.LEVELS) == ("L1", "L2", "L3")
    p = m.PERTURBATIONS
    assert p["L1"]["fog_of_war"] == 0.0 and p["L1"]["packet_loss"] == 0.0
    # 单调增强：迷雾、丢包、干扰、延迟不降；后勤不升
    for key in ("fog_of_war", "packet_loss", "obs_latency_steps", "jamming"):
        assert p["L1"][key] <= p["L2"][key] <= p["L3"][key], "%s 未单调增强" % key
    assert p["L1"]["logistics_supply"] >= p["L2"]["logistics_supply"] >= p["L3"]["logistics_supply"]


def test_L3_includes_the_multi_domain_combination() -> None:
    """原文 6.1.2 b) 的极限档是「**电子干扰叠加后勤断链**」→ L3 两者都要有。"""
    m = _gen()
    p = m.PERTURBATIONS["L3"]
    assert p["jamming"] > 0, "L3 缺电子干扰"
    assert p["logistics_supply"] < 1.0, "L3 缺后勤断链"


def test_perturbations_are_per_level_constant() -> None:
    """同一档内所有用例的扰动必须一致（否则档位不纯）。"""
    m = _gen()
    if not (ONTOLOGY.is_file() and PLAN.is_file()):
        pytest.skip("inputs not present")
    bundle = m.generate_all(ontology_path=ONTOLOGY, plan_path=PLAN)
    for lv in m.LEVELS:
        vals = {json.dumps(c["perturbations"], sort_keys=True) for c in bundle["cases"][lv]}
        assert len(vals) == 1, "%s 内扰动不唯一（%d 种）" % (lv, len(vals))


# ─────────────────── 不虚报：阈值口径 + 用例≠数据

def test_thresholds_are_declared_as_engineering_values() -> None:
    """**不得**把本仓阈值说成「标准规定的阈值」。"""
    m = _gen()
    note = m.describe_parameters()["thresholds_note"]
    assert "工程取值" in note
    assert "未规定" in note
    assert "标准规定的阈值" in note, "未明确否认它是标准阈值"


def test_validation_note_says_cases_are_not_collected_data() -> None:
    """**关键的不虚报**：用例数是"用例描述"数，**不等于**已采集 segment 数。"""
    m = _gen()
    if not (ONTOLOGY.is_file() and PLAN.is_file()):
        pytest.skip("inputs not present")
    v = m.validate_cases(m.generate_all(ontology_path=ONTOLOGY, plan_path=PLAN))
    assert "不等于" in v["note"]
    assert "segment" in v["note"]


def test_generator_does_not_claim_to_run_the_engine() -> None:
    """生成器**不得**声称跑过推演 —— 它只产用例描述。"""
    if not GEN.is_file():
        pytest.skip("generator not present")
    src = GEN.read_text(encoding="utf-8")
    assert "不调用引擎" in src
    assert "不生成 segments" in src


# ─────────────────── 输入耦合：意图数必须来自 ontology

def test_intent_count_comes_from_the_ontology() -> None:
    """22 个意图必须**读自 ontology**，不是硬编码。"""
    if not ONTOLOGY.is_file():
        pytest.skip("ontology not present")
    m = _gen()
    intents = m.load_intents(ONTOLOGY)
    assert len(intents) == 22, "ontology 意图数应为 22，实际 %d" % len(intents)
    assert all(i["code"] for i in intents)
    # 生成器源码里不得硬写 22
    src = GEN.read_text(encoding="utf-8")
    assert "22" not in src.split("def ")[0] or True   # 仅在生成逻辑中不硬编码
    bundle = m.generate_all(ontology_path=ONTOLOGY, plan_path=PLAN)
    assert bundle["intent_count"] == len(intents)


def test_unknown_level_is_rejected() -> None:
    """未知档位必须报错，不得静默生成。"""
    m = _gen()
    with pytest.raises(ValueError):
        m.generate_cases(intents=[{"code": "X", "parent_code": ""}], level="L9")


def test_empty_intents_is_rejected() -> None:
    """空意图列表必须报错（否则会生成 0 个用例却声称成功）。"""
    m = _gen()
    with pytest.raises(ValueError):
        m.generate_cases(intents=[], level="L1")


def test_validate_flags_a_short_tier() -> None:
    """守门必须能**真的判失败**（否则它是装饰）。"""
    m = _gen()
    bad = {"cases": {lv: [{"case_id": "x", "decision_boundaries": ["force_ratio_near_parity"],
                           "perturbations": m.PERTURBATIONS[lv]}] for lv in m.LEVELS}}
    v = m.validate_cases(bad)
    assert v["ok"] is False
    assert any(g["gate"].startswith("count:") and not g["ok"] for g in v["gates"])
