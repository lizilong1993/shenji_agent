"""I06 步骤 5：奖励消融（判据 2）与偏差告警（判据 3）。

## 本轮查清的两个事实（都写进了测试）

**事实一**：那 4 组消融配置是**死配置** —— `reward_shaping` / `penalty_scale`
在 `training/`、`ai/`、`scripts/` 下被 **0 个 `.py`** 引用，
`training/selfplay.py` 里连 `reward_shaping` 都没出现。
**故"跑那 4 组自博弈"会得到 4 份完全相同的输出**，不构成消融。

**事实二**：配置注释里的「需引擎鉴权」是错的（第 49/52 轮实测鉴权不阻塞本地推演）。

## 我在本轮的自身缺陷（也写进了测试）

`evaluate_config` 第一版把 `steps_remaining` 写死 `0`，而 `time_discount(gamma, 0)` = `gamma**0` = **1**，
于是 **gamma 完全不起作用** —— `reward_no_discount` 与 `reward_full` 得到**逐位相同**的结果，
看起来像"时间折扣无影响"，其实**根本测不到**。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUNNER = REPO / "scripts" / "run_reward_ablation.py"
CONFIG_DIR = REPO / "configs" / "reward_ablation"
REPORT = REPO / "experiments" / "shenji_agent" / "reward_ablation" / "report.json"
DOC = REPO / "docs" / "reward_ablation_report_20261010.md"
SELFPLAY = REPO / "training" / "selfplay.py"
MANIFEST = REPO / "ai" / "weights" / "model_manifest.json"

if RUNNER.is_file() and str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _runner():
    if not RUNNER.is_file():
        pytest.skip("runner not present")
    import importlib.util
    spec = importlib.util.spec_from_file_location("rra", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────── 事实一：原配置是死配置（回归守卫）

def test_selfplay_still_has_no_reward_shaping_handling() -> None:
    """**这条守着"死配置"这个事实**：若哪天 selfplay 真读了 reward_shaping，
    那么"直接跑 selfplay 即可消融"就成立了，本文档与执行器都需重写。
    """
    if not SELFPLAY.is_file():
        pytest.skip("selfplay.py not present")
    src = SELFPLAY.read_text(encoding="utf-8")
    assert "reward_shaping" not in src, (
        "selfplay.py 已开始处理 reward_shaping —— 执行器的前提变了，需重写文档")


def test_runner_actually_reads_the_configs() -> None:
    """执行器必须**真的读取** `configs/reward_ablation/*.yaml`（否则仍是死配置）。"""
    if not RUNNER.is_file():
        pytest.skip("runner not present")
    src = RUNNER.read_text(encoding="utf-8")
    assert "reward_ablation" in src
    assert "yaml.safe_load" in src
    assert "_apply_config" in src


def test_configs_no_longer_claim_engine_auth_is_required() -> None:
    """配置里的「需引擎鉴权」注释是**错的**，已订正 —— 不得回归。"""
    if not CONFIG_DIR.is_dir():
        pytest.skip("configs not present")
    for f in sorted(CONFIG_DIR.glob("*.yaml")):
        t = f.read_text(encoding="utf-8")
        assert "鉴权不通过" not in t, "%s 又写回了错误的鉴权说明" % f.name
        assert "不需要引擎鉴权" in t, "%s 未说明不需要鉴权" % f.name


def test_configs_warn_that_selfplay_alone_is_ineffective() -> None:
    """配置必须写明"仅跑 selfplay 无效"，否则读配置的人会走错路。"""
    if not CONFIG_DIR.is_dir():
        pytest.skip("configs not present")
    for f in sorted(CONFIG_DIR.glob("*.yaml")):
        t = f.read_text(encoding="utf-8")
        assert "run_reward_ablation" in t, "%s 未给出正确执行入口" % f.name


# ─────────────────── 我的缺陷：gamma 必须可见

def test_gamma_actually_affects_the_result() -> None:
    """**核心回归**：不同 gamma 必须给出**不同**结果。

    我第一版把 `steps_remaining` 写死 0 → `gamma**0 = 1` → gamma 无效，
    `reward_no_discount` 与 `reward_full` 逐位相同。这条断言把它钉住。
    """
    if not REPORT.is_file():
        pytest.skip("report not present")
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    full = r.get("reward_full", {}).get("metrics_by_horizon") or {}
    nod = r.get("reward_no_discount", {}).get("metrics_by_horizon") or {}
    if not full or not nod:
        pytest.skip("report lacks horizon data")
    # 在 h>0 处两者**必须不同**
    assert full["500"]["sum_return"] != nod["500"]["sum_return"], (
        "gamma=0.99 与 gamma=1.0 在 h=500 给出相同结果 —— gamma 又失效了")
    # 且 no_discount 在各 horizon 上应**恒定**（不打折）
    vals = {nod[h]["sum_return"] for h in nod}
    assert len(vals) == 1, "reward_no_discount 各 horizon 结果不一致: %s" % vals


def test_report_has_multiple_horizons() -> None:
    """必须跑 >0 的 horizon —— 否则测不到时间折扣。"""
    if not REPORT.is_file():
        pytest.skip("report not present")
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    hs = r.get("horizons") or []
    assert hs, "report 未记录 horizons"
    assert any(int(h) > 0 for h in hs), "只跑了 h=0，gamma 无从体现"


# ─────────────────── 判据 2：≥3 组 + 完成率

def test_at_least_three_configs_were_evaluated() -> None:
    """`I06` 判据 2 要求「覆盖 ≥3 组配置」。"""
    if not REPORT.is_file():
        pytest.skip("report not present")
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    names = [k for k, v in r.items()
             if isinstance(v, dict) and "metrics_by_horizon" in v]
    assert len(names) >= 3, "只有 %d 组配置: %s" % (len(names), names)
    # 三要素应各有对应组
    for want in ("reward_full", "reward_no_discount", "reward_no_penalty"):
        assert want in names, "缺少消融组 %s" % want


def test_report_includes_task_completion_rate() -> None:
    """判据 2 要求「记录任务完成率与平均回报」—— 两者都要有。"""
    if not REPORT.is_file():
        pytest.skip("report not present")
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    tc = r.get("task_completion") or {}
    assert tc.get("segments", 0) > 0, "无任务完成率数据"
    assert 0.0 <= tc.get("achievement_share", -1) <= 1.0
    # 平均回报
    full = r.get("reward_full", {}).get("metrics") or {}
    assert "mean_reward" in full


def test_task_completion_rate_is_declared_config_independent() -> None:
    """**关键口径**：完成率取自已发生的推演，**不随奖励配置变化**。

    不写清这一点，会被误读成"改了奖励就改了完成率"。
    """
    if not REPORT.is_file():
        pytest.skip("report not present")
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    note = (r.get("task_completion") or {}).get("note") or ""
    assert "不随奖励配置变化" in note, "未声明完成率与配置无关"


def test_report_declares_its_scope_boundary() -> None:
    """**不得**把信号层面消融说成策略性能结论。"""
    if not REPORT.is_file():
        pytest.skip("report not present")
    r = json.loads(REPORT.read_text(encoding="utf-8"))
    sn = r.get("scope_note") or ""
    assert "奖励信号层面" in sn
    assert "不是" in sn and "策略" in sn


# ─────────────────── 判据 3：偏差告警（**未通过，如实记录**）

def test_doc_records_criterion_three_as_failed() -> None:
    """判据 3 **未通过** —— 文档必须如实记录，不得含糊。"""
    if not DOC.is_file():
        pytest.skip("report doc not present")
    d = DOC.read_text(encoding="utf-8")
    assert "intent_negative_move_bias" in d
    assert "未通过" in d, "未声明判据 3 未通过"
    assert "−0.5627" in d or "-0.5627" in d, "未记录实测的 move bias 值"
    # 必须点明它来自真实数据口径，不能只用"噪声口径"解释掉
    assert "on_real_data" in d


def test_doc_states_i06_is_still_incomplete() -> None:
    """判据 3 未过 ⇒ `I06` 未完成。不得因为判据 1/2 达成而宣称完成。"""
    if not DOC.is_file():
        pytest.skip("report doc not present")
    d = DOC.read_text(encoding="utf-8")
    assert "仍未完成" in d
    for claim in ("I06 已完成", "三条判据均已达成", "全部达成"):
        assert claim not in d, "出现过度声明 %r" % claim


def test_doc_records_the_penalty_dominance_finding() -> None:
    """§2.2 的发现（惩罚项压过对齐项）是判据 2 的实质结论，必须记录。"""
    if not DOC.is_file():
        pytest.skip("report doc not present")
    d = DOC.read_text(encoding="utf-8")
    assert "主导" in d or "压过" in d
    assert "5.3.3" in d, "未回引标准条款"


def test_doc_does_not_pretend_to_be_policy_ablation() -> None:
    """文档必须显式声明边界。"""
    if not DOC.is_file():
        pytest.skip("report doc not present")
    d = DOC.read_text(encoding="utf-8")
    assert "奖励信号层面" in d
    assert "策略性能" in d
