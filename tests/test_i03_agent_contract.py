"""I03：31006 五方法契约的适配层。

判据
----
1. 远程 `python3 -m pytest tests/ -q` 含新增用例全部通过；
2. 五方法可被 `TrainEnv` 侧调用且返回结构符合 `docs/agent_contract.md`；
3. 未启用 LLM 时 `load_llm_model()` 返回「未启用」状态且**不抛异常**。

设计约束（来自本条的【风险与回滚】）
------------------------------------
「适配层只转发不改内部实现并可用配置整体禁用」——
故测试里有一条**专门断言转发计数**：`step()` 必须真的把调用传给被包装的 `Agent`，
而不是在本层自己算一个动作。否则适配层就成了主链路之外的**第二个决策入口**。

⚠ 本文件**不依赖 git**：适配层与撤销实验里出现过"仓库无 .git 时测试失败"
（评审实测 3 failed，`git rev-parse` 返回 128）。故此处只用内存对象，
不调用任何需要提交号的东西。
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ai.adapters import (APPENDIX_C_CONTRACT, FIVE_METHOD_CONTRACT, AgentAPI,
                         LLMAdapter)
from ai.adapters.llm_adapter import STATUS_DISABLED, STATUS_UNAVAILABLE

# 31006 5.3 / 附录 A.3 点名的五方法
REQUIRED = ("setup", "reset", "step", "load_llm_model", "parse_natural_language_command")


# ─────────────────────────── ① 五方法齐备（判据 1/2 的前半）

@pytest.mark.parametrize("name", REQUIRED)
def test_five_methods_exist_and_are_callable(name: str) -> None:
    assert hasattr(AgentAPI, name), "缺少 31006 要求的五方法之一: %s" % name
    assert callable(getattr(AgentAPI, name))


def test_contract_constant_matches_required_methods() -> None:
    assert tuple(FIVE_METHOD_CONTRACT) == REQUIRED


def test_appendix_c_methods_are_covered() -> None:
    """31007 附录 C 旁证：`__init__` / `save` / `load`。"""
    assert "__init__" in APPENDIX_C_CONTRACT
    for name in ("save", "load"):
        assert callable(getattr(AgentAPI, name))


# ─────────────────────────── ② 纯转发，不改内部实现

def _fake_agent():
    """一个记录被调用次数的假 Agent。**不需要真 Agent**，故不依赖引擎/权重。"""
    return mock.MagicMock(name="FakeAgent")


def test_setup_reset_step_forward_to_wrapped_agent() -> None:
    """三个转发方法必须**真的调用**被包装的 Agent（否则是本层自己在决策）。"""
    fake = _fake_agent()
    api = AgentAPI(fake)
    api.setup({"faction": 0})
    api.reset()
    api.step({"operators": []})
    fake.setup.assert_called_once_with({"faction": 0})
    fake.reset.assert_called_once_with()
    fake.step.assert_called_once_with({"operators": []})


def test_forwarded_counters_expose_use() -> None:
    fake = _fake_agent()
    api = AgentAPI(fake)
    api.setup({})
    api.step({})
    api.step({})
    assert api.forwarded["setup"] == 1
    assert api.forwarded["step"] == 2
    assert api.forwarded["reset"] == 0


def test_step_returns_agent_result_unchanged() -> None:
    """**不得改写**主链路的返回：`step` 的结果原样透出。"""
    marker = [{"actor": 0, "type": 5, "obj_id": 1}]
    fake = _fake_agent()
    fake.step.return_value = marker
    api = AgentAPI(fake)
    assert api.step({}) == marker


def test_step_never_invents_actions_when_agent_returns_none() -> None:
    """`Agent.step` 异常路径返回非 list；适配层**不得**把它变成"看起来成功"。"""
    fake = _fake_agent()
    fake.step.return_value = None
    api = AgentAPI(fake)
    assert api.step({}) == []


def test_adapter_does_not_import_policy_main_chain() -> None:
    """【风险与回滚】头号风险是「适配层反向渗入主链路」。

    做法：扫源码，适配层**不得**导入 `ai.policy`（那才是决策主链路）。
    允许导入 `ai.agent`（包装对象），因为转发必须拿到它。
    """
    src = (REPO / "ai" / "adapters" / "agent_api.py").read_text(encoding="utf-8")
    assert "from ai.policy" not in src and "import ai.policy" not in src, \
        "适配层导入了决策主链路 ai.policy —— 会造成策略漂移风险"
    assert "from ai.agent import Agent" in src, "应只包装 Agent"


# ─────────────────────────── ③ 可整体禁用

def test_disabled_adapter_does_not_touch_agent() -> None:
    """「可用配置整体禁用」：禁用时**不得触碰**被包装的智能体。"""
    fake = _fake_agent()
    api = AgentAPI(fake, enabled=False)
    assert api.step({"operators": []}) == []
    api.setup({"faction": 0})
    api.reset()
    fake.step.assert_not_called()
    fake.setup.assert_not_called()
    fake.reset.assert_not_called()


def test_disabled_state_is_reported() -> None:
    api = AgentAPI(_fake_agent(), enabled=False)
    assert api.contract()["enabled"] is False
    api2 = AgentAPI(_fake_agent(), enabled=True)
    assert api2.contract()["enabled"] is True


# ─────────────────────────── ④ 判据 3：未启用 LLM 的降级路径

def test_load_llm_model_disabled_returns_status_without_raising() -> None:
    """**判据 3**：未启用时返回结构化「未启用」状态，**不抛异常**。"""
    api = AgentAPI(_fake_agent())
    got = api.load_llm_model()          # 不得抛
    assert isinstance(got, dict)
    assert got["status"] == STATUS_DISABLED
    assert got["loaded"] is False
    assert got["reason"], "必须给出可读原因"


def test_load_llm_model_returns_fixed_field_set() -> None:
    """返回字段固定，调用方无需 try/except 即可判断。"""
    got = LLMAdapter().load_llm_model()
    for key in ("status", "loaded", "model_path", "reason", "capabilities"):
        assert key in got, "缺字段 %s" % key
    assert isinstance(got["capabilities"], dict)
    assert got["capabilities"]["model_inference"] is False


def test_load_llm_model_missing_file_degrades_not_raises(tmp_path: Path) -> None:
    """**启用了但模型不存在** → 降级为 unavailable，而不是抛异常。"""
    api = AgentAPI(_fake_agent(), llm_adapter=LLMAdapter(enabled=True))
    got = api.load_llm_model(str(tmp_path / "nope.bin"))
    assert got["status"] == STATUS_UNAVAILABLE
    assert got["loaded"] is False


def test_load_llm_model_does_not_claim_loaded_without_inference() -> None:
    """占位实现**不得**谎报 `loaded=True`（那会让调用方以为推理可用）。"""
    got = LLMAdapter(enabled=True).load_llm_model("/etc/hostname")   # 文件存在
    assert got["loaded"] is False
    assert got["status"] == STATUS_UNAVAILABLE, "文件在也不该报 loaded"


# ─────────────────────────── ⑤ 判据 2：自然语言解析的返回结构

def test_nl_parse_returns_required_fields() -> None:
    """步骤 3 点名要求：意图 / 目标对象 / 参数 / 置信度 / 是否降级。"""
    got = AgentAPI(_fake_agent()).parse_natural_language_command(
        "对目标发起打击", obj_id=101, target_obj_id=202)
    for key in ("intent", "target", "params", "confidence", "degraded", "action"):
        assert key in got, "缺字段 %s" % key
    assert got["intent"] == "attack"
    assert got["degraded"] is True, "无模型推理，必须如实标 degraded"
    assert 0.0 <= got["confidence"] <= 1.0


def test_nl_parse_action_matches_step_action_shape() -> None:
    """**步骤 3 的硬要求**：输出动作格式必须与 `step` 接受的一致。

    `step` 接受的是平台动作 dict（`actor`/`type`/`obj_id`/`target_obj_id`），
    字段名与本仓 `ai/action_safety.py` 的动作字典同源。
    """
    got = AgentAPI(_fake_agent()).parse_natural_language_command(
        "打击", actor=0, obj_id=101, target_obj_id=202)
    act = got["action"]
    assert isinstance(act, dict)
    assert set(act) <= {"actor", "type", "obj_id", "target_obj_id"}, \
        "动作里出现了平台文档之外的自造字段: %s" % sorted(act)
    assert act["type"] == 2, "打击应映射到平台 type=2"


def test_nl_parse_unmatched_text_does_not_invent_an_action() -> None:
    """解析不出意图时**不得编一个动作**，且置信度为 0。"""
    got = AgentAPI(_fake_agent()).parse_natural_language_command("随便说点什么")
    assert got["intent"] is None
    assert got["action"] is None
    assert got["confidence"] == 0.0
    assert got["degraded"] is True


def test_nl_parse_missing_target_yields_no_action() -> None:
    """攻击类命令缺目标 → 返回不可用动作，而不是编一个 `target_obj_id`。"""
    got = AgentAPI(_fake_agent()).parse_natural_language_command("打击", obj_id=101)
    assert got["action"] is None
    assert got["confidence"] == 0.0


def test_nl_parse_handles_none_text() -> None:
    """容错：`None` 不得抛异常。"""
    got = AgentAPI(_fake_agent()).parse_natural_language_command(None)
    assert got["intent"] is None and got["degraded"] is True


# ─────────────────────────── ⑥ 契约自检出口（取证通道）

def test_contract_reports_each_method_kind() -> None:
    """取证出口必须能区分**转发**与**占位**，否则无法审计"哪些是真实现"。"""
    api = AgentAPI(_fake_agent())
    c = api.contract()
    assert c["five_methods"]["setup"]["kind"] == "forward"
    assert c["five_methods"]["reset"]["kind"] == "forward"
    assert c["five_methods"]["step"]["kind"] == "forward"
    assert c["five_methods"]["load_llm_model"]["kind"] == "placeholder"
    for name in REQUIRED:
        assert c["five_methods"][name]["present"] is True


def test_signatures_match_forwarding_targets() -> None:
    """对齐签名（步骤 4 的机器版本）：转发的三个方法与 `Agent` 同名方法参数一致。"""
    from ai.agent import Agent

    for name in ("setup", "reset", "step"):
        api_params = list(inspect.signature(getattr(AgentAPI, name)).parameters)[1:]
        agent_params = list(inspect.signature(getattr(Agent, name)).parameters)[1:]
        assert api_params == agent_params, (
            "%s 的参数与 Agent 不一致: %s vs %s" % (name, api_params, agent_params))
