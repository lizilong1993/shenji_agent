"""31006 五方法契约的**薄适配层**（I03 步骤 1）。

标准依据
--------
`31006 5.3` 与附录 `A.3`：智能体侧需提供
`setup` / `reset` / `step` / `load_llm_model` / `parse_natural_language_command`。
`31007` 附录 C 另要求 `__init__` / `save` / `load`（本层一并转发，作旁证覆盖）。

本层的职责边界（**必须守住**）
------------------------------
**只转发，不复制策略逻辑。**

⚠ 首版头部写"三个方法体各只有一行 `return`" —— 那是**失实**的：
`step` 里有一处 `None -> []` 兼容（评审指出）。准确表述是
"**不实现任何决策逻辑**"；`None` 的兼容是既有 `Agent.step` 的约定，不是新决策。

【风险与回滚】点名的头号风险是「适配层反向渗入主链路造成策略漂移」。
故本文件**不实现任何决策逻辑**：`setup`/`reset`/`step` 三个方法体各只有一行 `return`，
直接交给既有 `Agent` 实例。若将来有人想在这里"顺手改一下动作"，那是在主链路上
开第二个决策入口 —— 本层的存在意义恰恰是**避免**这件事。

可整体禁用
----------
`AgentAPI(..., enabled=False)` 时 `step()` 返回 `[]` 且不触碰被包装的智能体，
满足【风险与回滚】的「适配层可用配置整体禁用」。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ai.agent import Agent
from ai.adapters.llm_adapter import LLMAdapter

# 五方法契约（31006 5.3 / 附录 A.3）+ 附录 C 三方法，用于自检与文档对照。
FIVE_METHOD_CONTRACT = (
    "setup",
    "reset",
    "step",
    "load_llm_model",
    "parse_natural_language_command",
)
APPENDIX_C_CONTRACT = ("__init__", "save", "load")


class AgentAPI:
    """把 `Agent` 包装成 31006 可调用的五方法接口。

    Parameters
    ----------
    agent:
        既有 `Agent` 实例。**必须由调用方构造**，本层不替他决定 seed/日志等级；
        传 `None` 时惰性新建（方便标准侧按契约直接 `AgentAPI()`）。
    enabled:
        `False` 时整层禁用（`step()` 恒返回 `[]`），用于回滚与对照实验。
    llm_adapter:
        可注入，便于测试与将来替换为真实 LLM 实现（只换实现，不换接口）。
    """

    def __init__(self, agent: Optional[Agent] = None, *, enabled: bool = True,
                 llm_adapter: Optional[LLMAdapter] = None) -> None:
        self.enabled = bool(enabled)
        self._agent: Optional[Agent] = agent
        self.llm = llm_adapter if llm_adapter is not None else LLMAdapter()
        # 覆盖五方法**与**附录 C 的 save/load —— 文档自称"调用可审计 … 每个方法"，
        # 首版只记五方法，save/load 无法审计（评审指出）。
        self.forwarded: Dict[str, int] = {
            name: 0 for name in (FIVE_METHOD_CONTRACT + ("save", "load"))}

    # -- 内部 ---------------------------------------------------------------

    def _require_agent(self) -> Agent:
        """惰性构造。**不吞异常**：构造失败要显式暴露，不能假装智能体就绪。"""
        if self._agent is None:
            self._agent = Agent(seed=0)
        return self._agent

    @property
    def agent(self) -> Optional[Agent]:
        """被包装的智能体（只读）。**不提供 setter** —— 换智能体应新建适配层。"""
        return self._agent

    def contract(self) -> Dict[str, Any]:
        """契约自检：哪些方法在、哪些是转发、哪些是占位。"""
        return {
            "five_methods": {
                name: {
                    "present": hasattr(self, name),
                    "kind": ("forward" if name in ("setup", "reset", "step")
                             else "placeholder" if name == "load_llm_model"
                             else "rule_based_placeholder" if name == "parse_natural_language_command"
                             else "unknown"),
                }
                for name in FIVE_METHOD_CONTRACT
            },
            "appendix_c": {
                name: hasattr(self, name) for name in APPENDIX_C_CONTRACT
                if name != "__init__"
            },
            "enabled": self.enabled,
            "forwarded": dict(self.forwarded),
        }

    # -- 五方法契约（前三个：**纯转发**）--------------------------------------

    def setup(self, setup_info: Dict[str, Any]) -> Any:
        """转发到 `Agent.setup()`。本层不解读 `setup_info`。"""
        self.forwarded["setup"] += 1
        if not self.enabled:
            return None
        return self._require_agent().setup(setup_info)

    def reset(self) -> Any:
        """转发到 `Agent.reset()`。"""
        self.forwarded["reset"] += 1
        if not self.enabled:
            return None
        return self._require_agent().reset()

    def step(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """转发到 `Agent.step()`。**禁用时返回空列表**，不去动被包装的智能体。"""
        self.forwarded["step"] += 1
        if not self.enabled:
            return []
        # 透传：`Agent.step` 正常返回 list；其异常路径自己返回 []。
        # 首版把**任何**非 list 都强转成 []，于是类型 bug 被伪装成"没有动作"
        # （评审指出）。现在只对 `None` 做兼容（那是 `Agent.step` 的既有约定），
        # 其余非 list 原样透出 —— 让上游看见真实的类型问题，而不是被我们抹平。
        result = self._require_agent().step(observation)
        if result is None:
            return []
        return result

    # -- 五方法契约（后两个：占位实现，见 llm_adapter）------------------------

    def load_llm_model(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """转发到 `LLMAdapter.load_llm_model()`。**未启用时返回结构化状态，不抛异常**（判据 3）。

        ⚠ `enabled=False` 时**本层也要禁用 LLM 半边**：首版只在适配层拦住前三个方法，
        LLM 两个方法仍照常转发，于是"整层禁用"只禁了一半（评审指出）。
        """
        self.forwarded["load_llm_model"] += 1
        if not self.enabled:
            return {"status": "disabled", "loaded": False, "model_path": None,
                    "reason": "适配层已禁用（AgentAPI(enabled=False)）",
                    "capabilities": {"natural_language": False, "model_inference": False}}
        return self.llm.load_llm_model(*args, **kwargs)

    def parse_natural_language_command(self, text: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """转发到 `LLMAdapter.parse_natural_language_command()`（规则占位解析）。"""
        self.forwarded["parse_natural_language_command"] += 1
        if not self.enabled:
            return {"intent": None, "target": None, "params": {}, "confidence": 0.0,
                    "degraded": True, "method": "disabled", "action": None,
                    "text": "" if text is None else str(text),
                    "reason": "适配层已禁用（AgentAPI(enabled=False)），不解析命令"}
        return self.llm.parse_natural_language_command(text, *args, **kwargs)

    # -- 31007 附录 C 旁证覆盖（同样只转发）----------------------------------

    def save(self, path: str) -> Any:
        self.forwarded["save"] += 1
        if not self.enabled:
            return None
        return self._require_agent().save(path)

    def load(self, path: str) -> Any:
        self.forwarded["load"] += 1
        if not self.enabled:
            return None
        return self._require_agent().load(path)
