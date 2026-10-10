"""31006 五方法契约的**薄适配层**（I03 步骤 1）。

标准依据
--------
`31006 5.3` 与附录 `A.3`：智能体侧需提供
`setup` / `reset` / `step` / `load_llm_model` / `parse_natural_language_command`。
`31007` 附录 C 另要求 `__init__` / `save` / `load`（本层一并转发，作旁证覆盖）。

本层的职责边界（**必须守住**）
------------------------------
**只转发，不复制策略逻辑。**

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
        self.forwarded: Dict[str, int] = {name: 0 for name in FIVE_METHOD_CONTRACT}

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
        result = self._require_agent().step(observation)
        # `Agent.step` 在异常路径返回 []，此处**保持原样**，不把失败伪装成成功。
        return result if isinstance(result, list) else []

    # -- 五方法契约（后两个：占位实现，见 llm_adapter）------------------------

    def load_llm_model(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """转发到 `LLMAdapter.load_llm_model()`。**未启用时返回结构化状态，不抛异常**（判据 3）。"""
        self.forwarded["load_llm_model"] += 1
        return self.llm.load_llm_model(*args, **kwargs)

    def parse_natural_language_command(self, text: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """转发到 `LLMAdapter.parse_natural_language_command()`（规则占位解析）。"""
        self.forwarded["parse_natural_language_command"] += 1
        return self.llm.parse_natural_language_command(text, *args, **kwargs)

    # -- 31007 附录 C 旁证覆盖（同样只转发）----------------------------------

    def save(self, path: str) -> Any:
        if not self.enabled:
            return None
        return self._require_agent().save(path)

    def load(self, path: str) -> Any:
        if not self.enabled:
            return None
        return self._require_agent().load(path)
