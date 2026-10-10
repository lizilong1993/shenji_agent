"""31006 五方法契约的适配层（I03）。

**本包不参与策略主链路**：`ai/policy.py`、`training/` 均不导入它。
它的存在是为了让标准侧按契约调用，而不是让本仓内部多一条决策路径。
若将来有人从主链路导入本包，那就是【风险与回滚】里说的"反向渗入"。
"""
from ai.adapters.agent_api import APPENDIX_C_CONTRACT, FIVE_METHOD_CONTRACT, AgentAPI
from ai.adapters.llm_adapter import LLMAdapter

__all__ = ["AgentAPI", "LLMAdapter", "FIVE_METHOD_CONTRACT", "APPENDIX_C_CONTRACT"]
