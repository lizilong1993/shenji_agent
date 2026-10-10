"""决策总线最小可用形态（I19）：消息总线 + 数据总线 + 实例注册。

**旁路定位**：本包不 import 任何训练主链路，也不被其强制依赖；
`enabled=False` 时全部写操作静默丢弃，主链路行为与不存在本包时一致。
"""
from .data_bus import DataBus, SCHEMA_ID as DATA_SCHEMA_ID
from .messages import DEFAULT_TOPIC, Message, MessageBus, Consumer, SCHEMA_ID as MESSAGE_SCHEMA_ID
from .registry import DEFAULT_TTL_SECONDS, InstanceRecord, Registry

__all__ = [
    "Message", "MessageBus", "Consumer", "DEFAULT_TOPIC", "MESSAGE_SCHEMA_ID",
    "DataBus", "DATA_SCHEMA_ID",
    "Registry", "InstanceRecord", "DEFAULT_TTL_SECONDS",
]
