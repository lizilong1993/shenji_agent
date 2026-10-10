"""消息总线：消息类型、JSONL 落盘与幂等消费（I19 步骤 1/4）。

定位（务必先读）
----------------
I19【风险与回滚】原文：「引入总线抽象会改变训练主链路调用方式。
回滚：**总线以旁路方式实现（可关闭），主链路仍可直接调用 `TrainEnv`**」。

故本模块**不 import 也不修改**任何训练主链路（`training/selfplay.py`、`ai/agent.py`…），
是一个**可独立启停的旁路**。它既可被主链路选择性调用，也可完全不调用。

31006 4.4.1 要求消息总线支持 JSON / Protocol Buffers / 批量消息
--------------------------------------------------------------
本实现**只做 JSON + 批量（JSONL）**，并**预留 Protobuf 位**：
`SCHEMA_ID` 与消息的 `schema` 字段声明格式，将来换 Protobuf 只需换序列化器，
消费端靠 `schema` 判别，不必改消息语义。**差距写在 `docs/bus_contract.md`。**

幂等为什么落在「消费端」
------------------------
总线的重连语义是 at-least-once（可能重投），所以**去重必须在消费端做**。
本实现用 `msg_id` 去重，并把已消费的 id 持久化到 `.consumed` 文件 ——
**进程重启后仍幂等**，而不是只在内存里去重。
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

SCHEMA_ID = "shenji.bus.message/1"
DEFAULT_TOPIC = "shenji.decisions"


@dataclass
class Message:
    """一条总线消息。字段刻意与 I19 步骤 1 的要求一一对应。"""

    msg_type: str                     # 消息类型，如 instance.register / decision / heartbeat
    payload: Dict[str, Any] = field(default_factory=dict)
    msg_id: str = ""                  # 幂等键
    topic: str = DEFAULT_TOPIC
    instance_id: str = ""             # 发出者
    ts: float = 0.0                   # 发出时刻（epoch 秒）
    seq: int = 0                      # 发出者内部序号，便于发现乱序/丢包
    schema: str = SCHEMA_ID
    # 预留 Protobuf：将来 payload 可由 pb 序列化后放这里，消费端按 schema 判别
    payload_encoding: str = "json"

    def __post_init__(self) -> None:
        if not self.msg_id:
            # uuid4 而非「时间+随机」：同一毫秒内的两条消息也必须不同
            self.msg_id = uuid.uuid4().hex
        if not self.ts:
            self.ts = time.time()

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def from_json(line: str) -> "Message":
        d = json.loads(line)
        known = {f for f in Message.__dataclass_fields__}  # type: ignore[attr-defined]
        return Message(**{k: v for k, v in d.items() if k in known})


class MessageBus:
    """JSONL 文件总线：append 即发布，逐行读即订阅。

    为什么用文件而不是 socket/Redis：I19 要求「最小可用」，且【风险与回滚】要求可关闭。
    文件后端**零外部依赖**（远程无 Redis、无容器运行时），
    也天然满足「持久化传输」；将来换 Redis 只需替换本类，契约不变。
    """

    def __init__(self, root: Path, topic: str = DEFAULT_TOPIC, *, enabled: bool = True):
        self.root = Path(root)
        self.topic = topic
        self.enabled = enabled
        self._lock = threading.Lock()          # 多线程/多进程同写时保证整行原子
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        return self.root / ("%s.jsonl" % self.topic)

    def publish(self, msg: Message) -> Optional[str]:
        """追加一条消息。`enabled=False` 时**静默丢弃**（这就是「可关闭」）。"""
        if not self.enabled:
            return None
        line = msg.to_json() + "\n"
        with self._lock:
            # 单次 write 一行，配合 O_APPEND：多写者不会互相截断
            fd = os.open(str(self.log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
        return msg.msg_id

    def publish_batch(self, msgs: Iterable[Message]) -> int:
        """批量发布（31006 4.4.1 的「批量消息」）。"""
        n = 0
        for m in msgs:
            if self.publish(m) is not None:
                n += 1
        return n

    def read(self, *, since_offset: int = 0) -> Iterator[tuple]:
        """从字节偏移开始逐行读，产出 (offset_after_line, Message)。

        返回偏移量是为了**断线重连**：消费者记住上次偏移，重连后从这里继续，
        既不丢消息也不从头重放（见 `Consumer`）。
        """
        if not self.log_path.exists():
            return
        with self.log_path.open("rb") as fh:
            fh.seek(since_offset)
            while True:
                pos = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                yield fh.tell(), Message.from_json(line.decode("utf-8"))


class Consumer:
    """幂等消费者：按 `msg_id` 去重，并把进度持久化以支持断线重连。

    两个持久化文件（都在 `state_dir` 下，按 topic + 消费者名分开）：
      - `.offset`   已消费到的字节偏移（重连用）
      - `.consumed` 已消费的 msg_id 集合（幂等用）
    """

    def __init__(self, bus: MessageBus, name: str, state_dir: Optional[Path] = None):
        self.bus = bus
        self.name = name
        self.state_dir = Path(state_dir) if state_dir else bus.root / "consumers"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._offset_path = self.state_dir / ("%s.%s.offset" % (bus.topic, name))
        self._consumed_path = self.state_dir / ("%s.%s.consumed" % (bus.topic, name))
        self.offset = self._load_offset()
        self._consumed: Set[str] = self._load_consumed()
        self.stats = {"seen": 0, "accepted": 0, "duplicates": 0}

    # --- 持久化的小工具（都做原子写，避免半截状态） ---------------------
    def _load_offset(self) -> int:
        try:
            return int(self._offset_path.read_text(encoding="utf-8").strip() or 0)
        except Exception:  # noqa: BLE001
            return 0

    def _load_consumed(self) -> Set[str]:
        try:
            return set(self._consumed_path.read_text(encoding="utf-8").split())
        except Exception:  # noqa: BLE001
            return set()

    def _save(self) -> None:
        for p, data in ((self._offset_path, str(self.offset)),
                        (self._consumed_path, "\n".join(sorted(self._consumed)))):
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, p)

    # --- 主入口 ---------------------------------------------------------
    def poll(self) -> List[Message]:
        """拉取自上次偏移以来的新消息，去重后返回。

        **重复消息不返回**，但会推进偏移（否则下次还要再读一遍）。
        """
        out: List[Message] = []
        for off, msg in self.bus.read(since_offset=self.offset):
            self.stats["seen"] += 1
            self.offset = off
            if msg.msg_id in self._consumed:
                self.stats["duplicates"] += 1
                continue
            self._consumed.add(msg.msg_id)
            self.stats["accepted"] += 1
            out.append(msg)
        self._save()
        return out

    def rewind_to(self, offset: int) -> None:
        """故意把偏移退回，用于**演示重连重投**（幂等应使结果不变）。"""
        self.offset = max(0, int(offset))
        self._save()
