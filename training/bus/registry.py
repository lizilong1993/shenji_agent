"""多环境实例注册与调度（I19 步骤 3）+ 断线重连（步骤 4）。

定位：**旁路**。本模块不 import 任何训练主链路，也不假设 `TrainEnv` 的存在 ——
实例只是一个带 `instance_id` 的自愿登记者。

心跳与「判死」为什么分开算
--------------------------
实例崩溃时不会发「我下线了」。所以存活判定必须由**时间**推出：
`last_seen` 距今超过 `ttl` 即视为失联。这是分布式系统里唯一可行的做法，
也意味着**判死必然滞后**（最多一个 TTL）—— 这一点写进契约，避免使用方误以为它是实时的。
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .messages import Message, MessageBus

TOPIC_REGISTRY = "shenji.registry"
DEFAULT_TTL_SECONDS = 30.0


@dataclass
class InstanceRecord:
    instance_id: str
    started_ts: float
    last_seen_ts: float
    status: str = "alive"                  # alive / suspect / dead / done
    assigned_task: Optional[str] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)

    def age(self, now: Optional[float] = None) -> float:
        return (now if now is not None else time.time()) - self.last_seen_ts


class Registry:
    """基于 JSONL 的实例注册表（与消息总线同一套存储原语）。

    每次状态变化**追加**一条记录而非改写文件；`snapshot()` 折叠出最新视图。
    追加的理由：多个实例可能同时写同一个文件，追加 + O_APPEND 不会互相覆盖，
    而「读-改-写」会。
    """

    def __init__(self, root: Path, *, ttl_seconds: float = DEFAULT_TTL_SECONDS,
                 enabled: bool = True):
        self.bus = MessageBus(Path(root), TOPIC_REGISTRY, enabled=enabled)
        self.ttl_seconds = float(ttl_seconds)
        self.enabled = enabled

    # ------------------------------------------------------------------ 写入
    def _emit(self, msg_type: str, payload: Dict[str, Any], instance_id: str) -> None:
        if not self.enabled:
            return
        self.bus.publish(Message(msg_type=msg_type, payload=payload,
                                 topic=TOPIC_REGISTRY, instance_id=instance_id))

    def register(self, instance_id: str, capabilities: Optional[Dict[str, Any]] = None,
                 now: Optional[float] = None) -> InstanceRecord:
        ts = now if now is not None else time.time()
        rec = InstanceRecord(instance_id=instance_id, started_ts=ts, last_seen_ts=ts,
                             capabilities=capabilities or {})
        self._emit("instance.register", asdict(rec), instance_id)
        return rec

    def heartbeat(self, instance_id: str, now: Optional[float] = None) -> None:
        self._emit("instance.heartbeat",
                   {"instance_id": instance_id, "last_seen_ts": now if now is not None else time.time()},
                   instance_id)

    def assign(self, instance_id: str, task_id: str) -> None:
        self._emit("instance.assign", {"instance_id": instance_id, "task_id": task_id}, instance_id)

    def finish(self, instance_id: str, status: str = "done") -> None:
        self._emit("instance.finish", {"instance_id": instance_id, "status": status}, instance_id)

    # ------------------------------------------------------------------ 读取
    def snapshot(self, now: Optional[float] = None) -> Dict[str, InstanceRecord]:
        """折叠全部记录，得出每个实例的最新状态，并按 TTL 判定存活。"""
        t = now if now is not None else time.time()
        out: Dict[str, InstanceRecord] = {}
        for _, msg in self.bus.read():
            p = msg.payload or {}
            iid = p.get("instance_id") or msg.instance_id
            if not iid:
                continue
            if msg.msg_type == "instance.register":
                out[iid] = InstanceRecord(
                    instance_id=iid,
                    started_ts=float(p.get("started_ts", msg.ts)),
                    last_seen_ts=float(p.get("last_seen_ts", msg.ts)),
                    capabilities=p.get("capabilities") or {})
            elif iid not in out:
                # 先收到心跳/分配（注册记录已滚出或乱序）也要能建视图
                out[iid] = InstanceRecord(instance_id=iid, started_ts=msg.ts, last_seen_ts=msg.ts)
            rec = out[iid]
            if msg.msg_type == "instance.heartbeat":
                rec.last_seen_ts = max(rec.last_seen_ts, float(p.get("last_seen_ts", msg.ts)))
            elif msg.msg_type == "instance.assign":
                rec.assigned_task = p.get("task_id")
            elif msg.msg_type == "instance.finish":
                rec.status = str(p.get("status", "done"))

        for rec in out.values():
            if rec.status == "done":
                continue
            rec.status = "alive" if rec.age(t) <= self.ttl_seconds else "dead"
        return out

    def alive(self, now: Optional[float] = None) -> List[InstanceRecord]:
        return [r for r in self.snapshot(now).values() if r.status == "alive"]

    def pick(self, now: Optional[float] = None) -> Optional[InstanceRecord]:
        """最小调度：在存活且未分配任务的实例中，取 startup 最早的一个。

        选最早而非随机，是为了**可复现** —— 调度的结果不应依赖随机数。
        """
        cands = [r for r in self.alive(now) if not r.assigned_task]
        if not cands:
            return None
        return sorted(cands, key=lambda r: (r.started_ts, r.instance_id))[0]
