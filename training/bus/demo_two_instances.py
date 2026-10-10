#!/usr/bin/env python3
"""I19 步骤 5 演示：2 个环境实例并行运行，总线可观测、消息可去重。

**为什么用线程而不是多进程**：本演示要验证的是**总线的语义**
（注册/心跳/调度/幂等），不是进程模型。线程共享同一解释器，
能把「同一毫秒内的并发写」这个真正的风险点暴露出来（`O_APPEND` 单行写），
而多进程只会把日志变乱、并引入与被测目标无关的平台差异。
跨进程的正确性另由「闭-开-再读」的持久化测试覆盖。

用法：
    python3 -m training.bus.demo_two_instances [--root DIR] [--ticks N]
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from .data_bus import DataBus
from .messages import Message, MessageBus, Consumer
from .registry import Registry


def _run_instance(instance_id: str, ticks: int, bus_root: Path, data_root: Path,
                  registry: Registry, out: Dict[str, Any]) -> None:
    """一个环境实例：注册 → 跑若干 tick（发决策消息 + 写态势）→ 心跳 → 完成。"""
    reg = registry.register(instance_id, capabilities={"ticks": ticks, "agent": "rule"})
    bus = MessageBus(bus_root, "shenji.decisions")
    dbus = DataBus(data_root, instance_id, shard_bytes=4096)

    for t in range(ticks):
        bus.publish(Message(msg_type="decision", instance_id=instance_id, seq=t,
                            payload={"tick": t, "action": "Move", "route_len": t % 3}))
        dbus.append(t, {"time": {"cur_step": t}, "operators": [{"obj_id": 1}]})
        if t % 4 == 0:
            registry.heartbeat(instance_id)
    registry.finish(instance_id)
    out[instance_id] = {"registered_at": reg.started_ts, "ticks": ticks,
                        "manifest_entries": dbus.manifest().get("total_entries", 0)}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="I19 demo: two environment instances on the bus.")
    ap.add_argument("--root", default=None, help="bus root (default: a temp dir)")
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--keep", action="store_true", help="do not delete the temp dir")
    args = ap.parse_args(argv)

    root = Path(args.root) if args.root else Path(tempfile.mkdtemp(prefix="shenji-bus-demo-"))
    bus_root, data_root = root / "bus", root / "data"
    registry = Registry(root / "registry", ttl_seconds=30.0)

    print("bus root: %s" % root)
    results: Dict[str, Any] = {}
    threads = [
        threading.Thread(target=_run_instance, name=iid,
                         args=(iid, args.ticks, bus_root, data_root, registry, results))
        for iid in ("env-a", "env-b")
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    # --- 总线可观测：两个实例各自的消费者互不干扰，且都能拿全 ---
    print("\n=== 实例注册与调度 ===")
    snap = registry.snapshot()
    for iid in sorted(snap):
        r = snap[iid]
        print("  %-6s status=%-6s ticks=%-4s task=%s" % (iid, r.status, results.get(iid, {}).get("ticks"), r.assigned_task))
    alive = registry.alive()
    print("  存活实例: %s" % [r.instance_id for r in alive])

    print("\n=== 消息总线（按实例消费）===")
    total = 0
    for iid in ("env-a", "env-b"):
        c = Consumer(MessageBus(bus_root, "shenji.decisions"), "cons-%s" % iid)
        got = [m for m in c.poll() if m.instance_id == iid]
        total += len(got)
        print("  %-6s 收到 %d 条决策消息（预期 %d）" % (iid, len(got), args.ticks))
        if got:
            print("         首条 payload=%s" % json.dumps(got[0].payload, ensure_ascii=False))
    print("  合计 %d 条（两条流合计 %d）" % (total, 2 * args.ticks))

    print("\n=== 幂等演示：同一消费者重连重投 ===")
    c = Consumer(MessageBus(bus_root, "shenji.decisions"), "cons-idem")
    first = c.poll()
    c.rewind_to(0)                     # 模拟断线重连后从头重投
    second = c.poll()
    print("  首次消费=%d  重投后再消费=%d  去重计数=%d" % (len(first), len(second), c.stats["duplicates"]))

    print("\n=== 数据总线（历史态势序列）===")
    for iid in ("env-a", "env-b"):
        m = DataBus(data_root, iid).manifest()
        raw = sum(s["raw_bytes"] for s in m["shards"])
        comp = sum(s["file_bytes"] for s in m["shards"])
        print("  %-6s entries=%-4s shards=%-3s raw=%dB -> file=%dB" % (
            iid, m["total_entries"], len(m["shards"]), raw, comp))

    print("\n=== 判据核对 ===")
    ok_parallel = len([r for r in snap.values() if r.status == "done"]) == 2
    ok_dedup = len(second) == 0 and c.stats["duplicates"] > 0
    print("  1. 2 个实例并行运行且消息/数据总线可观测: %s（耗时 %.3fs）" % (ok_parallel, elapsed))
    print("  2. 断线重连与消息去重演示通过: %s" % ok_dedup)

    if not args.keep and not args.root:
        shutil.rmtree(root, ignore_errors=True)
    return 0 if (ok_parallel and ok_dedup) else 1


if __name__ == "__main__":
    raise SystemExit(main())
