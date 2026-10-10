"""I19：消息总线 + 数据总线 + 实例注册（最小可用形态）。

用临时目录，不落任何产物到仓库。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.bus import (Consumer, DataBus, InstanceRecord, Message, MessageBus,
                          Registry)


# ------------------------------------------------------------------ 消息

def test_message_roundtrip_and_unique_ids() -> None:
    a, b = Message(msg_type="x"), Message(msg_type="x")
    assert a.msg_id != b.msg_id, "同一毫秒内的两条消息也必须不同"
    assert Message.from_json(a.to_json()).msg_id == a.msg_id
    assert a.schema == "shenji.bus.message/1"


def test_publish_and_consume(tmp_path: Path) -> None:
    bus = MessageBus(tmp_path)
    assert bus.publish_batch([Message(msg_type="d", seq=i) for i in range(3)]) == 3
    c = Consumer(bus, "w")
    assert [m.seq for m in c.poll()] == [0, 1, 2]
    assert c.poll() == []


def test_consumer_dedup_survives_reconnect(tmp_path: Path) -> None:
    """偏移退回模拟重连重投：去重必须仍然生效。"""
    bus = MessageBus(tmp_path)
    bus.publish_batch([Message(msg_type="d", seq=i) for i in range(4)])
    c = Consumer(bus, "w")
    assert len(c.poll()) == 4
    c.rewind_to(0)
    assert c.poll() == []
    assert c.stats["duplicates"] == 4


def test_consumer_dedup_survives_process_restart(tmp_path: Path) -> None:
    """换一个 Consumer 实例读同一个 state_dir —— 模拟进程重启。"""
    bus = MessageBus(tmp_path)
    bus.publish_batch([Message(msg_type="d", seq=i) for i in range(3)])
    c1 = Consumer(bus, "w")
    assert len(c1.poll()) == 3

    c2 = Consumer(bus, "w")          # 新进程：从磁盘恢复 offset 与 consumed
    assert c2.offset > 0
    assert c2.poll() == []


def test_consumers_are_isolated_by_name(tmp_path: Path) -> None:
    """去重是按消费者隔离的：另一个消费者仍应拿到全部消息。"""
    bus = MessageBus(tmp_path)
    bus.publish_batch([Message(msg_type="d") for _ in range(3)])
    assert len(Consumer(bus, "a").poll()) == 3
    assert len(Consumer(bus, "b").poll()) == 3


def test_disabled_bus_publishes_nothing(tmp_path: Path) -> None:
    """「可关闭」：关闭后不产生任何文件、也不报错。"""
    bus = MessageBus(tmp_path / "nope", enabled=False)
    assert bus.publish(Message(msg_type="d")) is None
    assert bus.publish_batch([Message(msg_type="d")] * 5) == 0
    assert not (tmp_path / "nope").exists()


# ------------------------------------------------------------------ 数据总线

def test_data_bus_shards_and_compresses(tmp_path: Path) -> None:
    db = DataBus(tmp_path, "env-a", shard_bytes=2048)
    for t in range(200):
        db.append(t, {"time": {"cur_step": t}, "pad": "x" * 40})
    m = db.manifest()
    assert m["total_entries"] == 200
    assert len(m["shards"]) > 1, "小阈值应产生多分片"
    assert m["compression"] == "gzip"
    for s in m["shards"]:
        assert s["sha256"] and s["entries"] > 0


def test_data_bus_state_at_tick(tmp_path: Path) -> None:
    db = DataBus(tmp_path, "env-a", shard_bytes=2048)
    for t in range(120):
        db.append(t, {"time": {"cur_step": t}})
    rec = db.state_at_tick(77)
    assert rec is not None and rec["state"]["time"]["cur_step"] == 77
    assert db.state_at_tick(99999) is None


def test_data_bus_disabled_writes_nothing(tmp_path: Path) -> None:
    db = DataBus(tmp_path, "env-a", enabled=False)
    assert db.append(0, {"x": 1}) is None
    assert not (tmp_path / "env-a").exists()


def test_data_bus_manifest_records_tick_range(tmp_path: Path) -> None:
    """manifest 的 tick 区间是 state_at_tick 跳过无关分片的依据。"""
    db = DataBus(tmp_path, "env-a", shard_bytes=1024)
    for t in range(50):
        db.append(t, {"pad": "y" * 60})
    shards = db.manifest()["shards"]
    assert shards[0]["tick_first"] == 0
    assert shards[-1]["tick_last"] == 49
    for s in shards:
        assert s["tick_first"] <= s["tick_last"]


# ------------------------------------------------------------------ 注册表

def test_registry_register_and_alive(tmp_path: Path) -> None:
    r = Registry(tmp_path, ttl_seconds=10.0)
    r.register("env-a", now=1000.0)
    r.register("env-b", now=1000.0)
    assert sorted(x.instance_id for x in r.alive(now=1005.0)) == ["env-a", "env-b"]


def test_registry_ttl_marks_dead(tmp_path: Path) -> None:
    r = Registry(tmp_path, ttl_seconds=10.0)
    r.register("env-a", now=1000.0)
    assert [x.status for x in r.snapshot(now=1005.0).values()] == ["alive"]
    assert [x.status for x in r.snapshot(now=1100.0).values()] == ["dead"]
    assert r.alive(now=1100.0) == []


def test_registry_heartbeat_extends_life(tmp_path: Path) -> None:
    r = Registry(tmp_path, ttl_seconds=10.0)
    r.register("env-a", now=1000.0)
    r.heartbeat("env-a", now=1008.0)
    assert [x.status for x in r.snapshot(now=1015.0).values()] == ["alive"]


def test_registry_assign_and_finish(tmp_path: Path) -> None:
    r = Registry(tmp_path, ttl_seconds=100.0)
    r.register("env-a", now=1000.0)
    r.assign("env-a", "task-1")
    r.finish("env-a")
    rec = r.snapshot(now=1001.0)["env-a"]
    assert rec.assigned_task == "task-1" and rec.status == "done"


def test_registry_pick_is_deterministic(tmp_path: Path) -> None:
    """选启动最早的实例，而不是随机 —— 调度结果不应依赖随机数。"""
    r = Registry(tmp_path, ttl_seconds=100.0)
    r.register("env-b", now=1002.0)
    r.register("env-a", now=1001.0)
    assert r.pick(now=1003.0).instance_id == "env-a"
    assert Registry(tmp_path / "again", ttl_seconds=100.0) is not None


def test_registry_disabled_writes_nothing(tmp_path: Path) -> None:
    r = Registry(tmp_path / "x", enabled=False)
    r.register("env-a")
    assert r.snapshot() == {}


# ------------------------------------------------------------------ 并发与旁路

def test_concurrent_publishers_do_not_corrupt_lines(tmp_path: Path) -> None:
    """并发写同一 JSONL：O_APPEND 单行写必须保证不出现半行。"""
    import threading
    bus = MessageBus(tmp_path)
    N, PER = 6, 40

    def worker(k: int) -> None:
        for i in range(PER):
            bus.publish(Message(msg_type="d", instance_id="w%d" % k, seq=i))

    ts = [threading.Thread(target=worker, args=(k,)) for k in range(N)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    lines = bus.log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == N * PER
    for ln in lines:
        json.loads(ln)          # 任一行解析失败即说明写被截断


def test_package_does_not_import_training_main_chain() -> None:
    """旁路定位的**机器可验证**检查：本包不得依赖训练主链路模块。"""
    bus_dir = Path(__file__).resolve().parents[1] / "training" / "bus"
    src = "\n".join(p.read_text(encoding="utf-8") for p in bus_dir.glob("*.py"))
    for banned in ("import training.selfplay", "from training.selfplay",
                   "import training.evaluate", "from training.evaluate",
                   "import ai.agent", "from ai.agent", "import ai.policy"):
        assert banned not in src, "旁路包不得依赖主链路: %s" % banned


def test_demo_runs_and_reports_both_criteria(tmp_path: Path) -> None:
    """端到端演示脚本自身要能跑通并打印两条判据为 True。"""
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run(
        [sys.executable, "-m", "training.bus.demo_two_instances",
         "--root", str(tmp_path / "demo"), "--ticks", "6"],
        cwd=str(repo), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2 个实例并行运行且消息/数据总线可观测: True" in r.stdout
    assert "断线重连与消息去重演示通过: True" in r.stdout
