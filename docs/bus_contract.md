# 决策总线最小可用形态（I19）

> 依据 `31006 4.4.1`：决策总线由**消息总线**与**数据总线**组成 ——
> 消息总线用 JSON / Protocol Buffers / 批量消息；
> 数据总线支持持久化传输、历史态势序列、多模态复盘数据、压缩与分片。

---

## 1. 定位：**旁路**，不是主链路

I19【风险与回滚】原文：

> 「引入总线抽象会改变训练主链路调用方式。回滚：**总线以旁路方式实现（可关闭），
> 主链路仍可直接调用 `TrainEnv`**。」

本实现严格遵守该约束，且把它做成了**可机器验证**的性质：

- 代码位于 `training/bus/`，**不 import** `training/selfplay.py`、`training/evaluate.py`、
  `ai/agent.py`、`ai/policy.py` 中的任何一个；
- `enabled=False` 时，**所有写操作静默丢弃**，不创建任何文件、不抛异常 ——
  主链路行为与「本包不存在」完全一致；
- 有一条测试（`test_package_does_not_import_training_main_chain`）
  **扫源码断言上述 import 不存在**，防止以后有人悄悄把主链路拉进来。

**因此本包可以随时整体删除，训练流程不受影响。**

---

## 2. 组成

| 模块 | 职责 | 对应 31006 4.4.1 |
| --- | --- | --- |
| `messages.py` | 消息类型、JSONL 落盘、批量发布、幂等消费 | 消息总线（JSON + 批量） |
| `data_bus.py` | 历史态势序列、分片、压缩、按 tick 取态 | 数据总线（持久化 + 历史序列 + 压缩 + 分片） |
| `registry.py` | 实例注册、心跳、TTL 判死、最小调度 | 多环境实例注册与调度 |
| `demo_two_instances.py` | 2 实例并行演示 + 幂等演示 | 判据 1、2 |

---

## 3. 消息格式（JSONL）

每行一条消息：

```json
{"msg_type":"decision","payload":{"tick":0,"action":"Move"},
 "msg_id":"…","topic":"shenji.decisions","instance_id":"env-a",
 "ts":1759…,"seq":0,"schema":"shenji.bus.message/1","payload_encoding":"json"}
```

**`schema` 与 `payload_encoding` 是为 Protobuf 预留的接口位。**
将来把 `payload` 换成 pb 序列化结果时，消费端按这两个字段判别即可，
**消息语义与主题不变**，无需改消费逻辑。

---

## 4. 幂等与断线重连

总线的重连语义是 **at-least-once**（可能重投），**所以去重必须落在消费端**。

`Consumer` 把两样东西**持久化到磁盘**（而非仅内存）：

| 文件 | 用途 |
| --- | --- |
| `<topic>.<name>.offset` | 已消费到的字节偏移 → 重连时从此继续（不丢、不从头重放） |
| `<topic>.<name>.consumed` | 已消费的 `msg_id` 集合 → 幂等 |

**为什么必须落盘**：只在内存里去重，进程重启后必然重复消费 ——
而「断线重连」这个场景本身就隐含了进程可能重启。

**去重按消费者隔离**：消费者 A 消费过的消息，消费者 B 仍应拿到。
两条独立的下游流不能因为共享去重集合而互相饿死。

---

## 5. 数据总线：为什么从第一天就分片

每个实例一个目录，序列按 `shard-NNNNN.jsonl.gz` 分片，配 `manifest.json` 记录
每片的**起止 tick、条目数、原始/文件字节数与 sha256**。

**分片的理由**：I15 已把「复盘文件 ≤100MB」写进 schema，
而 `I18` 对超限是**拒绝**。长对局的态势序列很容易越过该阈值，
所以存储层从第一天就分片，**避免将来为了过闸而重构格式**。
`manifest` 的 tick 区间使 `state_at_tick()` 能**先跳过无关分片**再片内查找。

---

## 6. 与 31005 内部接口规范的差距（判据 3 要求的书面说明）

| 维度 | 31005 内部接口规范 | 本实现 | 差距性质 |
| --- | --- | --- | --- |
| 消息编码 | Protocol Buffers / Avro（强 schema、二进制） | **JSON**（`payload_encoding:"json"`） | **未实现**；已预留 `schema`/`payload_encoding` 位 |
| 传输 | gRPC / MQTT（网络化、双向流） | **本地文件追加**（JSONL） | **未实现**；远程无容器运行时、无 Redis，网络中间件不可用 |
| 压缩 | 规范未指定 | **gzip**（标准库） | 可替换；`manifest.compression` 已记录 |
| 服务发现 | 注册中心 | **JSONL 注册表 + 文件心跳** | 简化形态 |
| 交付语义 | 未在本清单给定 | **at-least-once + 消费端幂等** | 已实现 |
| 批量 | 支持 | **`publish_batch`（逐行追加）** | 已实现（非单次批量帧） |

**升级路径**：本包的对外语义集中在三个类（`MessageBus` / `DataBus` / `Registry`）。
换 gRPC/MQTT 或 Protobuf 时**替换这三个类的实现即可**，调用方与契约不变。

---

## 7. 已知局限（不夸大）

1. **压缩在极小分片下可能「变大」**。演示里 40 条 × 2 片时
   `raw=5250B → file=5797B` —— gzip 有每流约 18 字节头 + 块填充，
   **片太小则压缩收益为负**。默认分片阈值 8MB（真实规模下压缩率约 50%，
   见测试里的 `shard_bytes=4096` 用例），但**小规模演示不代表真实压缩收益**。
2. **判死必然滞后**：实例崩溃不会发「下线」，只能靠 TTL 推断，
   **最长滞后一个 TTL（默认 30s）**。这是分布式的固有代价，不是实现缺陷。
3. **调度只做最小形态**：`pick()` 选启动最早的存活且未分配实例，
   **不做负载感知、不做抢占、不做重试**。
4. **多进程并发未实测**：演示用线程（共享解释器，能暴露并发写风险）；
   跨进程正确性由「关-开-再读」的持久化测试覆盖，**未做真实多进程压测**。
5. **未接持久化中间件**（Redis/Kafka）：环境不具备，与 I18 同一处境。
6. **`heartbeat` 未做自动周期发送**：由调用方显式调用，本包不启后台线程
   （那会让「可关闭」变复杂）。

---

## 8. 用法

```bash
# 2 实例并行演示（判据 1/2）
python3 -m training.bus.demo_two_instances --ticks 40

# 保留现场以观察产物
python3 -m training.bus.demo_two_instances --root /tmp/bus-demo --keep
```

```python
from training.bus import Message, MessageBus, Consumer, DataBus, Registry

bus = MessageBus(root / "bus")                  # enabled=False 即整体关闭
bus.publish(Message(msg_type="decision", instance_id="env-a", payload={"tick": 0}))

dbus = DataBus(root / "data", "env-a")
dbus.append(tick=0, state=frame)                # 自动分片 + gzip
assert dbus.state_at_tick(0)["state"] == frame

reg = Registry(root / "registry", ttl_seconds=30)
reg.register("env-a"); reg.heartbeat("env-a")
```

---

## 9. 本文件的边界

- 差距表（§6）是**结构性对照**，不是逐条条款比对 ——
  31005 内部接口规范的完整条文未在本工作区提供，故对照基于任务清单
  `I19`【具体步骤】第 6 条所列的 gRPC / MQTT / Avro 三项。
- **未做**：真实多进程压测、网络中间件接入、负载感知调度、
  Protobuf 实际编解码（仅预留位）。
