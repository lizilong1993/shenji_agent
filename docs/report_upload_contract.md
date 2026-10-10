# 对抗成绩与复盘文件回传契约（I18）

> 依据 `31006 5.1.2 数据回传系列接口`。
> **字段对齐的是仓库里的真实产物**，不是照抄标准条文 —— 实测范围见 §2。

---

## 1. 一句话

一次对局产生**一条**回传记录，按**确定性 `match_id`** 幂等落盘；
复盘文件在回传前**校验大小**（>100MB **拒绝**并提示分片）。

**当前后端：本地目录**（`--target`）。平台侧服务接口**未开放**，
故按 I18【风险与回滚】的既定口径：**本地目录先行，服务回传作为可选后端**。

---

## 2. 契约的字段从哪来（实测依据）

对真实实验目录 `experiments/shenji_agent_smoke/league_20260427_140817/` 实读：

| 文件 | 提供什么 |
| --- | --- |
| `match_results.json` | `matches[]`，**每场 21 个字段**：`scenario` / `bucket` / `map_id` / `order_index` / `paired_score` / `first_leg` / `second_leg` / `candidate_*` / `opponent_*` |
| `participants.json` | 双方 `artifacts` / `profile` / `user_name` / `source_commit_sha` / `code_origin` |
| `rating.json` | `overall`（含 `point_estimate`/`ci95`/`elo`/`count`）、`buckets`、`policy` |
| `promotion.json` | `accepted`、`failures[]`、各项覆盖率与回归量 |

**一个关键事实**：`match_results.json` 里**没有任何 replay 引用**
（实测 `replay` 字段出现 **0 次**）。
所以「哪场对局对应哪个复盘文件」**在现有产物里根本没有绑定** ——
这正是 31006 5.1.2 要求「对抗成绩回传」与「对抗复盘文件回传」**成对**出现、
而现状无法满足的原因。本契约用 **`match_id`** 承担这个绑定。

---

## 3. 回传记录（成绩）

```json
{
  "contract_version": "1.0.0",
  "match_id": "b077aba23909ae05",
  "uploaded_utc": "2026-10-10T00:00:00Z",

  "sides": {
    "candidate": {
      "user_name": "CurrentAI",
      "profile": "challenger",
      "artifacts": "/.../ai/weights",
      "source_commit_sha": "561af02b…",
      "code_origin": "git:561af02b…"
    },
    "opponent": { "...": "同上结构" }
  },

  "score": {
    "paired_score": 0.5,
    "first_leg":  { "candidate": 0.0, "opponent": 0.0 },
    "second_leg": { "candidate": 0.0, "opponent": 0.0 }
  },

  "scenario": "2230531221.json",
  "bucket": "complex",
  "map_id": 53,
  "order_index": 0,
  "run_dir": "/.../league_20260427_140817",

  "replay": {
    "path": "replay_1770632023.2594852.zip",
    "size_bytes": 3812007,
    "sha256": "…"
  }
}
```

### 字段与 31006 5.1.2 四要素的对应

| 31006 5.1.2 要求 | 本契约字段 |
| --- | --- |
| 对抗双方 | `sides.candidate` / `sides.opponent`（含身份、画像、来源提交号） |
| 对抗 ID | `match_id`（确定性，见 §4） |
| 成绩 | `score.paired_score` + 两腿明细 |
| 时间 | `uploaded_utc`（UTC，ISO 8601，秒级） |

`scenario` / `bucket` / `map_id` / `order_index` 是**追加的场景上下文**，
用于按场景或地图聚合，不影响四要素的完整性。

---

## 4. 对抗 ID 必须确定性 —— 这是幂等的前提

```
match_id = sha256("<scenario>|<map_id>|<order_index>|<candidate_sha>|<opponent_sha>")[:16]
```

**为什么不能用随机数或当前时间**：31006 5.1.2 要求重复回传不产生重复记录。
若 ID 含随机/时间成分，**同一场对局重跑脚本就会得到新 ID，幂等直接失效**。

固定源意味着：
- 同一场对局在**任何机器、任何时刻**算出的 ID 相同；
- **换场景或换任一方**提交号 → ID 必然不同（不会被误判为重复）。

---

## 5. 幂等语义

| 目标位置状态 | 行为 | 结果字段 |
| --- | --- | --- |
| 不存在 | 写入 | `written` |
| 已存在，且内容相同（**忽略 `uploaded_utc`**） | 跳过 | `skipped_identical` |
| 已存在，但内容不同 | **不覆盖**，记为冲突 | `conflicts`（并使退出码非 0） |

**比较时忽略 `uploaded_utc`**：重复回传的时间戳必然不同，
若把它计入比较，**每次重复回传都会被误报为冲突**。

**冲突为什么选择「不覆盖」而不是「覆盖」**：覆盖会**静默改写历史**。
同一 `match_id` 出现不同成绩，说明有一方数据被改过 —— 这是需要人看见的事件，不是可以自动收敛的状态。

---

## 6. 复盘文件大小校验

| 项 | 值 |
| --- | --- |
| 限额 | **100 MB**（`31006 5.1.2`，索引注**「建议」**） |
| 超限处置 | **拒绝回传**（退出码 2），并提示按 tick 区间分片 |
| 为什么与 I15 口径不同 | I15 的 schema 校验把超限当**提示**（只读，不伤害谁）；回传会把文件推给对端，**超限直接损害对端**，故必须挡在这里 |

---

## 7. 失败重试与原子性

- **原子写**：先写临时文件再 `os.replace`，避免对端读到半截 JSON；
- **重试**：目录瞬时不可写等**可恢复**故障重试 3 次；
- **不掩盖持续性错误**：重试耗尽即记 `failed` 并使退出码非 0；
- **权限**：产物落盘为 **0644**（`mkstemp` 默认 0600，会让换机/容器场景下别人读不到 —— 那正是本任务要消除的阻塞）。

---

## 8. 用法

```bash
# 只报告不写盘
python3 scripts/report_upload.py --run-dir <实验目录> --target <目标> --dry-run

# 本地目录回传
python3 scripts/report_upload.py --run-dir <实验目录> --target <目标>

# 绑定复盘文件（先过大小校验）
python3 scripts/report_upload.py --run-dir <实验目录> --target <目标> \
  --replay replays/replay_1770632023.2594852.zip

# 机器可读
python3 scripts/report_upload.py --run-dir <实验目录> --target <目标> --json
```

退出码：`0` 成功；`1` 存在冲突或失败；`2` 复盘文件超限被拒。

---

## 9. 与后续工作的接口

- **`I15`**：本契约的 `replay.sha256` 与 `size_bytes` 依赖 I15 定义的复盘容器；
  `match_id` 与复盘文件之间的绑定目前**靠调用方显式传 `--replay`**。
  **待 I15 写入端改造后**（尚未做），复盘文件内应自带 `meta.run_id`，
  届时可自动绑定，无需人工指定。
- **服务后端**：本脚本的后端是「本地目录」这一种。接平台接口时，
  只需替换落盘环节（`upload_to_dir` → HTTP 客户端），**契约与幂等语义不变**。

---

## 10. 本文件的边界

- **未接平台侧服务接口**（未开放），§8 演示的是**本地目录后端**；
- **未做真实跨机演练**（无第二台目标机）；端到端演练在**同一远程工作区**内完成；
- **`replay` 字段**在演练中绑定的是仓库内既有的那个历史复盘文件，
  而非该场对局真正产生的复盘（原局复盘未留存）—— 这一点演练记录里已注明。
