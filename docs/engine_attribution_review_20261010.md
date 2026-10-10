# 引擎/鉴权归因复核（2026-10-10）

> 起因：第 49 轮实测推翻了一处我重复多轮的结论 ——
> 「采集需闭源引擎、鉴权不通过、故不可做」。
> **鉴权确实失败，但本地推演照常完成。**
> 故本轮**逐项复核**所有以「引擎/鉴权/闭源」为由标为阻塞或未做的条目。

---

## 0. 一句话结论

**「引擎鉴权」不是一道统一的墙。** 它分成两类，此前被我混为一谈：

| 类别 | 是否受鉴权影响 | 证据 |
| --- | --- | --- |
| **本地推演与数据采集** | **不受影响** | 第 49 轮实跑：`state=completed`、1000 步、8 张表 + `status.json` |
| **平台侧能力**（官方对手、线上成绩、平台指标） | **受影响** | 无 `.engine_config`，运行打印 `did not pass authentication` |

**故：凡以「引擎」为由标为阻塞的条目，必须逐项重新判定属于哪一类。**

---

## 1. 复核结果

| 条目 | 原标状态 | 原阻塞理由 | 复核结论 |
| --- | --- | --- | --- |
| `I06` 步骤 2/3/5 | 待批准 | 「被引擎鉴权阻塞」 | **归因不成立** —— 本地可推演；真正卡点是**采集配置未进战斗**（见 §2） |
| `D1` | 待批准 | 采集/发布 | **归因不成立** —— 本地可采集（第 49 轮已实跑） |
| `C3` | 待批准 | 与 `I06` 同源（消融） | **归因不成立** —— 消融是训练侧，本地可跑；但受 `I06` 标签问题牵连 |
| `P1` 横向对比层 | 待批准 | 「外部对手不可获得」 | **归因成立** —— 属平台侧能力，**确实受影响** |
| `C2` | **已完成** | —— | 它是**把鉴权前置条件文档化**，本就不依赖鉴权成功 |
| `I02`/`I12`/`C6`/`C7` | 阻塞-P0-2 | P4 部署 | **与引擎无关** —— 卡在 P0-2 部署形态（另一件事） |
| `P2` | 阻塞-P0-2 | 「torch、闭源 whl、鉴权」 | **部分不成立** —— 第 49 轮证明鉴权不阻塞本地跑测试 |

---

## 2. `I06` 的真实卡点（本轮查清）

`I06` 判据 1 要求 `objective_label_counts.high_confidence>0` 且 `model_trained=true`。
当前 manifest 值：

```json
"objective_label_counts": {"total": 1480, "high_confidence": 0,
                           "medium_confidence": 1453, "low_confidence": 27}
"objective_metrics": {"model_trained": false,
  "reason": "existing artifacts yielded no high-confidence score/capture/damage labels..."}
```

### 2.1 高置信标签的定义（`scripts/build_objective_labels.py` README）

> Confidence is **high for actual score/capture/damage events**,
> medium for distance progress, and low for heuristic/final-outcome-only labels.

### 2.2 **高置信所需的字段，采集数据里是有的**

实测了一个**既有成功场次**的 `events.jsonl`（135 条事件 / 8 类）：

| 事件类型 | 条数 | 是否 objective 相关 |
| --- | --- | --- |
| `MOVE` / `STOP` | 41 / 34 | 否 |
| `SHOOT` | 25 | 间接 |
| **`ADJUDICATION`** | **25** | **是 —— 引擎裁决** |
| `SEIZE` | 3 | **是 —— 夺控** |
| 其余 | 7 | 否 |

**`ADJUDICATION` 的 payload 完整携带 objective 字段**（实测）：

```json
{"att_obj_id": 14905, "attack_color": 1, "target_color": 0,
 "target_obj_id": 14911, "damage": -1, "att_obj_blood": 3,
 "distance": 18, "cur_step": 200, "type": "直瞄射击", "wp_id": 36}
```

`damage`、`target_obj_id`、`cur_step`、`attack_color` 都在 ——
**`damage_dealt`/`damage_taken`/`kill_*` 完全可由此导出。**

### 2.3 **但标签器根本不读它**

静态检查 `scripts/build_objective_labels.py`：

| 关键字 | 是否出现在脚本里 |
| --- | --- |
| `ADJUDICATION` | **False** |
| `events.jsonl` | **False** |
| `judge_info` | **False** |
| `SEIZE` | **False** |

**→ 根因不是"引擎给不出数据"，而是"标签器没有读引擎已经给的数据"。**

> **⚠ 状态更新（第 51 轮）**：该根因**已修复**。
> `scripts/build_objective_labels.py` 新增 `labels_from_events()`，
> 并已把 `events.jsonl` 纳入输入发现与分派。
> **实测结果**：在既有 56 场上重跑标签采集 →
> `high_confidence` **由 0 升至 3181**（`label_count` 3702，
> `damage_kill_delta` 3004 + `occupy_success` 177，正/负样本 1679/1502），
> 即 **`I06` 判据 1 的前半已达成**。
> 本节保留原根因记录，因为它解释了**为什么**会有这次修复。
它读的是 self-play 的诊断产物（`--input` 指向 `experiments/.../objective_labels`），
其中 objective 字段**全为 null**：

```
{"action_type": "Shoot", "damage_dealt": null, "damage_taken": null,
 "kill_dealt": null, "kill_taken": null, "final_score_delta": 0.0,
 "label_confidence": "low", "label_kind": "final_outcome_credit", ...}
```

### 2.4 我本轮的一个**探针失败**（如实记录）

我用 `map_123` 跑了 600 步探针，**产物里只有 3 类事件、`ADJUDICATION=0`**：

```
END_DEPLOYMENT 1200 / DEPLOY_EMBARK 599 / EMBARK 1
```

**即：600 步全耗在部署阶段，从未进入战斗** —— 故没有裁决事件。
（这也说明**采集配置会影响能否产出 high-confidence 标签**，是第二个变量。）

> **故 `I06` 步骤 2 的修复点有两个，且都不是引擎**：
> ① 让标签器读 `events.jsonl` 的 `ADJUDICATION`/`SEIZE`；
> ② 确认采集配置能让对局进入战斗（否则没有裁决事件可读）。

---

## 3. 对清单的处置建议

| 条目 | 建议 |
| --- | --- |
| `I06` | **阻塞理由订正**：从「引擎鉴权」改为「标签器未消费 `ADJUDICATION` 事件 + 采集配置需进战斗」 |
| `D1` | **阻塞理由订正**：本地采集可用；真阻塞是 `Q7` 的换算口径 |
| `C3` | 随 `I06` 一并订正 |
| `P1` 横向对比层 | **维持**「待平台方配合」（归因成立） |
| `P2` | 阻塞理由中「鉴权」一项应删（不成立） |

---

## 4. 未做

| 项 | 状态 |
| --- | --- |
| 逐项复核（本文） | **✓** |
| **改标签器读 `ADJUDICATION`** | **✓ 第 51 轮已做**（`high_confidence` 0 → 3181） |
| **重跑标签采集使 `high_confidence>0`** | **✓ 第 51 轮已做**（3181） |
| 训练 objective 模型（步骤 3） | **✗ 未做** —— 实测**训练器本身不存在**（`supervised.py` 里 `objective` 出现 0 次，全仓无 objective 模型类） |
| 消融（步骤 5） | **✗ 未做** |
| 让探针进入战斗（换配置/想定） | **✗ 未做** |
| 清理探针产物 | **✓ 已清理**（避免占盘；数据记入本文） |
