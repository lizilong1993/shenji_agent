# objective 标签源与奖励函数：修复前基线（I06 步骤 1）

> I06 步骤 1 要求「固化现状证据……作为修复前基线写入复现文档」。
> 本文所有数字均为**远程实测**，不是从任务清单转抄。

## 1. `ai/weights/model_manifest.json` 的 objective 段（实测）

| 字段 | 实测值 |
| --- | --- |
| `objective_mode` | `heuristic_policy_score` |
| `objective_enabled` | `true` |
| `objective_label_counts` | `{"total": 1480, "high_confidence": 0, "medium_confidence": 1453, "low_confidence": 27}` |
| `objective_metrics` | `{"model_trained": false, "reason": "existing artifacts yielded no high-confidence score/capture/damage labels"}` |
| `objective_label_source` | `experiments/shenji_agent/objective_labels/latest/objective_labels.jsonl` |

**即：`high_confidence=0`、`model_trained=false` —— objective 模型从未真正训练。**

## 2. 但标签产物本身**已经**产出过高置信标签（实测，与 manifest 不一致）

| 产物 | label_count | 置信分布 | reward 分布 |
| --- | --- | --- | --- |
| `latest/summary.json` | 2344 | high 2266 / medium 78 | **zero 2266** / positive 78 / negative 0 |
| `latest_after_auth/summary.json` | 2344 | — | 同上，`reward_proxy_mean=2.53` |
| `after_smoke_20260429_032914/summary.json` | 13440 | high 13440 | **zero 13440**，`reward_proxy_mean=0.0` |

**两个结论：**

1. **manifest 是陈旧的**：它记录的是更早一次（`high_confidence=0`）的结果；
   `latest_after_auth` 之后其实已有高置信标签。
   故「标签源修不好」这个判断需要修正 —— **采集端修过一轮，是 manifest 没跟上**。
2. **真正的核心缺陷是奖励稀疏**：`latest_after_auth` 里 **2344 条中 2266 条 reward=0**（96.7%）。
   奖励只在**终局**由 `selfplay.py:673 _terminal_objective_score()` 产出一次，
   绝大多数步的监督信号是同一个 0 —— **这才是「无法训练」的直接原因**。

## 3. 奖励产出点的结构性证据

| 位置 | 事实 |
| --- | --- |
| `training/selfplay.py:673-674` | `_terminal_objective_score()` **只在 `while` 循环结束后**调用一次，每方一个标量 |
| `training/selfplay.py:232-245` | 该函数先取 `snapshot["score"]`，非零即返回；否则用城市/血量/存活数的**线性启发式** |
| `training/selfplay.py:541 _build_step_objective_records()` | **每步**其实都构造了 `progress`（含 `score_delta`/`city_value_delta`/`damage_dealt`/`kill_dealt` 等），但**没有被用作奖励** |

→ 稠密信号**已经采到了**，只是没被塑形进奖励。这正是 I06 步骤 4 要补的。

## 4. 引擎依赖（决定哪些步骤在无鉴权环境里做不到）

| 步骤 | 是否需要引擎 | 现状 |
| --- | --- | --- |
| 1 固化基线 | 否 | ✅ 本文 |
| 2 修标签源（重跑采集） | **是** | ❌ `.engine_config` 不存在、鉴权不通过 |
| 3 训练 objective 模型 | **是**（需新标签） | ❌ 同上 |
| 4 奖励函数 | 否（纯函数） | ✅ `training/reward.py` + 26 条测试 |
| 5 消融实验（自博弈对局） | **是** | ❌ 配置已就位，**未跑过** |

鉴权现状见 `docs/deployment_auth.md`。

## 5. 修复前基线的用途

消融报告必须与本文对比才有意义。特别地：

- 若修好奖励后 `objective_label_counts.high_confidence` **仍为 0**，说明问题不在奖励塑形；
- 若新标签里 **reward=0 的比例仍接近 96.7%**，说明稠密化没生效；
- `objective_metrics.model_trained` 必须由 `false` 变 `true`（判据 1）。

**本文写于 `training/reward.py` 落地之前/同期，其时 step 2/3/5 尚未执行。**
