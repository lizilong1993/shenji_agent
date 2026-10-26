# 资源预检、等待与恢复 P2

> 2026-10-26，分支 `feat/task-resource`。运行期始终离线；在线获取属于 P3，本阶段只做
> “已登记资源的选择 + 缺口分级 + 暂停/恢复”。

## 1. 流程

```text
POST /api/sim/runs
  └─ 创建 run（queued）→ 子进程 runner
       1. 本地登记资源（不联网）：缺登记时自动扫描仓库产物与默认 envpack
       2. 案例数据早检（case_roads）——缺失立即 waiting_resources
       3. 生成想定（日期/任务）→ 计算需求（envpack/case_roads/rolecards/weapons/platforms/…）
       4. 解析资源计划：
            ready     → running → 推演 → completed
            degraded  → running（警告写入 run.resources/summary）→ completed
            blocked   → waiting_resources（不建世界、不推演）
POST /api/sim/runs/<id>/resolve
  └─ 写入 decisions.json → 状态回 queued → 重新拉起 runner
       接受过期数据/补齐资源后：ready 或 degraded → 推演；仍 blocked → 继续等待
```

状态机新增 `waiting_resources`：

```text
queued → waiting_resources → queued → running → completed
waiting_resources / running → failed / cancelled
```

`waiting_resources` 占用 active 槽位（同一案例不能并行发起），重启服务时不会被
`_cleanup_stale_runs` 误判为僵尸任务，删除前需先取消。

## 2. 需求与分级（`resources/resolver.py`）

| 资源 | 默认级别 | 触发条件 |
|---|---|---|
| envpack | critical | 总是（空间权威） |
| case_roads | critical | 案例有 theatre |
| rolecards | critical | `engine.use_role_cards` |
| weapons | critical | 总是 |
| platforms | critical（multi） | 总是；10 个平台文件全部需要且逐一校验 |
| intel_bundle / weather / commander_service | advisory | 场景显式引用/启用时 |

选择规则：

1. 匹配 scope：名称（theatre）与 bbox（包含场景锚点 + 所有任务点；无 bbox 的资源视为全局）。
2. 不使用“晚于想定日期 +1 天”的数据（point-in-time）。
3. 在可用候选中取 `as_of` 最新；随后校验内容哈希，损坏的候选跳过并记为 integrity 缺口。
4. 超过该类型 TTL（envpack 30 天、情报 1 天、DEM/GEBCO 365 天）标记 stale。时效参考日为
   `min(想定日期, 今天)`：未来想定不会被当前数据误判过期；历史想定的过期建议是调整日期或
   提供历史快照，而不是在线刷新。critical → `blocked`；advisory → `degraded` 继续。
   日期过滤只作用于世界数据类（envpack/case_roads/水深/天气/情报），角色卡/平台/武器等
   代码数据不做历史日期过滤。
5. 用户可用 `use_cached_stale` 决策显式接受过期数据，计划转为 degraded 并留下警告。

`--envpack <path>` 是显式人工覆盖：登记为 `provider=cli-override`，跳过日期过滤，
但仍做完整哈希校验。

## 3. 决策

`POST /api/sim/runs/<id>/resolve`

```json
{"case_id": "taiwan_dense_transfer_v1",
 "decisions": [{"action": "use_cached_stale",
                "entry_id": "envpack:taiwan_strait",
                "note": "演示需要，接受 2026-10-23 快照"}]}
```

- 已实现：`use_cached_stale`、`cancel`（runner 将 run 置为 cancelled）。
- 预留（P3/P4）：`fetch_online`、`provide_file`；当前返回未解决缺口，计划保持 blocked。

## 4. 产物与可追溯

- `run.json.resources`：计划状态、plan_hash、选中条目（路径/哈希/as_of/quality/来源）、缺口、警告。
- `run.json.resource_gaps`：等待原因，供控制台直接渲染。
- `resources.json`：写入运行目录并随 `complete_run` 计入 checksums。
- `trajectory.json.provenance`：`resource_plan_hash`、`resources_status`。
- `summary.json.resources`：计划状态、plan_hash、警告列表。
- `/api/health.subchecks.sim_resources`：按 kind 计数的索引摘要（demo 模式也可用）。

## 5. 已验证

- 单元测试：`tests/test_resource_plan.py` 10 项（最新快照、未来数据拒绝、过期阻断/接受、
  advisory 降级、bbox 越界、多资源完整性、需求推导、摘要文案、waiting 状态机）。
- 全量：479 passed / 14 skipped。
- CLI：2027-01-01 想定 → waiting（envpack stale 70 天）→ decisions 接受 → completed/degraded。
- API（临时 8860）：launch 202 → waiting_resources（约 16 s，含想定构建与哈希）→
  resolve 200 → completed/degraded，警告写入 run.resources；`sim_resources` 健康子项正常。

## 6. 已知限制

- 预检发生在 runner 子进程内：发起后需等待想定构建（约 10–20 s）才看到 waiting，而非同步 202。
  后续可在服务端先做轻量 case_roads/envpack 预检来减少等待。
- `fetch_online`/`provide_file` 未实现（P3）。
- 失败/取消后不会自动清理 `tasks.request.json`/`decisions.json`，保留作为审计。
- 运行中资源需求变化不在本轮范围（仿真核心零网络、资源在开跑前钉死）。
