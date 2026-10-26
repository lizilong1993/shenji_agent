# P5 验收记录：任务驱动 + 资源管理

> 2026-10-26，分支 `feat/task-resource`（基于 main `d0fbbc1`）。验收环境：server155
> 上独立 worktree；测试端临时端口 8860–8864；8848 演示端与 8858 运行实例均未重启/未改动。

## 1. 用户七项验收

| # | 验收项 | 结果 | 证据 |
|---|---|---|---|
| 1 | 初始想定 + 用户任务自动推进 | 通过 | 60 tick CLI：U1（4001/4002 → 120.10/22.93）在 30 tick 到达，R-AIR 被裁剪为 4003/4004；轨迹含 “U1 开始/完成”；换任务/换日期轨迹与哈希变化 |
| 2 | 在线获取并缓存 | 通过（受网络限制） | Terrarium DEM 在线抓取 21×21 网格 → 写入缓存 → 离线回放逐值一致；envpack 刷新路径用注入 OSM/DEM 源验证“重建→登记新版本→旧版本保留”；真实 Overpass 在当前 egress 下不可达（探针 4.0 s=False，查询 61 s 超时），质量门拒绝登记不完整包 |
| 3 | 离线自动切换 | 通过 | `--resource-mode offline` 全程无网络；注册表/缓存回放；DEM 离线回放 `fetched_network=False` 且结果一致；缺失/过期按 P2 分级等待 |
| 4 | 缓存与想定日期匹配 + 时效提示 | 通过 | 解析器测试：未来快照拒绝、最新 as_of 选择、过期 critical 阻断、advisory 降级；未来想定日期不再把当前数据误判过期（参考日=min(想定日, 今天)）；UI 显示 stale 原因与 `quality=derived` |
| 5 | 非必要资源缺失提示并继续 | 通过 | advisory stale 计划 degraded；接受过期 envpack 的运行 completed/degraded，警告写入 run.resources 与 summary；地形 fallback 在控制台显示“降级数据” |
| 6 | 必要资源缺失暂停/处理/恢复 | 通过 | 历史日期（2026-09-01）→ waiting_resources（envpack 无该时点快照，pid=None）；并发两次 resolve 一次 200、一次 409；`provide_file` 指向环境包后 completed/ready，resource_gaps 清空 |
| 7 | 稳定性与一致性 | 通过 | 全量 `pytest tests`：493 passed / 14 skipped；同 seed 同资源计划两次 20 tick 的 frames/events/outcome 哈希一致（59ae1fb9…），plan_hash 一致（b273cf33…）；demo 模式仍不加载 sim |

## 2. 端到端记录

- CLI：`scripts/run_situation_case.py --scenario-date ... --tasks-file ... --resource-mode ...`
  - 未来日期 2027-01-01 + offline：completed/ready（当前快照 2026-10-23，参考日=今天）。
  - 历史日期 2026-09-01：waiting（envpack no_point_in_time），提供环境包后完成。
  - 任务坐标越界（121.5, 24.5）：在解析前被拒绝，提示 “outside envpack bbox”，不会被伪装成资源缺口。
- API（临时端口）：launch 202 → waiting（pid=None）→ resolve（并发一次成功一次 409）→ completed；
  provide_file 路径限制在允许根目录内，越界返回 400。
- UI（二级页面，Playwright）：无 pageerror/console error；想定日期、任务编辑、资源模式可选择；
  等待运行可打开“处理资源”面板，显示原因/建议/命令与 fetch_error；提供路径后完成；
  资源列显示 ready/degraded、警告条数与降级数据项数。

## 3. 资源一致性

- 运行前钉住资源计划（plan_hash 写入 `run.json.resources`、`resources.json`、
  `trajectory.provenance.resource_plan_hash`）。
- 计划内禁止混用不同 bbox/schema/快照：envpack 作为整体选择，多文件平台数据逐一校验。
- 同一计划重复运行确定性：资源计划不变时推演结果哈希一致。
- 资源索引支持 `SITUATION_RESOURCES_DIR` 环境变量隔离（验收/测试不再污染主索引）。

## 4. 未完成/限制

- 真实在线 envpack 重建未能在当前网络下端到端验证（Overpass 被 egress 挂起）；代码路径与质量门已用注入源覆盖。
- 运行中追加任务未实现（用户确认仅开局下达）。
- 任务类型仍为 move_to/seize/hold；运行中动态资源需求不在本轮（资源开跑前钉死）。
- 分支尚未合并 main、未重启 8858/8848。
