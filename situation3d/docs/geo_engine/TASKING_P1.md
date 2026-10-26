# 想定日期与开局任务 P1

> 2026-10-26，分支 `feat/task-resource`。只覆盖测试端（8858）启动推演；运行中追加任务、
> 资源预检与前端二级页面分别属于后续阶段。

## 1. 想定日期

- `POST /api/sim/runs` 与 `scripts/run_situation_case.py` 接受 `scenario_date`（`--scenario-date`），
  ISO 8601，可带时区；日期缺省为**当前 UTC 时刻**（runner 只解析一次，保证快照、回放时钟、
  run 清单一致）。
- 写入位置：`run.json.config.scenario_date`、`scenario.json.start_time_utc`、
  `trajectory.json.start_time_utc`。
- 本阶段日期只改变时钟与 provenance，不启用天气/昼夜模型（`weather_enabled` 仍由场景显式配置），
  因此不改变既有推演结果；资源时效匹配在 P2 使用同一日期。

## 2. 任务请求

```json
{
  "mode": "user",
  "side": 0,
  "tasks": [
    {"id": "U1", "side": 0, "units": [4001, 4002],
     "task": "move_to",
     "objective": "北部空域",
     "lon": 120.150, "lat": 23.000,
     "after": ["R-CARRIER-1"], "start_s": 0,
     "pace": "normal", "risk": "normal", "arrive_m": 250}
  ]
}
```

- `task` 支持 `move_to | seize | hold`；`pace ∈ dash/normal/cautious`；
  `risk ∈ low/normal/high`；`lon/lat` 与 `objective` 至少给一个，两者都给时以坐标为准。
- `mode=user`（有任务时的默认）：**按单位覆盖**——用户任务涉及的己方单位从默认任务中移除；
  默认任务被全部覆盖则删除，部分覆盖则剩余单位继续执行；若删除的任务是其它默认任务的前置，
  直接拒绝（避免依赖悬空）。
- `mode=mixed`：默认任务全部保留；用户任务与同侧默认任务单位重叠时拒绝。
- `mode=default`：不允许携带任务。
- 单方视角：`side` 缺省 0（红方），蓝方保持案例内置 doctrine。导演视角可在每条任务上显式写
  `side`，或在请求级设置；`mode=user` 的覆盖范围始终是 `side` 指定的那一侧。

### 校验规则（`sim/tasking.py`）

未知单位、阵营不符、单位重复编组、未知任务类型、未知目标名、`lon/lat` 缺一、坐标超出 envpack bbox、
未知前置任务、依赖成环、`start_s` 超出想定时长、非法 pace/risk/arrive_m、重复任务 id 均以
`TaskError` 拒绝，并在 CLI 中打印可读原因；API 启动的 run 会以失败状态结束并保留日志（P2 预检会前移）。

## 3. 调用方式

```bash
python3 scripts/run_situation_case.py --case taiwan_dense_transfer_v1 --ticks 60 \
  --scenario-date 2026-10-26T08:00:00+08:00 \
  --tasks-file /tmp/tasks.json \
  --envpack /path/to/envpacks/taiwan_tw2_v1 \
  --runs-dir /tmp/sim_runs
```

```bash
curl -s -X POST localhost:8858/api/sim/runs -H 'Content-Type: application/json' -d '{
  "case_id": "taiwan_dense_transfer_v1", "seed": 20261023, "ticks": 60,
  "scenario_date": "2026-10-26T00:00:00Z",
  "task_mode": "user", "side": 0,
  "tasks": [{"id": "U1", "units": [4001, 4002], "task": "move_to",
             "lon": 120.10, "lat": 22.93}]
}'
```

（API 的 `mode` 字段由 `task_mode` 传入；上例的 `mode` 键建议改用 `task_mode`。）

## 4. 已验证

- `python3 -m pytest tests/test_tasking.py -q`：26 项通过。
- 端到端 60 tick（seed 默认、真实 envpack）：用户任务 U1 使 4001/4002 的轨迹改变并在 30 tick
  到达 120.10/22.93；默认 R-AIR 被裁剪为剩余 4003/4004；`U1 开始/完成` 事件入轨迹；
  `validation.passed=true`。
- 非法请求（未知单位、`mode=default` 携带任务）在创建 run 前即被拒绝，退出码 1。
- 缺省日期运行时，`run.json`/`scenario.json`/`trajectory.json` 三处起始时间完全一致。

## 5. 未覆盖

- 运行中追加/修改任务（用户已确认仅开局下达）。
- 更丰富任务类型（patrol/observe/escort/resupply/withdraw）。
- UI 任务编辑（P4，仅二级页面）。
- 任务点可通行性预检（当前只做 bbox 与语法校验，路线可行性由引擎执行时判定）。
