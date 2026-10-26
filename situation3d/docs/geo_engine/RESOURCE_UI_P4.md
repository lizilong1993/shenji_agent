# 测试端控制器 UI P4

> 2026-10-26，分支 `feat/task-resource`。只改二级页面 `web/sim.html` 与
> `web/src/simConsole.js`；主页面（index.html/app.js）与 8848 演示端未改动。

## 1. 新增能力

- **想定日期**：`datetime-local` 输入，默认当前本地时间；提交时转 ISO UTC，
  缺省由 runner 取当前 UTC 时间。
- **资源模式**：auto（默认）/ offline / online，随 `POST /api/sim/runs` 的
  `resource_mode` 传入。
- **指挥方**：红方/蓝方（默认红方）；单方视角，导演视角可通过任务 `side` 扩展。
- **开局任务编辑器**：单位 ID（逗号分隔）、任务类型（move_to/seize/hold）、目标名或经纬度、
  开始秒、节奏；任务模式 user（覆盖所涉单位）/ mixed（保留默认，冲突报错）。
  未添加任务时不发送 tasks，执行案例内置 doctrine。
- **运行表资源列**：显示 ready/degraded/等待资源；degraded 显示提示条数；
  waiting 显示缺口资源类型。
- **资源处理面板**：对 `waiting_resources` 运行展示缺口原因、建议路径/命令（含网络探测/质量门
  失败原因与降级数据提示），并提供：
  - 接受该缓存数据（use_cached_stale，仅当缺口带 entry_id）；
  - 在线刷新（fetch_online）；
  - 提供环境包路径（provide_file，服务器绝对路径，且必须位于 `SITUATION_RESOURCE_ROOTS`
    或默认 envpacks/var/resources 目录内）；
  - 取消运行（cancel）。
- 轮询：queued/running 每 3 s，waiting_resources 每 10 s。

## 2. 验证

Playwright（临时测试服务器 8861/8862，独立 runs 目录；未触碰 8858/8848）：

1. 页面加载：案例下拉出现，无 pageerror / console error；默认日期已填。
2. 添加任务（4001,4002 → 120.10/22.93，offline 模式）发起：API 202，run.config 记录
   `scenario_date=2026-10-26T05:27:00.000Z`、`tasking={mode:user,side:0,tasks:1}`。
3. 想定日期 2026-09-01 + offline + 20 tick：运行进入“等待资源”，资源列显示 envpack 与
   “候选快照晚于想定日期”提示；点击“处理资源”打开缺口面板。
4. 点击“提供环境包路径”并输入允许目录内的 envpack：resolve 成功，运行完成；
   资源列显示 ready + “降级数据”提示；表格出现“完成”。
5. 全程无 pageerror；截图用于目视检查布局。

## 3. 限制

- 任务编辑器只支持 3 种任务类型；不支持运行中追加。
- 提供环境包路径需要用户在服务器上准备好目录；浏览器无法上传大文件。
- UI 仍为测试端专用，demo 模式 `/api/sim/*` 依然 403。
- 资源列的 ready/degraded 来自 run 状态索引（`_entry` 新增 resources/resource_gaps 字段），
  列表返回体积略有增加。
