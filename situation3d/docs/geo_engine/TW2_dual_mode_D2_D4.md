# TW2 双模式 D2–D4：控制台、选定发布、引擎战损与节点（2026-10-23）

> 承接 `TW2_dual_mode_D1.md`。分支 `feat/tw2-dual-mode`，worktree
> `/remote-home/lizilong/shenji_agent_tw2`；8848 演示端未改动。

## 1. 测试端控制台（D2）

启动（测试端）：

```bash
cd situation3d
SITUATION_DIR=$PWD/web/data/situations \
python3 server/serve.py --port 8876 --mode test --runs-dir var/sim_runs
# 打开 http://127.0.0.1:8876/sim.html
```

页面提供：案例选择、seed/时长、发起推演；运行列表（状态、seed、事件/开火/毁伤、
双方损失、校验、角色徽标）；查看、设为最优、指定保留/取消、删除。运行中每 3 秒轮询，
列表显示当前 tick。操作有二次确认。

新增 API（仅 test 模式；demo 下 403/404）：

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/sim/cases` | 可推演案例 |
| POST | `/api/sim/runs` | `{case_id, seed, ticks}`，spawn 子进程，返回 run_id |
| GET | `/api/sim/runs?case_id=` | 运行列表与指标 |
| GET | `/api/sim/runs/<id>?case_id=` | 运行详情 |
| POST | `/api/sim/runs/<id>/select` | `{confirm:true, note}` 设为最优 |
| POST | `/api/sim/runs/<id>/pin` | `{replace?}` 指定保留（第二份需替换） |
| POST | `/api/sim/runs/<id>/unpin` | 取消指定保留 |
| POST | `/api/sim/runs/<id>/delete` | `{confirm:true}` 删除（best/pinned 受保护） |

同一案例同时只允许一个 queued/running；`_watch_run` 监督子进程，异常退出时写 failed。
演示端静态 `/sim.html`、`/src/simConsole.js` 在 demo 模式返回 404。

## 2. 选定结果发布（D3）

只有 **best（设为最优）且 validation.passed** 的运行可发布：

```bash
python3 scripts/promote_run.py --case taiwan_dense_transfer_v1 \
  --run-id <run_id> --runs-dir var/sim_runs \
  --demo-root /remote-home/lizilong/shenji_agent_demo --dry-run
# 校验通过后加 --confirm 执行
```

- `promote` 校验 selection、validation、checksums，复制 `trajectory.json` 到
  `<demo>/situation3d/web/data/situations/<case>.json`，发布清单写到 `<demo>/situation3d/web/data/demo_release.json`（**不在** situations 目录内，避免被案例扫描器当成案例）
  （run_id/sha256/engine_commit/scenario_hash/seed），旧文件备份 `.bak_prev_*`；
- 复制后再次核对 sha256；不调用引擎；
- `ops/demo_update.sh --with-data` 若发现 `web/data/demo_release.json`，会校验演示文件哈希并
  **跳过**该案例从 main 的覆盖，避免把选定结果冲掉；
- 8848 的实际切换仍按里程碑、经用户同意后执行，本次未发布。

## 3. 引擎战损上屏（D4）

`web/src/targetStatus.js`：当 `data.provenance.runtime === "engine_run"` 时，优先使用
`frames[].strength`（引擎 `blood/max_blood`）计算血条/四档状态；其他案例完全走原
“告警暴露度”逻辑，旧案例显示不变。已用探针页验证 1.0/0.55/0 → 正常/受损/损毁。

`trajectory.json` 现在内嵌 `nodes[]`（关键态势节点）；前端只在数据含 nodes 时渲染
底部“关键节点”按钮条，点击跳到对应 tick。旧案例不出现该条。

## 4. 验证

- 全量 `pytest situation3d/tests`：**340 passed / 14 skipped**；
- 新增 `tests/test_run_store.py`（显式 run_id、审计）、`tests/test_promote.py`（选定/校验/
  篡改/备份）、`tests/test_server_api.py`（test 列表/选定/钉住；demo 403/404）；
- 手动：测试端发起 10 tick 推演（~30 s 含建库），列表/选定/钉住/取消全通；
  `/sim.html` headless 加载显示运行行；`?run=` 页面无 JS 异常，节点条显示
  “首次交火 · 0.1h / 推演结束 · 0.2h”；demo 模式 `/sim.html` 404；
- 1440 tick 全量运行：148 事件 / 44 开火 / 20 毁伤，红 14 / 蓝 6，validation 通过。

## 5. 剩余

- 合并 main / 推送 GitHub / 更新 8848：等待代码评审与用户发布确认；
- 运行库 1440 tick 约 7 分钟；后续可做 LOS 缓存/并行；
- `inference_frames` 仍为空（右侧威胁面板暂无数据）；交通让行与亚单元瞬态仍有少量记录；
- 演示端实际发布前需执行 D3 的 `--dry-run` 并确认选定的 run_id。
