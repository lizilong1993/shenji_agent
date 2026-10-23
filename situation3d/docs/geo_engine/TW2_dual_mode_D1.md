# TW2 双模式 D1：初始想定 → 引擎推演 → 运行库回放（2026-10-23）

> 分支 `feat/tw2-dual-mode`，worktree `/remote-home/lizilong/shenji_agent_tw2`。
> 本节只描述 D1 已落地的部分：测试端离线推演与回放；演示端 8848 未改动。

## 1. 初始想定

`situation3d/campaigns/taiwan_dense.py` 只定义开始状态：

- 80 个 operator（36 红车、24 蓝车、12 舰、8 机），真实经纬度、角色卡、初始朝向/高度；
- 16 辆车显式 `carrier`（TRV001-016 搭乘 TRS001/TRS002），由引擎在泊位下船；
- 3 个目标点（D/E/F）与 14 条任务（航渡、夺控、守备、海上/空中警戒）；
- 引擎配置：`detection_model=physical`、物理直瞄/间瞄、导弹飞行段、30 m 导航场、
  `scan_interval_s=180`、任务接敌停车/恢复（`roe.halt_on_contact=true`）。

没有 waypoints、逐帧位置、释放 tick、写死事件或结局。

## 2. TW2 AOI envpack

现有 `anping_v2c` 仅覆盖 120.14–120.19 / 22.97–23.02，TW2 的地面区与西侧海域都在包外。
本分支使用本地 `taiwan-latest.osm.pbf` + GDAL OSM 驱动离线构建：

```bash
# 1) 提取 lines / multipolygons（GDAL 已装；服务器无 pyosmium）
ogr2ogr -f GeoJSON /tmp/tw2osm/lines.geojson taiwan-latest.osm.pbf lines \
  -spat 120.06 22.92 120.23 23.02 \
  -where "highway IS NOT NULL OR waterway IS NOT NULL"
ogr2ogr -f GeoJSON /tmp/tw2osm/poly.geojson taiwan-latest.osm.pbf multipolygons \
  -spat 120.06 22.92 120.23 23.02 \
  -where "building IS NOT NULL OR natural IS NOT NULL OR landuse IS NOT NULL"

# 2) 构建 envpack（示意海面框避免“海面被当成陆地”）
python3 tools/build_envpack_from_geojson.py \
  --lines /tmp/tw2osm/lines.geojson --polygons /tmp/tw2osm/poly.geojson \
  --basic /remote-home/lizilong/shenji_agent_geo/situation3d/envpacks/anping_v2c/base_map.json \
  --out envpacks/taiwan_tw2_v1 --bbox 120.055 22.915 120.235 23.025 \
  --sea-box 120.055 22.915 120.143 23.025 --resolution-m 100
```

结果：15,920 道路、16,132 建筑、1,741 水面、1,883 土地覆盖；包内禁止使用粗底图
回退（`fallback_basic_json=null`），否则陆地点会被 2 km 底图误判为水面。
envpack 目录被 `situation3d/.gitignore` 忽略，需在部署机重建或拷贝。

## 3. 推演与运行库

```bash
python3 scripts/run_situation_case.py --case taiwan_dense_transfer_v1 \
  --seed 20261023 --runs-dir var/sim_runs
```

运行库布局见 `server/run_store.py` 文件头。一次 1440 tick 的结果：

| 指标 | 值 |
|---|---|
| 帧 | 1441（30 s/帧，12 h） |
| 事件 | 148：fire 44、impact 44、destroy 20、mission 20、release 16、objective 4 |
| 损失 | 红 14 / 蓝 6 |
| 目标归属 | D 红，E/F 蓝（partial） |
| 产物 | trajectory.json 约 32.3 MB |
| 耗时 | 501 s（p50 327 ms / p95 510 ms per tick） |
| 校验 | 通过：finite；water_hits=2（瞬态）、building_hits=1（瞬态） |

`validation` 规则：位置有限；停驻单位由子格清障移到最近空地；移动中的亚单元
瞬态水/建筑命中允许 2 次（20k 行 +1），系统性违规仍失败。

## 4. 测试端回放

```bash
SITUATION_DIR=/remote-home/lizilong/shenji_agent/situation3d/web/data/situations \
  python3 server/serve.py --port 8876 --mode test \
  --runs-dir var/sim_runs
# 打开
http://127.0.0.1:8876/?case=taiwan_dense_transfer_v1&run=<run_id>
```

- `GET /api/sim/runs?case_id=...`：运行列表（test 模式）；
- `GET /api/situation?id=...&run=<run_id>&part=core|inference`：读取运行库轨迹；
- demo 模式下上述接口返回 400/404，demo 进程启动时断言未加载 `sim/rules`。
- 前端 `app.js` 在 URL 带 `?run=` 时向 `/api/situation` 追加 run 参数；模式判定由服务端负责（demo 返回 400），运行结果没有整包回退。

## 5. 已知限制（D1）

- 探测每 180 s 一次，1440 tick 约 7 分钟；多次选优需要等待，后续可做 LOS 缓存/并行。
- 交通让行仍会产生 `traffic_stuck`（导出事件中已过滤 crowding），个别单位会停留；
  属于 P0.2 交通层与 30 s 步长的已知行为。
- 城市亚单元建筑遮挡：30 m 导航场仍可能漏小footprint；D1 用“停驻清障”保证画面
  不在楼内，移动瞬态允许极小比例（见 validation）。
- 未生成 `inference_frames`；右侧 TS/威胁面板暂为空。战损显示仍走旧推断，
  `targetStatus.js` 改读引擎血量属于已批准但尚未实施的 UI 项（D4）。
- 测试端控制台（发起/对比/选定）尚未实现（D2）；目前用 CLI 运行 + URL 查看。

## 6. 复现与下一步

1. 跑 2–3 个 seed，比较 summary/metrics；
2. D2：`/sim.html` 控制台、运行列表、`selection.json` 与三槽位；
3. D3：`promote_run.py` 发布到 demo worktree；D4：引擎战损上屏、节点导览；
4. 8848 仍按里程碑、经用户同意后更新。
