# 在线获取与缓存刷新 P3

> 2026-10-26，分支 `feat/task-resource`。仿真核心仍然零网络：在线获取只发生在
> 预检/解析阶段，成功后登记为新的资源版本，再由同一解析器选中。

## 1. 模式

`--resource-mode` / API `resource_mode`：

| 模式 | 行为 |
|---|---|
| `auto`（默认） | 只有出现 **critical envpack 缺口**、且想定日期不早于今天（`no_point_in_time` 除外）时尝试联网刷新；先做 3.5 s Overpass 探针，失败则保留现有资源并进入 waiting；成功则重建、登记、重新解析 |
| `offline` | 永不联网；只用注册表中已有资源（缺失/过期按 P2 分级处理） |
| `online` | 与 auto 相同的刷新，但探针失败视为错误（仍会等待用户） |
| 决策 `fetch_online` | 即使本次为 offline，也强制尝试一次在线刷新 |
| 决策 `provide_file` | 登记用户提供的 envpack 目录（`manual_override`，跳过日期过滤，仍做哈希校验） |

非 envpack 资源（case_roads/rolecards/platforms/weapons）均为仓库内数据，
在线刷新不在本轮范围；损坏/缺失时用 `scripts/resource.py scan` 或从仓库恢复。

## 2. 在线重建（`resources/providers.py`）

- 模板：现有 envpack 的 `manifest.json` + `base_map.json` + grid（不改变 AOI/网格/分辨率）。
- 数据源：复用 `geodata` 现有实现——Overpass（`OsmSource`）、Terrarium DEM（`DemSource`）、
  OSM PBF（`PbfOsmSource`，需要 pyosmium）与 GEBCO 可后续接入；HTTP 代理沿用环境变量。
- **公开下载优先**：Overpass 不可用时，可用 Geofabrik `taiwan-latest.osm.pbf`
  （`scripts/resource.py download-pbf <url>`，urllib 走 `https_proxy`）先落到
  `cache/osm_pbf/`，再用 `fetch --osm-pbf <path>` 重建。runner 的 auto 模式会复用
  `SITUATION_OSM_PBF` 或缓存里 7 天内的同名 PBF，不自动下载（避免无提示的大流量）。
- 输出：`<store>/cache/envpacks/<pack_id>@<UTC时间戳>/`；原始响应缓存在 `<store>/cache/_cache`。
- 登记：内容哈希 + `as_of=重建日期` + `provider=online-refresh`，entry_id 带时间戳
  （`envpack:<pack>@<stamp>`），旧版本保留，历史 run 的路径/哈希不受影响。
- 同一模板同一天只重建一次：已存在 `refreshed_from` 且 `as_of=今天` 的版本时直接复用。
- **质量门**：若重建结果的 roads 层仍是 `fallback-basic`（Overpass 不可达/返回不完整），
  默认拒绝登记，并提示检查网络/代理或使用离线包；只有显式 `--allow-degraded` 才登记
  `quality=derived` 的包。
- **回退策略继承**：模板 manifest 的 `fallback_basic_json=null` 会原样保留在新包中，
  防止 pipeline 默认的 2 km `base_map.json` 在 AOI 内重新引入建筑/地类（实测若不继承，
  20 tick 运行出现 922 个 building_hits；继承后降到 3 个瞬态命中）。

## 3. CLI

```bash
python3 scripts/resource.py fetch envpack:taiwan_strait --mode auto --timeout 20
python3 scripts/resource.py fetch envpack:taiwan_strait --allow-degraded
python3 scripts/resource.py provide /path/to/envpack --entry-id envpack:manual

# 公开下载 + PBF 重建（pyosmium 环境）
python3 scripts/resource.py download-pbf \
  https://download.geofabrik.de/asia/taiwan-latest.osm.pbf
/remote-home/lizilong/jev_gpu/venv/bin/python scripts/resource.py fetch \
  envpack:taiwan_strait --osm-pbf <cache>/osm_pbf/taiwan-latest.osm.pbf --mode online
```

## 4. 运行时行为与可追溯

- 自动刷新只发生一次/每次 runner 调用；成功/失败都写入 `run.json.config.resource_fetch_error`
  与等待摘要 `summary.fetch_error`。
- 选中条目（含新版本）写入 `resources.json` 与 `run.json.resources`，`trajectory.provenance`
  记录 `resource_plan_hash`/`resources_status`。
- 离线回退与 P2 相同：缓存条目按 `as_of` 选择；过期由用户决策。

## 5. 网络现状（server155，2026-10-26 经本机 Clash 反向隧道复测）

- 反向隧道：`ssh -R 27980:172.16.10.254:7897 situ155`，服务器侧 `https_proxy=http://127.0.0.1:27980`。
- **Overpass**：隧道 CONNECT 成功（HTTP 200），但微小查询 30 s 无响应；直连探针 4.0 s 为 False、
  真实查询 61 s 超时。结论：当前 Clash 出口下 Overpass 仍不可用。
- **Geofabrik PBF（公开下载）**：经同一代理下载 `taiwan-latest.osm.pbf` 312.7 MB 成功
  （HTTP 200/307 跟随，urllib 流式写入 `.part` 后原子改名）。
- **Terrarium DEM**：服务器直连可用（在线抓取 → 缓存 → 离线回放逐值一致）。
- 用 PBF + 在线 DEM 实测重建 TW2 envpack：32.3 MB、`osm=pbf`、`dem=online`、`fallback=false`、
  `quality=measured`；16,856 建筑 / 67,157 道路边。20 tick 运行 completed，p95 1.99 s/tick，
  `building_hits=3`（瞬态上限 2，接近通过；主要来自新数据建筑贴边）。
- 因此 auto 模式在本机不会走 Overpass；要么显式下载 PBF 后重建，要么接受离线包/等待。

## 6. 已验证

- `tests/test_resource_fetch.py` 5 项：
  - 注入 OSM/DEM 源成功重建 → 新版本登记、旧条目保留、as_of=重建日期、quality=measured；
  - OSM 不完整 → 拒绝登记（质量门），注册表条目数不变；
  - `allow_degraded` → 登记 derived；
  - `provide` 标记 `manual_override`；
  - 探针失败路径。
- 全量 510 passed / 14 skipped（合并 main 后）；新增 PBF 下载/重建测试。
- CLI 离线模式：`fetch envpack:taiwan_strait --mode offline` 在数秒内被质量门拒绝，
  注册表保持原条目。

## 7. 未覆盖

- 真实在线的端到端重建（当前网络不可达；需要代理可用后复测）。
- case_roads/情报包的在线刷新；动态天气源。
- 运行中刷新（资源在开跑前钉死）。
