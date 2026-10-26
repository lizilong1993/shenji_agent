# 在线获取与缓存刷新 P3

> 2026-10-26，分支 `feat/task-resource`。仿真核心仍然零网络：在线获取只发生在
> 预检/解析阶段，成功后登记为新的资源版本，再由同一解析器选中。

## 1. 模式

`--resource-mode` / API `resource_mode`：

| 模式 | 行为 |
|---|---|
| `auto`（默认） | 只有出现 **critical envpack 缺口**时尝试联网刷新；先做 3.5 s Overpass 探针，失败则保留现有资源并进入 waiting；成功则重建、登记、重新解析 |
| `offline` | 永不联网；只用注册表中已有资源（缺失/过期按 P2 分级处理） |
| `online` | 与 auto 相同的刷新，但探针失败视为错误（仍会等待用户） |
| 决策 `fetch_online` | 即使本次为 offline，也强制尝试一次在线刷新 |
| 决策 `provide_file` | 登记用户提供的 envpack 目录（`manual_override`，跳过日期过滤，仍做哈希校验） |

非 envpack 资源（case_roads/rolecards/platforms/weapons）均为仓库内数据，
在线刷新不在本轮范围；损坏/缺失时用 `scripts/resource.py scan` 或从仓库恢复。

## 2. 在线重建（`resources/providers.py`）

- 模板：现有 envpack 的 `manifest.json` + `base_map.json` + grid（不改变 AOI/网格/分辨率）。
- 数据源：复用 `geodata` 现有实现——Overpass（`OsmSource`）、Terrarium DEM（`DemSource`）、
  OSM PBF（如配置）与 GEBCO 可后续接入；HTTP 代理沿用环境变量。
- 输出：`<store>/cache/envpacks/<pack_id>@<UTC时间戳>/`；原始响应缓存在 `<store>/cache/_cache`。
- 登记：内容哈希 + `as_of=重建日期` + `provider=online-refresh`，entry_id 带时间戳
  （`envpack:<pack>@<stamp>`），旧版本保留，历史 run 的路径/哈希不受影响。
- **质量门**：若重建结果的 roads 层仍是 `fallback-basic`（Overpass 不可达/返回不完整），
  默认拒绝登记，并提示检查网络/代理或使用离线包；只有显式 `--allow-degraded` 才登记
  `quality=derived` 的包。

## 3. CLI

```bash
python3 scripts/resource.py fetch envpack:taiwan_strait --mode auto --timeout 20
python3 scripts/resource.py fetch envpack:taiwan_strait --allow-degraded
python3 scripts/resource.py provide /path/to/envpack --entry-id envpack:manual
```

## 4. 运行时行为与可追溯

- 自动刷新只发生一次/每次 runner 调用；成功/失败都写入 `run.json.config.resource_fetch_error`
  与等待摘要 `summary.fetch_error`。
- 选中条目（含新版本）写入 `resources.json` 与 `run.json.resources`，`trajectory.provenance`
  记录 `resource_plan_hash`/`resources_status`。
- 离线回退与 P2 相同：缓存条目按 `as_of` 选择；过期由用户决策。

## 5. 网络现状（server155）

- `probe_online` 使用一条极小 Overpass 查询（POST，3.5 s）而不是 TCP 连接：
  TCP/TLS 能连通不代表请求可用（egress 代理会挂起）。
- 实测：探针 4.0 s 返回 False；真实 Overpass 查询 61 s 后超时。因此 auto 模式在本机
  不会发起完整重建，会在 4 s 内判定离线并进入 waiting，附带 “网络探测失败” 摘要。
- Copernicus/GEBCO 的直连可达性沿用既有 `geodata` 结论；若 Overpass 经代理可用，
  重建路径无需改代码。

## 6. 已验证

- `tests/test_resource_fetch.py` 5 项：
  - 注入 OSM/DEM 源成功重建 → 新版本登记、旧条目保留、as_of=重建日期、quality=measured；
  - OSM 不完整 → 拒绝登记（质量门），注册表条目数不变；
  - `allow_degraded` → 登记 derived；
  - `provide` 标记 `manual_override`；
  - 探针失败路径。
- 全量 484 passed / 14 skipped。
- CLI 离线模式：`fetch envpack:taiwan_strait --mode offline` 在数秒内被质量门拒绝，
  注册表保持原条目。

## 7. 未覆盖

- 真实在线的端到端重建（当前网络不可达；需要代理可用后复测）。
- case_roads/情报包的在线刷新；动态天气源。
- 运行中刷新（资源在开跑前钉死）。
