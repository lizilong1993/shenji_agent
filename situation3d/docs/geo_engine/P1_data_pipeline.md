# P1 数据管线：在线与离线双路径（2026-10-21）

> 目标：同一套环境包构建代码同时支持**在线抓取**与**离线缓存回放**；运行时只读本地环境包，不访问网络。

## 1. 架构

```
                 ┌─────────────── 在线 ───────────────┐
 Overpass OSM ──▶│ OsmSource  →  raw JSON 写入 Cache  │──┐
 Terrarium DEM ─▶│ DemSource  →  PNG     写入 Cache  │──┤
                 └────────────────────────────────────┘  │
                                                          ▼
                 ┌─────────────── 离线 ───────────────┐  normalization
 Cache（同一目录）│ OsmSource/DemSource 只读缓存       │──▶（OSM features / DEM grid）
                 └────────────────────────────────────┘  │
                                                          ▼
                                     envpack: manifest + terrain/landcover/water/buildings/roads
                                                          │
                                                          ▼
                                     env/pack_env.EnvPackEnvironment（运行时）
```

- `mode=online`：始终抓取并写缓存（用于更新数据）。
- `mode=offline`：只读缓存；缺失即 `OfflineCacheMiss`，不会偷偷联网。
- `mode=auto`：优先缓存，缺项才联网；任一源失败时按 manifest 记录回退到 `basic.json` 图层。

缓存键由 `(kind, bbox/瓦片坐标)` 的 SHA1 决定，在线与离线共用同一目录；同一缓存下两次构建的图层文件逐字节一致（manifest 的 `mode/provenance` 除外）。

## 2. 代码

| 文件 | 作用 |
|---|---|
| `geodata/cache.py` | 通用磁盘缓存与离线缺失异常 |
| `geodata/osm.py` | Overpass 查询（多镜像）、原始 JSON 缓存、OSM→道路/水域/建筑/土地利用归一化 |
| `geodata/dem.py` | Terrarium DEM 瓦片抓取/缓存、PNG 解码、规则栅格采样 |
| `geodata/pipeline.py` | 双路径共享的 envpack 构建：terrain grid、landcover/water/building polygons、road graph、manifest/provenance |
| `scripts/build_envpack.py` | CLI：`--mode online|offline|auto`、`--cache`、`--bbox`、`--resolution-m`、`--no-osm/--no-dem` |
| `env/pack_env.py` | 运行时 `EnvironmentService`：地形双线性、多边形点查询、建筑遮挡、道路吸附、缓存 LOS |
| `sim/scenario.py` / `scripts/run_geo_demo.py` | `build_world(..., envpack=...)`、`--envpack` 直接消费环境包 |

## 3. 使用

```bash
# 在线：抓取 DEM/OSM，写入共享缓存
python3 scripts/build_envpack.py --basic web/data/campaign_maps/map_9001/basic.json \
  --out envpacks/anping --theatre anping --mode online \
  --cache /tmp/geocache --bbox 120.14 22.97 120.19 23.02 --resolution-m 300

# 离线：同一缓存回放，不访问网络
python3 scripts/build_envpack.py --basic web/data/campaign_maps/map_9001/basic.json \
  --out envpacks/anping_offline --theatre anping --mode offline \
  --cache /tmp/geocache --bbox 120.14 22.97 120.19 23.02 --resolution-m 300

# 运行时消费环境包
python3 scripts/run_geo_demo.py --case taiwan_landing \
  --envpack envpacks/anping --ticks 300 --dt 30 --out /tmp/tw_envpack
```

## 4. 验证结果（2026-10-21，server155）

- **代理方式**：服务器直连外网被阻断；使用本机 Clash（`172.16.10.254:7897`）经 SSH 反向隧道
  （`ssh -R 27980:172.16.10.254:7897`）供 server155 使用，设置 `HTTPS_PROXY=http://127.0.0.1:27980` 后
  Overpass/Terrarium 均可达。在线构建与离线回放都在同一次缓存上完成。
- **在线构建（真实数据）**：Anping bbox（120.14/22.97/120.19/23.02），300 m 网格；产出道路 **10,257 节点 /
  12,576 边**、水域 **759** 个多边形、建筑 **2,665** 个、土地利用 **283** 块；`provenance.osm=online`、
  `dem=online`、`fallback=false`。
- **离线回放**：同一缓存 `mode=offline`，五类图层（terrain/landcover/water/buildings/roads）sha256 **全部相同**，
  `provenance` 为 `offline-cache`。
- **持久化**：真实环境包保存在服务器 `situation3d/envpacks/anping_demo`，共享缓存
  `situation3d/envpacks/geocache_anping`（`.gitignore` 已排除，可复现重建）。
- **运行时**：台海案例 33 单位、120 tick、dt=30 s 使用真实离线包，p50 60.9 ms / p95 95.3 ms（数据丰富包；
  基础 basic.json 环境同场景约 12 ms）。建筑遮挡已改为线段-多边形相交；多边形/道路加了 0.01°/0.02° 空间桶索引。
- 测试：`tests/test_geo_engine_core.py` 含双路径管线（注入 fake OSM/DEM、离线缺失、OSM 归一化、pack 运行时采样）。

## 5. 已知限制

1. **OSM 在线可达性**：服务器直连外网仍被阻断，需要本机 Clash/SSH 反向隧道；代码已支持多镜像与缓存，
   代理就绪时 `--mode online` 即可。Geofabrik 大范围下载仍建议 `.osm.pbf` + osmium 导入器（尚未实现）。
2. **DEM**：使用 Terrarium z10（约 150 m/像素）演示；未接 Copernicus GLO-30、GEBCO/ENC 水深。
3. **几何**：建筑/水域多边形为 OSM 原始 ring，未做拓扑修复/简化；桥隧仅记录标签，通行谓词暂未使用。
4. **环境包格式**：JSON 图层（grid/polygons/graph），不是 COG/FlatGeobuf；接口已按 manifest 抽象，后续可替换为
   GDAL/osmium/Shapely 产出的正式图层而不改运行时调用方。
5. **建筑遮挡 LOS** 已改为线段-多边形相交 + 建筑顶高程比较（不再点采样）；超长雷达线按 bbox 预筛。
6. **缓存损坏**：JSON/PNG 损坏会隔离为 `.corrupt` 并按源失败回退，不会让构建崩溃。
