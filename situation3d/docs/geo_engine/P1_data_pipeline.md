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

1. **OSM 在线可达性**：服务器直连 Overpass/Geofabrik 仍被阻断（Overpass 经 egress 代理返回 406，Geofabrik
   超时）；需要本机 Clash/SSH 反向隧道，或改走本地 `.osm.pbf`（已实现，见 §6）。
2. **DEM**：默认仍为 Terrarium z10（约 150 m/像素）；Copernicus GLO-30 已接入（见 §6），用
   `--dem-source copernicus` 切换。GEBCO 水深已接入，ENC/港口测深仍未实现。
3. **几何**：建筑/水域多边形为 OSM 原始 ring，未做拓扑修复/简化；桥隧仅记录标签，通行谓词暂未使用。
4. **环境包格式**：`msgeo.envpack.v2` 已在 JSON 图层基础上增加 5 km 分块文件（landcover/water/buildings
   的 `chunks/` + manifest 索引），运行时兼容 v1/v2 但仍整包加载，按需加载分块尚未实现；仍不是
   COG/FlatGeobuf，接口已按 manifest 抽象。
5. **建筑遮挡 LOS** 已改为线段-多边形相交 + 建筑顶高程比较（不再点采样）；超长雷达线按 bbox 预筛。
6. **缓存损坏**：JSON/PNG 损坏会隔离为 `.corrupt` 并按源失败回退，不会让构建崩溃。

## 6. 增量：PBF / GLO-30 / GEBCO / envpack v2（2026-10-21）

新增数据源与模块：

| 源 | 模块 | 访问方式 | 说明 |
|---|---|---|---|
| Geofabrik `.osm.pbf` | `geodata/osm_pbf.py` | 本地文件（需 pyosmium） | ways 转 Overpass `out geom` 形状，复用 `normalize`；原始结果缓存到 `osm_pbf` |
| Copernicus GLO-30 | `geodata/dem_copernicus.py` | AWS S3 COG + `gdal_translate /vsicurl` | 30 m 1″ 瓦片窗口转 AAIGrid 缓存；纯 Python 双线性采样 |
| GEBCO 2026 15″ | `geodata/bathymetry.py` | `download.gebco.net` queue API | 提交 bbox→GeoTIFF→AAIGrid；`water.depth.json` 栅格 |
| 建筑高度回退 | `geodata/height_fallback.py` | 本地多边形 GeoJSON（WGS84/3857 自适应） | 用 `height`/`h` 填充无 OSM 高度的建筑，>150 m 视为尖刺 |

验证（Anping bbox 120.14/22.97/120.19/23.02，300 m 网格，`--osm-pbf --dem-source copernicus --height-geojson <GBA>`）：

- PBF 导入 6,877 条 way（highway 2,941、building 2,665、natural 927、landuse 394、waterway 66），全量扫描 91 s；
  道路图 **10,257 节点 / 12,576 边**，与 Overpass 包完全一致；最大连通分量 **96.8%**（≥95% 达标）。
- 地形 20×20、水深 20×20，均无空值；GEBCO 水深范围 -23.0～+11.6 m；海上取样点由默认 20 m 改为
  GEBCO 实测（如 10.8 m）。
- 建筑高度：OSM 标签 120（height）+ 252（levels），GBA 回退填充 376，合计真实高度覆盖 **28.1%**
  （目标 80% 未达标：Overture 本区仅 5.9% 有 height/num_floors；tw_lod1 GBA 瓦片质心匹配率 14%，存在偏移）。
- envpack **v2**：manifest `msgeo.envpack.v2`，landcover/water/buildings 写入 5 km 分块（`chunks/`）并带索引；
  `EnvPack.load` 与 `EnvPackEnvironment` 兼容 v1/v2。
- 在线/离线回放：六类图层 sha256 全部一致，所有分块文件逐字节一致。

环境注意：pyosmium 装在 `/remote-home/lizilong/jev_gpu/venv`（py310，需
`LD_LIBRARY_PATH=/remote-home/lizilong/py310-libs`）；GEBCO 的 urllib 请求需
`SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())')`；Geofabrik 经本机 Clash
（`172.16.10.254:7897`）下载。运行时（8848、演示脚本）不依赖上述工具链，仍只用系统 python。
