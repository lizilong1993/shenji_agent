# 多数据源并行获取与自动回退（P6）

> 2026-10-26。目标：资源获取不绑定单一提供方或单一方式；并行探测、按质量/速度排序、
> 失败自动切换；同时支持“逐瓦片下载”和“下载完整栅格后本地切片”；统一缓存与资源登记。
> 代码新增：`resources/sources.py`、`resources/acquire.py`、`resources/processing.py`、
> `resources/tilemath.py`；CLI 新增 `tiles` / `probe-tiles` / `raster-tiles`。

## 1. 现状分析（改造前）

- 矢量/环境：`geodata`（Overpass/PBF、Terrarium/Copernicus、GEBCO）已有 online/offline/auto
  缓存，但 Overpass 端点串行尝试、无速度测量；envpack 刷新一次只用一种方法。
- 瓦片：`tools/fetch_assets.py` 内置 3 个影像源（Esri/EOX/GIBS）和 Terrarium/WSF3D，但由 CLI
  参数**手工选源**；逐瓦片多线程下载，无探测、无测速、无自动回退；瓦片集不进资源注册表。
- 服务端：天地图瓦片代理有本地缓存，离线 mbtiles/XYZ 直发；与资源层无统一登记。
- 处理：envpack 有归一化管线；瓦片只有“下载即用”，没有“完整栅格 → 本地切片”的通用路径。
- 缓存：`geodata.CacheStore` 以 kind/key 缓存原始响应；`web/tiles` 只靠文件存在跳过。

## 2. 统一获取架构

```text
TilesetSpec (name/bbox/zooms/layer)
   │
   ├─ sources.candidates_for_tiles()
   │     xyz:   esri / eox-s2 / gibs-modis / terrarium
   │     raster: 任意 GeoTIFF/COG URL（默认 COG 窗口裁剪，失败回退整文件下载）
   │     file:  本地已有瓦片集
   │
   ├─ acquire.probe_parallel()  —— 并行探测（8 线程）
   │     可用性 / 延迟 / 采样吞吐 / 内容 magic / 质量标注
   │
   ├─ rank: (quality measured>derived>default>unknown, 采样速度降序, 优先级, source_id)
   │
   ├─ 依次尝试（失败/过慢/质量不合格 → 下一个）
   │     xyz    → 多线程逐瓦片下载，404 记 absent，失败率超阈值即切源
   │     raster → COG 窗口裁剪（/vsicurl range）或整文件下载 → gdalwarp 裁剪
   │              → gdaldem hillshade（单波段/terrain）→ gdal2tiles → TMS→XYZ y 翻转
   │     file   → 复制本地瓦片集
   │
   └─ 质量检查 → manifest.json（来源/尝试/校验/统计）→ 原子安装到
      var/resources/cache/tiles/<name>/<spec_hash>/ → 注册 tile_layer 资源
```

缓存复用：同一 `name + spec_hash` 且条目校验通过时直接返回（不再探测/下载）；
`--mode online` 强制刷新，`--force` 同理；`--mode offline` 只用缓存。
所有尝试（含探测失败/低于 min_bps/批量失败）写入 manifest 与资源 notes，可追溯。

## 3. 质量与回退规则

- 探测：XYZ 取 bbox 中心瓦片（z=min(max_zoom,10)）验证 HTTP/图片 magic 并测速；栅格取
  Range 前 256 KB 验证 TIFF magic 并测速；本地路径直接可用。
- 批量下载：单瓦片 30 s 超时；404 视为“该处无瓦片”；HTTP 5xx/坏内容计入失败；
  失败数超过 `max(1, 10%)` 或没有任何可用瓦片 → 切换下一源。
- 栅格：先尝试 COG 窗口（`gdalwarp -te_srs EPSG:4326 -te ...` 经 `/vsicurl`），
  不是 COG 时自动回退整文件下载（`download_file`，支持 `min_bps` 慢速中止）。
- 瓦片统计：数量、字节、非空率（≥128 B 且图片 magic）；非空率 <1% 判质量不合格。
- 回退发生在三个层次：探测不可用 / 批量下载失败 / 质量检查失败。

## 4. 实测（server155，经本机 Clash 反向隧道；2026-10-26）

| 场景 | 结果 |
|---|---|
| `probe-tiles` Esri/EOX/GIBS（并行） | 三者均可用；Esri 约 11 KB/s、EOX 3.2 KB/s、GIBS 1.5 KB/s，Esri 排第 1 |
| XYZ 影像 z8-10（4 瓦片） | 自动选 Esri，3.0 s 完成（32 KB，4/4 非空） |
| 多源回退（坏 raster + Esri/EOX/GIBS） | 坏 raster 探测失败、GIBS 探测失败；EOX 胜出并完成，耗时 12.7 s；尝试记录完整 |
| 缓存复用 | 第二次同规格获取 0.21 s，无网络请求 |
| Terrarium 逐瓦片 z8-11（6 瓦片） | 135.5 s（该时段直连很慢），证明逐瓦片路径对逐请求延迟敏感 |
| Copernicus DEM 整文件下载 + hillshade + 切片 | 25.8 MB 源文件，49.9 s 完成（6 瓦片，derived） |
| Copernicus DEM COG 窗口 + hillshade + 切片 | **14.7 s**，同样 6 瓦片；比整文件路径快约 3.4 倍 |

结论：小 AOI 且瓦片源可用时 XYZ 最快；瓦片服务慢/限流或需要栅格专有数据时，COG/整栅格
+ 本地切片更稳；COG 窗口进一步省掉整文件下载，是综合最优路径。

## 5. 测试

`tests/test_acquisition.py`（7 项）：
- 并行探测排序（fast > slow，broken 淘汰）；
- 探测成功但批量下载失败 → 自动回退到第二源（attempts 记录 flaky=failed/fast=ok）；
- 缓存复用（关掉 HTTP 服务后仍命中）；
- offline 无缓存报错；
- manifest/注册表/内容校验；
- 合成 GeoTIFF RGB → 本地 gdal2tiles 切片；
- 单波段高程 → hillshade 切片（quality=derived）。

全量回归：见提交时的 `pytest tests` 结果（新增 7 项）。

## 6. 限制与后续

- 速度探测只取一个样本；同一源不同瓦片延迟波动大时，批量阶段仍可能切换（已按失败率/质量兜底）。
- 默认按“质量相同则速度优先”排序，因此更快但许可更严格的源可能被选中；用 `--sources`
  显式限定许可可接受的源。
- 瓦片集目前写入资源缓存并注册 `tile_layer`，尚未自动挂到 web/tiles 或前端图层；
  发布到演示仍需单独步骤（UI/服务端路径）。
- COG 路径依赖 GDAL（`gdalwarp/gdal2tiles/gdaldem`）；缺失时自动回退整文件下载或报错。
- 参考并借鉴了 VoxCity（MIT）的 COG `/vsicurl` 窗口裁剪思路；未复制其代码。
