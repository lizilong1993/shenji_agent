# 正式数据集适配器：GBA / Overture / CHMv2（P7）

> 2026-10-26。项目范围是全中国，台湾与中印边界只是演示案例；本阶段把 VoxCity 调研中
> 有价值的公开数据源做成**本系统正式资源适配器**，而不是把 VoxCity 作为运行时依赖。

## 1. 新增资源类型

| kind | 内容 | 默认级别 | 来源 |
|---|---|---|---|
| `building_footprints` | GeoJSON 建筑轮廓 + `height_m`/`height_var_m`/`region` | advisory | GBA（默认）、Overture |
| `canopy_height` | 树冠高度 GeoTIFF | advisory | Meta/WRI CHMv2 |
| `landcover` | 地类栅格（ESA WorldCover 2021 v200，10 m） | advisory | ESA WorldCover |

统一输出：

```text
cache/datasets/<kind>/<name>/<spec_hash>/
    features.geojson  |  canopy_height.tif
    manifest.json         # s3d.vector.v1 / s3d.raster.v1
```

登记进 `var/resources/index.json`，带 content_hash/bbox/as_of/license/quality/来源；
resolver 与现有 envpack/瓦片共享在线-离线/时效/门控机制。

## 2. 适配器

- **GBA（Global Building Atlas LOD1）**：5° GeoParquet 瓦片（`data.source.coop`），
  命名 `e115_n25_e120_n20.parquet`；pyarrow 按 bbox 结构过滤，shapely WKB 解析，
  按 AOI 裁剪并标准化。单瓦片 100–180 MB，已验证直连约 5–17 MB/s。
- **Overture**：通过可选 `overturemaps` 包（STAC + S3）查询 `building`，geometry WKB
  与 height/num_floors/class 标准化；本机实测查询在 150 s 内未返回（S3/STAC 慢），
  因此 GBA 为默认，Overture 作为备选适配器保留。
- **CHMv2（Meta/WRI 1 m 树冠）**：zoom-10 quadkey COG，GDAL `/vsicurl` 窗口裁剪
  （沿用 P6 的 COG 路径），输出 Web Mercator GeoTIFF，CC BY 4.0。
- **ESA WorldCover 2021 v200（10 m 地类）**：3° COG 瓦片，GDAL 窗口裁剪；2026-10-26
  厦门 0.02° 实测 7.1 s / 61 KB（240×240），CC BY 4.0。

CLI：

```bash
resource.py buildings --name xiamen-buildings --bbox 118.0,24.4,118.1,24.5 --sources gba,overture
resource.py canopy    --name xiamen-canopy    --bbox 118.0,24.40,118.02,24.42 --resolution 10
resource.py enrich-envpack --envpack envpack:taiwan_strait \
  --buildings building_footprints:tw-anping-buildings --name tw2-gba
```

可选依赖（pyarrow/shapely/overturemaps）只在适配器内部导入；在 py310 venv 中运行，
系统 python3.8 缺少 pyarrow 时给出明确错误。GBA/ CHMv2 不需要账号；GBA 许可需按数据集页确认。

## 3. 与引擎集成（不是只“下载”）

`resources/enrichment.py` + `resource.py enrich-envpack` 实现 VoxCity 的“按重叠面积补全
建筑高度”思路：

- 读取 envpack `buildings.json`，只处理 `height_source=default` 的建筑；
- 用 GBA/Overture 建筑多边形建立 STRtree，按代表点/重叠面积匹配，写入 `height_m`，
  `height_source=gbd_overlap`；
- 复制 envpack 为新版本，保留原包与历史 run；新版本 manifest 记录
  `provenance.building_heights`（来源/匹配数/填充数）；
- 新版本登记为独立 `envpack:<name>`，resolver 按 `as_of` 选择。

树冠资源目前作为独立 `canopy_height` 资源登记，尚未接入 envpack/感知层；可作为后续
城市体素/植被遮蔽/微气候约束的输入，不需要引入 VoxCity 运行时。

## 4. 实测（server155，2026-10-26）

| 数据/流程 | 结果 |
|---|---|
| GBA 瓦片 e115_n25_e120_n20 | 181.8 MB，下载 10.6 s（约 17 MB/s），2,351,272 行 |
| 厦门 AOI 0.1° 建筑 | 6,365 栋，height coverage 100%，均值 13.63 m；适配器+登记 29.5 s |
| 台湾 Anping AOI 建筑 | 23,260 栋，height coverage 100%，均值 9.75 m；5.6 s |
| CHMv2 厦门 0.02° | 10.3 s，1864×2047 单波段 GeoTIFF，3.8 MB，CC BY 4.0 |
| WorldCover 厦门 0.02° | 7.1 s，240×240 单波段 GeoTIFF，61 KB，CC BY 4.0 |
| envpack 高度补全 | 14,590 栋 default 中填充 13,651 栋（93.6%），新版本 38.6 MB |
| 新 envpack 推演 | 20 tick completed，plan ready；selected `envpack:tw2-gba`；validation passed（building_hits=1） |
| Overture 实时查询 | 150 s 未返回，记录为可选源；不影响 GBA 默认路径 |

## 5. 测试

`tests/test_vector_adapters.py`（6 项）：spec/hash、统计、GeoJSON+注册、GBA 瓦片命名、
CHMv2 quadkey、建筑源回退（gba 失败→overture）、树冠 raster manifest/注册。
`tests/test_enrichment.py`（1 项）：default 高度补全、原包不变、新版本注册与校验。
全量回归见提交时 `pytest tests`。

## 6. 限制与后续

- GBA 5° 瓦片较大；按 AOI 只下载相交瓦片，但仍可能 100–180 MB；可将 GBA 转成按 AOI
  裁剪的 GeoParquet 缓存以进一步省流量（后续）。
- GBA 高度为 LOD1 估算值，9.6% 左右的 OSM 非默认高度与 GBA 会有差异；enrichment 只填
  `default`，不覆盖已有 `tag/levels` 高度。
- Overture 实时查询受网络影响；如需要可预下载 Overture release 到本地再由 pyarrow 读取。
- canopy/landcover 尚未进入引擎感知/遮蔽；目前作为独立资源登记，供后续城市体素/植被/微气候使用。
- 所有数据许可证逐层记录；GBA/Overture/CHMv2 的具体条款在部署对外使用前需复核。
