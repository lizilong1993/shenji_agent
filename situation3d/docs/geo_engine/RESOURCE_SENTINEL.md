# Sentinel-2 L2A 影像资源（2026-10-27）

## 数据路径

- STAC：Earth Search AWS（`https://earth-search.aws.element84.com/v1/search`），collection
  `sentinel-2-l2a`；按 AOI bbox + 想定日期向前 `window_days` 查询，读 `eo:cloud_cover`。
- 资产：AWS Open Data `sentinel-cogs` 的 `TCI.tif`（真彩色 COG，10 m）；GDAL
  `/vsicurl` 只读 AOI 窗口，不下载整景。
- 选择：优先 `cloud <= max_cloud`，再按云量升序、日期距目标最近排序；没有满足云量时
  选云量最低的场景并标记 `quality=derived`。
- 大 AOI：`--mosaic` 按 `grid:code`（MGRS 瓦片）分组，每组选云量最低/日期最近的场景，
  用 GDAL 把多景 TCI 窗口镶嵌到同一 bbox；源清单写入 `scene_ids`。
- 登记：`satellite_imagery:<name>`，manifest `s3d.raster.v1`，记录 scene id / datetime /
  cloud / query；许可为 Copernicus Sentinel data terms。

## 用法

```bash
python3 scripts/resource.py sentinel --name xiamen-s2 \
  --bbox 118.0,24.4,118.02,24.42 \
  --date 2025-12-31 --window-days 365 --max-cloud 30 --resolution 10
```

生成地图瓦片（Sentinel 场景 → XYZ，失败自动回退 EOX cloudless/Esri）：

```bash
python3 scripts/resource.py sentinel-tiles --name xiamen-s2 \
  --bbox 118.0,24.4,118.02,24.42 --zooms 10-13 \
  --date 2025-12-31 --fallback-sources eox-s2,esri
```

需要代理访问 Sentinel S3 时（server155 当前环境）：

```bash
ssh -f -N -R 27980:172.16.10.254:7897 situ155
export https_proxy=http://127.0.0.1:27980 HTTPS_PROXY=http://127.0.0.1:27980
export GDAL_HTTP_PROXY=http://127.0.0.1:27980
```

## 实测（2026-10-27，server155 新节点）

- STAC 查询厦门 AOI：返回 `S2B_50QNM_20251208`，云量约 0.0038；日期/云量选择正常。
- 经本机 Clash 代理：TCI COG 窗口抓取 + 登记 **8.1 s**，223×223、3 波段、约 151 KB。
- 复用缓存场景生成瓦片：`sentinel-tiles` 0.8 s，z10-12 共 3 张、12.6 KB，tile 源为 `sentinel2`。
- 中印边界 0.8°×0.8° `--mosaic`：4 个 MGRS 场景镶嵌，2969×2969、26 MB，约 136 s（经代理）；
  适合 400 km 战区尺度，不需逐景手工拼。
- 无代理直连 `sentinel-cogs` S3 时多次出现 `TIFFFillTile` 部分读取错误（got < expected），
  说明当前 egress 到该 S3 不稳定；生产应配置代理或预取缓存。

## 回退与限制

- `sentinel-tiles` 已内置兜底：Sentinel 获取/切片失败时自动改用 EOX Sentinel-2 cloudless 2020
  或 Esri XYZ 源；也可单独走 `resource.py tiles --sources eox-s2`。
- 单场景：当前不跨景镶嵌；AOI 跨两景时可能只覆盖部分，需要后续加 mosaic。
- 缓存键包含 `bbox/date/window_days/max_cloud/resolution`（`DatasetSpec.query`），不同日期
  不会误复用；`--force` 可刷新。
- Sentinel-1 SAR、Planetary Computer/CDSE OAuth 源尚未接入；可复用同一 STAC/COG 框架。
