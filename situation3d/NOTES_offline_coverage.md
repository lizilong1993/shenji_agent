# 离线瓦片覆盖：两个战区

## 现状（2026-10-08 实测）

`web/tiles/` 是**生成产物、不进 git**（`.gitignore` 第 18 行）。当前规模约 3.0 GB：

| 目录 | 体积 | 内容 |
|---|---|---|
| `imagery/` | 782 MB | Esri World Imagery，z0–17 |
| `terrain/` | 2.2 GB | Terrarium 高程，z0–14 |
| `tdt/` | 20 MB | 天地图代理缓存 |
| `wsf3d/` | 39 MB | DLR 建筑高度，z13 |
| `gh3d/` | 6.0 MB | Google 3D 高度 |
| `places/` | 1.9 MB | Natural Earth 离线地名 |

## 覆盖模型：**只有一个 `region_rectangle`**

`manifest.coverage.imagery` 里是 `global_rectangle`（z0–6，补过之后等于全球）
+ `region_rectangle`（z7+，**只有一个矩形**）+ `per_zoom` 逐层统计。该 manifest 由
`tools/analyze_tile_coverage.py` **扫描实际瓦片**生成。

理解这一点很重要：战区层是按**当前案例的 theatre bbox** 建 provider 的
（实测层 1 的 `rect = [77, 31.5, 81.5, 36.5]` 正是中印），所以战区层本身不写死范围；
但 manifest 里那份 `region_rectangle` 是台海那批抓取留下的，**不描述中印**。
也就是说：**加了第二个战区之后，manifest 的单矩形描述已经不准**。

## 本轮补的缺口：中印战区此前**完全没有离线覆盖**

补之前，中印案例（拉达克）一张本地瓦片都没有，底图完全依赖天地图 —— key 失效或断网时
那个案例就是空白。而且战区层与 7 个热点层**已经按中印 bbox 在请求**，只是每次都 404。

### 第一段：z0–11（战区级）

```
python3 tools/fetch_assets.py imagery --bbox 77.0,31.5,81.5,36.5 \
    --min-zoom 0 --max-zoom 11 --source esri --out web/tiles --workers 8 --no-manifest
python3 tools/fetch_assets.py terrain --bbox 77.0,31.5,81.5,36.5 \
    --min-zoom 0 --max-zoom 11 --out web/tiles --workers 8 --no-manifest
```

干跑与实际都是 **1272 张影像 + 1272 张地形**；逐张核对**应有 1272、缺 0**。
实测新增影像 268 张、地形 323 张（其余因 z0–5 是全球层、与既有瓦片重合而跳过）。

**功能验证**：天地图**关掉**、只留离线底图，逐级压低相机：

```
高度 150 km   请求 71  失败 0  maxLevel 9   层级 {1:5,2:3,3:3,4:3,5:10,6:10,7:6,8:6,9:25}
高度  30 km   请求 13  失败 0  maxLevel 11  层级 {10:4,11:9}
```

即中印的 z9–11 **确实由本地瓦片供应**，零失败。

### 第二段：z12（7 个热点）

层清单显示中印 7 个热点层的 `maxLevel` 是 **12**，只抓到 z11 的话选中热点压近会 404。
按热点并集 bbox 补 z12：

```
python3 tools/fetch_assets.py imagery --bbox 77.4,32.5,79.7,36.5 \
    --min-zoom 12 --max-zoom 12 --source esri --out web/tiles --workers 8 --no-manifest
python3 tools/fetch_assets.py terrain --bbox 77.4,32.5,79.7,36.5 \
    --min-zoom 12 --max-zoom 12 --out web/tiles --workers 8 --no-manifest
```

* 干跑 1512 + 1512；影像 **1311 新增 / 201 已存在 / 0 失败**，地形 **1512 全部已存在**
* 逐张核对：影像与地形 **1512 应有、缺 0**
* 体积：影像 731 → **782 MB**（+51 MB），地形不变

**功能验证的诚实口径**：把相机压到 12 km 与 **5 km**、只留离线底图 —— 两次都
**零失败**，但 `maxLevel` 都停在 **11**，**我没有演示出一次 z12 请求**。
所以准确说法是：
* 「z12 瓦片**在盘上齐全**」—— 逐张核对过，确定；
* 「z12 请求**不会 404**」—— 只证到"没观察到失败"，**没有正向证到**。
（可能是这个视口按屏幕误差只需 z11，也可能是 hotspot provider 的请求没进
`imageryStats` —— 后者是这套计数的已知边界，见下。）

## ⚠ 已知不一致：manifest 只描述一个区域

`manifest.json` 只有一个 `bbox` 与一个 `region_rectangle`，两次抓取都用 `--no-manifest`
（**故意不写**，否则会把台海那份描述覆盖成中印的）。后果：manifest 仍只描述台海，
而部署已覆盖两个战区。运行时它被用来
（a）拼一句状态文案、（b）给 provider 声明覆盖矩形 ——
其中 (b) 走的是 `global_rectangle`（全球），所以**没有功能影响**；
**但描述确实已经不准**。

**要么**把 manifest 改成支持多区域（`regions: {taiwan_strait: {...}, india_china: {...}}`），
**要么**明确它就是"主抓取批次的记录"。前者更诚实，但改它要动
`tools/analyze_tile_coverage.py` 与 `assets.js` 的读取点，本轮**没有做**。

## 仍未覆盖

中印的**城区高清**（台海那一档 imagery z13–17，其中 z15–17 约 444 MB）没有抓。
台海 terrain z12 单层就有 936 MB，做全对等大约再加 **1 GB 量级**。
需要时按需补，命令同上把 `--max-zoom` 提到 14（影像）/12（地形）——
**先跑 `--dry-run` 拿准确瓦片数再决定**，本文两段就是这么做的。