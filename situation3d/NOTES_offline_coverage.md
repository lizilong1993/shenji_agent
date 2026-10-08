Warning: Permanently added 'server-10-254-1-155' (ED25519) to the list of known hosts.
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

## ⚠ 订正：manifest **本来就支持多战区**，而且早就声明了中印覆盖

我在本文早先的版本里写过"manifest 只有一个 bbox、不支持多区域、`region_rectangle`
只描述台海"—— **那是错的**。查实后：

`manifest.theatres` 是一张**按战区索引**的表，里面**两个战区都在**：

```json
"taiwan_strait": { "bbox": [115.5,20.0,122.5,26.5],
                   "coverage": { "imagery": {"min_zoom":0,"max_zoom":11},
                                 "terrain": {"min_zoom":0,"max_zoom":10}, ... } },
"india_china":   { "bbox": [77.0,31.5,81.5,36.5],
                   "coverage": { "imagery": {"min_zoom":0,"max_zoom":11},
                                 "terrain": {"min_zoom":0,"max_zoom":10}, ... } }
```

`assets.js` 的 `theatreList()` **优先读 `m.theatres`**（manifest 赢），只有在拿不到时才
退回代码里写死的 `THEATRE_BBOX`。所以"烘焙一个战区就能让前端生效、不必改代码"这条
通路**本来就是通的**。

**这把本轮那个缺口说明得更准确、也更严重**：

> manifest **早就声明** `india_china` 有 `imagery z0-11` / `terrain z0-10` 覆盖，
> **而瓦片实际不存在** —— 每次请求都 404。

也就是说这不是"少烘焙了一个战区"，而是**声明与事实不符**：元数据说覆盖到了，
盘上什么都没有。所以本轮的抓取不是"新增覆盖"，而是**把已经声明的东西补成真的**。

**`--no-manifest` 这个选择仍然是对的**，但理由要改：不是为了"避免覆盖掉台海"，
而是因为本次抓的是**战区级 z0-11 + 热点 z12**，而 `analyze_tile_coverage.py` 生成的
manifest 还会重算 `coverage`/`tile_counts`/`per_zoom` —— 手工抓取后重跑它才该更新 manifest，
手动拼容易写出半真半假的数。**要更新就重跑那个工具，不要手改。**

### 仍然对不上的两处（都无害，但记下来）

* `theatres.india_china.coverage.imagery.max_zoom = 11`，而中印 **7 个热点层要 z12**
  （实测层清单 `maxLevel: 12`）。manifest 声明的是**战区级**层级，热点层级另有一套；
  本轮已把 z12 瓦片补齐，所以事实层面没问题，**只是 manifest 的这个字段不描述热点**。
* `terrain max_zoom = 10`，而本轮实际抓到了 **z11 与 z12** —— 多抓了一层，无害。

## 仍未覆盖

中印的**城区高清**（台海那一档 imagery z13–17，其中 z15–17 约 444 MB）没有抓。
台海 terrain z12 单层就有 936 MB，做全对等大约再加 **1 GB 量级**。
需要时按需补，命令同上把 `--max-zoom` 提到 14（影像）/12（地形）——
**先跑 `--dry-run` 拿准确瓦片数再决定**，本文两段就是这么做的。

## 声明 vs 实际：全量核对（2026-10-08，已通过）

把 `manifest.theatres` 里**每一条声明**逐张对照盘上文件（按 Web Mercator 算出该 bbox
该有哪些瓦片，再逐个 `os.path.isfile`）：

```
taiwan_strait  imagery  z0-11  应有  2259  缺 0  OK
taiwan_strait  terrain  z0-10  应有   619  缺 0  OK
taiwan_strait  wsf3d    z0-10  应有   619  缺 0  OK
india_china    imagery  z0-11  应有  1272  缺 0  OK
india_china    terrain  z0-10  应有   336  缺 0  OK
india_china    wsf3d    z0-10  应有   336  缺 0  OK
```

**这一类的缺口至此封闭**：本轮之前 `india_china` 的三项**全部声明了而一张都没有**
（见上文），现已补齐并逐张核对。

两点说明：
* 台海 z10–11 盘上的瓦片数**多于**这里的 2259 —— 多出来的是热点/城区批次抓的，
  战区级声明仍然满足。
* 中印多抓了一层（`terrain` 声明 z0-10，实际抓到 z11；`imagery` 另加热点 z12），
  属**多给不少给**，无害。

复查用的脚本逻辑很短（Mercator 求 x/y 范围 + `os.path.isfile`），值得在每次改
`manifest` 或重抓之后跑一遍 —— **声明与事实不符这类问题不会报错，只会静默 404。**