# 离线瓦片覆盖：两个战区

## 现状（2026-10-08 实测）

`web/tiles/` 是**生成产物、不进 git**（`.gitignore` 第 18 行）。当前规模约 2.9 GB：

| 目录 | 体积 | 内容 |
|---|---|---|
| `imagery/` | 731 MB | Esri World Imagery，z0–17 |
| `terrain/` | 2.2 GB | Terrarium 高程，z0–14 |
| `tdt/` | 20 MB | 天地图代理缓存 |
| `wsf3d/` | 39 MB | DLR 建筑高度，z13 |
| `gh3d/` | 6.0 MB | Google 3D 高度 |
| `places/` | 1.9 MB | Natural Earth 离线地名 |

## 本轮补的缺口：中印战区此前**完全没有离线覆盖**

`manifest.json` 的 `bbox` 是 `[115.5, 20.0, 122.5, 26.5]` —— 就是台海战区。
中印案例（拉达克）没有任何本地瓦片，**底图完全依赖天地图**；天地图 key 失效或断网时，
那个案例就是空白。

已补 z0–11：

```
python3 tools/fetch_assets.py imagery --bbox 77.0,31.5,81.5,36.5 \
    --min-zoom 0 --max-zoom 11 --source esri --out web/tiles --workers 8 --no-manifest
python3 tools/fetch_assets.py terrain --bbox 77.0,31.5,81.5,36.5 \
    --min-zoom 0 --max-zoom 11 --out web/tiles --workers 8 --no-manifest
```

* 干跑与实际都是 **1272 张影像 + 1272 张地形**
* **逐张核对：应有 1272，缺 0**（影像与地形都是）
* 实测新增：影像 268 张、地形 323 张（其余因 z0–5 是全球层、与既有瓦片重合而跳过）
* 体积增加约 **64 MB**（717→731 MB / 2.1→2.2 GB）

### 功能验证（不是只看文件在不在）

把天地图**关掉**、只留离线底图，再压到战区中心 —— 此时网络上什么都取不到：

```
案例=india_china_west   天地图 visible=False
瓦片请求=52  失败=0  minLevel=1  maxLevel=8
层级直方图={"1":5,"2":3,"3":3,"4":3,"5":10,"6":10,"7":6,"8":12}
tilesLoaded=true
```

即中印案例现在**有可用的离线底图**了。

## ⚠ 已知不一致：manifest 只描述一个世纪

`manifest.json` 只有**一个** `bbox` 字段，而它这次是用 `--no-manifest` 抓的 ——
**故意不写**，否则会把台海的 bbox 覆盖成中印的，把台海那份描述弄坏。

后果：manifest 仍写着台海 bbox，而部署里其实覆盖了两个战区。
运行时它只被用来拼一句状态文案（`离线瓦片 <bbox>`），**没有功能影响**，但描述已经不准了。

**要么**把 manifest 改成支持多区域（`regions: {taiwan_strait: {...}, india_china: {...}}`），
**要么**明确它就是"主抓取批次的记录"。前者更诚实，但改它要动 `OfflineAssets` 的读取点。

## 尚未覆盖的部分

中印的**热点/城区高清**（台海那一档是 imagery z12–17 + terrain z12）**没有抓**。
台海这一档本身就占大头：影像 z15–17 约 444 MB、地形 z12 约 936 MB。
按面积比（中印/台海 = 0.56）**做全对等大约再加 1.4–1.7 GB**。

需要的话按需补，命令同上把 `--max-zoom` 提到 14（影像）/12（地形）即可。
`--dry-run` 可以先出准确瓦片数再决定 —— 本次就是这么先量后抓的。