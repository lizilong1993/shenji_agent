# situation3d — 三维战场态势生成系统

自研三维推演引擎 + CesiumJS 前端。面向真实 WGS84 地理与真实装备型号，覆盖
**海 / 陆 / 空**三域、**红 / 蓝**双方，具备**三级态势自动推理**与**军标可视化**，
并且**完全离线运行**。

```
engine/     推演内核：地理计算、军标目录、想定与航路、三级态势推理、演示案例
scripts/    生成前端所需的数据集（每案例一份）
web/        CesiumJS 前端（离线瓦片、军标、推理面板、案例菜单）
server/     零依赖静态服务（缓存策略 + gzip）
tools/      资产抓取与栅格化（fetch_*.py / build_gh3d_tiles.py）、色带标定、
            远程同步、无头取证与运行时读数（capture.py / inspect_remote.py）
```

---

## 1. 运行

```bash
# 生成全部演示案例数据
python3 situation3d/scripts/gen_demo_trajectory.py

# 启动静态服务（默认 8848）
sh situation3d/tools/restart_server.sh 8848
# 浏览器打开 http://<host>:8848/
```

前端深链参数（同时用于截图取证）：

| 参数 | 作用 |
|---|---|
| `?case=<id>` | 指定演示案例 |
| `?tick=300` | 首帧前定位到某帧 |
| `?autoplay=1` | 立即开始推演 |
| `?panel=R01` | 打开指定目标的信息面板 |
| `?hotspot=6` | 选中第 N 个热点区域 |
| `?terrain=1` `?overlay=1` `?buildings=1` `?sensors=1` | 打开对应模型开关 |
| `?cam=lon,lat,height[,pitch]` | 指定机位（诊断用） |
| `?diag=1` | 左上角叠加运行时读数 |
| `?cache=0` | 停用前端瓦片缓存（对照用） |
| `?components=0\|1` | 地图组件全关 / 全开 |
| `?components=a,b` / `?components=-a` | 地图组件白名单 / 排除 |
| `?compass=0\|1` `?scalebar=0\|1` `?viewpoint=0\|1` | 单项地图组件开关 |
| `?follow=<目标ID>` | 直接进入目标跟随 |
| `?overview=0\|1` | 鸟瞰图显示 / 隐藏 |
| `?places=0\|1` | 地名图层显示 / 隐藏 |
| `?situation=input\|manage\|download` | 直接打开态势面板的某个标签页 |
| `?unfollow=1` | 跟随一次后立即解除（验证参考系解锁） |

---

## 2. 演示案例

案例注册表在 `engine/cases.py`，新增案例只改一处，数据生成脚本自动跟进。

| id | 名称 | 覆盖 |
|---|---|---|
| `joint_full` | 联合态势演示（海陆空全要素） | 12 目标 / 10 热点 |
| `sea_convoy` | 海上编队穿越与护航 | 7 目标 / 5 热点 |
| `air_control` | 空中巡逻与预警指挥 | 6 目标 / 4 热点 |
| `coastal_defense` | 沿海要地防空反导 | 7 目标 / 5 热点 |

红方：052D 驱逐舰、054A 护卫舰、903 补给舰、鹰击-62 岸舰导弹营、红旗-9B
远程防空营、歼-16、空警-500。
蓝方：成功级护卫舰、沱江级巡逻舰、爱国者-3 防空连、F-16V、E-2K。

每个目标都带真实型号、军标图形键、传感器型号与探测距离、武器有效射程。

---

## 3. 三级态势推理（`engine/inference.py`）

| 级别 | 回答的问题 | 主要输出 |
|---|---|---|
| L1 目标级 | 这个平台在做什么、被威胁到什么程度 | 状态、意图、威胁度 0–100、告警等级、最近对方目标、被探测数量 |
| L2 编组级 | 红蓝双方编组之间是什么关系 | 编队形态、质心、跨度、与对方同域编组的最小距离、对抗关系 |
| L3 区域级 | 每个热点区域内部是什么态势 | 态势等级 1–4、双方兵力数、主导方 |

**威胁度口径**（三项相加，逐项可查）：

```
威胁度 = 装备权重(0–35) + 接近度(0–50) + 覆盖度(0–9)

接近度 = 50 · exp(-2 · d / R_射程)      R 取最近对方目标的武器有效射程
覆盖度 = Σ 3 · (1 - d / R_探测)          只计探测范围内的对方传感器
```

用**武器射程**而不是"影响半径"算接近度是关键：预警机的雷达看得远但没有武器，
否则一架远处的预警机会把全场威胁度顶到"警告"。这一版之前就踩过这个坑。

推理规则与阈值全部随数据集导出（`inference_model` 段），前端不复制一份口径。

---

## 4. 离线资产与 LOD

### 4.1 资产族与四级覆盖

影像与高程都不是"一个范围一个层级"，而是**分级覆盖**——每一级只在自己需要的范围
下载，否则要么全球没有底图，要么为了几平方公里的米级影像铺满整个战区。

| 级别 | 范围 | 影像 | 高程 | 影像分辨率 |
|---|---|---|---|---|
| 全球 | 全球 | z0–6 | z0–10 | 2.4 km → 0.7 km/px |
| 战区 | 115–125E, 18–29N | z0–11 | z11–12 | 150 m → 76 m/px |
| 热点区域 | 8 个热点框 | z0–12 | z11–12 | 约 19 m/px |
| 重点区域 | 4 个城区 AOI | z13–17 | z13–14 | 9.5 m → **1.2 m/px（米级）** |

| 族 | 目录 | 来源 |
|---|---|---|
| 影像底图 | `web/tiles/imagery/{z}/{x}/{y}.jpg` | Esri World Imagery |
| 高程 DEM | `web/tiles/terrain/{z}/{x}/{y}.png` | Mapzen / AWS（Terrarium 编码） |
| 建筑高度 | `web/tiles/wsf3d/buildingheight/...` | DLR WSF3D v02 |

**热点框做了包含去重**：平潭、澎湖落在「台湾海峡」框内，重复下载只浪费请求。
去重后是 8 个框——注意台北框的东边界 121.75 超出台湾海峡的 121.00、高雄框的南边界
22.45 低于 23.00，**两者其实都不被包含**（手算容易搞错，交给脚本判断）。

**高程不需要无限加深**：Terrarium 源是 30 m SRTM，z12 已经是 38 m/px，再往下只是
上采样、不增加信息。所以高程只补到 z12（战区与热点）与 z14（重点区域）。

覆盖范围与层级区间全部记在 `web/tiles/manifest.json`，前端 provider 的 `rectangle`
与 `maximumLevel` 直接取自这里。

```bash
python3 tools/fetch_base_imagery.py                # 全球 + 战区/热点高程 + 米级影像
python3 tools/fetch_base_imagery.py --only urban   # 只补重点区域的米级影像
```

`web/tiles/manifest.json` 记录**真实覆盖范围与层级区间**，前端 provider 的
`rectangle` 直接取自这里。

### 4.2 LOD 的关键一条：provider 必须声明真实覆盖范围

底图只下载了台海 AOI。如果 provider 仍声明覆盖全球，Cesium 会为域外瓦片持续
发请求、持续 404，并因此**卡在 z0–z1 不再细化**——实测一次加载发出 500 次重试、
整幅底图退化成一片渐变。把 `rectangle` 设成 manifest 里的真实范围后，请求全部
命中已下载瓦片，层级正常推进到 z7。

### 4.3 其余 LOD 手段

- **高程**：分级覆盖下"请求层级取不到"是常态（重点区域有 z14，别处只有 z10）。
  如果取不到就返回平地，在覆盖范围之外放大时地形会**塌成平面**，反而比粗层级更差。
  所以 `heightmapFor` 从目标层级**逐级上溯**，用最近的有数据的祖先瓦片降采样——
  这正是地形金字塔本身的行为。
- **城区高清影像**：相机高于 200 km 时不挂载 z13–17 图层，否则每个表面瓦片都
  会去要一批用不上的瓦片；低于阈值再挂上，飞高时摘掉。
- **影像四层叠加**：全球 z0–6 → 战区 z0–11 → 热点 z0–12 → 城区 z13–17，按"粗 → 细"
  依次 `addImageryProvider`（Cesium 的 index 0 在最底层）。热点层每个框一个 provider
  并带自己的 `rectangle`，所以只会请求框内的瓦片；与战区层重叠的部分因为 URL 相同
  会命中 `TileCache` 的内存缓存，代价很小。
- **建筑体块**：按相机高度分 4 档（粗网格/细网格、采样步长 8/4/2/1），已生成的
  档位缓存复用，来回缩放不重算。
- **解码缓存**：高程与 WSF3D 瓦片的解码结果都有 LRU 上限。

### 4.4 资产管线

```bash
# 台海全域（影像 z0-11、高程 z0-10、WSF3D z0-10）
python3 tools/fetch_assets.py all --bbox 115,18,125,29

# 城区高清（影像 z12-14、WSF3D z13），城市区列表见 engine/scenario.py 的 URBAN_AOI
sh tools/fetch_urban_wsf3d.sh

# 归一化 manifest —— 必须最后跑
python3 tools/finalize_manifest.py --bbox 115,18,125,29
```

**manifest 归属权是强制约定**：`bbox` / `imagery` / `terrain` / `wsf3d` 属于
全域抓取，由 `finalize_manifest.py` 权威写入；`urban_*` 属于城区抓取。城区抓取
一律带 `--no-manifest`，否则会把全域条目覆盖成一个城区的范围——那会让底图
provider 只声明覆盖一座城市，现象是"地图坏了"，而不是"manifest 写错了"。

---

## 5. WSF3D 建筑高度还原

DLR GeoService 返回的是**调色板 PNG**：黑色为无数据，蓝色渐变为建筑高度。
直接当图层叠加会盖住底图（无数据是纯黑、不透明），所以：

1. **叠加显示**用 `ImageryLayer.colorToAlpha` 把黑色扣掉。
2. **三维体块**需要真实高度，做法是**实测标定**而不是猜：
   `tools/calibrate_wsf3d.py` 对每个采样像元，取其**像元中心**的经纬度去问
   WMS `GetFeatureInfo`，拿到真实的 `GRAY_INDEX`（米），与该像元的颜色配对，
   最后输出**颜色→高度查表**（`height_ramp.json`）。
   - 调色板是离散的，查表比在单个通道上做线性拟合准得多（线性拟合残差 6.8 m）。
   - 必须在 z13 标定：z10 一个像元约 153 m，粗于 WSF3D 的 90 m 源，重采样会
     把颜色和数值搅在一起。

无数据值为 `-32767`，在栅格里渲染为纯黑。

---

## 6. 前端瓦片缓存

`web/src/tilecache.js`。目标是"浏览过的瓦片不再重复向服务器取"，分两层：

- **内存 LRU**：512 张已解码 canvas，渲染路径直接命中，不触发解码。
- **IndexedDB**：原始字节，跨会话持久，上限 6000 条，按写入时间淘汰。

### 6.1 设计原则：缓存永远不能挡住渲染

第一版把缓存放在渲染路径上（`requestImage` 里 `await` 取图），结果很糟：只要
取图链路任一环卡住（IndexedDB 打开、fetch、图片解码），Cesium 就会因为等这一张
瓦片而**完全不渲染表面瓦片**，整幅底图变黑。实测现象是 `surface tiles 0`、
`imagery requests 1`——一次请求把整条流水线堵死。

现在分成两条互不影响的路径：

```
渲染路径（同步）  requestImage 只查内存 LRU，命中即返回；未命中交给 Cesium
                  自己的加载器。全程没有 await。
预热路径（异步）  后台取图、解码、写内存与 IndexedDB。慢/失败/卡住都只影响
                  "下次能不能命中"，不影响这一次能不能画出来。
```

持久化命中靠**启动后把 IndexedDB 搬回内存**。但预热同样要讲纪律：

- 只在底图空闲时进行（由 `globe.tileLoadProgressEvent` 归零驱动）；
- 一次只搬 16 张，串行解码；
- 预热取图并发上限 2。

预热无节制地并发会占满浏览器每域 6 条连接，把 Cesium 自己的影像请求饿死——
同样是"底图变黑、日志里什么错都没有"。

### 6.2 另一层是 HTTP 缓存

`server/serve.py` 对 `/tiles/` 与 `/vendor/` 返回
`Cache-Control: public, max-age=604800, immutable` + `ETag`，源码文件保持
`no-store`。案例 JSON 是几 MB 的重复文本，开 gzip 后约 470 KB（约 1/20）。
服务端 accept backlog 调到 256：默认的 5 挡不住浏览器 6 条并发连接 + Cesium
的批量瓦片请求，被丢弃的连接在页面里表现为 `Failed to fetch` 和底图降级。

---

## 7. 取证与验证

```bash
# 一张截图（本地驱动 + 远程渲染 + 校验 + 重试）
python tools/capture.py --url "/?case=joint_full&tick=800" --out shots/x.png
```

`tools/capture.py` 有两道判据，因为截图会以两种不同方式出错：

1. **有没有渲染** —— 中心区域亮度；空场景接近 4。
2. **细化到不到位的层级** —— 瓦片还没细化完的地球仍然"亮"，只是把粗层级拉伸
   铺满，看起来是一片渐变，平均亮度分辨不出来，但**边缘能量**可以：细化完成的
   画面约 1.9，粗层级占位约 0.13。

**取帧方式已经换掉**：不再用 `--screenshot` + `--virtual-time-budget`，改走
DevTools 协议（导航 → 等瓦片队列排空 → `Page.captureScreenshot`，不使用虚拟时间）。
旧方式的失败在低空表现为"整幅黑"，而且会被误读成渲染缺陷——详见 §9.5 与
`tools/capture.py` 的模块文档。`--keep-failed` 仍会保留被拒绝的帧。
`tools/remote_shot.sh` 只作参考保留。

### 7.1 运行时读数：只有 DevTools 协议靠得住

截图只能判断"画面对不对"，判断不了"**为什么**是粗的"——它从页面里带不回任何信息。
两条被实测否掉的替代路径：

| 手段 | 实测结果 |
|---|---|
| `--dump-dom` | **不可用**。它在 load 事件 + 虚拟时间预算结束时序列化 DOM，而 WebGL 页面此刻还没栅格化完。同一个 URL，截图能正常渲染，DOM 里却是 `surface tiles 0`、`imagery levels min=0 max=1`、`tiles rendered false`——描述的是首帧之前的状态，不是之后。 |
| 把 `--virtual-time-budget` 调大 | **无效**。预算算的是虚拟时间，而瓦片解码与 SwiftShader 栅格化消耗真实时间：预算从 40 s 提到 120 s，`imagery requests` 只从 10 涨到 12。 |

`tools/inspect_remote.py` 因此改走 `--remote-debugging-port`，自己实现 DevTools 的
WebSocket 客户端（**纯标准库**：容器里没有 pip、没有 node、没有 npm），在**瓦片队列
排空之后**读页面状态；探针在导航前注入，钩住
`UrlTemplateImageryProvider.prototype.requestImage`——应用自己的 `instrument()`
只包了底图 provider，城区高精度层在它眼里是看不见的。

实测输出（`?cam=121.57,25.03,6000,-45`，台北上空 6 km）：

```
camera            lon=121.57 lat=25.03 height=6 km
globeTilesLoaded  true          pageErrors 0
imageryRequests   413
imageryLevels     z13:224  z14:17  z11:33  其余 z0-z10
按 provider 分组（rectangle / maxLevel）：
  115,18,125,29                 maxLevel=11 ->  29 次，最高 z11（战区底图）
  121.505,24.973,121.625,25.093 maxLevel=17 ->  43 次，最高 z14（台北）
  120.24,22.57,120.36,22.69     maxLevel=17 ->  34 次，最高 z13（高雄）
  118.03,24.42,118.15,24.54     maxLevel=17 ->  69 次，最高 z13（厦门）
  119.24,26.02,119.36,26.14     maxLevel=17 ->  95 次，最高 z13（福州）
```

这一份读数直接证明：**四个城区高精度 provider 都在被真实请求**（而不只是声明了
`maximumLevel`），且 6 km 机位下主力层级是 z13、局部细化到 z14。

---

## 8. 目标跟随与地图组件

### 8.1 地图组件（指南针 / 比例尺 / 视点坐标）

**Cesium 1.145 没有内置指南针，也没有内置比例尺**（内置 widget 只有 HomeButton /
SceneModePicker / BaseLayerPicker / Geocoder / NavigationHelpButton / Timeline /
Animation / FullscreenButton / VRButton / InfoBox / SelectionIndicator /
ProjectionPicker / CreditDisplay），三项全部自建，归在图层管理第四类「地图组件」下，
默认显示，复选框控制加载与隐藏。

- **指南针**：取"盘面 N 指向屏幕上的北"语义（"指针指向相机朝向"没有信息量——屏幕
  上方本身就是相机朝向）。朝向**自算**而不用 `camera.heading`：那个 getter 内部会
  临时改写 `camera._transform` 并两次 `updateMembers`，10 Hz 高频读取有额外开销。
  点击复位正北（pitch 会先 clamp，否则接近 ±90° 时"点了没反应"）。
- **比例尺**：取屏幕两点 → `getPickRay` + `globe.pick` → `EllipsoidGeodesic` →
  按 1/2/5×10ⁿ 取整。**取样点必须在比例尺实际绘制处，不能用屏幕中心**——3D 倾斜
  视图下近端与远端的每像素米数可差数倍。三级兜底：`globe.pick` **只对已渲染瓦片
  求交**，未加载时返回 undefined → `IntersectionTests.rayEllipsoid` 解析求交 →
  视锥解析式 → 全失败则隐藏。用测地线量测不需要手工 `cos(lat)` 校正。
- **视点坐标**：`camera.positionCartographic`，即**相机自身**的经纬高（**椭球高**，
  不是海拔）。与右侧「地理位置信息」面板分工不同——后者是**光标落点**。实测
  900 km 机位下两者相差约 2.2° 纬度，证明确实是两个不同的量。

### 8.2 目标跟随

开关在图层管理里；开启后点击任一目标即以其为中心持续跟随。

**用 `camera.lookAt` + `HeadingPitchRange` 逐帧驱动，不用 `viewer.trackedEntity`**：
后者的 `entity.viewFrom` **只在首帧生效**（跟踪中改视角必须解绑重绑），且 3D 跟踪会
强制 `enableTilt = false`，做不到"随航向旋转 + 俯仰可控"。

**航向公式**（由 1.145 源码 `offsetFromHeadingPitchRange` 推出，实测已复核）：
`offset_ENU = (−r·cos p·sin h, −r·cos p·cos h, −r·sin p)`，即 `heading` 是**相机视线
方位角**，相机**位置**方位角 = `heading + 180°`。要让相机位于目标右后上方：
`heading = 目标航向 − 30°`。

实测复核（`?case=joint_full&tick=800&follow=R06`）：相机高度 13.15 km = 8500 +
15000·sin18°；相机相对目标 Δ北 4.8 km、Δ东 13.4 km → 水平 14.2 km = 15000·cos18°，
方位角 70.3°；R06 末段航向 279° → 公式预期 `279+180−30 = 69°`。**吻合。**

视角参数按域分档（海 3.5 km/−22°、陆 4.0 km/−30°、空 15 km/−18°，均偏右 30°）。
跟随相机挂在 `scene.preRender`（`Scene.render` 顺序为 preUpdate → update → preRender
→ draw，在这里改相机本帧生效）。目标静止（暂停或到达航路终点）时跳过，既省掉每帧的
`lookAt` 也不再刷 `camera.changed`。

**三条必须遵守的纪律**：

1. **`detachFollow()` 内必须 `camera.lookAtTransform(Matrix4.IDENTITY)`**。
   `camera.lookAt` 会把 `camera.transform` 锁在目标 ENU 上，
   `ScreenSpaceCameraController` 随即把 `_globe` 置 undefined、`_ellipsoid` 换成
   单位球 → 缩放与贴地钳制失效、**可以穿到地面以下**。而 `setView` / `flyTo`
   **自己不会解锁**（它们保存并恢复原 transform），所以所有视角跳转入口都要先经
   `detachFollow()`。诊断浮层的 `unlocked` 项就是这一条的判据。
2. **"拖动即解除"不能用 `camera.changed` 或 `moveStart/moveEnd`**——这两个事件的
   判定会比较 `camera.transform`，而跟随每帧都在改它，会被跟随本身持续触发。改为在
   canvas 上捕获 `pointerdown/move/up` + `wheel`，用 4 px 阈值区分"点击切换目标"与
   "拖动解除"；按下即 `suspend`，让拖拽第一像素起相机就归用户。
3. **跟随时 `camera.changed` 每帧触发，会把建筑体块的 400 ms 防抖反复重置而永不执行**。
   改为 `if (state.follow.id) return;`，并在 `detachFollow()` 收尾时补一次
   `refreshBuildings()`。

### 8.3 一个真实 Bug 的根因（已修）

页面加载后 Cesium 弹出 `An error occurred while rendering. Rendering has stopped. /
TypeError: a.then is not a function`。

根因：`web/src/tilecache.js` 的 `attach()` 在内存命中时**同步返回 `HTMLCanvasElement`**，
而 Cesium 1.145 的 `ImageryLayer._requestImagery().doRequest()` 对 `requestImage()` 的
返回值**直接调用 `.then()`**（只有 `defined()` 守卫，**没有** `Promise.resolve()` 包装）。
canvas 没有 `.then` → 渲染循环内同步抛错 → CesiumWidget 判定渲染已停止。

修复：命中分支 `return Promise.resolve(hit);`。

---

## 9. 空态、图层管理与菜单栏

### 9.1 空态（初始化不加载任何态势）

打开页面**不加载任何案例**，只保留底图 / 地形 / WSF3D / 地图组件 / 鸟瞰图。
用户从顶栏「态势案例」选择后才加载。菜单里另有「清除当前态势」回到空态。

空态下：目标列表与热点表显示占位文案，播放/步进/时间轴/导出全部禁用，
目标跟随开关置灰，状态栏显示「空态 · 未加载态势」。

两个实现要点：

- **`state.data` 的守卫必须逐处加**。最要命的是 `positionFor()`——它被 Cesium 的
  `CallbackProperty` **每帧求值**，空态下每帧抛一次未捕获异常，直接污染状态栏。
  其余（`renderTargetList` / `renderHotspotTable` / `setTick` / `renderTransport` /
  `exportData` / `updateGeoReadout` / `selectHotspot` / `refreshBuildings`）各有
  兜底分支。
- **`bindUI()` 前移到 `initViewer()` 之后**。它注册的 `tileLoadProgressEvent` /
  `camera.changed` / `preRender` / `ScreenSpaceEventHandler` 都不依赖案例数据；
  放在 `loadCase` 之后会让磁盘预热、城区影像挂载、地图组件与跟随驱动全部被推迟
  到案例加载完。

配套新增 `web/data/base.json`（与案例无关、启动即加载）：城区 AOI + 军标目录 +
默认机位 + 区域名。这两样原本只随案例数据集下发，空态下建筑体块与符号渲染都需要
它们。放进 `cases.json` 也可以，但 `cases.index()` 目前零依赖，而
`scenario.build_dataset` 正是靠函数内 import 才避开循环依赖——让菜单索引反向依赖
`scenario` / `symbology` 是把这条边固化错。

### 9.2 图层管理（四大类，全部可折叠）

**项目内所有可展示要素都归入这四类**，各由一个复选框控制加载与显隐：

| 类别 | 开关 |
|---|---|
| **态势图层** | 热点区域、计划航路、区域级态势横幅、鸟瞰图 |
| **目标图层** | 可见目标、目标标牌、历史轨迹、传感器覆盖、目标元数据面板、红方/蓝方/海上/陆上/空中（带计数）、目标跟随 |
| **底图** | 影像底图、城区高清影像、地形模型（DEM）、建筑高度图层（WSF3D）、建筑体块（WSF3D 三维）、气象模型 |
| **地图组件** | 指南针、比例尺、视点坐标 |

顶栏「图层管理」菜单提供 全部显示 / 全部隐藏 / 恢复默认 三个批量操作。
**「全部隐藏」刻意不关底图与地图组件**——把底图关掉只会得到一片黑球。
**「目标跟随」是模式开关而不是图层，刻意不列入批量操作**——把它一起关掉等于
隐式解除跟随，语义混乱。

「目标元数据面板」开关关掉后，点击目标只做选中与高亮，不弹出信息面板。

### 9.3 菜单栏结构

| 类型 | 名称 | 二级内容 |
|---|---|---|
| 菜单 | **态势案例 ▾** | 4 个案例（名称 + 描述 + 目标数/热点数）；分隔线；**清除当前态势** |
| 菜单 | **图层管理 ▾** | 全部显示 / 全部隐藏 / 恢复默认 |
| 菜单 | **视图 ▾** | 复位视角（回默认机位 + 解除热点锁定与跟随） |
| 按钮 | **鸟瞰图**（toggle，按下态高亮） | — |
| 按钮 | **态势自动推演**（播放/暂停） | — |
| 按钮 | **导出数据** | — |

**必须留在菜单栏的四项**及理由：

1. **态势案例** —— 空态下这是唯一的态势加载入口，没有它系统无法进入可用状态。
2. **态势自动推演** —— 全局最高频操作，任何时刻都要一键可达。
3. **图层管理** —— 面板入口 + 全部要素显隐的批量操作，属全局控制。
4. **导出数据** —— 数据出口。

**已移除**：「态势输入」（空壳，只滚动目标列表）、「位置信息」（并入 视图 → 复位视角）、
「瓦片缓存」面板（统计并入 `?diag=1`）、独立的「模型开关」区块（并入图层管理四类）、
「天气控制」按钮（气象模型并入「底图」）。

### 9.4 地名图层

**数据来源**：**Natural Earth 1:10m Populated Places**（`ne_10m_populated_places`）。
选它的三个理由：**公有领域**（官方声明无需署名，不引入授权风险）；自带 `SCALERANK`
0–10 的重要性分级，正是分级显示需要的 LOD 提示；含 `NAME_ZH` 中文字段，省去翻译。

**离线存储**：`web/tiles/places/places.json`，约 926 KB（gzip 后约 250 KB，随静态服务
走 HTTP 缓存），**7342 个地名**。由 `tools/fetch_places.py` 抓取并转换：

```bash
python3 tools/fetch_places.py                     # 全球
python3 tools/fetch_places.py --bbox 108,18,146,36  # 只保留一个范围
```

紧凑键名：`n`=中文名（缺失时用英文名）`e`=英文名 `x`/`y`=经纬度 `r`=scalerank
`p`=人口 `c`=国家 `a`=一级行政区 `k`=类别。数据已按 `(scalerank, -人口)` 排序，
所以前端"取前 N 个"就是"最重要的 N 个"，不需要再排序。

**为什么自己写 shapefile 解析**：远程容器是 python3.8 且**没有 pip**，装不了
`pyshp`/`geopandas`。shapefile 与 dBASE 都是很简单的二进制格式，用标准库读一百多行
就够，也免得多一份依赖。

**渲染与交互**：用 `Cesium.LabelCollection` 而不是 `viewer.entities`——7000 多个地名
如果都建成 Entity，光是实体管理开销就压垮帧率。这里维护一个**固定大小的标签池**
（260 个），每次更新只把"该显示的那些"重新绑定到池对象上，不新建、不销毁。
样式为深色字 + 白色描边（陆地和海面都读得清），并带 `scaleByDistance` 让远处标签
略小。开关在图层管理「底图」组里。

**缩放/平移时的更新策略**——两级筛选：

1. **按相机高度分档**（`LOD` 表）：

   | 相机高度 | 最大 scalerank | 全球候选数 |
   |---|---|---|
   | > 6000 km | 1 | 68 |
   | 3000–6000 km | 2 | 186 |
   | 1500–3000 km | 3 | 522 |
   | 700–1500 km | 4 | 1128 |
   | 300–700 km | 6 | 2443 |
   | 100–300 km | 7 | 5525 |
   | < 100 km | 8 | 6768 |

2. **屏幕空间贪心去重**：按重要性排序后依次放置，与已放置标签的屏幕包围盒重叠的
   跳过。这一步是必需的——按 rank 筛选后，长三角、珠三角这种密集区仍会糊成一团。

更新挂在 `camera.changed` 上，**节流 250 ms**，并且只有"相机高度跨过档位"或
"移动超过约 0.005°"时才重算；静止时完全不工作。投影上千个点到屏幕坐标是这里唯一
的重活，所以不放在每帧的 `preRender` 里。实测 900 km 机位下显示 88 个标签。

> 一个容易漏的初始化：HTML 里的 `checked` 只是初始显示，`change` 事件不会自动触发，
> 而地名数据是懒加载的——不显式调一次 `setEnabled(true)` 就会"默认勾选却什么都不
> 显示"。诊断浮层的 `places total / shown / enabled` 就是这一条的判据。

### 9.5 鸟瞰图

> **曾经的"20–300 km 无头黑屏"是取证工具的缺陷，不是渲染缺陷——已修复。**
>
> 旧结论（记录于 2026-09-20 上午）：无头 SwiftShader 下相机高度约 20–300 km 时
> 主视图整幅变黑，900 km 与 6 km 正常，`?overview=0` 同样黑，因此判断是"这个高度
> 渲染不出来"，并把真实 GPU 复核列为待办。
>
> **这个判断是错的。** 用 DevTools 协议从页面内部读运行时状态，20 km 机位下页面
> 一直正常出帧：
>
> ```
> 20 km   raf/frames 持续增长（约 5 fps）  globe.tilesLoaded true
>         ctxLost false   pageErrors []
> ```
>
> 页面一直在渲染，黑的是**截图**。根因是旧取证链路用
> `--screenshot` + `--virtual-time-budget`：虚拟时间快进的是页面定时器，而瓦片解码
> 与 SwiftShader 栅格化消耗的是真实时间。低空视野里瓦片数量多几百张，预算早在画面
> 画完之前就到期，`--screenshot` 于是合成了一张还没画过的画布。
>
> 改为走 DevTools（导航 → 等瓦片队列排空 → `Page.captureScreenshot`，全程不使用
> 虚拟时间）后，**上述"必黑"的高度第一次尝试就全部出图**：
>
> | 相机高度 | 旧链路 | 新链路（首次尝试） |
> |---|---|---|
> | 300 km | ❌ luma 3.3 | ✅ luma 83.0、edge 2.938 |
> | 20 km | ❌ luma 3.3 | ✅ luma 73.3、edge 13.671 |
> | 12 km | ❌ luma 3.0 | ✅ luma 81.5、edge 4.498 |
> | 6 km | ✅ | ✅ luma 96.4、edge 13.238 |
> | 900 km | ✅ | ✅（改动前基线） |
>
> 见 `tools/capture.py` 的模块文档。**教训：截图工具的失败不要记成渲染的失败**——
> 先问"页面自己知道什么"，再问"图为什么不对"。

`web/src/overview.js`。第二个 Cesium Viewer，`SCENE2D`，240×172，位于中间窗口左下角，
显示真实地球缩略图（z0–z6），并用黄色闭合折线画出**主相机当前视野框**，随相机联动。
点击鸟瞰图可把主视角跳转过去（跳转前先解除跟随）。

四个实现要点：

- **必须用 `baseLayer`**：Cesium 1.107 起 `Viewer` 的 `imageryProvider` 选项被移除且
  **静默忽略**，不传 baseLayer 会回落到 Ion 在线底图，既违反离线要求也会因缺 token 报错。
- **不与主视图共用 provider 实例**：主视图的 provider 被 `TileCache.attach` 与
  `instrument` 各包了一层，共用会让 2D 的请求与 404 计入主视图的 `imageryStats`，
  `?diag=1` 全部失真；`errorEvent` 也是共享 Event，无法按 viewer 区分。
- **`setView(Rectangle)` 必须在容器尺寸稳定之后**：`SCENE2D` 下 `setView` 会按调用
  那一刻的画布尺寸反算正交视锥宽度。容器刚解除 `display:none` 就 setView，画布尺寸
  还没生效，地球会缩在一角、其余留白。所以 `show()` 里先 `resize()` 再 `setView`，
  并用一次 `setTimeout(0)` 补一次。
- **不用 `requestRenderMode`**：实测在无头 SwiftShader 下该视图会停在黑屏（瓦片已
  加载完但画布未重绘）。这块画布只占主视图像素的约 1.5%，不值得为它承担按需渲染的
  不确定性。

---

## 10. 已知限制

- **底图与城区的覆盖范围是"分级"的**，不是均匀的：全球 z0–6、战区 z0–11、
  9 个热点框 z0–12、4 个城区 AOI z13–17（z17 约 1.2 m/px）。四个城区
  （台北 / 高雄 / 厦门 / 福州）**都已下齐** z12–17 影像与 z13 WSF3D，实测计数：
  z13 每个城区 12–16 张（0.3° 框全覆盖）、z14 42–49、z15 156、z16 528–575、
  z17 2112–2205，均等于各自 0.3° 框的完整瓦片数。
  城区框外回落战区层（z11），所以城区 AOI 的矩形边界在低空可见——这是分级
  覆盖的固有特征，不是缺失。
- **无头软件光栅化环境下底图细化慢**（瓦片解码与 SwiftShader 栅格化吃真实时间），
  取证时看到的是这台机器的速度，不是页面逻辑。前端已加"正在加载瓦片…剩余 N"徽标
  把这段过程显式化。**取帧已改用 DevTools 等待瓦片排空**，所以"首次截图常要重试"
  这个老现象不再成立（见 §7 / §9.5）。
- **WSF3D 是 90 m 分辨率的城市体量**，不是单体建筑模型；三维体块表达的是街区
  尺度的建筑高度分布。
- 高程 z10 在城区尺度会被拉伸；底图 z11 在城区尺度同理，靠城区 z12–14 补。
- **建筑体量的第二个数据源（3D-GloBFP）已打通到端到端**，见 §11。它不是 WSF3D
  的平替，两个源各有短板：WSF3D 全球连续但粗，3D-GloBFP 逐建筑但覆盖不均
  （中国大陆可用，台湾在数据集里近乎为空）。当前 manifest 已切到 `gh3d`。

---

## 11. 建筑高度数据源：WSF3D 与 3D-GloBFP

三维体块当前的栅格是 WSF3D（DLR，90 m，逐聚落平均高度）。z13 一个像元约 17 m，
所以**源头分辨率就是瓶颈**：城市读起来是一团平滑的凸起，CBD 与外围的高度对比
被抹掉。

`tools/build_gh3d_tiles.py` 打通了第二个源：**3D-GloBFP**（Che et al., ESSD 16,
5357–5374, 2024），**逐建筑矢量 + `Height` 属性（米）**，按 z13 栅格化后每像元
是"该像元内建筑的最大高度"，天际线因此能被表达出来。

### 11.1 关于"v4"：公开渠道只有 v2

调研（Zenodo REST / DataCite / OAI-PMH / figshare API / ArcGIS Online / 中英文检索）
的结论：**3D-GloBFP 的公开最新版是 2025-05-22 的 v2**，Zenodo concept DOI
`10.5281/zenodo.11391076`（记录 `15487037`），**查不到 v3/v4 的任何发布证据**。
如果需求文档里写死了"v4"的指标，需要先澄清出处。

v2 的形态（已实测下载解包验证）：

| 项 | 实测值 |
|---|---|
| 类型 | 逐建筑 shapefile（`.shp/.dbf`），**没有栅格、没有 volume 字段** |
| 属性 | `Height`（N,24，米，6 位小数）= 2020 年前后建筑高度估计值 |
| 许可 | **CC BY 4.0**（可商用、可再分发，仅需署名） |
| 体量 | 全库 10 个 figshare PART、2554 个分幅、约 198 GB（zip） |
| 分幅 | 1.25°（1658 格）/ 2.5°（612）/ 5°（193）/ 10°（92），中国按省命名 |
| 在线服务 | **没有 WMS/WMTS/XYZ**，只能整包下载后本地栅格化 |
| 台海窗口 115–125°E/18–29°N | 35 个分幅，约 **1.61 GB** |
| 实测样例 | 厦门/泉州 `2274_117.5_23.75_118.75_25.0_fujianCN.zip`：112.6 MB，解压 277 MB，**1 523 341 栋** |

**覆盖陷阱（已实测）**：`2277_118.75_22.5_120.0_23.75_taiwanCN.zip` 只有 **0.2 MB /
1911 栋**——台湾西部在这个数据集里近乎为空。所以**这个源只覆盖大陆的厦门与福州，
不能用来替换台北/高雄的 WSF3D**。上线前必须逐图幅统计 feature 数，而不是"有文件
就算有数据"。

### 11.2 为什么是"构建期栅格化"，而不是运行期读栅格

容器是 python3.8 + 无 pip + 无 GDAL + 无 numpy。所以：

```
figshare zip → .shp/.dbf → 逐建筑 (lon, lat, Height)
             → z13 像元 → PNG 瓦片 + height_ramp_gh3d.json
```

* **DBF 用标准库直读**（32 字节头 + 定长记录，`struct` 足够）。真正的难点是几何。
* **SHP 多边形只取质心**，且用面积加权的三角形法（顶点平均会把 L 形/L 形凹多边形
  的重心拉偏）。z13 像元 17 m 下，轮廓形状不再携带额外信息。
* 输出**刻意与 WSF3D 同构**：`<layer>/<z>/<x>/<y>.png` + 一张颜色→高度查表，
  前端因此可以把两者当成同一类数据，而不是特例。
* **调色板必须与查表严格一致**：第一版按线性插值上色，而查表只有 10 个停靠点，
  于是插值出来的颜色解不回去，整格误差一个色带。现在颜色**取最近停靠点**——
  PNG-8 本来也只有 256 色，把 0–400 m 按 0.25 m 细分需要 1601 个调色板项。
* **下载必须校验字节数**。这里的出口代理会在传输中途断掉却仍返回 `200`，也会直接
  拒绝 `urllib`（`SSL: UNEXPECTED_EOF_WHILE_READING`，一个字节都拿不到）而放行
  `curl`（同一代理下 112 MB / 28 s 完成）。所以用 `curl --retry-all-errors -C -`
  续传，并且**只有字节数与目录一致才接受**——截断的 zip 会当作有效数据解析，
  建筑静默缺失。

### 11.3 实测结果

```
$ python3 tools/build_gh3d_tiles.py build --grid 2274 --zoom 13 --out /tmp/gh3d_real
    1523341 building records, fields=Height
    1481948 buildings with a usable height      (97.3% 有效)
    1481948 centroids resolved
    1481948 buildings binned into z13 cells
building mean 11.0 m / max 171.4 m
cell heights (max): p90 20.2 m  max 171.4 m   cells 1 300 712   tiles 893
```

**逐建筑数据在 17 m 像元上仍然只有 p90 = 20 m 的像元级最大高度**，说明该区域内
绝大多数建筑确实低矮；最大值 171.4 m 与天际线里的高层一致。**取最大而非平均**是
这里的关键口径：一个 CBD 像元里 1 栋 150 m 塔楼 + 20 栋 8 m 建筑，均值只有 15 m，
塔楼会被平均掉。

```bash
# 列出台海窗口的所有分幅与体积
python3 tools/build_gh3d_tiles.py list --bbox 115,18,125,29

# 建单个图幅（测试用）
python3 tools/build_gh3d_tiles.py build --grid 2274 --zoom 13 --out web/tiles/gh3d
```

下载缓存与目录在 `~/.cache/gh3d`（`GH3D_CACHE` 可覆盖）。

### 11.4 前端切换与一个被量出来的静默 bug

切换只需改 manifest：

```json
"building_blocks": { "source": "gh3d", "layer": "buildingheight",
                     "ramp": "./tiles/gh3d/height_ramp_gh3d.json" }
```

`assets.js` 的 `blockSource()` 据此换目录与查表文件，缺省保持 `wsf3d`——这样即使
gh3d 瓦片没构建出来，体块也不会整块消失。

切过去之后发现体块**根本不出现**，而且不报错。用
`window.SituationHost.debugBuildings()`（新增，把体块链路的每个短路条件逐条报出来）
定位到三个原因：

1. **`dom.modelBuildings` 这个 DOM 缓存引用与文档脱钩**：实测
   `document.getElementById('modelBuildings').checked === true` 而
   `dom.modelBuildings.checked === false`、`isConnected === false`，同时
   `querySelectorAll('#modelBuildings').length === 1`（页面上只有一个）。
   于是 `refreshBuildings()` 第一行的开关判断直接返回。改为 `checkboxOn(id)`
   一律现取 DOM。
2. **案例装载完不触发生成**：`refreshBuildings()` 只挂在"选中热点 / 解除跟随 /
   相机高度变化"上，所以"一进页面就带 `?buildings=1`"或"先开开关再选案例"都不出
   体块。在 `applySituationData()` 收尾补一次。
3. **`?buildings=1` 排在 `?cam=` 之前**：体块按*当前机位*挑最近的城区 AOI，取证时
   定位到厦门却按默认机位选城市。把体块钩子移到 `?cam=` 之后。

修好后的实测（`?cam=118.09,24.48,12000,-40&buildings=1`）：

```
buildingsKey  "2:0"
switchNote    建筑体块已显示：「中国 厦门」LOD far（网格 24×24，采样步长 8），
              265 个体块，用时 223 ms。相机高度变化时会自动切换 LOD。
rampSource    3D-GloBFP per-building vector rasterised to max height per cell
```

> 教训：**DOM 缓存引用会失效，判断"开关开没开"必须现取**；而"整个功能静默不工作"
> 这类问题，靠看渲染结果是查不出来的——`debugBuildings()` 把短路条件摆出来之后
> 一眼就定位到了。

