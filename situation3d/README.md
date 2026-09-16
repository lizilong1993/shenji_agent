# situation3d — 三维战场态势生成系统

自研三维推演引擎 + CesiumJS 前端。面向真实 WGS84 地理与真实装备型号，覆盖
**海 / 陆 / 空**三域、**红 / 蓝**双方，具备**三级态势自动推理**与**军标可视化**，
并且**完全离线运行**。

```
engine/     推演内核：地理计算、军标目录、想定与航路、三级态势推理、演示案例
scripts/    生成前端所需的数据集（每案例一份）
web/        CesiumJS 前端（离线瓦片、军标、推理面板、案例菜单）
server/     零依赖静态服务（缓存策略 + gzip）
tools/      资产抓取、色带标定、远程同步、无头取证
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

### 4.1 资产族

| 族 | 目录 | 层级 | 来源 |
|---|---|---|---|
| 影像底图 | `web/tiles/imagery/{z}/{x}/{y}.jpg` | z0–11（全域）+ z12–14（城区） | Esri World Imagery |
| 高程 DEM | `web/tiles/terrain/{z}/{x}/{y}.png` | z0–10 | Mapzen / AWS（Terrarium 编码） |
| 建筑高度 | `web/tiles/wsf3d/buildingheight/...` | z0–10（面）+ z13（城区） | DLR WSF3D v02 |

`web/tiles/manifest.json` 记录**真实覆盖范围与层级区间**，前端 provider 的
`rectangle` 直接取自这里。

### 4.2 LOD 的关键一条：provider 必须声明真实覆盖范围

底图只下载了台海 AOI。如果 provider 仍声明覆盖全球，Cesium 会为域外瓦片持续
发请求、持续 404，并因此**卡在 z0–z1 不再细化**——实测一次加载发出 500 次重试、
整幅底图退化成一片渐变。把 `rectangle` 设成 manifest 里的真实范围后，请求全部
命中已下载瓦片，层级正常推进到 z7。

### 4.3 其余 LOD 手段

- **高程**：超过下载层级时复用祖先瓦片并降采样，不会出现空洞。
- **城区高清影像**：相机高于 200 km 时不挂载 z12–14 图层，否则每个表面瓦片都
  会去要一批用不上的瓦片；低于阈值再挂上，飞高时摘掉。
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

第二道判据是把 `--virtual-time-budget` 从"祈祷参数"变成可复现流程的关键。
`--keep-failed` 会保留被拒绝的帧，便于直接看诊断浮层而不是只看到一个失败。

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

### 9.4 鸟瞰图

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

- **城区高清影像目前只有台北**（`中国台湾省 台北`）。高雄、厦门、福州的
  z12–14 影像与 z13 WSF3D 尚未下载完，切到这几个城市时城区图层为空。
  补齐：`sh tools/fetch_urban_wsf3d.sh`。
- **无头软件光栅化环境下底图细化很慢**，首次截图常需要重试。真实浏览器有 GPU
  时明显更快。前端已加"正在加载瓦片…剩余 N"徽标把这段过程显式化。
- **WSF3D 是 90 m 分辨率的城市体量**，不是单体建筑模型；三维体块表达的是街区
  尺度的建筑高度分布。
- 高程 z10 在城区尺度会被拉伸；底图 z11 在城区尺度同理，靠城区 z12–14 补。
- **3D-GloBFP v4** 可作为 WSF3D 之外的建筑体量来源。当前的三维体块管线只依赖
  "建筑高度栅格 + 颜色→高度标定表"两件事，替换数据源只需新增一个同构的抓取
  与标定适配器，`web/src/assets.js` 不用改。
