# situation3d — 三维态势生成系统

课题三子课题3 下一阶段系统。目标：把现有 wargame 二维仿真态势升级为**基于 Cesium 三维地球的真实世界态势生成系统**。

## 目标口径（已与需求方确认）

| 项 | 决策 |
|---|---|
| 三维引擎 | CesiumJS（Apache-2.0，成熟 + 开源） |
| 推演内核 | **自研**：复现庙算陆战引擎语义 + 新增海空算子，面向真实 3D 地球与真实装备型号 |
| 场景范围 | 台海为主 + 热点区域清单 |
| 真实性口径 | 图像真实（MAR20 / ShipRSImageNet / SatSAM2）+ 态势仿真 |
| 数据层 | PostgreSQL + PostGIS 存轨迹点，CZML 仅作派生输出 |

## 目录结构

```
situation3d/
├── web/                       三维世界前端（CesiumJS）
│   ├── index.html             页面骨架：顶栏 / 左侧图层树 / 中央地球 / 右侧信息 / 底部推演控制
│   ├── src/
│   │   ├── app.js             地球初始化、轨迹加载与时间轴驱动回放
│   │   └── style.css          界面样式
│   ├── data/                  生成的态势数据（gitignore，见下）
│   └── vendor/cesium/         CesiumJS 1.145.0 预构建版（gitignore，由脚本获取）
├── engine/
│   ├── geo.py                 WGS84 大地测量：haversine / 方位角 / 球面插值 / 折线采样
│   └── scenario.py            台海演示想定定义与逐帧轨迹生成
├── scripts/
│   └── gen_demo_trajectory.py 生成 web/data/demo_taiwan_strait.json
├── server/
│   └── serve.py               静态服务入口（无第三方依赖）
└── tools/                     构建与取证工具
    ├── fetch_cesium.sh        下载并展开 CesiumJS 1.145.0 到 web/vendor/
    ├── bootstrap_remote_headless.sh  远程容器无头 Chromium 依赖补齐（一次性）
    ├── remote_shot.sh         远程侧单次截图
    └── capture.py             本地驱动：截图 + 像素校验 + 失败重试
```

## 首次准备

`web/vendor/` 与 `web/data/` 均不入库（45 MB 第三方 + 可确定性重建的数据）：

```bash
sh  situation3d/tools/fetch_cesium.sh                 # 取 CesiumJS
python3 situation3d/scripts/gen_demo_trajectory.py    # 生成演示态势
```

## 启动方式

在远程容器内（`/remote-home/lizilong/shenji_agent`）：

```bash
setsid nohup python3 situation3d/server/serve.py --port 8848 </dev/null >/tmp/s3d.log 2>&1 &
```

浏览器访问 `http://<server>:8848/`。

支持两个确定性入口参数，供深链与自动取证使用：

| 参数 | 作用 |
|---|---|
| `?tick=360` | 首帧前跳转到指定帧（0 – 1439） |
| `?autoplay=1` | 打开即开始播放 |

## 渲染取证

`tools/capture.py` 在本地驱动、在远程渲染。之所以不直接用 `--screenshot`：Cesium 的球体栅格化与瓦片解码消耗的是**真实时间**，而 `--virtual-time-budget` 计的是**虚拟时间**，同一预算在不同次运行中会分别产出"已渲染"与"未渲染"（球体全黑）两种结果。因此每次截图都用像素校验（视口中心 240×240 区域的亮度），未通过则重试。

```bash
python3 situation3d/tools/capture.py --url "/?tick=360" --out tools/shots/tick_360.png
```

远程容器若被重建，需先补齐无头 Chromium 的系统库与字体：

```bash
ssh <host> 'sh -s' < situation3d/tools/bootstrap_remote_headless.sh
```

> 该容器无 root、无系统 Chromium、无任何字体。缺字体时 Skia 会直接 abort
> （`SkFontMgr_FontConfigInterface.cpp: Not implemented`），且中文标签全部显示为豆腐块。

## 当前状态

- **P0 环境基线** — 已完成（远程 git、分支策略、引擎数据字典）
- **P1 三维世界骨架** — 已完成：三维地球加载台海区域、8 个热点、5 个目标、1440 帧轨迹回放，全部经截图取证
- P2 海域域层与联合推演 — 未开始
- P3 真实型号库与图像接入 — 未开始
- P4 对接现有态势生成链路（LSTM / GCN） — 未开始
- P5 热点区域与验收 — 未开始

### P1 遗留问题

**底图分辨率不足，不足以支撑"真实场景"验收。**

CesiumJS 内置的 Natural Earth II 离线底图最高只到 z2，一级瓦片覆盖 45°×45°。
台海视角约 6°×4°，即单张瓦片被放大约 7 倍：

- z2 瓦片加载完成时（见 `tools/shots/tick_1439.png`）：海岸线、地形晕渲可辨，能看出陆/海/山地，
  但没有真实卫星纹理，港口、城市、航道等目标级要素完全不可见。
- 瓦片未就绪时（其余几张）：退化为陆/海大色块。

需要决策底图来源，二选一（或组合）：

| 方案 | 说明 | 代价 |
|---|---|---|
| A. 离线预取高分辨率瓦片 | 只取台海 AOI，z0–z12 约数百张瓦片，落盘随仓库外置分发 | 需确认瓦片源授权；AOI 外无底图 |
| B. 接入在线影像服务 | 画面质量最好，随取随用 | 依赖外网；离线/内网环境下不可用 |

> 注：`tools/shots/` 下的截图由 `tools/capture.py` 生成，像素校验只判定"球体是否已渲染"，
> 不判定影像 LOD。同一参数的多次捕获可能分别落在粗层级与 z2，属正常现象。

### 取证截图

| 文件 | 内容 |
|---|---|
| `tools/shots/verified_ticks.png` | tick 0 / 360 / 600 / 720 / 1200 / 1439 六时点对比 |
| `tools/shots/tick_*.png` | 各时点原始截图（1600×900） |

## 数据说明

`web/data/demo_taiwan_strait.json` 是**基于真实地理航路构造的仿真态势**，不是真实 AIS/ADS-B 记录。
航路点取自台海真实地理位置（平潭、澎湖、高雄、厦门等），速度与高度取自真实装备量级。
按已确认口径，态势数据为仿真生成，图像数据后续接入真实数据集。

## 引擎数据基准

自研引擎复现陆战裁决语义的权威基准见
[`../docs/wargame_engine_data_dictionary.md`](../docs/wargame_engine_data_dictionary.md)。
