# 目标运动与动态特效（CZML）

2026-10-19 实现，分支 `feat/czml-motion`。三个文件构成一条链路：

| 文件 | 职责 |
| --- | --- |
| `web/src/targetMotion.js` | 把案例逐帧状态编译成 CZML（位置 + 姿态采样），装载 `CzmlDataSource`，把采样属性挂到既有目标实体；由 `tickFloat` 驱动 `viewer.clock`。 |
| `web/src/targetFx.js` | 车轮节点旋转、船舶尾流、飞机尾迹；全部从同一个 `viewer.clock` 取时间。 |
| `web/src/entityLod.js` | 新增 `forwardAxis()` / `wheelNodes()` 两张资产表。 |

## 运动

* CZML 每目标一份：`position.cartographicDegrees`（线性插值）+
  `orientation.unitQuaternion`（四元数 slerp），epoch 为案例起点、采样间隔
  `tick_seconds`。31 目标 × 1441 帧的编译约 **65 ms**，装载约 **90 ms**。
* 姿态由轨迹给出：航向取帧数据 `heading`（罗盘角），缺失时用相邻点方位角；
  空中目标由高度差给俯仰、航向变化率给坡度（±35°），水面/地面只保留航向。
* `app.js` 的 `loop()` 每帧 `TargetMotion.syncClock(tickFloat)`；播放/暂停/
  拖动时间轴都只改 `state.tickFloat`，CZML 与特效自然同步。
* 跟随相机改用 `entity.position.getValue(viewer.clock.currentTime)`，与模型的
  帧间插值一致，避免近景“模型平滑、相机跳步”。

## 资产轴向（俯视标定，heading=90°、东=屏幕右）

| 资产 | 局部前向轴 |
| --- | --- |
| Kenney Watercraft（船） | `+X` |
| OGA MiG-15 | `-X` |
| Quaternius 坦克（armor） | `-Y` |
| Blender 程序化资产（apc/artillery/sam/coastal/aew/helicopter） | `-X` |

换资产必须重新标定并同步 `EntityLod.FORWARD_AXIS`；
`tools/entity_lod_test.js` 会校验每个 `MODEL_MAP` 键都有合法轴向。

## 特效

* **车轮**：`entity.model.nodeTransformations` 驱动 GLB 的 `wheel*` /
  `roadwheel*` 节点绕局部 Y 轴旋转；角速度 ∝ 当前速度（视觉同步，上限 30 rad/s）。
  节点名/半径在 `EntityLod.WHEEL_NODES`。履带坦克无轮节点，改用 glTF 的
  `Tank_Forward` 动画（其余 3 段动画倍率置 0）。
* **尾流 / 尾迹**：两条 `PolylineGlowMaterialProperty` 折线，按 Cesium 时钟回溯
  各自 CZML 采样位置（14 段），船舶另按半船长后移；长度由速度与船/机长决定。
  **不用 ParticleSystem**：时间轴 900× 压缩时粒子发射与寿命的时间基准不一致，
  实测 8 架飞机会膨胀到 24 万粒子、帧率 0.9 fps、堆 1.2 GB；折线方案稳定在
  基线帧率、堆内存不增长（20 s 播放保持 125 MB）。
* 任意时刻暂停：粒子/轮转角/尾迹都冻结；旋翼与雷达动画通过 `_multiplier=0`
  冻结（1.145 的公开 `multiplier` 为只读）。

## 验证

* `node tools/entity_lod_test.js`：104 项通过（含轴向/轮节点用例）。
* 浏览器（8851，Chromium/SwiftShader）：CZML 31/31 绑定；半帧位置插值 = 138.6 m
  （18 kn × 15 s）；姿态 slerp 半帧 8.45°（整帧 16.9°）；播放/暂停/拖动时钟一致；
  车轮节点矩阵随角度旋转；尾流 14 点、尾迹随速度显示；切换案例后旧数据源销毁。
* 相对帧率（同机位、同代理 A/B）：全球 4.98 vs 5.06，海面 1.6 km 3.29 vs 2.76，
  空中 300 m 3.66 vs 3.53 —— 特效对帧率无可测劣化；堆内存 +18 MB（CZML 采样）。
