# 目标动态特效 P2（第一批）

2026-10-26 · 分支 `feat/target-models-v3` · 工作树 `/remote-home/lizilong/shenji_agent_models`

在 P0/P1 资产基础上补充四类与运动状态绑定的特效；不改数据、不重启 8848/8858。

## 1. 已实现

### 1.1 舰艏浪

`targetFx.js` 在原有舰尾流之外，为每艘水面舰新增一条 `wake-<id>-bow` 折线：

- 回溯 CZML 当前时刻前 4 段位置，沿航向前移 `0.45×舰长`，略高于水面 1.2 m；
- 速度 >4 kn、相机高度 <50 km、距离 <60 km 时显示；
- 与尾流共用时间源，暂停/拖动同步；每条船只多 1 个 primitive。

### 1.2 地面扬尘

`targetFx.js` 为每个地面目标新增一个 `dust-<id>` billboard：

- 位置在目标后方 `0.35×车长`，贴地 +0.6 m；
- 速度 >2 kn、相机高度 <30 km、距离 <25 km 且正在播放时显示；
- Billboard.scale 按像素倍率 0.12→0.30（约 8–19 px），使用 canvas 沙尘贴图；**不是米制**，避免 12 km 视距整屏棕雾；
- 80 目标场景最多 80 个 billboard，无逐粒子开销。

### 1.3 高空凝结尾迹

`targetFx.js` 在原有尾气折线之外新增 `exhaust-<id>-contrail`：

- 高度 >7000 m 且速度 >50 kn 时显示；
- 宽 20 m、低透明度白色发光折线，复用同一份 CZML 回溯位置；
- 低空仍显示原有尾气，不重复叠加。

### 1.4 VLS 燃气羽流

`targetCombatFx.js` 新增 `spawnVlsPlume()`：

- 052C/052D/054A/055 等带 VLS 的舰发射 `导弹/反舰/火箭` 时，在炮口/发射位置沿地表法线叠加 4 团火光/烟团（0–16.5 m）；071 等无 VLS 平台不触发；
- 复用现有瞬时实体池（上限 80），900–1260 ms 自动回收；
- 不依赖 ParticleSystem，暂停/拖动时间轴时与现有特效一致。


### 1.5 通信链路（数据链）

`targetCombatFx.js` 新增选中目标的数据链弧线：

- 选中目标后，取同阵营、距离在 `min(influence_km,30 km)` 内最近的 6 个友军；
- 用 `PolylineDashMaterialProperty` 画低亮度青色虚线，端点由 `currentGeo()` 每帧回调，目标移动时链路跟随；
- 选中目标变化或每 10 tick 重建拓扑；每帧只更新 6 条链路的端点；过滤 <50 m 的零长度链路；无选中/无友军时不建实体；
- `TargetCombatFx.debug().comms` 可查当前链路数。

## 2. 验证

- `node --check targetFx.js`、`node --check targetCombatFx.js` 通过。
- Cesium 1.145 + headless Chromium 加载 `taiwan_dense_transfer_v1`（`autoplay=1`，tick=300）：
  - 默认 12 km 相机、无 `panel` 参数：`TargetFx.debug()` 中 12 艘水面舰 `show=true` 且 `bow=true`；8 架飞机 `show=true`；60 个地面目标中 54 个移动目标 `dust=true`，6 个静止目标（TRV023–028）`false`（符合门控）。
  - 页面无 JS `pageerror`；控制台 404 为 8859 工作树缺少 `web/tiles` 离线瓦片，与特效/模型无关。
  - 用 `?panel=TRS001` 选中时相机拉到 120 km，四类运动特效会被高度门控关掉；此时 `TargetCombatFx.debug().comms = 6`、扫描扇区 `range=60000`，无页面错误。选中后链路每 10 tick 重建，时间轴跳转不会保留过期拓扑。
- 凝结尾迹在 tick=300 未触发（该案例飞机高度 <7000 m），逻辑已就位；VLS 在当前两个案例中没有海基导弹事件，未做端到端触发验证。

## 3. 未完成

- **轮系/履带**：HQ-9 的 `$wheel0..7` 节点共用一个远离轮心的原点，直接旋转会绕世界原点甩出；需要 Blender 重导出轮心节点或扩展 pivot 补偿。其余轮式模型轮子烘焙在车身，无法单独旋转。
- **雷达旋转/炮塔/旋翼**：Z-10/HQ-16/YJ-62 已有；055、075、AH-64、HIMARS 等新资产尚未拆关节。
- **加力/起落架**：未做；现有尾气/凝结尾迹先覆盖高速飞行。

## 4. 下一批

1. 轮系：从备份源文件给 ZTD-05/EQ240/HIMARS/HQ-9 拆 `wheel_*` 节点并登记 pivot；
2. 055/075 的雷达/炮塔节点；AH-64 旋翼；
3. LOD1（1.2–1.5 km）与实例化，真机 GPU 验收。
