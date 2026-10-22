# P1.4 3D 感知：几何射线、传感器与航迹（2026-10-22）

> 结论先行：新增 `sim/perception.py` 作为可选物理探测模型（`detection_model="physical"`）：
> 视线为 3D 地形/真实建筑足迹/烟幕射线，传感器有距离、方位/俯仰视场与距离衰减，探测概率
> 叠加目标特征（RCS/IR/声学，来自平台数据 v1）、能见度/海况、隐蔽与运动；探测用
> seed+时间桶+双方 uid 的确定性哈希掷骰，并带 60 s 保持；探测结果融合为分方航迹（TTL 300 s）。
> 观察表/spotting 模型保留为 legacy 默认。山脊、街区、烟幕、视场、距离单调、确定性均有用例。
> 物理模式 P6 envpack 运行：直瞄 66、事件 393、p50 75 ms（与 table 路径同量级）。

## 1. 问题（ARCHITECTURE_GAPS G4）

陆上观察用 200 m 格距离表与整数减半掩体；`env.pack_env` 的 LOS 是 30 m 采样 + 固定掩体高度；
`sim/sensors.py` 为名称匹配的固定距离 × 质量因子，无视场、无探测概率曲线、无目标特征与
海况/能见度衰减。探测对几何与姿态不敏感。

## 2. 方案

### 2.1 几何（`perception.los_blocked`）

- 地形：沿视线按 ≤30 个采样点（长距≥100 m 步长）插值，视线高度低于地形+0.5 m 即遮挡；
- 建筑：`env.building_intersects`（真实足迹 + 高度，20 km 内启用）；
- 烟幕：场景 `obscurants`（圆形柱，时间窗/半径/顶高/衰减），视线在柱内且高度低于顶高时
  对光学/红外/ESM 乘衰减（默认 0.02，可配置）。
- 海/空长距使用粗步长（≤30 采样），避免 300 km 雷达对每对目标做细采样。

### 2.2 传感器与概率（`perception.detection_probability`）

- `SensorSpec`：kind（optics/ir/radar/sonar/esm）、range、方位视场（默认光学 60°、红外 45°、
  雷达/声呐/ESM 360°）、俯仰视场、p0、距离衰减尺度、安装角（相对车体朝向）、被动标志；
- 来源：实体传感器（`sim.sensors.Sensor`）优先，否则读取平台数据 v1 的 `sensors`
  （如坦克 optics 4 km、步兵 2 km、岸舰雷达 180 km）；
- 概率 = p0·exp(-d/scale)·特征·环境·烟幕·隐蔽：雷达按 RCS（√(rcs/10)，0.3–2.5），红外按 IR
  特征，声呐按声学，光学按尺寸等级；能见度不足时线性衰减；海况降低声呐；隐蔽（森林/居民地/
  静止伪装）降低光学/红外；移动目标 +10%；
- 视场：按目标方位与车体朝向+安装角比较，超出半视场即 0（雷达/声呐全向）；
- 几何遮挡 → 0（声呐不做地形/建筑遮挡）。

### 2.3 探测判定与航迹

- 每对观察者-目标按 10 s 时间桶计算一次确定性掷骰
  `blake2b(seed:bucket:observer_uid:target_uid)`，`roll < p_total` 即探测成功；
  探测后保持 60 s，避免每 tick 闪烁；同一 tick 内多传感器概率合并 `1-∏(1-p_i)`；
- `World.tracks`：按方融合为航迹（uid/位置/首末发现时间/置信度/传感器种类），300 s 未更新
  即淘汰；`World.tracks_for(side)` 返回排序后的只读列表；
- 传感器扫描选择器沿用 `enable_radio_share`（物理模式可选无线电共享）。

### 2.4 接入

- `detection_model="physical"`（`engine.detection_model` 或 `scripts/run_geo_demo.py
  --detection-model physical`）；`WorldRuleContext.can_detect` 物理模式直接读 `world.seen`，
  `_best_target`/交战沿用既有链路；
- 场景新增 `obscurants`: `{lon, lat, radius_m, top_m, base_m, from_s, to_s, attenuation}`；
- 默认仍为 `table`（legacy）：切换默认需要案例标定与对拍，P1.5 火力物理化后再评估。

## 3. 验收

- 新增 `tests/test_perception.py` 20 项：山脊遮挡/抬高越顶、街区遮挡/绕开、烟幕衰减与
  时间窗、方位/俯仰视场、距离单调与零距、特征缩放、能见度衰减、平台传感器回退与覆盖、
  seen/航迹、几何改变探测、跨世界确定性、航迹过期、保持到期不续期、舰艇海面高度基准、
  目标域门控（声呐不探空、光电/雷达不探潜）、建筑异常 fail-closed、5 km 窄山脊、
  ESM 不受烟幕影响、World.step 集成（烟幕开/关）。
- 全量 **251 项测试通过（3 skipped）**。
- P6 envpack physical 运行（1440 tick）：393 事件、海空 289、直瞄 66、间瞄 10、毁伤 17、
  下船 3、目标归属变化；p50 75 ms / p95 119 ms，117 s。
- 默认 table 路径未受影响（407 事件/直瞄 66；基准事件哈希不变）。

### 3.1 独立评审修正（2026-10-22）

- **海面高度基准（blocker）**：舰艇传感器取“海面+桅杆（20 m）”、水下取潜望镜高（10 m）、
  目标高度按域取值，修复舰船 alt 为海底高程导致的 LOS 恒阻塞；
- **目标域门控（blocker）**：声呐只探水面/水下，光电/雷达不探潜航目标；
- **保持语义（blocker）**：只有掷骰成功才续期 60 s 保持并更新航迹，held 期间不刷新位置，
  杜绝一次幸运探测永久锁定；
- **几何健壮性**：`building_intersects` 异常按遮挡处理（fail-closed）；距离 0 不再判 0；
  地形采样上限 120 点、近距 30 m，长距海/空改用雷达地平线截断（4.12·(√h₁+√h₂) km）；
- **环境因子**：ESM 不受烟幕/能见度影响；能见度 0 视为完全遮蔽；隐蔽并入场景
  `concealment` 字段；同型多传感器不再互相去重；平台记录缓存避免重复加载；
- **接入**：物理模式不再强制无线电共享（按 `enable_radio_share`）；obscurants 解析做
  数值/范围校验并与 top/base 排序。

## 4. 限制与后续

- 概率参数与特征值为演示估计；雷达已加地平线截断但未建模海杂波/多径，声呐为简化声学，
  ESM 未与干扰耦合（P2 环境动态/频谱对抗）；>20 km 且双方均非陆上时忽略地形遮蔽（由地平线主导）。
- 传感器安装角固定在车体前向或全向，尚无独立指向控制与扫描周期；航迹位置取真值（未加
  测距/测角噪声与滤波），融合为按目标 uid 的简单聚合。
- `obscurants` 为静态时间窗，未与风/扩散/战斗烟尘联动（P2）。
- 物理模式默认关闭：案例锚点/编成对探测敏感，需在 P1.5 火力物理化与校准后统一切换。

## 5. 复现

```bash
cd situation3d
python3 -m pytest tests/test_perception.py -q
python3 scripts/run_geo_demo.py --case taiwan_landing \
    --envpack envpacks/anping_v2c --detection-model physical \
    --ticks 360 --dt 30 --out /tmp/tw_phys
```
