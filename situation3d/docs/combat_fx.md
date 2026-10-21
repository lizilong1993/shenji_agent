# 交战特效（combat-fx）

2026-10-21。代码：`web/src/targetCombatFx.js`；事件生成：`tools/combat_events.py`。

## 事件契约

案例 JSON 顶层 `events[]` 新增三类事件（与既有 campaign 事件同 schema）：

```json
{"id":"CF000123","tick":226,"stage_id":"S2","type":"fire","source":"combat",
 "producer":"combat.synth.v1","actors":["B02","R10"],"weapon":"反舰导弹",
 "shooter_id":"B02","target_id":"R10","location":[120.168911,22.883489,0],
 "target_location":[120.170268,23.019797,0],"effects":{"before":{},"after":{}}}
{"id":"CF000124","tick":228,"type":"impact","result":"hit","damage":24,
 "actors":["B02","R10"],"shooter_id":"B02","target_id":"R10",
 "location":[120.170268,23.019797,0],"cause_event_ids":["CF000123"]}
{"id":"CF000125","tick":1300,"type":"destroy","actors":["B06"],
 "target_id":"B06","location":[120.174225,23.005427,20]}
```

- `fire`：时间 + 射手/目标位置 + 武器名；动画只依赖这些字段。
- `impact`：`result=hit|miss`，`damage` 为展示值；命中红闪/烟团。
- `destroy`：击毁时刻；与 `TargetStatus.at()` 的损毁时点一致。
- `combat` 顶层元数据记录 `model/event_count/disclaimer`，明确是**演示合成**。

## 合成规则（tools/combat_events.py）

- 只对交战案例（`case_id` 含 dense_transfer 的转运演示默认跳过）。
- 交战判定：敌对双方距离 ≤ 射手 `engage_km`；射手 alert≥2；按 domain 限定
  可打目标（海→海/空，防空→空，岸舰→海，陆上火炮/坦克→陆，空中→空/海/陆）。
- 射速冷却按武器（导弹 8–12 tick、炮弹 5–6 tick）；命中/未命中用 sha1 确定性。
- 击毁档位与 `targetStatus.js` 完全同口径（暴露度排序、6%/10%/22%、800 阈值），
  避免“图标已损毁但模型还在开火”。
- `python3 tools/combat_events.py --dry-run <case.json>` 先看统计；写入前自动备份
  `*.bak-combat-*`；重复运行会先删除本脚本产出的旧事件。

## 渲染（targetCombatFx.js）

- 开火：枪口闪光（billboard）+ 曳光/导弹弹体（CallbackProperty 沿 tickFloat 插值）
  + 枪口位置按型号长度/朝向表计算（模型局部前向 -X）。
- 命中：闪光/烟团 + `silhouetteColor` 红闪 260ms。
- 战损：`TargetStatus.at()` 的 HP 档位 → 模型压暗 + 三条烟柱 billboard；
  击毁：爆炸 billboard 群 + 冲击环 + 持续烟柱 + 焦黑。
- 扫描：仅**选中目标**且 `sensor_range_km>0` 时，显示以目标为圆心、半径
  `min(sensor_range,60km)` 的旋转扇区（polygon + 外圈），角度由 tickFloat 驱动，
  暂停/拖动同步；贴海/地 80m，空中目标用其高度。
- 性能：瞬时实体上限 80、烟柱 10（每柱 3 billboard）、弹体 24；60km/60km 高度裁剪。
- 不使用 `Cesium.ParticleSystem`：app 的 `viewer.clock.shouldAnimate=false`，
  ParticleSystem 不会发射；改用 billboard 回调，暂停时也一致。

## 已知边界

- 合成事件是演示，不是引擎实测交战；若引擎后续产出同 schema 事件，本模块直接消费。
- 击毁：舰船在 12 s 内下沉 7 m、飞机在 8 s 内降至地面，通过包装实体 position 实现；回拖/重置恢复原 CZML 位置。
- 已接入 `nodeTransformations` 关节：Z-10 主/尾旋翼（旋翼盘 Object_9/Object_10，已用带 pivot 补偿的旋转验证）、HQ-16 雷达、YJ-62 发射筒抬升。
- 代理炮塔（用户 2026-10-21 接受方案 2）：052C/054A/071/99A/M1A2 新增 `type_*_gunfx.glb` 代理炮塔，HQ-9 新增 `type_hq9_launcherfx.glb` 代理发射箱，作为独立 Entity 覆盖原炮位，开火时炮管后坐；不修改原模型，3 km 视距下用于表达舰炮/坦克炮动作。
- J-16 雷达罩为固定结构，没有机械关节；选中目标时用雷达扫描扇区表达探测，开火时用枪口/尾焰特效。
