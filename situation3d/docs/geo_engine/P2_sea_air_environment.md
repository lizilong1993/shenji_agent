# P2 海空动力学与环境动态（2026-10-22）

> 结论先行：海空交战改为**导弹飞行段**——发射事件携带飞行时间，命中/末端拦截在到达时刻
> 结算（目标 SAM/CIWS/AAA 可拦截），`instant` 模型保留兼容；飞机增加**升降率限制与爬升掉速**，
> 不再垂直瞬移；舰船受**海况**减速；世界新增**天气时间线与昼夜**，动态修改能见度/海况/风，
> 直接作用于感知与机动。P6 envpack 运行 734 事件（323 次海空发射、320 次弹着）、15 毁伤、
> p50 84 ms；全量 289 项测试通过。

## 1. 问题（ARCHITECTURE_GAPS G9/G6 部分）

舰船/飞机共用地面式路线与固定 AGL，无爬升率/能量约束；导弹即时 Pk 结算、无飞行时间与
末端防御；天气/能见度/海况是静态常量。

## 2. 方案

### 2.1 导弹飞行段（`rules/missile.py` + `rules/engagement.py`）

- 海空武器按种类给巡航速度（反舰 1000 km/h、舰空 3000、空空 4000、鱼雷 60 等），
  `tof_s = 距离/速度`（≥5 s）；发射事件 `sea_air_fire`（`model=flight`）记录
  `result="launched"`、`planned_hit`、`intercept_pk`、`intercepted`、`tof_s`、`impact_at_s`、
  `planned_damage`，不在发射 tick 造成伤害；
- `World._pending_impacts`：到 `impact_at_s` 的 tick 结算伤害/压制，追加
  `missile_impact` 事件（hit/miss/intercepted、血量、落点）；目标已毁则丢弃；
- 末端防御：目标的 SAM/CIWS/AAA（弹药/冷却可用）按 `1−∏(1−Pk·readiness)`（上限 0.9）
  计算拦截概率——射程按**拦截弹自身包线**（`min(发射距离, 拦截弹最大射程)`）判断，
  近防炮对所有反舰弹都能参与；发射时用同一 RNG 掷骰；确定性；
- `missile_model="instant"` 保留旧即时结算（`model=instant`）。

### 2.2 飞机/舰船动力学（`sim/motion.py`）

- 飞机：高度按 `climb_rate_ms/descent_rate_ms` 向路线高度逼近（垂直速度做加速度限制），
  不再用路线高度直接覆盖位姿；确实爬升时速度上限按爬升率折减（最多 −25%，平飞不折减）；
  航线起点位于地面时直接进入航线高度（空中任务起点）；遮挡/完成分支保留实际高度；
  航线可通行性仍按计划高度（terrain+AGL）校验，实际高度只影响表现/能量，避免爬升过程中
  被地形硬阻断；
- 舰船：按所在位置天气的 `sea_state` 缩放速度（每级 −4%，下限 50%）；
- 运动学其余部分（航向/曲率/制动）沿用 P0.2。

### 2.3 动态环境（`sim/world.py`）

- 场景 `weather_base`（能见度/海况/风）+ `weather` 时间线（`at_s` 后生效）、
  `start_hour_utc` 与 `utc_offset_h`（战区时区）；**只有显式配置天气/起始时刻时**才启用
  动态天气，每 tick `_update_environment` 写入 `env.weather`（能见度 = 基础×昼夜系数：
  昼 1.0、晨昏 0.6、夜 0.2；海况/风取时间线），并记录 `world.effective_weather`；
  未配置时世界从环境自身天气播种且不回写；
- 感知 `perception._env_factor` 使用能见度（光学/红外），机动使用海况；P1.4 的烟幕
  时间窗/衰减继续生效；
- `WorldRuleContext`/CZML 不受影响。

## 3. 验收

- 新增 `tests/test_p2_environment.py` 13 项：TOF 单调、末端拦截（有/无防空、CIWS 远距
  参与）、飞行发射/弹着与延迟结算、instant 兼容、飞机爬升率/下降率/平飞不折减、海况减速、
  天气时间线/昼夜/经 step 生效、未配置天气不污染环境。
- 全量 **289 项测试通过（3 skipped）**。
- P6 envpack（默认 flight 导弹）：734 事件（海空发射 323、弹着 320）、直瞄 6、
  `traffic_stuck` 65、p50 84 ms / p95 214 ms。

### 3.1 独立评审修正（2026-10-22）

- **升降瞬移（blocker）**：垂直钳位改为按目标高度方向（误差符号）而非垂直速度符号；方向
  反转时先减速再反向，不再一 tick 吸附。
- **平飞掉速（major）**：高度误差 <0.2 m 时目标垂直速度为 0，爬升折减仅在真爬升时生效。
- **末端拦截射程（major）**：按拦截弹自身包线判断（min(发射距离, 最大射程)），CIWS/SAM
  对远距反舰弹可参与。
- **天气污染（major）**：世界从 `env.weather` 播种，只有场景显式配置天气/起始时刻才回写；
  未配置时不修改共享环境。
- **空中地形 livelock（major）**：航线可通行性恢复按计划高度校验，实际高度只做表现/能量，
  避免爬升过程被硬阻断并静默重规划。
- 其它：flight 模式强制要求 `now_s`；`weather_base`/timeline 类型与数值校验（非法值不覆盖）；
  `utc_offset_h` 战区时区；`missile_impact` 补 pk/intercept_pk/impact_at_s/model；起飞判定改
  为航线起点；文档掉速上限修正为 −25%。

## 4. 限制与后续

- 导弹为"点到点 + 飞行时间 + 概率拦截"，无比例导引、三维拦截几何、诱饵/电子战耦合；
  Pk 基础值仍来自合成矩阵（P4 数据）。
- 末端拦截只在射程/弹药/冷却上判断，不消耗拦截弹（拦截弹数量未计入）。
- 天气只有能见度/海况/风标量；风未驱动烟幕漂移、海况未影响探测/武器精度，
  昼夜为示意分段（P3 可细化）。
- 甲板作业、舰载机起降、潜艇深度机动未建模；潜艇仍走简化运动。

## 5. 复现

```bash
cd situation3d
python3 -m pytest tests/test_p2_environment.py -q
python3 scripts/run_geo_demo.py --case taiwan_landing --envpack envpacks/anping_v2c \
    --ticks 1440 --dt 30 --out /tmp/tw_p2
# 旧即时结算对照：--missile-model instant
```
