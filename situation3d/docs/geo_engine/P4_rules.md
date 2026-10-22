# P4 规则补齐：桥隧 / 雷场 / 间瞄（2026-10-22）

> 基线：`feat/geo-engine`，提交 `eb20671`（桥隧）、`35165c2`（雷场/障碍）、`efb1a7f`（间瞄）；
> 全量 184 项测试通过（3 skipped）。本文记录语义、数据来源、验收与偏差。

## 1. 桥隧通行（`eb20671`）

- 数据：PBF/Overpass 归一化已保留 `bridge`、`tunnel`、`maxheight`、`maxweight`、`width`，
  并写入道路图每条 edge（P1 管线）。
- 平台参数：`PlatformProfile` 新增 `weight_t`、`vehicle_height_m`；`beam_m` 同时作为车辆宽度。
  现有演示值：履带 40 t / 2.8 m / 3.5 m，轮式 12 t / 2.2 m / 2.4 m，炮兵 15 t / 2.5 m / 2.5 m，
  两栖 25 t / 3.0 m / 3.2 m（演示估计，非实装数据）。
- 判定：`mobility.passability.passable` 通过 `env.road_edge_at(lon, lat, 25 m)` 取最近道路边，
  按 `maxweight ≥ weight_t`、`maxheight ≥ vehicle_height_m`、`width ≥ beam_m` 过滤；超限不可通行。
- `EnvPackEnvironment.road_edge_at` 用道路空间索引 + 点到线段距离（局部平面近似）实现。
- 规划：`GridPlanner` 只依赖 `passable`，因此 A* 自动绕开不可通过的桥隧；运动子步也复检。

## 2. 雷场 / 路障 / 工事（`35165c2`）

- 雷场表：`rules/data/minefield.json`（`msgeo.minefield.v1`）；`rules/minefield.py` 解析
  人员/车辆 2d6 行，以及按装甲类别（复合/重型/中型/轻型/无装甲）的战损修正表。
- 触发：实体进入雷场多边形时按 2d6 掷一次，伤害 = 基础值 + 装甲修正（≥0）；同一实体同一雷场只掷一次
  （`World._mine_hits`），事件 `mine_strike` 记录掷骰、基础/修正/总伤害。雷场不阻断通行。
- 场景格式（`sim/scenario.py`）新增：
  ```json
  "minefields": [{"name": "MF-1", "kind": "临时雷场", "owner": -1,
                  "ring": [[lon, lat], ...] }],          // 或 {"center":[lon,lat],"radius_m":300}
  "obstacles":  [{"name": "RB-1", "kind": "roadblock", "ring": [...] }]
  ```
  `owner` 与实体同方时不触发。
- 障碍：`roadblock` 阻断所有地面单位；`fortification` 只阻断车辆/两栖（步兵可过）。
  判定在 `World._can_enter_cell`，并通过 `GridPlanner.plan(..., blocked=...)` 参与寻路，避免“撞墙后原地重规划”。
- 简化与未实现：工事不提供掩体加成、无工程兵破障/扫雷、无布雷动作（仅场景预置）。

## 3. 间瞄 / 校射 / 散布（`efb1a7f`）

- 表：`rules/data/crt.json` 的 `无校射火力散布裁决` / `格内校射火力散布裁决`；距离分档
  0–20 / 21–40 / 41–60 / 61+（原表首列为 Excel 日期序列 46042，按 0–20 档解释），随机数 2–12；
  单元格语义：`命中` / `散布`（=1 格）/ `散布N格`。
- `rules/indirect_fire.py`：
  - `ScatterTables.lookup(observed, distance, roll)`；计划射击用无校射表，有观察/校射用格内校射表；
  - 未命中时从瞄准格出发随机游走 N 个相邻规则格（六角邻居），得到落点；
  - 落点格内**敌方**地面实体逐一用火炮 `TarType=0` 攻击等级表 + 共享 `resolve_crt_shot`
    （直瞄/间瞄共用 CRT 与修正）裁决；弹药/冷却每次射击只扣一次；
  - 事件 `indirect_fire`：瞄准格、落点格、掷骰、`hit/scatter`、散布格数、攻击等级、各目标子结果与总伤害。
- `rules/combat.py` 抽出 `resolve_crt_shot(..., consume=)`，直瞄原行为不变（golden replay 仍通过）。
- 世界接入：
  - 目标选择 `_best_target` 在无直瞄武器时用 `_indirect_can_reach` 判断间瞄射程；
  - `_engage`：地面目标且无直瞄解 → 自动间瞄（能探测则按校射表）；无目标但带 `aim_cell` 时按计划射击；
  - 动作 `{"type": "fire_indirect", "entity": id, "target": {"lon", "lat"}}`（`geoio.actions`）。
- 语义偏差：原版散布方向/落点分布未公开，本实现为随机游走；未实现射弹飞行时间、持续火力计划、
  压制恢复细化；`WpData` 的 `F0/F1/F2` 字段未参与裁决（P0 已确认不影响直瞄）；导弹飞行段/制导仍未实现。

## 4. 验收

| 项 | 结果 |
|---|---|
| 权威表逐项一致 | 散布表逐格解析单测（命中/散布/散布N格、四档距离）✓；雷场表 2d6 分布与装甲修正 ✓ |
| 单发分布 | 间瞄命中率、雷场伤害分布在 12 万次采样下与表推概率偏差 < 1 pp ✓（非正式 TOST） |
| 场景对拍 | 无 SDK 端不可对拍；用 golden replay（直瞄/海空）回归确认未改变既有裁决 ✓ |
| 单测 + fuzz | 新增 17 项 P4 单测；间瞄 2000 次随机射击无非法状态 ✓；全量 184 项通过 |
| 桥隧 | 超重/超高/超宽不可通行、寻路绕行单测 ✓ |

## 5. 已知限制

1. 桥隧限制目前作用于“距道路边 25 m 内”的位置；越野跨越河道仍靠通行谓词，不区分桥面/水面。
2. 雷场为场景预置数据，尚无布雷/扫雷工兵动作与覆盖度参数。
3. 间瞄未接入补给消耗与持续射速；一次决策一门炮一发。
4. 海空/导弹制导（飞行段、F0/F1/F2）仍为后续项。
