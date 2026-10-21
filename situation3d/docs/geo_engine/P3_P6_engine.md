# P3–P6 引擎核心实现与验证（2026-10-21）

> 分支：`feat/geo-engine`（基线 `wargame-baseline-6982407`）
> 本文件对应方案 §4/§5 的 P3–P6。P1 的完整 OSM/DEM/GEBCO 数据管线尚未实现，当前用
> `basic.json` 网格环境与 fallback 环境包跑通全链路（限制见 §6）。

## 1. 模块

```text
env/grid.py        MapGridEnvironment：basic.json 高程/cond/道路/河流 → 环境服务；格线 LOS
env/pack.py        msgeo.envpack.v1 manifest 校验与加载
mobility/profile.py     平台属性（domain/速度/地形系数/坡度/涉水/吃水/净空）
mobility/passability.py 硬谓词 + 软速度系数
mobility/planner.py     源图网格 A*（地面/两栖/水面/水下/空中）→ Route
sim/entity.py      Pose/Route/Entity/EntityRuleView（属性式视图）
sim/motion.py      连续弧长积分、加减速、50 m 子步碰撞、阻塞停驶
sim/world.py       固定步长主循环、订单、传感器、交战、堆叠、夺控、补给、上下船
sim/context.py     WorldRuleContext（规则层唯一输入）
sim/sensors.py     雷达/IR/声呐/ESM（合成参数）
sim/scenario.py    想定+地图加载、单位/武器/传感器装配、上船判定
rules/crt.py       权威 CRT 与战损修正查表（车辆/人员）
rules/combat.py    直瞄裁决（武器表→高差→遮蔽→2d6→CRT→修正）
rules/detection.py 观察表 + 掩蔽/地形减半 + 域外传感器分流
rules/engagement.py 海空交战矩阵与 Pk（参数沿用 MVP，标注近似）
rules/elevation.py 8×12 高差表（P0 已确认语义）
geoio/observation.py 席位过滤观测（geo + 可选 legacy hex）
geoio/actions.py    move/stop/fire + 旧 move_path 适配
geoio/czml.py       CZML 快照（位置/颜色/路线 polyline）
scripts/run_geo_demo.py  场景/案例运行 + CZML/观测/摘要
scripts/build_envpack.py basic.json → fallback envpack
scripts/bench_geo_engine.py 300/500 实体基准
```

依赖方向保持：`rules/` 只 import `rules/*` 与 stdlib；`mobility/` 依赖 `env/`；`sim/` 组合二者；`geoio/` 对外。`io/` 因与 Python 标准库冲突，包名改为 `geoio/`。

## 2. 规则实现要点

- 直瞄攻击等级：`WeaponTables[(WeaponID, TarType, ObjBlood)]` + 高差修正（仅仰射）+ 遮蔽（居民地 −1、丛林 −2），攻击力可为负。
- 骰点：攻击 `random1` 与战损 `random2` 均为 2d6；引擎 RNG 由 `seed` 固定，保证可复现。
- 车辆 CRT：先用（攻方车/班数，攻击等级）映射列，再用 `random1` 取原始战损；`random2` + 状态修正查装甲修正表，最终战损 = 原始 + 修正（<0 记 −1）。
- 人员 CRT：`random1`×攻击等级查表，`压` 为压制；修正表阈值 `≤0 / 1–7 / ≥8`。
- 装甲映射：`0 无装甲 / 1 轻型 / 2 中型 / 3 重型 / 4 复合`；已用 P0 探针的 200 条样本验证修正值。
- 观察：陆地按公开观察表（10/25/25，无人机 2）；海/空/水下改用雷达/IR/声呐/ESM 传感器（合成参数），掩蔽减半仅对车辆目标保留高地例外。
- 交战：海空矩阵与 Pk 沿用 MVP；距离改为米、冷却改为秒。
- 离散规则：规则格（200 m）判定堆叠 ≤4 与夺控（目标格 + 6 邻格无敌方地面单位）；同格堆叠是运动约束。

## 3. 运动与碰撞

- 位置是 `(lon, lat, alt)`；Route 由 A* 折线 + 分段限速构成；积分按弧长推进，加速度/减速度受限。
- 每步按 `v·dt / 50 m` 子步检查 `passable`（水深/坡度/建筑/净空/吃水），阻塞即停驶并重规划。
- 空中单位沿 Route 高度（地形 + AGL）飞行，检查地形净空与升限。
- 上下船：水面起步的陆军单位挂到同格友方舰船（`on_board`）；舰船进入可登陆格停稳后 150 m 内陆点卸载；运输舰被击毁则乘客损失。

## 4. 运行结果

公开想定 1231（12 单位，map_123）：

```text
ticks 300: p50 9.6 ms / p95 43 ms；直瞄射击 6–13 次，毁伤 1–2
```

台海案例（33 单位，map_9001，dt=30 s，1440 tick = 12 h）：

```text
ticks 1440, elapsed 20.8 s, p50 12.1 ms / p95 31.7 ms
sea_air_fire 326, destroyed 12, alive 21/33
CZML 22 packets, observation 12 KB
```

300 实体基准（60×60 格）：`p50 52 ms / p95 207 ms`；100 实体：`p50 8 ms / p95 17 ms`；500 实体：`p50 34 ms / p95 649 ms`（修复前后量级相近）。P50 低于 100 ms，P95 未达标，峰值出现在齐射/传感器扫描 tick；后续用 NumPy 空间索引/批量 LOS 优化（计划 §4.13 的优化顺序）。

## 5. 测试

- `tests/test_geo_engine_core.py`（21 项）：环境、通行、规划、运动、CRT、直瞄事件、堆叠、夺控、观测过滤、CZML、envpack、旧动作、300 tick fuzz 不变量。
- `tests/test_geo_engine_skeleton.py`（21 项）、`test_wargame_core_timebase.py`（4 项）、golden 回放（2 + 3 项）。
- 全量 `unittest discover`：125 项（含 golden 回放时 124 项执行 + 1 项可选 TW 重放）。

## 6. 已知限制（后续工作）

1. **P1 数据**：已完成在线（Overpass/Terrarium）/离线（共享缓存）双路径构建与运行时 `EnvPackEnvironment`，详见 `P1_data_pipeline.md`；当前 OSM 在线链路不可达、图层为 JSON 而非 COG/FlatGeobuf，水深仍为演示值。
2. **规则空白**：间瞄计划/校射/散布、雷场/路障/工事、导弹飞行段、制导/引导射击、压制恢复细化尚未实现；间接火力武器只有表数据。
3. **桥梁/隧道/航道宽度**：basic.json 无承载/净空/宽度字段，通行谓词未覆盖。
4. **传感器/海空武器**：参数为 MVP 合成值，未标定真实型号。
5. **性能**：300 实体 p95 134 ms，未达 100 ms 目标；需要空间索引与向量化。

以上限制与方案 §5.7 验收标准的差距已在 §4/§5 量化；P3/P4/P5 的引擎骨架、规则表对拍与 P6 台海回放已跑通。
