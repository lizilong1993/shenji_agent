# P6 案例：台海陆战贯通与中印西段（2026-10-22）

> 提交见 git log；全量 191 项测试通过（3 skipped）。

## 1. 台海联合登陆（`taiwan_landing`）

### 1.1 本次改动

- **锚点校正**：原 hex 锚点与 `map_9001` 实际海岸线不符（滩头落在海里）。按 100×100 / 2 km
  地图重新选取：滩头 5688、换乘海域 5788、外海 5784/5785、台南 6091、反突击出发地 6291、
  高雄 8294、海峡 2139、澎湖 2358 等。
- **两栖下船链路**：`1030–1032` 改为 amphibious 域，开局以乘客身份搭乘登陆舰（`sim/scenario.py`
  对 amphibious 深水单位自动登船）；登陆舰选择滩头 8 格内最近的**水面格**作为泊位（BFS），
  到达后 `World._sync_passengers` 在 2.2 km 内寻找可通行登陆点并下船，同时清空旧路线/目标。
- **分阶段推进**：`run_geo_demo._assign_orders` 读取实时目标归属；到达单位自动接续下一目标；
  同格偏移缩小到 0.0004°（≤200 m），保证夺取仍落在目标规则格；每 30 tick 复查一次。

### 1.2 结果

| 运行 | 事件 | 目标归属 | 步进 |
|---|---|---|---|
| 基础地图（无 envpack），1440 tick × 30 s | 349：海上/空中 234、**直瞄 82**、下船 3、目标 15、毁伤 15 | -1 / 0 / 1 | p50 148 ms |
| Anping envpack（`anping_v2c`），1440 tick × 30 s | 252：海上/空中 239、下船 3、目标 1、毁伤 9 | -1 / 0 / 1 | p50 574 ms（负载≈45；负载≈62 复测 1168 ms） |

基础地图运行满足“陆海空全链路有事件、目标归属变化”：登陆舰→下船→滩头夺控（OBJ-BEACH 转红）→
向内陆推进，蓝方装甲从 6291 前出反击并与红方地面交火（82 次直瞄）。

envpack 运行完成了下船与滩头夺控，但**没有地面直瞄**：规划网格是 2 km 地图格，而
`EnvPackEnvironment` 的水面/建筑是亚单元 OSM 数据，路面航线落在潟湖/水面上被 50 m 子步判为
不可通行，双方在安平城区附近反复“规划-阻塞”；城市建筑又遮蔽了 LOS。属于已知一致性缺陷。

## 2. 中印西段（`india_china_west`）

- 地图：`map_9002`，中心 79.1E/33.75N，100×100 / 4 km，DEM 为 Terrarium z9（本地瓦片），
  陆地为 Natural Earth；全部为虚构演示数据。
- 想定：红蓝各 10 个单位（坦克/战车/步兵/炮兵/防空/无人机/战斗机），目标为山口、高地、谷地、
  补给站与后方基地；四阶段：侦察对峙 → 夺取山口 → 高原反击 → 巩固/结局。
- 回放（1440 tick × 30 s）：事件 14（海空 8、直瞄 2、目标 2、毁伤 2），目标归属 -1/0/1，
  CZML 输出 18 个实体轨迹包 + document，`observation.json` 可读；步进 p50 2.2 ms。

复现：

```bash
# 地图（需要本地 z9 Terrarium 瓦片；瓦片在 web/tiles/terrain，未入库）
python3 scripts/build_campaign_map.py --profile india_china_west \
    --out web/data/campaign_maps/map_9002 --no-osm

python3 scripts/run_geo_demo.py --case india_china_west --ticks 1440 --dt 30 \
    --out /tmp/in_p6

python3 scripts/run_geo_demo.py --case taiwan_landing --ticks 1440 --dt 30 \
    --out /tmp/tw_p6_basic
# envpack 版（数据链验证）：
python3 scripts/run_geo_demo.py --case taiwan_landing --envpack envpacks/anping_v2c \
    --ticks 1440 --dt 30 --out /tmp/tw_p6_envpack
```

## 3. 验收与限制

- **2026-10-22 P0.1 更新**：envpack 运行改用 30 m 导航场（`mobility/nav_field.py`），
  "规划-阻塞"循环消失，地面直瞄 0→23、事件 252→338、毁伤 9→14，步进 p50 574→126 ms
  （1440 tick，seed 20261021）。本节以下为改前记录，保留用于对照。详见 `P0_1_nav_field.md`。
- **2026-10-22 P0.2 更新**：运动学与 OBB 碰撞落地后，envpack 运行地面直瞄 23→66、毁伤 18、
  事件 407、p50 81 ms；1440 tick 不变量探针穿模/瞬移/航向跳变均为 0。详见
  `P0_2_kinematics_collision.md`。
- 台海：基础地图全链路（陆/海/空事件 + 归属变化）✓；envpack 版下船+夺控 ✓，地面交火被
  规划/水面一致性与城市 LOS 阻断（降级记录）。
- 中印：想定可跑通并输出可回放 CZML ✓；事件数量偏少（接触线稀疏），后续可增加巡逻触发。
- 性能：envpack 全场景 p50 574–1168 ms（33 实体，随宿主负载波动）；本任务未继续优化，详见 PERFORMANCE.md。
- 复核修正：巡逻舰出生点改为近岸水域（原锚点在陆地）；目标格内偏移缩小到 ≤40 m；
  LOS 建筑高度早退在 envpack 缺少 terrain 层时自动禁用。
- 后续：envpack 规划网格细化（200 m）或水面/建筑一致性；城市环境下的观察规则（街巷通视）需另立规则。
