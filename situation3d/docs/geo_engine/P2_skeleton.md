# P2 骨架说明（2026-10-21）

> 分支：`feat/geo-engine`（基线 `wargame-baseline-6982407`）
> 范围：只搭数据契约与最小可测骨架，不含运动积分、碰撞、路径规划与 CRT 裁决（P3/P4）。

## 目录

```text
situation3d/
  geo/rule_grid.py      HexGrid / RuleGrid(200 m) / LegacyHexGrid(compat) 的底座
  geo/geodesy.py        (lon, lat) 约定的 WGS84 距离/插值（包装 engine.geo）
  compat/legacy.py      旧 row/col ↔ 经纬度；旧想定 cur_hex 映射
  compat/timebase.py    SDK frames-per-step 与秒的换算
  sim/entity.py         Pose / Route / Entity
  sim/world.py          实体注册、规则格占用、同格堆叠
  env/service.py        EnvironmentService 协议 + FlatEnvironment 桩
  rules/context.py      RuleEntityView / RuleContext / WeaponTable / Cover
  rules/weapons.py      权威 WpData 索引与攻击力查询（TarType 语义已校正）
  rules/elevation.py    8×12 高差修正（行=高差等级、列=距离；仅仰射生效）
  rules/data/*.json     从 SDK 提取的权威规则表（P0 产物）
```

依赖方向：`rules/` 只依赖 `rules/context.py`；`mobility/`（P3）依赖 `env/`；`sim/` 依赖两者；`io/` 对外。`rules/` 不 import `geo/env/sim`。

## 关键约定

- **RuleGrid 固定 200 m**，提供规则距离；`LegacyHexGrid` 按源图尺度只做旧坐标映射。两者不混用。
- **规则距离 = 立方坐标距离**。相邻格中心 200 m；距离 2 的锯齿格中心相距 346 m，欧氏距离除以 200 会算成 1。
- **TarType**：0 间瞄 / 1 对人员 / 2 对车辆 / 3 对空。对人员表的 `ObjBlood` 是**攻方车/班数**（1–5），不是目标血量；旧字典 1↔2 写反，均已由黑盒实测确认。
- **高差**：`elevation_modifier` 仅在仰射（攻方更低）时按 `[Δh等级-1][距离-1]` 查表，平射/俯射为 0；距离 >12 为 0。
- `WeaponTable` 要求长度 = `att_max + 1`，拒绝截断表；越界返回 `None`（不是 0）。
- 旧想定缺 `grid` 元数据时 `LegacyHexGrid.from_map_doc` 直接报错，不用 bbox 反推。

## 本次新增测试

```bash
cd situation3d
python3 -m unittest tests/test_geo_engine_skeleton.py   # 17 项
python3 -m unittest tests/test_wargame_core_timebase.py # 4 项
GEORULES_GOLDEN=1 python3 -m unittest tests/test_golden_replay.py # 2 项
```

覆盖：200 m 格尺度、六角↔经纬度往返 <1 m、200 m/1.5 km 双网格分离、旧想定坐标映射、权威表完整性、wp36 对车辆逐格值、越界与缺失组合、高差表验证点、同格堆叠、tick>1 现状锁定、300 tick 逐位回放。

## 下一步（P3 入口）

1. `mobility/` 通行谓词与路网/越野规划器，消费 `env.EnvironmentService`；
2. `sim/motion` 连续弧长积分 + 子步碰撞；
3. 规则格在“进入新格”时校验堆叠/桥梁容量（`World` 已提供占用查询）；
4. `rules/combat.py` 迁移时用 `RuleContext.rule_distance`，禁止直接 import `geo`；
5. 补 F0/F1/F2 制导标志探针；生成 `map_9001` 后补录台海 golden replay。
