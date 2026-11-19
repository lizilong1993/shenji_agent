# 可选规则块（案例显式开启）

四项都在 `sim/scenario.py::build_world` 末尾安装。**不写这些键时**，
`World.command_links` / `World.refuel` 为 `None`、`World.unit_postures` 为空字典，
`mission["formation"]` 为 `None`——所有路径与引入它们之前逐项一致
（`tests/test_golden_replay.py` 同 seed 逐键比对守住核心案例）。

## 1. 指挥链中断 `scenario["command_links"]`

```json
{
  "command_links": {
    "upper_cuts": [
      {"side": 1, "units": [12, 13], "from_s": 600, "until_s": 900,
       "note": "上级链路中断"},
      {"side": 0, "units": "all", "restore_when": {"all_alive": true}}
    ],
    "peer_cuts": [
      {"units": [21, 22], "from_s": 300, "until_s": 600}
    ]
  }
}
```

- `upper`：窗口内不给受影响单位下发**新的**指挥官命令。已下发的按原有到期规则自然失效；
  自动还击、人工命令都不受影响。
- `peer`：窗口内 `radio_share` 的这一跳不成立。共享仍是**单跳**，被遮的一跳不会绕道补上。
  一对 uid 必须同阵营，否则跳过并在 `World.command_links_warning` 里点名。
- 恢复：`until_s` 与 `restore_when.all_alive` 任一成立即恢复，先到者生效。
  两者都不写 = 窗口永不结束（解析时留一条警告）。

实现：`sim/command_links.py`、`sim/world.py::_apply_commander_orders`、`sim/spotting.py::radio_share`。
测试：`tests/test_command_links.py`。

## 2. 单位级交战姿态 `scenario["roe"]["unit_postures"]`

```json
{"roe": {"policy": "weapons_tight", "area_radius_m": 800,
         "unit_postures": {"12": "hold_fire", "15": "free"}}}
```

姿态只有三种：`free` / `hold_fire` / `hold_position`。它**只替换全场手册的 `policy` 一项**
（`free→weapons_free`，另两种→`hold_fire`），责任区半径、确认门槛、还击窗口照旧。
人工 `fire` 命令仍优先放行，禁止打击清单仍优先。

可以只写 `unit_postures` 而不写 `policy`——那时不装全场手册，只有被点名的单位受影响。
非法姿态名或不在编成内的 uid 只追加 `World.roe_warning`，不让案例加载失败。

实现：`sim/roe.py::UNIT_POSTURES/posture_policy`、`sim/world.py::_rules_for`。测试：`tests/test_unit_posture.py`。

## 3. 空中加油 `scenario["refuel"]`

```json
{"refuel": {"tanker": 31, "receiver": 32, "rate_per_s": 0.01,
            "supply": 0.5, "range_m": 300, "alt_tolerance_m": 300,
            "window_s": 600},
 "operators": [{"obj_id": 31, "name": "YY-20", "domain": "air", "fuel": 1.0},
               {"obj_id": 32, "name": "J-16", "domain": "air", "fuel": 0.4}]}
```

- 只支持**一对**，不做就近自动分配。两机须同阵营、同在空域、都未搭载。
- 开窗条件：水平距离 ≤ `range_m`、高度差 ≤ `alt_tolerance_m`、受油机未满油、余额 > 0，同时成立。
- 每步转移 `min(rate_per_s × dt, 受油缺口, supply)`；快进大步长按窗口剩余时长截断。
- 关窗（不再重开）：`window_s` 到期、任一方脱离会合条件、受油机满油、余额耗尽。
  事件 `refuel_start` / `refuel_end` 各写一条。
- **不引入通用飞行油耗**。地理引擎本来不扣油；缺油起点由 `operators[].fuel` 显式给出
  （缺省 1.0，与旧行为一致）。要「飞一段就没油」需另加油耗模型，不在本块范围。

实现：`sim/refuel.py`，挂在 `World._logistics`（`step` / `_motion_advance` / `advance_to` 共用）。
测试：`tests/test_air_refuel.py`。

## 4. 队形落点 `mission["formation"]`

```json
{"missions": [{"id": "M1", "side": 0, "units": [1, 2, 3, 4],
               "task": "move_to", "lon": 119.905, "lat": 23.105,
               "formation": {"spacing_m": 40}}]}
```

按 ⌈√n⌉ 栅格槽位、以「重心→目标」方向旋转，为每个单位算一个落点；落点仍要过
`_point_free()`，失败则整段退回旧的 `_passable_point()` 17 候选点。
`spacing_m` 缺省 40，≤ 0 视为非法并回退。`retask()` 新建的任务会保留该字段。

**与 GPRTS 原方案差一处**：它按运动层分组并做「最近单位优先」贪心；这里按 uid 顺序占槽，
适合同批出发的分队。分批抵达、需要重排的编队不在本块范围。

实现：`sim/mission.py::_slot_point / _formation_point / _approach_angle`。测试：`tests/test_formation_slots.py`。
