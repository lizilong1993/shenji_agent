# `wargame_core` 规格说明（P0/P1 草案）

> 状态：P0 规格提取 + P1 陆战最小闭环
> 代码：`situation3d/wargame_core/`
> 性质：clean-room 实现；只依据公开规则文档、公开想定/地图数据与可观察行为；不反编译、不复制闭源代码。
> 适用范围：两个虚构演示案例的陆战裁决；海空扩展见后续版本。

## 1. 目标与非目标

**目标**

- 用纯 Python 3.8 标准库实现一个可测试的六角格陆战内核。
- 兼容公开 `scenario.json` / `basic.json` 的关键字段，能直接加载公开想定作为测试向量。
- 覆盖两案例实际需要的陆战子集：机动、通视、观察、直瞄射击、伤害/压制、弹药、目标点、胜负。
- 为后续海面/空中扩展提供统一的 `Unit/GameState/Event` 模型。

**非目标（本版不做）**

- 不追求与闭源 `train_env` bit-exact；不承诺完整复刻 1800 秒实时制的所有规则。
- 不实现间瞄、引导射击、上下车、工事、布雷、通信、疲劳、电子干扰、空中/海上裁决（接口预留）。
- 不实现多智能体对抗；`auto_orders()` 只是演示用的确定性“参谋”。

## 2. 数据契约

### 2.1 地图 `basic.json`

已核实字段：

- `map_id`, `start_elev`, `ele_grade`, `coordinate_number_length`,
  `coordinate_number_mode`, `coordinate_system`, `upper_left_longitude`,
  `upper_left_latitude`, `max_row`, `max_column`, `scale`, `map_data`.
- 公开样例：`100 x 100`，`coordinate_system = oddr`，`scale = 200`（米/格），
  `ele_grade = 10`（米/高程级）。
- `map_data[row][col]` 的单元字段：`elev/e`、`cond`、`roads/rd`、`rivers/rv`、
  `neighbors/nb`、`pos`。
- `cond`：0 开阔地、1 丛林地、2 居民地、3 松软地、4 大河流、5 路障。
- `roads`：六方向，0 无、1 黑色道路、2 红色道路、3 黄色道路。
- 邻域顺序：右、右上、左上、左、左下、右下（从东开始顺时针）。

坐标：`pos = row * 100 + col`；`cur_hex` 四位数。自研内核内部可用 `(row, col)`，
导出兼容格式时再编码。

### 2.2 想定 `scenario.json`

已核实字段：

- `scenario_id`, `time {cur_step, tick, max_time}`, `cities[]`, `operators[]`。
- `cities[]`：`{coord, value, flag, name}`；`flag = -1` 未占领、`0` 红方、`1` 蓝方。
- `operators[]`：`obj_id/color/type/sub_type/name/basic_speed/armor/carry_weapon_ids/
  remain_bullet_nums/observe_distance/move_state/cur_hex/blood/max_blood/stop/value/...`。
- `color`：0 红、1 蓝（与 situation3d 的 red/blue 适配层转换）。

算子类型：`1` 步兵、`2` 车辆、`3` 飞机、`4` 工事、`5` 战略支援。
公开想定中已见 `sub_type`：0/1 坦克/战车、3 炮兵、4 无人战车、5 无人机、
6 武装直升机、7 巡飞弹、8 运输机、9 侦察战车、10 炮兵校射雷达、
11/12 工事、13 布雷车、14 扫雷车、15 防空高炮、16 防空导弹排。

### 2.3 自研扩展字段

`Unit` 在公开字段之外增加：

- `unit_type`：`infantry | vehicle | artillery | aa | recon | helicopter | uav | aircraft | fort`
- `domain`：`land | air`（后续 `sea`）
- `weapons`：`WeaponProfile` 列表
- `ammo`：`{weapon_key: rounds}`
- `suppressed / suppressed_ticks`：压制状态与剩余 tick
- `concealed / moving / marching / stacked`：状态修正
- `path / path_index`：当前路径

## 3. 已实现规则

### 3.1 机动

- 六角 A*；`vehicle` / `infantry` / `air` 三种模式；`march` 接口保留（仅道路）。
- 车辆地形代价：开阔 1、丛林 2、居民地 3、松软 4、大河流 4（仅道路可过）、路障不可过。
- 人员：地形不减速；高差 > 60 m 降半速；可过路障。
- 坡度：`|Δelev| / ele_grade`；> 5 不可过；1 级以上按 `1 + 0.5*(level-1)` 降速。
- 道路：沿道路（方向道路类型 > 0）按 1.0 代价；`march` 仅道路。
- 每 tick 预算 = `speed_kmh * 1000/3600 * tick_seconds / scale_m`，按边代价消耗。

### 3.2 通视与观察

- 相邻格永远通视。
- 连线高程线性插值；中间格高程 + 居民地/丛林地掩蔽加成 > 连线高程则阻断。
- 高低差：高者为观察点，加一个高程级（高看低）。
- 观察距离表（格）：

| 观察\被观察 | 步兵 | 车辆 | 直升机 | 无人机 |
|---|---|---|---|---|
| 步兵 | 10 | 25 | 25 | 1 |
| 车辆 | 10 | 25 | 25 | 1 |
| 直升机 | 10 | 25 | 25 | 1 |
| 无人机 | 2 | 2 | 0 | 0 |

- 掩蔽状态：被观察距离减半，但观察者高程更高时无效。
- 目标在居民地/丛林地：观察距离减半。

### 3.3 直瞄射击

- 武器攻击等级表来自公开裁决表（大/中/小直瞄炮、速射炮、轻武器、导弹、高炮等）。
- 修正：高程差、目标地形掩蔽、掩蔽状态、目标机动、堆叠、行军、射手机动/压制。
- 裁决：`score = attack_level + (2d6 - 7) - armor`；
  - 车辆：`score >= 9` 毁伤 2，`>= 6` 毁伤 1，`>= 3` 压制，否则无效；
  - 人员：`score >= 6` 毁伤 1，`>= 3` 压制，否则无效。
- 弹药：每次射击消耗 1 发；无弹不能射击。
- 事件：`direct_fire`，带 `attack_level/effective_attack_level/modifiers/roll/result/damage/target_blood`。
- **这是近似裁决**：公开表是矩阵式结果，本版用分数映射；`wargame_core_spec` 记录差异，后续可替换 `_result()`。

### 3.4 目标点与胜负

- 夺控：目标格内有本方地面单位、且相邻 6 格无敌方地面单位；flag 更新。
- 分值：目标点 value + 存活单位 value。
- 结束：一方无存活单位；或 `tick >= max_time` 按分值判胜/平。

## 4. 测试与运行

```bash
cd /remote-home/lizilong/shenji_agent_campaign
PYTHONPATH=situation3d python3 -m unittest discover -s situation3d/tests -p 'test_wargame_core*.py' -v

python3 situation3d/scripts/run_wargame_core_demo.py \
  --scenario land_wargame_sdk/Data/Data/scenarios/1231.json \
  --map land_wargame_sdk/Data/Data/maps/map_123/basic.json \
  --ticks 20 --verbose
```

已覆盖的测试：

- `test_hexmap_alignment`（坐标/邻域/距离/连线）
- `test_wargame_core_rules`（通视、机动、观察、直瞄、目标点、压制）
- `test_wargame_core_compat`（公开想定 1231：12 单位/2 目标/100×100；5 tick 确定性）

## 5. 已知近似与差异

| 项 | 本版 | 公开规则 | 影响 |
|---|---|---|---|
| 战斗结果 | 分数映射 | 攻击等级 × 随机数矩阵 | 伤害分布可能偏平滑 |
| 压制持续 | 2 tick | 规则文档未给固定秒数（需实测） | 需在 P1 校准 |
| 堆叠 | 只标记，不阻止进入 | 同格最多 4 个本方地面单位 | 需补 |
| 行军 | 接口保留，未接入自动 orders | 道路/武器锁定/75 s 转换 | 后续 |
| 间瞄 | 未实现 | 计划/校射/散布/持续 | 后续 |
| 上下车/工事/布雷 | 未实现 | 公开规则 | 后续 |
| 空中/海上 | 未实现 | 本版非目标 | P2 |

## 6. 兼容性与许可

- 公开想定 `1231.json` 可直接加载；地图 `map_123/basic.json` 直接加载。
- 本包不 import `train_env`，不依赖 wheel；Python 3.8 标准库可运行。
- 若要对拍闭源行为，需先确认 SDK 许可允许“行为观察 + 兼容实现”；对拍只比较可观察输出，不读取二进制。
- 事件/字段命名尽量沿用公开契约（`obj_id/color/type/sub_type/cur_hex/blood/...`），便于后续把自研状态导出为 situation3d `campaign.v3`。

## 7. 下一步

1. 用 `1231.json` 等公开想定做 20–60 tick 的定性对比，记录移动耗时/通视/伤害差异。
2. 补堆叠限制、行军模式、压制时长校准。
3. 接入 `engine/campaign.py` 的阶段循环与事件模型。
4. P2：`naval.py` / `air.py`，定义海空交战矩阵。
