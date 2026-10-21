# 目标角色卡数据包 v2（大类 / 小类）

本目录是按"大类角色卡 + 小类角色卡 + unknown 兜底"组织的第一版默认数据，供 `situation3d` 主工作树与 `feat/geo-engine` 工作树共用。数据为**演示/校核用默认值**，不是实装情报；每个小类都标注了置信度与来源。

## 1. 文件结构

| 文件 | 内容 |
|---|---|
| `classes.json` | 8 个大类角色卡：字段契约、共性配置、允许的装备类别、默认未知小类 |
| `subclasses/land_infantry.json` | 人员大类下的 6 个小类（含 unknown） |
| `subclasses/land_vehicle.json` | 车辆大类下的 19 个小类（含 unknown） |
| `subclasses/land_installation.json` | 工事/固定阵地大类下的 6 个小类（含 unknown） |
| `subclasses/sea_surface.json` | 水面舰艇大类下的 10 个小类（含 unknown） |
| `subclasses/sea_subsurface.json` | 潜艇大类下的 3 个小类（含 unknown） |
| `subclasses/air_aircraft.json` | 飞机大类下的 9 个小类（含 unknown） |
| `subclasses/air_helicopter.json` | 直升机大类下的 4 个小类（含 unknown） |
| `subclasses/air_uav.json` | 无人机大类下的 4 个小类（含 unknown） |
| `equipment.json` | 装备目录：40 种武器、32 种传感器、13 种通信设备 |
| `types.json` | 类型索引、旧军标图标别名、庙算 `type/sub_type` 映射 |
| `数值总表.md` | 由 JSON 自动生成的速查表 |

合计 **8 个大类、61 个小类**（其中 8 个为 `unknown` 兜底）。

## 2. 大类 / 小类 / unknown 规则

### 2.1 命名

```
大类：land.vehicle
小类：land.vehicle.mbt
未知：land.vehicle.unknown
```

- 大类是稳定的分类学节点，负责**字段契约**：哪些字段必须由小类给出、允许哪些装备类别、符号族、庙算 `type`。
- **大类卡不写随小类变化的数值**。例如 `land.vehicle` 不设置 `max_speed_ms`；坦克的 80 km/h 与运输车的 90 km/h 都由各自小类卡定义。`classes.json` 里的 `value_policy` 字段明确记录这一约束，校验器应拒绝大类卡出现 `motion.max_speed_ms`。
- 小类是能力的实际载体：速度包线、地形因子、装甲/血量、传感器、武器、载员、保障参数。
- `unknown` 是每个大类的强制兜底小类。当只能判定到"车辆/水面舰艇/飞机"而无法细分时，实例引用 `<class>.unknown`；它的 `confidence = "unknown"`、`unknown_policy.engage = false`，行为按保守下限约束（不主动交战、速度取低值、无武器）。

### 2.2 合并顺序

```
大类 shared（域、运动模式、默认 ROE、保障类别）
  < 小类 values（具体数值与装备）
  < 型号模板（可选，如同一小类下的 99A / M1A2）
  < 实例 overrides（部署参数与有限覆盖）
```

小类卡里的 `values` 是**完整快照**（便于直接阅读与单卡校验）；加载时大类 `shared` 作为默认，小类同名字段覆盖，数组整体替换。不要在加载器里假设 `values` 只含差异字段。

### 2.3 庙算类型映射

`types.json.sdk_map` 来自对庙算公开想定（`land_wargame_sdk/Data/Data/scenarios`，抽样 50 个想定、聚合 21 种算子）的实测：

| 庙算 type | 庙算 sub_type | 算子名称示例 | 映射到小类 |
|---|---|---|---|
| 1 步兵 | 2 | 步兵小队 / 步兵 | `land.infantry.rifle` |
| 1 步兵 | 16 | 防空导弹排 | `land.infantry.manpads` |
| 2 车辆 | 0 | 重型坦克 / 中型坦克 | `land.vehicle.mbt` |
| 2 车辆 | 1 | 重型战车 / 中型战车 / 轻型战车 | `land.vehicle.ifv` |
| 2 车辆 | 3 | 炮兵 | `land.vehicle.spg` |
| 2 车辆 | 4 | 无人战车 | `land.vehicle.ugv` |
| 2 车辆 | 9 | 侦察型战车 | `land.vehicle.recon` |
| 2 车辆 | 10 | 炮兵校射雷达 | `land.vehicle.radar` |
| 2 车辆 | 13/14 | 火箭布雷车 / 扫雷车 | `land.vehicle.engineer` |
| 2 车辆 | 15 | 防空高炮 | `land.vehicle.aaa` |
| 3 飞机 | 5 | 无人机 | `air.uav.tactical` |
| 3 飞机 | 6 | 武装直升机 | `air.helicopter.attack` |
| 3 飞机 | 7 | 巡飞弹 | `air.uav.loitering` |
| 3 飞机 | 8 | 运输机 | `air.aircraft.transport` |
| 4 工事 | 11 | 人员工事 | `land.installation.personnel_fort` |
| 4 工事 | 12 | 车辆工事 | `land.installation.vehicle_fort` |

未在样本中出现的 sub_type（如固定翼战斗机、舰艇）由 `type + name` 或显式 `subclass_id` 指定；名称启发式只允许出现在导入适配层，并记录 `source.kind = "derived"`。

`sdk_map` 是单值默认映射，存在多对一（例如 `2/1` 可能是 ifv/apc/truck/amphibious，`2/3` 可能是 spg/mlrs/towed/coastal）。导入时应先取默认，再按名称/role 或显式 `subclass_id` 细化；无法细化时使用该类 `unknown`，不要把默认映射当成确定身份。

模型字段说明：`reference_model` 是真实型号；`identity.model.asset` 是可渲染资产 ID。两者不一致时 `asset_approximation = true`，表示当前用同族/近似资产占位（例如 052D 暂用 052C 资产），资产为 `null` 时前端回退到军标图标。正式资产到位后替换 `asset` 并清除该标记即可。

## 3. 数值口径

每张小类卡同时保留两套口径，禁止混用：

1. `sdk` 块：庙算平台的实际字段（`type`、`sub_type`、`basic_speed_kmh`、`armor`、`max_blood`、`weapon_ids`、`observe_distance`；视想定样本而定，部分卡可能缺省 `weapon_ids`/`observe_distance`）。它用于与庙算裁决表、观察表、`WpData` 对齐，保证兼容层行为一致。
2. `values` 块：面向本系统连续仿真与可视化的物理值（SI 单位）。例如庙算所有车辆 `basic_speed = 36 km/h`，但物理上坦克 80 km/h、运输车 90 km/h，角色卡两者都记录、分场景使用。

具体约定：

- 速度：`max_speed_ms` / `cruise_speed_ms` 为内部单位；总表按域换算为 km/h 或节。
- 武器射程：`equipment.json` 的 `range_km` 为物理射程；`sdk.wpdata_id` / `sdk.tar_type` 指向庙算 `WpData` 行（1 格 = 200 m），裁决时按庙算表，可视化与连续运动按物理射程。
- 装甲：`armor_class ∈ {0,1,2,3,4}`，与 `rules/crt.py` 的"无/轻/中/重/复合装甲"一致。
- 血量：`max_blood` 对应庙算 `max_blood`（车辆/步兵算子数量口径），不是 HP 百分比。
- 观察：庙算 `observe_distance` 以"格"为单位，角色卡的传感器射程以 km 为单位；适配层按地图格宽换算。

补充两条语义：

- `sdk.weapon_ids` 保留庙算想定样本的原值（如步枪 5、两栖突击车 69、无人战车 74），**不保证与 `equipment.json` 一一对应**；供本系统裁决的映射以 `equipment.sdk.wpdata_id` / `sdk.tar_type` 为准。
- `shoot_on_move=false` 的语义是"射击时需停车/展开/上浮"，而不是"平台移动时绝对不能开火"；当前仅炮兵、反坦克小组、机动发射车（HQ-9B TEL、YJ-62 TEL）和弹道导弹保留该限制。

### 庙算实测样例（50 个想定聚合）

| 算子 | type/sub | basic_speed | armor | max_blood | 携带武器 ID | 观察距离（格） |
|---|---|---|---|---|---|---|
| 重型坦克 | 2/0 | 36 | 4 | 3–4 | 36,37,43,84 | 1/10/25 |
| 重型战车 | 2/1 | 36 | 2–3 | 3 | 4,43,54,56,69,73 | 1/10/25 |
| 中型战车 | 2/1 | 36 | 2 | 4 | 4,43,56,73 | 1/10/25 |
| 无人战车 | 2/4 | 36 | 0–2 | 3–4 | 4,43,56,74 | 0/10/25 |
| 炮兵 | 2/3 | 36 | 2 | 3 | 72 | 1/10/25 |
| 防空高炮 | 2/15 | 36 | 2 | 3 | 1 | 1/10/25 |
| 防空导弹排 | 1/16 | 5 | 0 | 3 | 2 | 1/10/25 |
| 步兵小队 | 1/2 | 5 | 0 | 3–4 | 29,35,5,71 | 1/10/25 |
| 武装直升机 | 3/6 | 200 | 2 | 1 | 43,56,73 | 1/10/25 |
| 无人机 | 3/5 | 100 | 0 | 1 | — | 0/2 |
| 巡飞弹 | 3/7 | 100 | 0 | 4 | 76 | 0/2 |
| 运输机 | 3/8 | 200 | 0 | 3 | — | 1/10/25 |
| 人员工事 | 4/11 | 0 | 2 | 5 | — | 1/10/25 |
| 车辆工事 | 4/12 | 0 | 2 | 5 | — | 1/10/25 |

## 4. 来源

### 4.1 庙算平台（本地实测，非网络）

- `zm:scenario_operators`：`land_wargame_sdk/Data/Data/scenarios/*.json`，抽样 50 个想定聚合；字段见 `docs/wargame_ref/observation_structure.txt`（算子 `type/sub_type/basic_speed/armor/A1/carry_weapon_ids/observe_distance/max_passenger_nums` 等）。
- `zm:weapons_0`：`situation3d/rules/data/weapons_0.json`（来源标注 `Data/weapons/weapons_0.xlsx`），武器 ID、`tar_type`、`att_min/att_max`（格）。
- `zm:observation`：`situation3d/rules/data/observation.json`，观察规则表（步兵/车辆/直升机/无人机）。
- `zm:wpdata`：`rules/data/wpdata.json`，80 行 `(weapon_id, tar_type, obj_blood)` 裁决表。

### 4.2 网络公开资料（2026-10-21 抓取）

| 来源 ID | 内容 | URL |
|---|---|---|
| `wiki:Type_99_tank` | 99A：80 km/h、越野 60、行程 500 km、125mm、42 发、3 乘员 | https://en.wikipedia.org/wiki/Type_99_tank |
| `wiki:Type_15_tank` | 15 式轻坦：105mm、5 km 炮射导弹 | https://en.wikipedia.org/wiki/Type_15_tank |
| `wiki:ZBD-04` | 04A：3+7 载员、100mm+30mm、两栖 | https://en.wikipedia.org/wiki/ZBD-04 |
| `wiki:Type_05_amphibious` | 05 式：公路 65 km/h、水上 30 km/h、行程 500 km | https://en.wikipedia.org/wiki/Type_05_amphibious_fighting_vehicle |
| `wiki:PLZ-05` | PLZ-05：55 km/h、155mm、40–70 km | https://en.wikipedia.org/wiki/PLZ-05 |
| `wiki:PHL-03` | PHL-03：12 管 300mm、70–130 km | https://en.wikipedia.org/wiki/PHL-03 |
| `wiki:HQ-9` | HQ-9B：260 km | https://en.wikipedia.org/wiki/HQ-9 |
| `wiki:YJ-62` | YJ-62：280 km（出口型）/400 km（62A） | https://en.wikipedia.org/wiki/YJ-62 |
| `wiki:Type_052D_destroyer` | 052D：30 kn、64 单元 VLS、346A 雷达 | https://en.wikipedia.org/wiki/Type_052D_destroyer |
| `wiki:Type_054A_frigate` | 054A：32 单元 HHQ-16 | https://en.wikipedia.org/wiki/Type_054A_frigate |
| `wiki:Type_056_corvette` | 056A：25 kn、1500 t、76mm、4×YJ-83、8×HHQ-10 | https://en.wikipedia.org/wiki/Type_056_corvette |
| `wiki:Type_022_missile_boat` | 022：双体导弹艇、8×YJ-83 | https://en.wikipedia.org/wiki/Type_022_missile_boat |
| `wiki:Type_071` | 071：600–800 陆战队员、4 艘气垫艇、4 架直升机 | https://en.wikipedia.org/wiki/Type_071_amphibious_transport_dock |
| `wiki:Type_903` | 903 综合补给舰 | https://en.wikipedia.org/wiki/Type_903_replenishment_ship |
| `wiki:Type_039A` | 039 系列：S20 出口型 18 kn、潜深 300 m | https://en.wikipedia.org/wiki/Type_039A/B/C_submarine |
| `wiki:J-16` | 歼-16：2120 km/h、升限 17300 m、航程 3000 km | https://en.wikipedia.org/wiki/Shenyang_J-16 |
| `wiki:KJ-500` | 空警-500：550 km/h、5700 km、12 h、对战斗机 470 km | https://en.wikipedia.org/wiki/Shaanxi_KJ-500 |
| `wiki:Z-10` | 直-10：300 km/h、升限 6400 m、航程 800 km | https://en.wikipedia.org/wiki/Changhe_Z-10 |
| `wiki:MQ-9` | MQ-9：482 km/h、升限 15420 m、续航 27 h | https://en.wikipedia.org/wiki/General_Atomics_MQ-9_Reaper |
| `wiki:Y-20` | 运-20：926 km/h、升限 13000 m、载重 66 t、航程 7800 km | https://en.wikipedia.org/wiki/Xi%27an_Y-20 |
| `doc:demo_estimates` | 同族估算/演示值（如轮式步战、指挥车、部分舰艇航速） | 本数据包 |
| `doc:observation_table` | 庙算观察规则表 | `docs/wargame_ref` |

## 5. 置信度与维护

- `high`：有公开型号数据或庙算实测支撑，可直接用于演示与校核。
- `medium`：同族估算或公开资料的近似值（如 054A 27 kn、071 22 kn、轮式步战 100 km/h），可接受但应在正式使用前复核。
- `low`：演示值（工程车、指挥车等），只用于占位。
- `unknown`：只确定到大类的保守卡，不作为能力依据；涉及交战时按 `unknown_policy` 处理。

修改规则：

1. 内置小类只增不改；需要调整数值时升 `revision`（本版为 v2 初稿，未写 revision 字段，正式入库时补）。
2. 新小类必须先挂到大类，并在 `types.json.classes[].subclasses` 注册；没有专属小类时先复用 `unknown`。
3. 任何新武器/传感器先加入 `equipment.json`，小类只引用 `ref`；禁止在小类里内联第二份参数。
4. 数值改动必须同步 `source.refs` 与 `confidence`；`数值总表.md` 由脚本重新生成。
5. 庙算兼容字段（`sdk`）与物理值（`values`）分开维护，不得互相覆盖。

## 6. 已知限制

- 庙算样本只有 50 个想定、21 种算子，固定翼战斗机/舰艇的 `sub_type` 未在样本中出现，映射依赖名称/显式类型。
- 舰艇航速、登陆舰/补给舰吨位等部分值来自常用公开值而非逐型号官方数据；每张卡的 `source.note` 与 `confidence` 记录了具体依据。
- 传感器探测距离按公开雷达/光电指标的量级取值，未建模探测概率、气象与电子对抗细节。
- `max_blood` 沿用庙算的算子数量口径，与"单车毁伤点数"不是同一概念；后续若切换为真实毁伤模型需另设字段。
- 未知小类的速度/装甲是保守占位值，不能当作情报；引擎应同时输出 `confidence=unknown` 以便界面提示。
- 资产与符号按**服务器主工作树（8848 当前版本）**校验：17 个 `type_*.glb` 均存在；静态符号清单覆盖 19 个基础键，`light_vehicle`/`truck`/`engineer_vehicle`/`transport_ship` 等使用前端内置近似符号。本机 `code/shenji_agent_feat3d` 副本较旧，不能作为校验依据。
