# 五维映射表与 31007 附录B 逐项对照（I05 步骤 1/2）

> 标准依据：`31007 3.7`、`5.2.3`、**附录B**（`表B.1` 态势 17 项、`表B.2` 动作 10 项）。
> **原文**：`.scratch/std/s31007.txt`（81,675 B）—— 表 B.1 见 `:917`，表 B.2 见 `:972`。
>
> **本文所有 P3 侧字段均为实测**（真实复盘帧 `replays/replay_1770632023.2594852.zip`
> + `docs/wargame_ref/observation_structure.txt`），不是从标准或注释推测。

---

## 0. 必须先说清的一件事（**本条的核心判断**）

`31007 附录B` 的标题是《**空战**算法与模型接口数据格式规范》。本仓 P3 是
**庙算·陆战指挥官**（陆战）。

两者**不是同一套域**：

| | 附录B（标准） | P3（本仓） |
| --- | --- | --- |
| 域 | 空战 | 陆战 |
| 主体 | `aircraft_dic` / `ship_dic` / `submarine_dic` | `operators`（含步兵/车辆/飞机/工事/战略支援） |
| 空间 | 经纬度 + `waypoint_dic`（航线点） | **六角格**（`cur_hex` 四位数）+ `move_path` |
| 组织 | `side_dic` + 任务字典 | `role_and_grouping_info`（席位/编队）+ `communication`（任务消息） |
| 传感器 | `sensor_dic`（雷达/红外/声纳） | `observalbe_distance` + `see_enemy_bop_ids`（**无传感器对象**） |

**因此本条【风险与回滚】说的「强行映射出错误语义」不是假设，而是默认结局。**
本文的做法是：**逐项标注 等价 / 近似 / N-A，并对每个 N-A 写明原因**，
**不允许留空，也不允许用"看起来差不多"凑一个映射**。

> 上一条同类教训（`I03`）：我曾把"本仓无原文"当成事实并据此搁置了一轮。
> 这次**先把原文翻出来**再判断 —— 结论是原文在手，但**域不同**，
> 所以"做不到"的部分是**真实差异**，不是"没找原文"。

---

## 1. P3 侧可用字段（实测基线）

### 1.1 帧顶层键（14 个，真实帧实测）

`actions` · `cities` · `communication` · `jm_points` · `judge_info` · `landmarks` ·
`operators` · `passengers` · `role_and_grouping_info` · `scenario_id` · `scores` ·
`terrain_id` · `time` · `valid_actions`

### 1.2 算子字段（84 个，真实帧实测）

覆盖：识别（`obj_id`/`color`/`type`/`sub_type`/`name`）、机动（`move_state`/`cur_hex`/
`cur_pos`/`speed`/`move_path`/`stop`）、火力（`carry_weapon_ids`/`remain_bullet_nums`/
`C2`/`C3`/`weapon_cool_time`/`weapon_unfold_state`）、损伤（`blood`/`max_blood`/`armor`）、
疲劳压制（`tire`/`keep`）、空地（`altitude`/`target_altitude`/`changing_altitude`）、
雷达（`activating_radar`/`radar_activated`）、工事（`in_fort`/`entering_fort`/`exiting_fort`）、
工程（`laying_mine`/`roadblock_*`/`comnode_*`）、电子战（`disrupt_*`/`disrupted*`）、
伪装（`disguised`/`disguising`）、聚合解聚（`forking`/`unioning`）、
载员（`passenger_ids`/`get_on_*`/`get_off_*`）。

（完整 84 项见 `docs/wargame_ref/observation_structure.txt`。）

---

## 2. 五维映射表（`I05` 步骤 1）

五个维度按 `I05` 原文：**作战实体状态 / 传感器 / 任务 / 战场环境 / 后勤保障**。

### 维度一：作战实体状态

| 标准侧概念 | P3 来源字段 | 转换规则 | 缺失情况 |
| --- | --- | --- | --- |
| 实体标识 | `operators[].obj_id`（int） | 直取 | — |
| 阵营 | `operators[].color`（0 红 / 1 蓝） | 直取 | — |
| 类型 | `operators[].type`（1 步兵 / 2 车辆 / 3 飞机 / 4 工事 / 5 战略支援） | 直取 | **标准按空战机型分**，P3 按陆战兵种分 → **类别体系不同** |
| 细分类型 | `operators[].sub_type` | 直取 + 枚举表 | — |
| 位置 | `operators[].cur_hex`（六角格四位数） | 若需经纬度须**经地图投影换算** | **标准用经纬度，P3 用格号 → 非直取** |
| 速度 | `operators[].speed`（格/s）、`basic_speed`（km/h） | 单位换算 | 标准为节（kn）→ **量纲不同** |
| 航向 | — | — | **N/A：P3 无航向字段**（六角格机动无连续朝向） |
| 状态 | `move_state` / `stop` / `tire` / `keep` / `target_state` | 直取 | — |
| 完好度 | `blood` / `max_blood` / `armor` | 直取 | — |
| 高度 | `altitude` / `target_altitude` | 直取 | 标准空战高度为核心；P3 仅部分单位有 |
| 载员 | `passengers[]`、`passenger_ids`、`on_board`、`car` | 直取 | 标准无对应（空战无载员概念）→ **P3 侧超出标准** |

### 维度二：传感器

| 标准侧概念 | P3 来源字段 | 转换规则 | 缺失情况 |
| --- | --- | --- | --- |
| 传感器对象清单 | — | — | **N/A：P3 无 `sensor_dic` 这类传感器对象** |
| 探测距离 | `operators[].observalbe_distance`（可被观察的基础距离） | 直取 | 标准按雷达/红外/声纳分型；**P3 只有一个标量** |
| 已探测目标 | `operators[].see_enemy_bop_ids`（可见敌方算子 ID） | 直取 | 标准为 `contact_dic`（含识别等级/威胁）；**P3 只有 ID 列表** |
| 识别等级 | — | — | **N/A：P3 不提供 identification level** |
| 雷达开关 | `activating_radar` / `radar_activated` | 直取 | 标准按传感器型号；**P3 为单一雷达布尔** |
| 电子干扰 | `disrupt_*` / `disrupted` / `disrupted__globally` | 直取 | 标准无对应 → **P3 侧超出标准** |

### 维度三：任务

| 标准侧概念 | P3 来源字段 | 转换规则 | 缺失情况 |
| --- | --- | --- | --- |
| 任务清单 | `communication[]`（`type` 207 进攻 / 208 防御 / 209 侦察 / 210 集结） | 按 type 分类 | 标准为 `mssnpatrol_dic` / `mssnstrike_dic` 两个独立字典 → **结构不同** |
| 任务目标位置 | `communication[].hex` | 直取 | — |
| 任务起止时间 | `communication[].start_time` / `end_time` | 直取 | — |
| 执行单位 | `communication[].unit_ids` | 直取 | 标准在 `assign_unit_to_mission` 动作里；**P3 在任务消息里** |
| 任务航线 | `communication[].route` | 直取 | 标准为 `add_wayline` 独立对象 + `assign_wayline_to_mission`；**P3 内联在任务里** |
| 侦察半径 | `communication[].radius`（type 209） | 直取 | 标准无直接对应 |
| 作战条令 | — | — | **N/A：P3 无 `doctrine_dic`（交战规则/武器使用条令）** |
| 编组角色 | `role_and_grouping_info`（faction / role / operators） | 直取 | 标准为 `side_dic` → **粒度不同** |

### 维度四：战场环境

| 标准侧概念 | P3 来源字段 | 转换规则 | 缺失情况 |
| --- | --- | --- | --- |
| 地图标识 | `terrain_id` | 直取 | — |
| 想定标识 | `scenario_id` | 直取 | — |
| 时间 | `time.cur_step` / `tick` / `max_time` / `max_step` / `stage` | 直取 | — |
| 天气 | — | — | **N/A：P3 无 `weather` 字段**（`I05` 标准依据点名 `weather` 为 `str`） |
| 夺控点 | `cities[]`（`coord`/`value`/`flag`/`name`） | 直取 | 标准无对应 → **P3 侧超出标准** |
| 地标/工程 | `landmarks`（`roadblocks` / `minefields` 含通路） | 直取 | 标准无对应 → **P3 侧超出标准** |
| 间瞄点 | `jm_points[]` | 直取 | 标准无对应 → **P3 侧超出标准** |
| 区域 | — | — | **N/A：P3 无 `add_area` 式独立区域对象** |

### 维度五：后勤保障

| 标准侧概念 | P3 来源字段 | 转换规则 | 缺失情况 |
| --- | --- | --- | --- |
| 弹药 | `remain_bullet_nums`、`C2`（普通弹药）、`C3`（导弹） | 直取 | 标准为 `magazine_dic`（弹药库）+ `loadout_dic`（挂载）+ `mount_dic`（挂架）三件套 → **P3 只有算子级计数** |
| 武器清单 | `carry_weapon_ids`、`weapon_dic`（P3 侧无独立武器字典） | 仅 ID 列表 | **标准有 `weapon_dic`（射程/命中率）；P3 的武器属性在 `valid_actions` 里按动作给** |
| 挂架/挂载 | — | — | **N/A：P3 无 `mount_dic` / `loadout_dic` 结构** |
| 补给能力 | — | — | **N/A：P3 无补给/再装填模型** |
| 疲劳 | `tire` / `tire_accumulate_time` | 直取 | 标准无对应 → **P3 侧超出标准** |
| 压制 | `keep` / `keep_remain_time` | 直取 | 标准无对应 → **P3 侧超出标准** |
| 弹头存量 | `extra.warhead_remain_num` | 直取 | 标准无对应 |

### 五维小结

| 维度 | 等价映射 | 近似映射 | N/A | P3 侧超出标准 |
| --- | --- | --- | --- | --- |
| 作战实体状态 | 7 | 2 | **1**（航向） | 1（载员） |
| 传感器 | 3 | 2 | **2**（传感器对象、识别等级） | 1（电子干扰） |
| 任务 | 5 | 1 | **1**（作战条令） | 0 |
| 战场环境 | 4 | 0 | **2**（天气、区域） | 3 |
| 后勤保障 | 4 | 1 | **2**（挂架挂载、补给） | 3 |

**没有任何一项留空。** N/A 共 **8 项**，每项都写了原因。

---

## 3. 表 B.1 逐项映射（态势 17 项）

标准列：`编号 / 参数名 / 类型 / 含义`。**17 项，逐项标注，不允许空。**

| # | 标准参数名 | 类型 | 标准含义 | 判定 | P3 对应 / N-A 原因 |
| --- | --- | --- | --- | --- | --- |
| 1 | `side_dic` | dict | 各方信息（ID、立场） | **近似** | `role_and_grouping_info`（席位→faction/role/operators）+ `operators[].color`。**P3 无「方」对象**，角色由席位推导 |
| 2 | `aircraft_dic` | dict | 己方与敌方飞机对象 | **近似** | `operators[]` 中 `type == 3`（飞机）。**P3 的飞机只是 5 类之一**，与标准的"以飞机为主体"不同 |
| 3 | `ship_dic` | dict | 舰艇信息 | **N/A** | **P3 无舰艇兵种**（`operators[].type` 枚举为 步兵/车辆/飞机/工事/战略支援，无舰艇）。陆战域不含海上单位 |
| 4 | `submarine_dic` | dict | 潜艇信息 | **N/A** | 同上，且**P3 无水下域**概念 |
| 5 | `sensor_dic` | dict | 传感器状态与性能（雷达/红外/声纳） | **N/A** | **P3 无传感器对象**。最接近的是算子标量 `observalbe_distance` 与布尔 `radar_activated` —— 那是**单位属性**，不是**可列举的传感器实体**，故不构成等价 |
| 6 | `contact_dic` | dict | 探测到的目标列表及属性（识别等级/位置/威胁） | **近似** | `operators[].see_enemy_bop_ids`（可见敌方 ID 列表）+ `jm_points[]`（来袭弹道）。**缺"识别等级"与"威胁评估"** |
| 7 | `doctrine_dic` | dict | 各单元条令（武器使用、交战规则） | **N/A** | **P3 无条令对象**。规则在引擎侧，not exposed 在 observation 里 |
| 8 | `mssnpatrol_dic` | dict | 巡逻任务 | **近似** | `communication[]` 中 `type == 209`（侦察任务，含 `radius`）。**P3 无独立"巡逻任务"类型**，`209` 是最近的对应 |
| 9 | `mssnstrike_dic` | dict | 打击任务 | **近似** | `communication[]` 中 `type == 207`（进攻任务）与 `208`（防御）。**P3 无"打击任务"独立类型**，进攻任务是最近对应 |
| 10 | `loadout_dic` | dict | 挂载（武器类型、剩余弹药、状态） | **近似** | `carry_weapon_ids`（武器 ID 列表）+ `remain_bullet_nums`（按类型剩余）+ `C2`/`C3`。**P3 按弹药类型计数，不按挂载配置** |
| 11 | `mount_dic` | dict | 挂架（挂载编号与可用性） | **N/A** | **P3 无挂架对象**（陆战单位无挂架概念） |
| 12 | `magazine_dic` | dict | 弹药库（储量、补给能力） | **N/A** | **P3 无弹药库实体，也无补给模型**。弹药是算子级计数（`remain_bullet_nums`），没有"库"这一层 |
| 13 | `weapon_dic` | dict | 武器列表（种类、射程、命中率） | **近似** | `carry_weapon_ids`（仅 ID）+ `valid_actions` 里按动作给出的武器参数。**P3 的武器属性不集中在帧里**，射程/命中率在 `valid_actions` 与引擎侧 |
| 14 | `waypoint_dic` | dict | 飞行/航行路径点 | **近似** | `move_path`（计划机动路径，六角格号列表）+ `communication[].route`。**P3 用格号，标准用经纬度点** |
| 15 | `weather` | **str** | 天气（影响探测与武器） | **N/A** | **P3 无 `weather` 字段**（实测帧 14 个顶层键里没有）。这是标准 17 项中**唯一标 `str` 的项**，P3 侧完全没有对应概念 |
| 16 | `response_dic` | dict | 系统返回（执行状态、警告、错误） | **近似** | `actions[].error`（`code` + `message`）+ `judge_info[]`（裁决）。**P3 把错误内联在历史动作里**，不是独立响应字典 |
| 17 | `logged_messages` | dict | 推演日志（可选调试，不直接决策） | **近似** | `communication[]`（`type` 204 聊天 / 205 渲染 / 334 发射任务）+ `actions[]`（含错误）。**P3 无独立"日志"通道**，通信与历史动作合起来覆盖部分语义 |

### 表 B.1 统计

| 判定 | 数量 | 编号 |
| --- | --- | --- |
| 等价映射 | **0** | — |
| 近似映射 | **10** | 1, 2, 6, 8, 9, 10, 13, 14, 16, 17 |
| N/A | **7** | 3, 4, 5, 7, 11, 12, 15 |

复核：`0 + 10 + 7 = 17` ✓（与标准「表 B.1 共 17 项」一致）

**重要结论：表 B.1 的 17 项里，P3 侧**没有一个**可以判"等价映射"。**
原因不是实现不足，而是**域不同**：标准的 17 项是围绕"空战平台 + 传感器 + 任务 + 挂载"
组织的，而 P3 是围绕"陆战算子 + 夺控点 + 六角格"组织的。

---

## 4. 表 B.2 逐项映射（动作 10 项）

| # | 标准参数名 | 类型 | 标准含义 | 判定 | P3 对应 / N-A 原因 |
| --- | --- | --- | --- | --- | --- |
| 1 | `add_mission` | dict | 添加任务（类型/名称/目标） | **近似** | `communication[]` 的 `type 207/208/209/210`（下达作战任务）。**P3 是"消息"而非"add_*"动作**；且平台文档明确 `valid_actions` **不含队长动作**（下达任务属队长动作） |
| 2 | `set_mission_status` | str | 启用/禁用任务 | **N/A** | **P3 无任务启停动作**。任务由 `start_time`/`end_time` 界定，没有独立"激活/禁用"操作 |
| 3 | `set_mission_doctrine` | dict | 设置任务条令 | **N/A** | P3 无条令概念（与表 B.1 第 7 项同因） |
| 4 | `add_area` | list | 添加区域（多个经纬度点） | **N/A** | **P3 无区域对象**。最接近的是 `communication[].radius`（侦察半径），但那是**任务属性**而非**可增删的区域实体** |
| 5 | `add_wayline` | list | 添加预设航线（航点/速度/高度） | **近似** | 机动动作 `type 1`（`move_path`，六角格路径）+ `communication[].route`。**P3 无"预设航线"独立对象**，路径内联在动作/任务里 |
| 6 | `assign_area_to_mission` | str | 将区域绑定到任务 | **N/A** | 无区域对象（见 #4），故无从绑定 |
| 7 | `assign_wayline_to_mission` | str | 将航线绑定到任务 | **N/A** | 无独立航线对象（见 #5）。P3 的任务自带 `route`，**不需要绑定动作** |
| 8 | `assign_target_to_mission` | str | 给任务分配攻击目标 | **近似** | `communication[]` 的任务消息带 `hex`（目标位置）；攻击目标经 `valid_actions` 的 `type 2` 带 `target_obj_id` 指定。**P3 无"任务↔目标"显式绑定动作** |
| 9 | `assign_unit_to_mission` | str | 将平台加入任务 | **近似** | `communication[].unit_ids`（执行任务的单位列表）。**P3 在任务消息里直接带单位**，没有独立的 assign 动作 |
| 10 | `set_unit_status` | dict | 启动/关闭平台系统（雷达、武器） | **近似** | 雷达：动作 `type 11`（武器锁定）/`12`（武器展开）/`17`（开启炮兵校射雷达）；状态：`type 6`（切换状态，`target_state` 0–4）。**P3 把"平台系统"拆成多个具体动作**，不是一个统一的 `set_unit_status` |

### 表 B.2 统计

| 判定 | 数量 | 编号 |
| --- | --- | --- |
| 等价映射 | **0** | — |
| 近似映射 | **5** | 1,5,8,9,10 |
| N/A | **5** | 2,3,4,6,7（+ 见下复核） |

复核：5 + 5 = **10** ✓

**结论同表 B.1：10 项里**没有一个**可判"等价映射"。**
标准的 10 项是**任务/区域/航线/绑定**式的高层指挥动作，
而 P3 的 42+ 动作是**算子级战术动作**（机动/打击/上下车/工事/布雷…）。
两者的**动作粒度差一层**。

---

## 5. 差异说明（判据 3：陆战/空战差异逐项有说明）

**统一说明**（适用于所有 N/A 与近似项）：

1. **域不同**：附录B 面向**空战**（`aircraft_dic`/`ship_dic`/`submarine_dic`/`waypoint_dic`），
   P3 面向**陆战**（算子/六角格/夺控点）。**同一概念名在两域指的不是同一件事。**
2. **粒度差一层**：标准 B.2 是**任务级指挥动作**（add/assign/set），
   P3 是**算子级战术动作**（42+ 个 `valid_actions`）。**不是"缺动作"，是"层级不同"。**
3. **空间表示不同**：标准用**经纬度**（`position: [lat, lon]`），P3 用**六角格号**（`cur_hex` 四位数）。
   换算需地图投影，**不是无损直取**。
4. **P3 侧有标准未覆盖的内容**：夺控点（`cities`）、地标工程（`landmarks`）、
   间瞄点（`jm_points`）、疲劳（`tire`）、压制（`keep`）、电子战（`disrupt_*`）——
   这些在附录B 里**没有对应项**，属 P3 超出标准的部分，**不应强行塞进标准的 17/10 项里**。

---

## 6. 未做的部分（如实列出）

| 项 | 状态 |
| --- | --- |
| 本文（`I05` 步骤 1/2：五维映射 + B.1/B.2 逐项） | **已完成** |
| 动作字典安全双校验（步骤 3） | **已完成**（第五十轮，`ai/action_safety.py` + 45 条测试） |
| 单测（步骤 4） | **已完成**（同上） |
| 把本文与差异说明合并送评审（步骤 5） | **未做**（无人评审） |
| 判据 1 的"无空白或待定项" | **达成**：17+10 项全部有判定；N/A 项**逐项写了原因**，不是留空 |
| 判据 3 的"陆战/空战差异逐项有说明" | **达成**：§5 给出统一说明，且每项 N/A/近似都写了具体原因 |

**本文的用途是"说明为什么不能合并"，不是"提供一份合并方案"。**
