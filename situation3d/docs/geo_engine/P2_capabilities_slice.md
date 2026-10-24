# P2 目标能力目录垂直切片

状态：2026-10-24，分支 `feat/entity-catalog`（基于 `4dc9e1d`）。新增 `situation3d/capabilities/` 与 14 项测试，未修改角色卡数据、未接入引擎默认路径，现有 370 项测试通过（14 跳过）。

## 1. 定位

对应总体报告 P2：把 61 个小类角色卡投影成带来源的 `EntityType` 能力目录，并为物理门提供数据驱动实现。角色卡仍是唯一数据源；目录是旁路视图，解决 G4 中“能力/环境门靠零散代码、数值无来源、缺失值行为不一致”的问题。

本切片只做目录与门，不改行为：`resolve_type(key, mode=...)` 双读保留 legacy 有效卡字典，新代码可按需取 `EntityType`。

## 2. 模块

```
situation3d/capabilities/
  model.py    SourceRef / Quantity / EntityType / GateResult
  catalog.py  角色卡 -> EntityType 投影、CapabilityCatalog、resolve_type 双读
  gates.py    slope / wading / bridge / sea_state / ukc / runway 六类门
situation3d/tests/test_capabilities_v1.py
```

依赖 `engine.rolecards`（只读），不依赖引擎运行时。

## 3. 数量与来源

`Quantity` 保存 `value`、`unit`、`source`（角色卡 refs、note、confidence）。`value is None` 表示卡未给出该量，门据此走 `unknown_policy`，不做默认值猜测。

现有卡可投影的规范键：

| 组 | 键 |
| --- | --- |
| 尺寸 | `dim.length_m` / `dim.width_m` / `dim.height_m` |
| 速度 | `speed.max_ms` / `speed.cruise_ms` |
| 运动 | `motion.accel_ms2` / `motion.decel_ms2` |
| 地形 | `terrain.gradient_max_deg` / `terrain.side_slope_max_deg` / `terrain.wading_m` / `terrain.swim_m` |
| 海上 | `sea.draft_m` / `sea.ukc_m` / `sea.beam_m` / `sea.sea_state_max` / `sea.depth_max_m` |
| 空中 | `air.ceiling_m` / `air.runway_m` / `air.climb_ms` / `air.descent_ms` / `air.agl_m` / `air.min_clearance_m` |
| 防护 | `protection.armor_class` / `protection.max_blood` |
| 质量 | `mass.t`（当前卡均未给出，按未知处理） |

尺寸优先取 `identity.dimensions`，缺失时回退 `motion.length_m`/`motion.sea.*`。注意既有 `mobility/card_profile.py` 只读 `motion.length_m`，对现卡实际总是落到域默认值；这是已记录差距，本切片未改动，以免影响金标准。

## 4. 门语义

| 门 | 环境输入 | 实体量 | 判定 |
| --- | --- | --- | --- |
| `slope` | `gradient_deg`、`side_slope_deg` | 对应最大坡度 | 环境值 > 实体上限则 deny |
| `wading` | `water_depth_m` | `terrain.swim_m`（有则用）否则 `terrain.wading_m` | 环境水深 > 上限则 deny |
| `bridge` | `bridge_max_weight_t`/`height_m`/`width_m` | `mass.t`、`dim.height_m`、`dim.width_m` | 实体尺寸/质量 > 桥限则 deny |
| `sea_state` | `sea_state` | `sea.sea_state_max` | 环境海况 > 上限则 deny |
| `ukc` | `water_depth_m` | `sea.draft_m + sea.ukc_m`、`sea.depth_max_m` | 水深不足或超过下潜上限则 deny |
| `runway` | `runway_length_m` | `air.runway_m` | 需求 > 可用长度则 deny |

缺失值统一返回 `decision=unknown`；阻塞与否由未知策略决定：默认 `deny`，可由卡内 `values.gates.unknown_policy` 覆盖（白名单 `deny` / `advisory` / `allow_with_penalty` / `allow`，非法值回退 `deny`）。角色卡既有的顶层 `unknown_policy` 是行为对象（`combat`/`engage` 等），与本切片的门策略无关，未复用。`GateResult.blocking` 只在 `deny` 或 `unknown+deny` 时为真，调用方可按 advisory 记警告。

同一门内有多个检查项时按 `deny > unknown > pass` 聚合：一个量缺失不会掩盖另一个已确定的违规。`ukc` 的深度上限是可选项：卡未给出 `sea.depth_max_m` 时只检查下界（draft+UKC），水面舰因此不受影响；潜艇卡均给出上限。

## 5. 双读与兼容

- `resolve_type(key, mode="legacy")` 返回与既有 `RoleCardRegistry.effective` 完全相同的有效卡字典，行为不变。
- `resolve_type(key, mode="new")` 返回 `EntityType`。
- `CapabilityCatalog` 只读投影，测试验证构建前后注册表与有效卡深比较不变。

## 6. 验证

```
cd situation3d
python3 -m unittest tests/test_capabilities_v1.py -v
```

覆盖：61 个小类全部可投影、已知量均有单位与来源、坦克/驱逐舰关键量、注册表四张表不被修改且 `sdk` 无共享引用、非法门策略回退、legacy/new 双读、六类门的边界与缺失值、缺量不掩盖违规、两栖泳深优先、潜艇深度上限、未知门报错。

## 7. 已知限制与下一步

- 引擎未接线：`sim`/`mobility` 仍走 `card_profile`，本切片仅提供可替换的数据层。接线需在 `card_profile` 与通行层引入模式开关，并以金标准测试证明 legacy 哈希不变。
- 门策略尚无数据：61 张卡都没有 `values.gates.unknown_policy`，当前全部按默认 `deny` 处理；advisory 语义需后续补数据或由调用方逐次覆盖。
- 质量数据缺失：61 张卡均无 `weight_t`，桥梁限重门目前只会返回 unknown；需补装备/参考型号数据并加来源。
- 未覆盖报告中的 MLC 语义、森林密度分级、软地压强、逐值类型化（`Quantity` 之外的枚举/区间）；这些需要新增卡字段，不在本切片内改数据。
- `sea_state` 门尚未接入 `envmodel` 的海况查询；待环境分支合并后由网关调用。
