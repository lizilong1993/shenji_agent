# P2 目标能力目录垂直切片

状态：2026-10-24 起始，分支 `feat/entity-catalog` 已合并 main；2026-10-26 在 `feat/catalog-engine` 增加可选引擎接线。新增 `situation3d/capabilities/`、`tests/test_capabilities_v1.py`（14 项）与 `tests/test_catalog_engine.py`（9 项）；未修改角色卡数据，默认路径行为不变，现有 426 项测试通过（14 跳过）。

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
python3 -m unittest tests/test_capabilities_v1.py tests/test_catalog_engine.py -v
```

覆盖：61 个小类全部可投影、已知量均有单位与来源、坦克/驱逐舰关键量、注册表四张表不被修改且 `sdk` 无共享引用、非法门策略回退、legacy/new 双读、六类门的边界与缺失值、缺量不掩盖违规、两栖泳深优先、潜艇深度上限、未知门报错。

## 7. 引擎接线（可选）

想定 `engine.catalog_mode` 控制投影方式：

- `legacy`（默认，含缺省/非法值）：`profile_from_card(card, key)` 完全保持旧算法；61 张卡的 30 个 `PlatformProfile` 字段与合并前逐字段一致，测试 `test_legacy_fields_match_old_rules_for_all_cards` 锁定。
- `new`：`profile_from_card(card, key, entity=EntityType)` 用目录量覆盖尺寸与门限——`length_m/width_m/height_m` 取 `identity.dimensions`，陆上/水面 `beam_m` 取 `sea.beam_m` 或车宽，`vehicle_height_m` 与 `weight_t` 取卡值（质量目前均未知，保持 0）。空中平台不设置 beam。
- 场景 GLB 覆盖 `length_m/width_m` 时，`new` 模式同步 `beam_m`，避免碰撞宽度与道路限宽门不一致。
- 目录在 `sim/rolecards_bridge` 内按进程缓存；`entity_type_for` 失败时场景回退 `legacy`。

## 8. 已知限制与下一步

- 引擎接线仍是 opt-in：新想定需显式设置 `catalog_mode: "new"`；报告计划中的“新想定默认 new、旧想定 pin legacy”尚未切换。
- 门策略尚无数据：61 张卡都没有 `values.gates.unknown_policy`，当前全部按默认 `deny` 处理；advisory 语义需后续补数据或由调用方逐次覆盖。
- 质量数据缺失：61 张卡均无 `weight_t`，`new` 模式 `weight_t` 仍为 0，桥梁限重门只返回 unknown；需补装备/参考型号数据并加来源。
- 未覆盖报告中的 MLC 语义、森林密度分级、软地压强、逐值类型化（`Quantity` 之外的枚举/区间）；这些需要新增卡字段，不在本切片内改数据。
- `sea_state` 门尚未接入 `envmodel` 的海况查询；待环境分支合并后由网关调用。
