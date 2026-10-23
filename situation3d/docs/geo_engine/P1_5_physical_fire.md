# P1.5 物理火力默认化（2026-10-22）

> 结论先行：直瞄默认切换到物理裁决（`direct_fire_model="physics"`），所有直瞄武器均有
> 弹道/穿深数据（23 个武器 ID 全覆盖）；间瞄默认走弹道弧+落点散布+破片
> （`indirect_fire_model="physics"`）；毁伤引入机动/火力/传感器组件。CRT 与散布表保留为
> 兼容/校准开关。P6 envpack 默认物理火力：452 事件、10 次直瞄、15 毁伤、p50 62 ms；
> 同场景 CRT 对照为 396 事件、65 次直瞄、3 次间瞄、18 毁伤、p50 79 ms（物理弹一发
> 击穿/毁伤更高，射击次数显著减少）。全量 270 项测试通过。

## 1. 问题（ARCHITECTURE_GAPS G5）

默认 `direct_fire_model=crt`：2d6 查表 + 修正表；间瞄为散布表随机游走；海空为 Pk 矩阵；
毁伤只有单一血值，无组件损坏；`physics_direct_fire` 仅可选且只覆盖 15 个直瞄武器。

## 2. 方案

### 2.1 直瞄物理默认（`rules/physics_direct_fire.py`）

- 补齐 `rules/data/ballistics.json`：全部直瞄武器（含步兵轻武器 29、火箭筒 35、便携/车载
  导弹 2/3/5）都有口径、弹种、初速、100/1000/2000 m 穿深、散布、射程；
- 命中 = 两次高斯瞄准误差 ≤ 目标半径（尺寸等级）；入射几何 → 装甲面/RHA 面；
  等效厚度按弹种归一化角/跳弹角/斜率指数修正；穿深随距离衰减并带 0.75–1.25 随机；
- 组件损伤：穿透后按机动 35%/火力 30%/传感器 15%/结构 20% 命中子系统，
  `World.apply_component_damage` 降低组件健康（每次 0.35×伤害）；
  - 机动损伤线性降低运动速度上限（最低 20%）；
  - 火力损伤线性降低后续穿透伤害；
  - 传感器损伤缩短探测距离（P1.4 `specs_for`）；
- `direct_fire_model="crt"` 仍为兼容/校准模式；`"shadow"` 保留物理对拍字段。

### 2.2 间瞄物理默认（`rules/physics_indirect.py` + `ballistics_indirect.json`）

- 弹道：低弧解 `θ=½asin(gR/v²)`，飞行时间 `2v sinθ/g`；超出弹道/射程即无射击解；
- 散布：CEP（校射射减半）→ 高斯距离/方向误差（σ=CEP/1.1774，随距离缩放），
  合成落点经纬度，事件记录 TOF、CEP、σ、落点；
- 破片：爆半径内按 `1-(d/R)^1.5` 衰减，步兵/车辆不同基准，装甲等级削弱车辆伤害；
  车辆中弹同时随机损坏一个组件，步兵被压制 45 s；
- `indirect_fire_model="table"` 保留 P4 的散布表兼容模式。

### 2.3 默认与兼容

- `build_world` 默认 `direct_fire_model=physics`、`indirect_fire_model=physics`；
  `engine`/CLI 可切回 `crt`/`table`；`WorldRuleContext` 默认同步为 physics。
- 直瞄目标选择对纯间瞄单位增加物理射程可达性判断，避免"有间瞄却无目标"。
- 基准事件哈希按新语义重钉（300 `9e4e3d68467494e7`、500 `3590d3e46d1a37b1`）。

## 3. 验收

- 新增 `tests/test_physics_fire.py` 19 项：全直瞄/间瞄数据覆盖、穿深随距离下降、
  等效厚度随入射角上升、命中率随距离下降、组件损伤对运动/传感器的影响、组件归零失效、
  dry-run 不污染组件、防空武器不可对地、步枪可穿透无装甲、掩体降低命中、默认模型、
  TOF/校射 CEP/确定性、破片衰减与装甲保底、间瞄最小射程不回退、前观引导的端到端物理间瞄、
  世界间瞄事件同 seed 复现。
- 全量 **270 项测试通过（3 skipped）**。
- P6 envpack（默认物理火力）：452 事件、海空 292、直瞄 10、毁伤 15、
  `traffic_stuck` 115、`crowding` 16，p50 62 ms；基础图：6 直瞄、13 毁伤、2 目标。
- CRT 对照（同场景，`--direct-fire-model crt --indirect-fire-model table`）：
  396 事件、直瞄 65、间瞄 3、毁伤 18、p50 79 ms（物理弹单发穿透即重创，射击次数少）。

### 3.1 独立评审修正（2026-10-22）

- **shadow 污染（blocker）**：`dry_run` 不再写入组件损伤；shadow 只读。
- **间瞄可达性（blocker）**：`_indirect_reachable_physics` 支持 Entity/View 两种弹药接口，
  移除吞异常的裸 `except`；`_best_target` 首轮过滤同时考虑物理与表间瞄射程，混合单位不再漏射。
- **最小射程回退（blocker）**：物理模式下有剖面的火炮出界/冷却/无弹时不再回退散布表，
  表回退事件显式标注 `model="table"`。
- **传感器组件生效范围（major）**：表/spotting/sensor_detect 路径统一按组件健康缩放探测
  距离；组件归零=传感器失效。
- **组件归零语义**：机动 0 → 速度 0，火力 0 → 不能开火，传感器 0 → 无传感器；
  视图在组件变化后刷新。
- **直瞄数据/门控**：7.62 穿深提高到可穿透无装甲；纯防空武器（WpData 无地面 TarType）
  禁止对地直瞄；掩体（城市/森林/建筑）按比例缩小目标有效半径，物理路径不再忽略掩体。
- **间瞄**：破片对重甲目标保底 1 点；炮兵可使用本方融合接触（前观/营指挥）；observed
  按本方探测判定；事件补齐 `impact_cell/outcome/scatter_cells/att_level/target_blood`
  并只列爆半径内目标；RNG 统一走 `ctx.rng`；组件损坏仅在目标存活时记录。
- **可见性**：observation 实体字典输出 `components`；基准哈希按最终语义重钉。
- 文档与模块 docstring 同步为“物理默认、CRT 兼容”。

## 4. 限制与后续

- 弹道/破片/装甲数值仍为演示估计；海空与导弹仍走 Pk 矩阵（P2 导弹飞行段/制导）。
- 组件损伤为单次随机命中一个子系统，无装甲面-组件映射、无乘员/弹药架二次效应；
  毁伤仍未完全替代血值（结构血值保留，组件为附加维度）。
- 间瞄破片不考虑建筑/掩体遮挡与工事减伤（P2 环境/掩体模型）。
- 物理火力更致命导致存活单位更多、拥堵更明显（P6 `traffic_stuck` 115 次）；P1.6 运动学
  规划负责让行/绕行规则。
- P6 默认运行未触发间瞄（本方融合接触内没有 2–24 km 的间瞄目标）；前观引导的物理间瞄
  由单测端到端覆盖，案例锚点/编成调优暂缓。
- CRT/散布表继续作为校准与兼容路径，金标准回放仍可切换。

## 5. 复现

```bash
cd situation3d
python3 -m pytest tests/test_physics_fire.py -q
python3 scripts/run_geo_demo.py --case taiwan_landing --envpack envpacks/anping_v2c \
    --ticks 1440 --dt 30 --out /tmp/tw_p15
# CRT 对照：
python3 scripts/run_geo_demo.py --case taiwan_landing --envpack envpacks/anping_v2c \
    --direct-fire-model crt --indirect-fire-model table --ticks 1440 --dt 30 --out /tmp/tw_crt
```
