# engine/rolecards — P0 角色卡注册表

本模块把 `data/` 下的角色卡数据包接入 `situation3d`，**P0 阶段不改变任何行为路径**：

- `registry.py`：加载 `classes.json` / `subclasses/*.json` / `equipment.json` / `types.json`，提供 `effective()`、类型解析与校验。
- `validate.py`：结构校验 + 交叉校验（大类卡不得携带小类数值、`required_values` 可解析、装备引用、装备类别策略、两栖/吃水/升限、弹药上限等）。
- `legacy.py`：`card_from_legacy(target)` 把现有 `Target` 或密集转运目标定义转换为角色卡快照；旧 `speed_kn` 解释为巡航速度（`cruise_speed_ms = speed_kn × 0.514444`），保证后续接入时轨迹不变。
- `schema/*.json`：三类数据契约的 JSON Schema 文档（运行时校验不依赖 `jsonschema` 包）。

数据目录：

```
data/
  classes.json          8 个大类（字段契约，无数值上限）
  subclasses/*.json     61 个小类（含 8 个 unknown）
  equipment.json        40 武器 / 32 传感器 / 13 通信
  types.json            图标别名与庙算 type/sub_type 映射
```

使用：

```python
from engine.rolecards import load_registry, card_from_legacy

registry = load_registry()                 # 加载即校验，失败抛 RoleCardError
card = registry.effective("land.vehicle.mbt")
snapshot = card_from_legacy(target, registry)
```

P0 验收：注册表校验通过、所有旧目标都能映射到小类（不能细分则落到 `<class>.unknown`）、生成案例的逐帧哈希与基线一致。P1 才会让 `scenario.py` 读取卡值。

已知问题（P0 不修，避免改变现有行为）：`engine/geo.py::kmh_to_knots` 多除了 3600，导致 `armor`/`artillery` 等使用该函数的旧目标 `speed_kn` 接近 0（如 R04≈0.0067 kn），密集转运目标定义中的车辆 `speed_kn` 被四舍五入为 0（实体 `Entity.speed_mps` 仍是正确值）。`card_from_legacy()` 保留原值并在 `legacy` 块中记录；P1 接线前应单独修复该单位换算并重新生成基线。
