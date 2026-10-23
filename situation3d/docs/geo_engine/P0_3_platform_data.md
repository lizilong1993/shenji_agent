# P0.3 平台数据 v1（2026-10-22）

> 结论先行：新增 `situation3d/platforms/*.json`（10 个平台类）把尺寸、质量/功率、速度曲线、
> 转弯、地形/海/空限制、装甲面、特征、传感器与武器从代码常量搬到数据文件；
> `mobility/platform_data.py` 负责 schema 校验与 `PlatformProfile` 重建，`profile_for` 优先读
> 数据、常量作回退。JSON 重建结果与内置常量逐字段一致，P6 结果与 P0.2 完全相同。

## 1. 问题（ARCHITECTURE_GAPS G3 第一部分）

平台参数散落为代码常量（`mobility/profile.py`、角色卡适配器），型号差异只能手工改常数，
无法从数据推导/验证；质量、功率、装甲面、特征、传感器、武器没有物理数据载体。

## 2. 数据格式（`msplatform.v1`）

每个平台一个 JSON（`platforms/<key>.json`）：

| 块 | 内容 |
|---|---|
| `identity` | 中文名、domain（land/amphibious/surface/subsurface/air）、size_class |
| `dimensions` | 长/宽/高（m） |
| `mass` / `power` | 战斗全重（t）、功率（hp） |
| `motion` | 最大速度、加速度、制动减速度、偏航率、最小转弯半径、原地转向、地形速度因子、速度曲线 `[速度, 因子]` |
| `terrain` | 最大纵坡/侧坡、涉水/浮渡深度、两栖、道路约束 |
| `sea` / `air` | 吃水、UKC、最大潜深；最低离地/升限/AGL/爬升/下降率 |
| `road` | 限重、车高、车宽（桥隧判定） |
| `armor` | 装甲等级 + 四向 RHA 面（前/侧/后/顶，mm） |
| `signature` | RCS（m²）、红外、声学特征（相对值） |
| `sensors` / `weapons` | 传感器列表（类型/距离）、武器列表（weapon_id/弹药或海空武器） |
| `provenance` | 来源、置信度、说明（当前全部为演示估计） |

## 3. 加载与校验

- `mobility/platform_data.py`：
  - `validate_platform(doc)`：schema 版本、必填字段、类型/范围（尺寸>0、速度/加速度>0、
    偏航率 0.1–720°/s、地形因子 0–1、速度曲线单调且因子 0–1.5、装甲面/特征/限重非负、
    传感器/武器为对象列表），错误一次性聚合并抛 `PlatformDataError`；
  - `profile_from_record(doc)`：重建 `PlatformProfile`（缺失块用与常量一致的默认值）；
  - `load_platforms(dir)`：按文件名排序加载、拒绝重复 key、目录缺失返回空表（回退常量）；
  - `registry()/profile_by_key()/record_by_key()`：进程内惰性缓存，供 P1 传感器/武器读取。
- `mobility/profile.py`：`profile_key_for(unit_type, domain)` 抽出映射；`profile_for` 先查
  JSON 数据，缺失时回退 `PROFILES` 常量。
- `tools/gen_platforms.py`：从内置常量生成 v1 JSON，保证逐一对应（后续手改数据即可）。

## 4. 验收

- **重建一致**：`test_json_rebuilds_builtin_profiles` 用 `dataclasses.fields` 逐字段比较，
  10 个平台全部与内置 `PROFILES` 相等（`PlatformProfile` 值相等）。
- 测试：`tests/test_platform_data.py` 13 项（加载/校验/重建/内部映射/元数据/坏数据/重复 key、
  非对象块、NaN/Inf、provenance 必填、数据损坏回退、reload）；全量 **231 项通过（3 skipped）**。
- **行为对拍**：P6 envpack（1440 tick）与 P0.2 结果一致：事件 407、直瞄 66、毁伤 18、
  p50 81 ms（JSON 与常量逐字段相同，故行为不变）。

## 4.1 独立评审修正（2026-10-22）

- **校验健壮性**：各顶层块先做对象类型检查（`identity/motion/terrain/sea/air/road` 等非对象
  不再触发 `AttributeError`，统一报 `PlatformDataError`）；数值必须是有限数（拒绝 NaN/Inf）；
  `provenance.source/confidence` 必填；错误信息带文件名。
- **运行期回退**：数据目录损坏/非法时 `profile_by_key` 捕获 `PlatformDataError` 并告警，
  `profile_for` 回退内置常量，仿真不会因数据问题无法启动；`reload_registry` 可测试重置。
- **生成器**：路径从 `__file__` 推导（不再写死沙箱路径），增加 `main()`/`--out`，
  服务端重生成与入库 JSON 逐字节一致（`diff -r`）。
- **测试有效性**：新增哨兵 profile 验证 `profile_for` 确实优先返回数据（而非常量）；
  `wheeled` 补显式映射；`beam/height` 改为单一来源（`road.beam_m`、`dimensions.height_m`）。

## 5. 限制与后续

- 数值仍为演示估计（`provenance.confidence=low`）；P1 接入公开/估计数据时逐项替换并在
  `provenance` 标注来源。
- 质量/功率/装甲面/特征/传感器/武器已进入数据与注册表，但裁决层尚未消费（武器仍走既有
  weapon 表/角色卡）；P1 物理火力与 3D 感知将直接读取这些字段。
- `speed_curve` 目前是占位折线；P0.3 之后的载重/坡度耦合与 P1 运动学规划会使用它。
- 角色卡（`mobility/card_profile.py`）仍是覆盖层：卡片优先，未给出的字段按域回退；
  P1 可把角色卡并入平台数据选择逻辑。

## 6. 复现

```bash
cd situation3d
python3 -m pytest tests/test_platform_data.py -q
python3 tools/gen_platforms.py   # 重新生成 v1 数据（应与内置常量一致）
```
