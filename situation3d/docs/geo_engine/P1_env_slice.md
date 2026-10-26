# P1 环境垂直切片：v3 契约与查询闭环

> 分支：`feat/env-authority`（基于 `main` `4dc9e1d`，提交 `8a1000d`）  
> 日期：2026-10-24  
> 状态：切片原型，未合并、未改演示环境。环境包为 legacy 数据副本，不是最终数据质量。

## 1. 本轮做了什么

- 新增 `situation3d/envmodel/`：
  - `manifest.py`：校验 `msgeo.envpack.v3` 的 identity/bbox/必需图层/逐层 `source/license/confidence`。
  - `query.py`：`SpatialQueryV3` 把现有 `EnvPackEnvironment` 包装成 P0 契约（`Envelope`、`quality`、`source`、`confidence`、`as_of`、`at(time)`、`query_batch`、`nav_graph`、`cost_surface`、`obstacles`、`road_graph`、`tile`、`los`、`passability`）。
  - `obstacles.py`：建筑/障碍 facade，复用现有建筑空间索引与 3D 段/多边形高度检查。
  - `open_pack(root)`：一次加载 manifest + `EnvPackEnvironment`。
- 新增 `scripts/upgrade_envpack_v3.py`：把 legacy envpack 复制并写成 v3 清单；不重投影、不改变图层内容。
- 新增 `scripts/check_envmodel_v3.py`：抽样检查水体/水深、建筑高度质量、道路覆盖与 LOS 可见率。
- 新增 `scripts/check_paths_v3.py`：从道路节点采样起终点，跑 `NavField` 路径、`route_ok` 与逐点 Gate 检查；默认 300 条，可跑 10k。
- 用 `taiwan_tw2_v1` 和 `anping_v2c` 生成 v3 包（gitignored），验证真实包的 `open_pack`、时间绑定、水体/水深、建筑障碍与 LOS。
- 新增 `envmodel/gates.py` + `SpatialQueryV3.passability()`：建筑、水深、坡度、道路限重/限高/限宽门，返回原因码与 `unknown` 标记；`unknown_policy` 从 manifest `gates` 读取（bridge/tunnel 默认 advisory，water/building/slope/road 默认 deny）。
- 新增 `BridgeSet`：从 OSM `bridge=true` 的 128 条道路边生成低置信桥面（甲板高=端点地形+8 m，容差 25 m）；桥面点跳过水深门，但仍检查桥重/桥高/桥宽。
- 新增 `envmodel/policy.py`：加载/校验 `situation3d.policy.v1`（要求 `approval_status=approved`），把 policy `gates` 的 `unknown_policy` 覆盖到 manifest 默认值；`open_pack(..., policy_path=...)` 可直接使用。
- 新增 `envmodel/authorization.py`：通用 ROE + 阵营/目标域/武器类别矩阵授权（`hold` 禁火、`return_fire`/`defensive` 需受威胁、`weapons_free` 查矩阵），不包含目标选择或战术规则。
- `sim/world.py` 增加可选 `World.policy` 与 `authorize_engagement()`：在目标选定后、开火前检查授权；`policy=None` 保持旧行为，`sim/scenario.py` 可从想定顶层 `policy` 块注入。武器类别优先用 `wpdata_id -> rolecard category` 映射（cannon/atgm/sam 等），缺失时回退 `kind`。
- `mobility/nav_field.py` 增加可选 `safe_fallback`（默认 False）：平滑/圆角路径未通过 `_points_ok` 时回退原始 A* 顶点并吸附首尾单元中心；默认行为不变，10k 一致性检查显式启用。
- 新增 `tests/test_envmodel_v3.py`：19 项测试，全部通过。

## 2. 关键语义

- `water()`：如果 `hydrography.depth_kind=unknown`，即使 legacy 包有默认 20 m 水深，也返回 `depth_m=None, quality=unknown`；避免把缺省值伪装成实测。
- `building()`：如果 manifest `buildings.height_kind=default`，返回 `quality=default`；数据质量由 manifest 显式表达。
- `los()`：同时返回 `path_visible` 与 `building_blocked`，不是单一布尔；当前 `path_visible` 复用 `EnvPackEnvironment.line_of_sight`（已含建筑/覆盖高度），所以它是“路径可见”，不是纯地形可见；纯地形/植被分离留 P1 真实数据阶段。
- `at(time_s)`：同一视图的所有 `Envelope.as_of` 一致；动态环境后续接 `EnvironmentEvent`。
- `query_batch()`：批量接口，避免调用方逐点 Python 调用成为瓶颈。

## 3. 验证结果

```bash
cd situation3d
python3 -m unittest tests.test_envmodel_v3 -v
# Ran 26 tests ... OK
python3 -m unittest tests.test_envmodel_v3 tests.test_geo_engine_core tests.test_planning tests.test_kinematics tests.test_nav_field -q
# Ran 382 tests ... OK (14 skipped)
```

真实包抽样（`envpacks/taiwan_tw2_v3`）：

```text
pack taiwan_strait 3.0-p1-slice layers [buildings, hydrography, landcover, obstacles, roads, terrain, water]
120.16 23.0  ele -3.0 derived  water quality=not_applicable  road=True
120.10 22.93 ele -21.0 derived water is_water=True depth=None quality=unknown
obstacles 16132 source=derived-from-buildings confidence=low
```

真实 Anping 包（`envpacks/anping_v3`，来自 `anping_v2c` 的离线 OSM/DEM/GEBCO 缓存）：

```text
pack anping 3.0-p1-slice layers [bathymetry, buildings, hydrography, landcover, obstacles, roads, terrain, water]
water sample: 145/300 points; depth range 2.3–21.7 m (GEBCO, derived)
building height quality: default 1889, derived 670, measured 101
LOS visible: 12/200 at 2 m, 144/200 at 30 m
obstacles: 2665
NavField(30m) cold build ~8.2 s; cached ~0.013 s; 1k 路径一致性见 §3.1
```

注意：TW2 抽样点的 `ele=-3/-21` 来自 legacy 地形层（`source=legacy-basic`，100 m）；Anping 的 300 m 地形/GEBCO 水深也不是生产精度。这暴露的是数据质量，不是查询实现错误。

### 3.1 路径一致性（300 条探索，统一 1–3 m 权威）

`scripts/check_paths_v3.py --pack envpacks/anping_v3 --routes 300`（道路节点起终点，30 m NavField，`blocked` 排除建筑中心，`safe_fallback=True`；失败时侧移，再做 ±120 m 窗口、4 m 网格、按车宽半宽膨胀建筑的局部 A*，最后用 3 m 精确 Gate 复验）：

```text
planned/accepted 60
no_path 240
gate_rejected 164
accepted route_ok 60 (100%)
accepted gate_denied 0 / 295 sampled
mean_length 221.8 m; plan_p95 70.3 ms
```

本轮的权威统一改动：

- `NavField.route_ok` 先走 30 m 超覆盖；失败时按 3 m 步长沿路线采样，在 mask 判阻处用精确几何复核（建筑中心与半车宽 8 向偏移、细地类覆盖后的水深、车辆等级白名单内的 OSM 桥面、道路限重限高限宽），只有精确几何也阻断才判失败。`motion_point_ok` 与 `sim/motion._footprint_ok` 使用同一精确回退，修复路线在运动层不再被粗 mask 拦停。
- `evaluate_gates` 与导航 mask 共用同一规则：细地类多边形覆盖 2 km 水深回退；水深已知时按涉水判定（两栖用 `swim_depth_m`，其余用 `wading_depth_m`）；水深为空时按 landcover 因子（water/rock 为 0）与道路绑定判定。
- 桥面只认车辆等级道路（`VEHICLE_BRIDGE_CLASSES`，含 link/road 类），人行道/自行车道/台阶桥不再豁免水深；建筑检查先于桥面豁免，桥 25 m 内也不能穿建筑。
- `validate_route`/`repair_route`/`plan_valid_route` 默认复验步长收紧到 3 m，首阻搜索与候选复验同一步长；陆上/两栖平台追加半车宽建筑净空检查（中心 + 8 向偏移，与局部规划器一致）；`sim/motion` 的 footprint 样本在 mask 判阻时同样走精确回退。

结果：accepted 路线的 `route_ok` 从 78.3% 提升到 100%，`gate_denied` 保持 0。代价是接受数从 113 降到 60（约 20%）：半车宽建筑净空与更严格的 landcover/水深规则拒绝了大量贴建筑或跨窄水带的候选路线，`no_path` 增至 240。当前 30 m 全局 A* 并不做足迹级走廊规划，这是后续提高合法路线成功率的首要改造点。

## 4. 已知限制

1. `taiwan_tw2_v3` / `anping_v3` 仍是 legacy JSON 图层：Anping 是离线 OSM/DEM/GEBCO 缓存（300 m 地形），TW2 是基础图回退 100 m；未接入 COG/GeoParquet/3D Tiles。
2. `nav_graph()` 直接构造 `NavField(30 m)`；未做按需加载/LOD/缓存签名，冷构建约 8 s，缓存命中约 0.013 s，性能不是生产级。
3. `road_graph()` 返回 legacy `(nodes, edges)`；尚无车道/信号/交通状态。
4. `ObstacleSet` 复用现有建筑空间索引；`BridgeSet` 只是 OSM `bridge` 边的低置信甲板（地形+8 m），尚未加入真实桥面高程、桥下净空、树冠、隧道内壁、动态障碍。
5. `passability()` 已接建筑/水深/坡度/地类/道路限重限高限宽门，`unknown_policy` 已从 manifest `gates` 和已批准 `Policy.gates` 读取；`World.authorize_engagement()` 已接通用 ROE/矩阵与 rolecard 武器类别，但尚未接入任务/战术层与目标选择。
6. `safe_fallback` 只在同一 30 m mask 内回退；两个候选都失败时返回 None。精确回退解决了“mask 误阻但几何合法”的路线判定，但起终点仍可能被吸附到 30 m 单元中心，`no_path` 中的一部分来自该吸附与粗连通性。
7. `tile()` 只返回图层路径元数据；未实现 COG/PMTiles/3D Tiles 范围读取。
8. 3 m 点采样仍可能漏过窄于步长的水域/建筑条带；建筑净空用“中心 + 8 向半车宽偏移”近似车身矩形，未按航向做 OBB 求交；真实包上的净空还依赖 OSM footprint 精度（部分建筑多边形压路），接受率对数据质量敏感。
9. `route_ok` 的快速路径（30 m 超覆盖通过）不再复核 Gate；`plan_hybrid` 可能返回 Gate 拒绝的粗路线。`check_paths_v3` 先过滤 Gate，因此 100% 指标是“Gate 合法集的路线权威一致性”，不是双向等价证明。

## 5. 下一步

1. 把 Anping 的 `terrain` 换成 COG（NLSC/Copernicus）、`buildings/roads` 换成 GeoParquet/FlatGeobuf；保留 `source/license/confidence`。
2. 把 `ObstacleSet` 扩展为建筑体块 + 桥面 + 树冠 + 动态事件；加 BVH/空间桶。
3. 把 `passability()` 的 `unknown_policy` 接到 `Policy` 对象/审批/场景覆盖；统计合法路线成功率/假阻断率。
4. 导航权威已统一（3 m 精确回退 + 共用 landcover/水深/桥面规则 + 半车宽净空），`route_ok` 100%；下一步把 300 条扩展到 10k 回归，并把 30 m 全局 A* 升级为足迹级走廊规划，以提升约 20% 的合法路线接受率。
5. 在测试端 8858 做只读查询验证，不改 8848/前端。

> 远程代码位置：`/remote-home/lizilong/shenji_agent_env`（worktree），分支 `feat/env-authority`，提交 `8a1000d` + … + `d12c7a9` + `7d44e41` + 本轮权威修正提交。本地 D 盘副本只含本说明，代码以远程 worktree 为准。
