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

### 3.1 路径一致性（300 条探索，accepted-only + 3 m 局部绕行）

`scripts/check_paths_v3.py --pack envpacks/anping_v3 --routes 300`（道路节点起终点，30 m NavField，`blocked` 排除建筑中心，`safe_fallback=True`；失败时先 `repair_route` 侧移，再做 160 m 窗口的 3 m 局部 A*，最后用 10 m 精确 Gate 复验）：

```text
planned/accepted 109
no_path 191
gate_rejected 132
accepted route_ok 97 (89.0%)
accepted gate_denied 0 / 717 sampled
mean_length 304.2 m; plan_p95 78.6 ms
```

3 m 局部绕行把接受数从 91 提到 109、拒绝从 175 降到 132，且接受路线仍保持 0 建筑/水域穿越；但接受率只有约 36%。剩余拒绝主要是 160 m 窗口/40 m 边距不足、车辆宽度未膨胀、以及长街区/水域无法一次绕开。下一步应扩大局部窗口、按车辆半宽缓冲障碍，并处理 `route_ok` 的 mask 与精确 Gate 不一致（89% vs 100%）。

## 4. 已知限制

1. `taiwan_tw2_v3` / `anping_v3` 仍是 legacy JSON 图层：Anping 是离线 OSM/DEM/GEBCO 缓存（300 m 地形），TW2 是基础图回退 100 m；未接入 COG/GeoParquet/3D Tiles。
2. `nav_graph()` 直接构造 `NavField(30 m)`；未做按需加载/LOD/缓存签名，冷构建约 8 s，缓存命中约 0.013 s，性能不是生产级。
3. `road_graph()` 返回 legacy `(nodes, edges)`；尚无车道/信号/交通状态。
4. `ObstacleSet` 复用现有建筑空间索引；`BridgeSet` 只是 OSM `bridge` 边的低置信甲板（地形+8 m），尚未加入真实桥面高程、桥下净空、树冠、隧道内壁、动态障碍。
5. `passability()` 已接建筑/水深/坡度/道路限重限高限宽门，`unknown_policy` 已从 manifest `gates` 和已批准 `Policy.gates` 读取；`World.authorize_engagement()` 已接通用 ROE/矩阵与 rolecard 武器类别，但尚未接入任务/战术层与目标选择。
6. `safe_fallback` 只在同一 30 m mask 内回退；两个候选都失败时返回 None。它不能保证精确建筑足迹/水域门，30 m 单元格与亚单元足迹仍不一致；起终点也可能被吸附 30 m 单元中心。
7. `tile()` 只返回图层路径元数据；未实现 COG/PMTiles/3D Tiles 范围读取。

## 5. 下一步

1. 把 Anping 的 `terrain` 换成 COG（NLSC/Copernicus）、`buildings/roads` 换成 GeoParquet/FlatGeobuf；保留 `source/license/confidence`。
2. 把 `ObstacleSet` 扩展为建筑体块 + 桥面 + 树冠 + 动态事件；加 BVH/空间桶。
3. 把 `passability()` 的 `unknown_policy` 接到 `Policy` 对象/审批/场景覆盖；统计合法路线成功率/假阻断率。
4. 处理 accepted-only 过滤暴露的 132 条 `gate_rejected`/191 条 `no_path`：扩大局部 3 m 绕行窗口、按车辆半宽膨胀障碍，处理长街区/水域；同时收敛 `route_ok` 89% 与精确 Gate 100% 的 mask 差异。
5. 在测试端 8858 做只读查询验证，不改 8848/前端。

> 远程代码位置：`/remote-home/lizilong/shenji_agent_env`（worktree），分支 `feat/env-authority`，提交 `8a1000d` + `0352255` + `7e66492` + `d1c5d6e` + `23de8e5` + `e37324c` + `476e299` + `dc4dcec` + `ce20b89` + `a2b8d13` + `443661c` + `6cd5de0` + `4b2467a` + `545e3b7` + `3f7ee5e` + `437b092` + `c8603bd` + `d19fb2b` + `bba2ef3` + `dfc6883` + `c3589df` + `d83bd2d` + `2aece9d` + `fbd4adc` + `c48c7b2`。本地 D 盘副本只含本说明，代码以远程 worktree 为准。
