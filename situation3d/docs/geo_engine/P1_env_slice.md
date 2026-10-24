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
- 用 `taiwan_tw2_v1` 和 `anping_v2c` 生成 v3 包（gitignored），验证真实包的 `open_pack`、时间绑定、水体/水深、建筑障碍与 LOS。
- 新增 `envmodel/gates.py` + `SpatialQueryV3.passability()`：建筑、水深、坡度、道路限重/限高/限宽门，返回原因码与 `unknown` 标记。
- 新增 `tests/test_envmodel_v3.py`：13 项测试，全部通过。

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
# Ran 13 tests ... OK
python3 -m unittest tests.test_envmodel_v3 tests.test_geo_engine_core tests.test_planning tests.test_kinematics -q
# Ran 68 tests ... OK
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
NavField(30m) build: ~8.2 s; 8/8 sampled routes route_ok=True
```

注意：TW2 抽样点的 `ele=-3/-21` 来自 legacy 地形层（`source=legacy-basic`，100 m）；Anping 的 300 m 地形/GEBCO 水深也不是生产精度。这暴露的是数据质量，不是查询实现错误。

## 4. 已知限制

1. `taiwan_tw2_v3` / `anping_v3` 仍是 legacy JSON 图层：Anping 是离线 OSM/DEM/GEBCO 缓存（300 m 地形），TW2 是基础图回退 100 m；未接入 COG/GeoParquet/3D Tiles。
2. `nav_graph()` 直接构造 `NavField(30 m)`；未做按需加载/LOD/缓存签名，构建约 8 s，性能不是生产级。
3. `road_graph()` 返回 legacy `(nodes, edges)`；尚无车道/信号/交通状态。
4. `ObstacleSet` 复用现有建筑空间索引；尚未加入树冠、桥面/桥下净空、隧道内壁、动态障碍。
5. `passability()` 已接建筑/水深/坡度/道路限重限高限宽门，但 `unknown_policy` 仍是固定保守策略，未从 manifest/Policy 读取。
6. `tile()` 只返回图层路径元数据；未实现 COG/PMTiles/3D Tiles 范围读取。

## 5. 下一步

1. 把 Anping 的 `terrain` 换成 COG（NLSC/Copernicus）、`buildings/roads` 换成 GeoParquet/FlatGeobuf；保留 `source/license/confidence`。
2. 把 `ObstacleSet` 扩展为建筑体块 + 桥面 + 树冠 + 动态事件；加 BVH/空间桶。
3. 把 `passability()` 的 `unknown_policy` 接到 manifest/Policy；统计合法路线成功率/假阻断率。
4. 增加 `nav_graph` 的 30 m/2 m 双层与缓存签名；用真实包跑 10k 条路径的穿越/route_ok 检查（当前只做了 8 条 smoke）。
5. 在测试端 8858 做只读查询验证，不改 8848/前端。

> 远程代码位置：`/remote-home/lizilong/shenji_agent_env`（worktree），分支 `feat/env-authority`，提交 `8a1000d`。本地 D 盘副本只含本说明，代码以远程 worktree 为准。
