# P1 环境垂直切片：v3 契约与查询闭环

> 分支：`feat/env-authority`（基于 `main` `4dc9e1d`）  
> 日期：2026-10-24  
> 状态：切片原型，未合并、未改演示环境。环境包为 legacy 数据副本，不是最终数据质量。

## 1. 本轮做了什么

- 新增 `situation3d/envmodel/`：
  - `manifest.py`：校验 `msgeo.envpack.v3` 的 identity/bbox/必需图层/逐层 `source/license/confidence`。
  - `query.py`：`SpatialQueryV3` 把现有 `EnvPackEnvironment` 包装成 P0 契约（`Envelope`、`quality`、`source`、`confidence`、`as_of`、`at(time)`、`query_batch`、`nav_graph`、`cost_surface`、`obstacles`、`road_graph`、`tile`、`los`、`passability`）。
  - `obstacles.py`：建筑/障碍 facade，复用现有建筑空间索引与 3D 段/多边形高度检查。
  - `open_pack(root)`：一次加载 manifest + `EnvPackEnvironment`。
- 新增 `scripts/upgrade_envpack_v3.py`：把 legacy envpack 复制并写成 v3 清单；不重投影、不改变图层内容。
- 用 `taiwan_tw2_v1` 生成 `envpacks/taiwan_tw2_v3`（gitignored），验证真实包的 `open_pack`、时间绑定、水体未知深度、建筑障碍与 LOS。
- 新增 `tests/test_envmodel_v3.py`：10 项测试，全部通过。

## 2. 关键语义

- `water()`：如果 `hydrography.depth_kind=unknown`，即使 legacy 包有默认 20 m 水深，也返回 `depth_m=None, quality=unknown`；避免把缺省值伪装成实测。
- `building()`：如果 manifest `buildings.height_kind=default`，返回 `quality=default`；数据质量由 manifest 显式表达。
- `los()`：同时返回 `path_visible` 与 `building_blocked`，不是单一布尔；便于区分“路径不可见”和“是否由建筑触发”。当前 `path_visible` 复用 `EnvPackEnvironment.line_of_sight`（已含建筑/覆盖高度），所以它是“路径可见”，不是纯地形可见；纯地形/植被分离留 P1 真实数据阶段。
- `at(time_s)`：同一视图的所有 `Envelope.as_of` 一致；动态环境后续接 `EnvironmentEvent`。
- `query_batch()`：批量接口，避免调用方逐点 Python 调用成为瓶颈。

## 3. 验证结果

```bash
cd situation3d
python3 -m unittest tests.test_envmodel_v3 -v
# Ran 10 tests ... OK
```

真实包抽样（`envpacks/taiwan_tw2_v3`）：

```text
pack taiwan_strait 3.0-p1-slice layers [buildings, hydrography, landcover, obstacles, roads, terrain, water]
120.16 23.0  ele -3.0 derived  water quality=not_applicable  road=True
120.10 22.93 ele -21.0 derived water is_water=True depth=None quality=unknown
obstacles 16132 source=derived-from-buildings confidence=low
```

注意：抽样点的 `ele=-3/-21` 来自 legacy 地形层（`source=legacy-basic`，100 m），不是真实 DEM；这正是切片要暴露的数据质量问题，不是查询实现错误。

## 4. 已知限制

1. `taiwan_tw2_v3` 只是 legacy 图层副本：地形仍是基础图回退 100 m，水域无水深，建筑高度多为默认值；未接入 NLSC/Copernicus/OSM 真实 v3 数据。
2. `nav_graph()` 直接构造 `NavField(30 m)`；未做按需加载/LOD/缓存签名，性能不是生产级。
3. `road_graph()` 返回 legacy `(nodes, edges)`；尚无车道/信号/交通状态。
4. `ObstacleSet` 复用现有建筑空间索引；尚未加入树冠、桥面/桥下净空、隧道内壁、动态障碍。
5. `passability()` 只返回 `allowed` + 粗略 reason；未接 P0 的 `Gate`/`unknown_policy`。
6. `tile()` 只返回图层路径元数据；未实现 COG/PMTiles/3D Tiles 范围读取。

## 5. 下一步

1. 用真实 DEM/OSM/土地覆盖生成一个 5–10 km 的 v3 包；把 `terrain` 换成 COG、`buildings/roads` 换成 GeoParquet/FlatGeobuf。
2. 把 `ObstacleSet` 扩展为建筑体块 + 桥面 + 树冠 + 动态事件；加 BVH/空间桶。
3. 把 `passability()` 接 `Gate`/`unknown_policy`，并统计合法路线成功率/假阻断率。
4. 增加 `nav_graph` 的 30 m/2 m 双层与缓存签名；用真实包跑 10k 条路径的穿越检查。
5. 在测试端 8858 做只读查询验证，不改 8848/前端。
