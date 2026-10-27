# 目标模型 P3（LOD1 第一批）

2026-10-26 · 分支 `feat/target-models-v3` · 工作树 `/remote-home/lizilong/shenji_agent_models`

对密集案例主要高面模型（ZTD-05/EQ240/Perry/J-16/KJ-2000/KJ-500/E-2D/052C）增加 LOD1，并在 1.2–3 km 相机高度切换；8848/8858 未改动。

## 1. LOD1 资产

| 资产 | LOD0 | LOD1 | 说明 |
|---|---:|---:|---|
| `type_ztd05.glb` → `lod1_ztd05.glb` | 8,868 面 / 0.12 MB / 5 prims | **4,138 面 / 0.08 MB / 5 prims** | gltf-transform simplify 0.20/error 0.05 + meshopt |
| `type_eq240.glb` → `lod1_eq240.glb` | 8,601 面 / 0.43 MB / 6 prims | **5,215 面 / 0.34 MB / 6 prims** | simplify 0.35/error 0.02 + webp + meshopt |
| `type_perry.glb` → `lod1_perry.glb` | 29,395 面 / 1.25 MB / 9 prims | **13,451 面 / 0.97 MB / 9 prims** | simplify 0.25/error 0.02 + webp + meshopt |
| `type_j16.glb` → `lod1_j16.glb` | 19,262 面 / 1.50 MB / 16 prims | **6,146 面 / 1.04 MB / 16 prims** | simplify 0.25/error 0.02 + webp + meshopt |
| `type_kj2000.glb` → `lod1_kj2000.glb` | 23,299 面 / 0.74 MB / 8 prims | **10,863 面 / 0.35 MB / 8 prims** | 512 贴图 + simplify 0.3 + webp + meshopt |
| `type_kj500.glb` → `lod1_kj500.glb` | 23,293 面 / 0.74 MB / 8 prims | **10,804 面 / 0.35 MB / 8 prims** | 512 贴图 + simplify 0.3 + webp + meshopt |
| `type_e2d.glb` → `lod1_e2d.glb` | 23,079 面 / 0.92 MB / 4 prims | **15,686 面 / 0.40 MB / 4 prims** | 512 贴图 + simplify 0.3 + webp + meshopt |
| `type_052c.glb` → `lod1_052c.glb` | 29,413 面 / 0.99 MB / 8 prims | **16,288 面 / 0.52 MB / 8 prims** | 512 贴图 + simplify 0.3/error 0.05 + meshopt |

LOD1 与 LOD0 同朝向、同原点、同 yaw fix；`lod1_*` 文件不参与 `type_*` 的预算/许可/bboxes 统计。

## 2. 运行时切换

- `EntityLod.lod1UrlFor(def)` 登记 ZTD-05/EQ240/Perry/J-16/KJ-2000/KJ-500/E-2D/052C；其他型号返回 null；052D 自 2026-10-27 起使用自建 `type_052d`，暂无 LOD1。071/054A/055 因 simplify 后仍约 2.2–3.2 万面、收益不足，暂不登记。
- `app.js` 新增 `applyTargetModelLod()`：
  - 相机高度 **1.2–3 km** 时，对登记型号把 `entity.model.uri` 切到 LOD1；
  - 低于 1.2 km 回 LOD0；高于 3 km 回 LOD0（此时通常已回 2D 图标）；
  - 由 `camera.changed` 现有的 250 ms 节流调用；另加 500 ms 相机高度兜底（程序化 setView 不触发 changed 时也能切换）；
  - 只改 URI，不重建实体；方向/位置/标牌/轨迹属性不变。
- 3 km 以上仍由现有 2D 图标形态接管；1.2 km 以下保持 LOD0。
- **行为变更**：基线 4dc9e1d 的 `EntityLod.DEFAULTS` 是 `1e9`（always show models）；本分支按用户确认恢复 3000/3600 m，高空改回 2D 图标。合并前需知悉这是全局 UI 行为变化。

## 3. 验证

- `node --check web/src/app.js`、`entityLod.js` 通过；`entity_lod_test.js` 218/218。
- Cesium 1.145 + headless Chromium 实测（`taiwan_dense_transfer_v1`）：
  - 相机 2000 m：80 个有模型目标中 **69 个**（ZTD-05×36 + EQ240×24 + Perry×3 + J-16×2 + 空警-500×2 + E-2D×2）URI 为对应的 `lod1_*`；
  - 相机 800 m：0 个 LOD1；相机 4000 m：0 个 LOD1；
  - 页面无 `pageerror`；LOD1 资源请求正常返回。
- 估算：密集案例在 1.2–3 km 区间，六型合计面数 **745,125→379,753（−49%）**，draw calls 不变（各型 LOD1 的 primitive 数同 LOD0）；KJ-2000 的 LOD1 不在当前案例中，供其他案例使用。

## 4. 已知限制

- 八型有 LOD1；071/054A/055 简化收益不足未做；056A 面数过低不需要。
- LOD1 不减 primitive，只减三角面；draw calls 需靠实例化/合批。
- 69 个目标在阈值处同时换 URI，资源已缓存但仍可能有一次加载抖动；未做真机 GPU 帧时间验收。
- LOD1 的 EQ240 车轮细节更简化，1.2 km 近边界需目视复核。

## 5. 下一批

1. 071/054A/055 需更合适的减面参数/网格；
2. ZTD-05 已用材质合并把 prims 5→2（2026-10-27，见 §6）；EQ240 需贴图图集或实例化；
3. 真机 GPU（RTX 3060/4060 级）测 1.5/3/6 km 的 FPS、draw calls、首帧时间；
4. 轮系/雷达/炮塔关节与 052D 细化/LOD1 继续排期。

## 6. 2026-10-27 ZTD-05 材质合并（draw calls）

密集案例里 36 台 ZTD-05 的 5 个 primitive 共用同一张调色板贴图，仅 roughness/metallic 因子不同。把其中 4 个材质并入第 1 个（保留玻璃 BLEND），并把 TEXCOORD 统一为 float 以便 join，LOD0/LOD1 都从 5 → **2 prims**：

| 资产 | 面数 | prims | 体积 |
|---|---:|---:|---:|
| `type_ztd05.glb` | 8,868（不变） | 5 → **2** | 127.5 → 121.3 KB |
| `lod1_ztd05.glb` | 4,138（不变） | 5 → **2** | 75.9 → 70.3 KB |

- 密集案例（80 目标）draw calls 由 **506 → 398（−21%）**（36 × 3）；1.2–3 km 的 LOD1 带同样减少 108。
- 工具：`tools/modelgen/merge_materials.py`（默认演练，`--apply` 前写备份）；材质合并后 roughness 统一为材质 0 的值（0.78），Blender 对比渲染与 Cesium headless 加载均无可见差异。
- EQ240 的 6 个材质各有独立贴图（11 张），不做合并；需要贴图图集（离线烘焙）或实例化才能继续压 draws。
- 验证：预算 40/40、许可 40 条 7 风险、entity_lod 221/221、全量 pytest（见提交说明）；ZTD-05 LOD0/LOD1 在 Cesium 均 200、无 pageerror。
