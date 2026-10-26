# 目标模型 P3（LOD1 第一批）

2026-10-26 · 分支 `feat/target-models-v3` · 工作树 `/remote-home/lizilong/shenji_agent_models`

对密集案例占比最大的 ZTD-05 与 EQ240 增加 LOD1，并在 1.2–3 km 相机高度切换；8848/8858 未改动。

## 1. LOD1 资产

| 资产 | LOD0 | LOD1 | 说明 |
|---|---:|---:|---|
| `type_ztd05.glb` → `lod1_ztd05.glb` | 8,868 面 / 0.12 MB / 5 prims | **4,138 面 / 0.08 MB / 5 prims** | gltf-transform simplify 0.20/error 0.05 + meshopt |
| `type_eq240.glb` → `lod1_eq240.glb` | 8,601 面 / 0.43 MB / 6 prims | **5,215 面 / 0.34 MB / 6 prims** | simplify 0.35/error 0.02 + webp + meshopt |

LOD1 与 LOD0 同朝向、同原点、同 yaw fix；`lod1_*` 文件不参与 `type_*` 的预算/许可/bboxes 统计。

## 2. 运行时切换

- `EntityLod.lod1UrlFor(def)` 只登记 ZTD-05/EQ240；其他型号返回 null。
- `app.js` 新增 `applyTargetModelLod()`：
  - 相机高度 **1.2–3 km** 时，对登记型号把 `entity.model.uri` 切到 LOD1；
  - 低于 1.2 km 回 LOD0；高于 3 km 回 LOD0（此时通常已回 2D 图标）；
  - 由 `camera.changed` 现有的 250 ms 节流调用，不新增帧循环；
  - 只改 URI，不重建实体；方向/位置/标牌/轨迹属性不变。
- 3 km 以上仍由现有 2D 图标形态接管；1.2 km 以下保持 LOD0。

## 3. 验证

- `node --check web/src/app.js`、`entityLod.js` 通过；`entity_lod_test.js` 206/206。
- Cesium 1.145 + headless Chromium 实测（`taiwan_dense_transfer_v1`）：
  - 相机 2000 m：80 个有模型目标中 **60 个**（ZTD-05×36 + EQ240×24）URI 为 `lod1_ztd05/lod1_eq240`；
  - 相机 800 m：0 个 LOD1；相机 4000 m：0 个 LOD1；
  - 页面无 `pageerror`；LOD1 资源请求正常返回。
- 估算：密集案例在 1.2–3 km 区间，两型合计面数 525,672→274,128（−48%），draw calls 不变（仍 5/6 prims）。

## 4. 已知限制

- 仅两型有 LOD1；071/Perry/J-16 等未做。
- LOD1 不减 primitive，只减三角面；draw calls 需靠实例化/合批。
- 60 个目标在阈值处同时换 URI，资源已缓存但仍可能有一次加载抖动；未做真机 GPU 帧时间验收。
- LOD1 的 EQ240 车轮细节更简化，1.2 km 近边界需目视复核。

## 5. 下一批

1. 071/Perry/J-16 等 LOD1；
2. 对 ZTD-05/EQ240 做实例化/共享 Model，进一步压 draw calls；
3. 真机 GPU（RTX 3060/4060 级）测 1.5/3/6 km 的 FPS、draw calls、首帧时间；
4. 轮系/雷达/炮塔关节与 052D 网格改造继续排期。
