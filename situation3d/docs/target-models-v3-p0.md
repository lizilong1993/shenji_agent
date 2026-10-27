# 目标模型 P0 优化结果

2026-10-26 · 分支 `feat/target-models-v3` · 工作树 `/remote-home/lizilong/shenji_agent_models`

本次执行报告《target-models-v3.md》的 P0 阶段，并同步完成密集案例占比最高的 ZTD-05 / EQ240 减面。**8848 演示未改动、未重启**；验证使用 8859 临时开发服务。

## 1. 改动内容

| 文件 | 改动 |
|---|---|
| `server/serve.py` | 新增 `ASSET_PREFIXES=("/assets/models/","/assets/fx/")`，资源以 `Cache-Control: public, max-age=600` 下发；ETag/Last-Modified 仍可在过期后复用 |
| `web/src/entityLod.js` | 恢复 `DEFAULTS={enter3dHeight:3000, exit3dHeight:3600}`（此前为 `1e9`，导致 `entity_lod_test.js` 6 项失败） |
| `web/assets/models/type_*.glb` | 30 型重新导出：纹理 ≤1024 + WebP；静态件 `palette → join`；按预算 `weld/simplify`；静态模型 `meshopt` |
| `web/assets/models/specs.json` | 逐型号 class、公开长度/宽度/近似关系 |
| `web/assets/models/licenses.json` | 从 CREDITS.md 生成的许可台账（含 4 个风险标记） |
| `web/assets/models/manifest.json` | 30 型 sha256、大小、面数、primitive、材质、扩展、许可 |
| `tools/modelgen/optimize_assets.py` | 优化管线（可复跑，默认 dry-run） |
| `tools/model_budget_test.py` | 预算校验 + manifest 生成 |
| `tools/model_license_test.py` | 许可台账生成/校验 |

## 2. 量化结果

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| 30 型文件体积 | 61.6 MB | **17.48 MB** | −71.6% |
| 三角面 | 776,634 | **424,274** | −45.4% |
| primitive（draw calls） | 774 | **223** | −71.2% |
| 纹理字节 | 20.1 MB | **7.26 MB** | −63.9% |
| 台海密集案例（80 目标）draw calls | 1,582 | **506** | −68.0% |
| 台海密集案例三角面 | 3,004,938 | **937,250** | −68.8% |
| 中印案例（10 目标，9 个有模型）draw calls | 382 | **55** | −85.6% |
| 中印案例（10 目标，9 个有模型）三角面 | 234,507 | **110,369** | −52.9% |

主要单模型：052C 4.8→1.0 MB、071 5.5→0.9 MB、J-16 5.8→1.5 MB、PLZ-05 3.8→1.2 MB、ZTD-05 2.1→0.12 MB、EQ240 2.6→0.43 MB、PHL-03 2.1→0.18 MB / 228→2 primitives（MB 按 1024² 计）。

## 3. 验证

- `gltf-transform validate`：30/30 通过。
- `python3 tools/model_budget_test.py`：**ALL PASS (30 assets)**（P0 验收预算按当前可达值 + 余量，P1 重建后收紧）。
- `python3 tools/model_license_test.py`：30/30 有许可记录；风险项 4 个：`type_052c`/`type_kj2000`（DCS 提取物）、`type_ztd05`/`type_phl03`（Sketchfab Free Standard）。
- `node tools/entity_lod_test.js`：**183 通过 / 0 失败**。
- Cesium 1.145 + headless Chromium/SwiftShader：052C、ZTD-05、PHL-03、071、Z-10、J-16、EQ240、M1A2 均成功加载，页面无 `pageErrors`；WebP（`EXT_texture_webp`）与 Meshopt（`EXT_meshopt_compression`）解码正常。
- 8859 静态资源：`200/304` 为 `Cache-Control: public, max-age=600` + ETag；404 不再继承长缓存；`type_052c.glb` 1,039,244 B。
- 世界包围盒/原点：已用 `tools/model_bboxes_gltf.py` 重新生成 `web/assets/models/bboxes.json` 与 `engine/data/models/bboxes.json`（支持 meshopt）。抽查 052C/071/ZTD-05/EQ240/PHL-03：X/Z 包络与水线/地面原点不变；052C 顶高 43.73→41.76 m（桅杆尖）、ZTD-05 4.88→4.07 m（天线）、071 龙骨 -6.88→-6.53 m（0.35 m），属薄件减面，P1 需保护。

## 4. 备份与回滚

- 原始 GLB：`/remote-home/lizilong/model_backups/target-models-v3/20261026/`（36 个，124 MB）。
- 回滚：把备份目录的 `type_*.glb` 覆盖回 `web/assets/models/`，并回退 `serve.py`/`entityLod.js` 的本特性提交。
- 优化后的 GLB 使用 Meshopt/量化，Blender 不能直接再编辑；后续 P1/P2 请从备份原文件重新导出，不要以优化后文件为源。

## 5. 未完成/下一阶段

- P1：052D（复用 `build_052d()`）、KJ-500（复用 `build_kj500()`）、15 式、CM-11（M60A3 炮塔）、075、056A/903/磐石/基隆/沱江细节；按用户新口径可继续增加 CC-BY/NC 资产。
- P2：轮系/履带/炮塔/旋翼/雷达关节，艏浪/扬尘/加力/凝结尾迹/通信链路。
- P3：LOD1（1.2–1.5 km 切换）、实例化、真机 GPU 验收。
- `model_budget_test.py` 默认使用 P0 可达预算；`--strict` 使用报告 §5.1 的 P1 目标（当前 30 型中仍有多项不达标，属 P1 待办）。
