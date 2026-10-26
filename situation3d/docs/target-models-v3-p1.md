# 目标模型 P1（第一批）结果

2026-10-26 · 分支 `feat/target-models-v3` · 工作树 `/remote-home/lizilong/shenji_agent_models`

本次在 P0 基础上新增/替换 3 型目标模型；**052D 暂未替换**（见 §4），外部 CC-BY 模型因缺少 Sketchfab 下载令牌暂未拉取。8848/8858 未改动。

## 1. 新资产

| 资产 | 类型 | 做法 | 面数 | primitive | 体积 | 说明 |
|---|---|---|---:|---:|---:|---|
| `type_075.glb` | 075 两栖攻击舰 | `build_custom.py` 新增 `build_075()` 参数化自建 | 2,216 | 6 | 0.17 MB | 237×36 m，直通甲板/右舷舰岛/坞舱/升降机 |
| `type_kj500.glb` | 空警-500 | KJ-2000 缩比 0.78 | 23,293 | 8 | 0.74 MB | 36.3×39.4 m；机体/发动机仍是 Il-76 风格，属近似 |
| `type_ztq15.glb` | 15 式轻型坦克 | 99A 缩比 0.80 | 12,513 | 2 | 0.33 MB | 8.8×3.0×2.2 m；炮塔/火炮仍为 99A，属近似 |

三型均经 P0 同管线（纹理≤1024 WebP、palette/join、simplify、meshopt）处理，`gltf-transform validate` 通过。

## 2. 集成

- `entityLod.js` 新增/调整规则：
  - `075|两栖攻击|LHD` → `type_075`（优先于 071）；
  - `15 式|ZTQ-15` → `type_ztq15`（优先于 99A）；
  - `KJ-2000` 保持 `type_kj2000`；`空警-500|KJ-500` → `type_kj500`。
- `entity_lod_test.js` 期望值更新，**186 通过 / 0 失败**。
- `CREDITS.md` 新增 3 行：075 自建；KJ-500 为 DCS KJ-2000 缩比（上游风险）；15 式为 99A 缩比（CC-BY）。
- `licenses.json` 33 条，风险 5 条（052c/kj2000/kj500 DCS；ztd05/phl03 Free Standard）。
- `specs.json`、`manifest.json`、两份 `bboxes.json` 已更新为 33 型。

## 3. 量化

| 指标 | P0 后（30 型） | P1 后（33 型） |
|---|---:|---:|
| 文件体积 | 17.48 MB | **18.72 MB** |
| 三角面 | 424,274 | **462,296** |
| primitive | 223 | **239** |
| 台海密集案例 | 506 draws / 937,250 面 | **506 draws / 873,606 面**（KJ-500 替换 KJ-2000，省 6.4 万面） |

新增的 075 目前未进入现有案例目标；后续案例若使用 075，会直接命中 `type_075`。

## 4. 未完成 / 阻塞

- **052D**：试过在 052C 模型上叠加四块平板 AESA（见 `/tmp/build_052d_overlay2.py`），但受原模型舰桥曲率/坐标限制，面板会浮在甲板上方，近距离明显穿帮，未采用。`build_052d()` 的初版过于方块化，也不适合替换现有 052C。现阶段 052D 仍用 052C 近似；下一步需要：
  1. 提供 Sketchfab API token，下载更合适的 052D；或
  2. 在 Blender 中按 052C 舰桥表面重新拓扑/替换弧型雷达罩（需更细的网格编辑）。
- **外部 CC-BY 资产**：Sketchfab 下载端点返回 401（`Authentication credentials were not provided`），服务器上已无有效 token。M777/M60A3/AH-64D/HIMARS/055/J-20/P-3C 候选已调研并记录在评估报告 §3.3，但**尚未下载**。需要 token 或手工提供 GLB。
- **CM-11**：需要 M60A3 炮塔 donor；候选 [M60A3](https://sketchfab.com/models/3d298af87c2c4a0691e61ab8ee9ad803) 下载受阻，暂用 M48A5。
- **M777 / 056A / 903 / 磐石 / 基隆 / 沱江细节升级**：本轮未做，留待下一批。
- 15 式/KJ-500 的缩比模型只解决“尺寸/类型”问题，细节仍是 donor 型号；报告与 `specs.json` 已标注近似关系。

## 5. 验证

- `gltf-transform validate`：33/33 通过。
- `model_budget_test.py`：**ALL PASS (33 assets)**；`--strict` 仍显示 P1 目标缺口（15 式面数、J-16/PLZ-05 等），属下一批。
- `model_license_test.py`：33 条许可，5 条风险。
- `entity_lod_test.js`：186/186。
- Cesium 1.145 + headless Chromium：075、KJ-500、15 式均成功加载，页面无错误。
- 8859 已停止；8848/8858 仍在运行且未重启。

## 6. 下一批

1. 拿到 Sketchfab token 后：下载 M777、M60A3、AH-64D、HIMARS、055、J-20、P-3C，按 P0 管线归一/减面；
2. CM-11：M48A5 车体 + M60A3 炮塔 donor；
3. 052D：网格级改造或外部资产；
4. 056A/903/磐石/基隆/沱江细节升级；
5. P2 关节/特效（轮系、炮塔、旋翼、尾流、通信链路）。
