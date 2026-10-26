# 目标模型 P1 结果

2026-10-26 · 分支 `feat/target-models-v3` · 工作树 `/remote-home/lizilong/shenji_agent_models`

P1 分两批：第一批为自建/缩比近似（075、KJ-500、15 式）；第二批在拿到 Sketchfab token 后下载 8 个候选，接入 7 个 CC-BY 资产（M777/M60A3/AH-64D/HIMARS/055/J-20/P-3C；H/PJ26 作细节 donor）。**8848/8858 未改动**。

## 1. 新增/替换资产

| 资产 | 型号 | 做法 | 面数 | primitive | 体积 | 说明 |
|---|---|---|---:|---:|---:|---|
| `type_075.glb` | 075 两栖攻击舰 | `build_custom.py` 新增 `build_075()` 自建 | 2,216 | 6 | 0.17 MB | 237×36 m，直通甲板/右舷舰岛/坞舱/升降机 |
| `type_kj500.glb` | 空警-500 | KJ-2000 缩比 0.78 | 23,293 | 8 | 0.74 MB | 尺寸对了，机体/发动机仍是 Il-76 风格 |
| `type_ztq15.glb` | 15 式轻型坦克 | 99A 缩比 0.80 | 12,513 | 2 | 0.33 MB | 炮塔/火炮仍为 99A |
| `type_055.glb` | 055 型驱逐舰 | Sketchfab CC-BY，归一/减面/合批 | 33,958 | 5 | 0.27 MB | 180 m；1068→5 primitives |
| `type_p3c.glb` | P-3C 反潜巡逻机 | Sketchfab CC-BY，归一/减面 | 23,096 | 2 | 0.23 MB | 35.6×30.4 m |
| `type_ah64e.glb` | AH-64E 近似 | Sketchfab CC-BY AH-64D | 18,029 | 6 | 1.20 MB | 17.73 m 旋翼直径；后续可换 E 型细节 |
| `type_himars.glb` | M142 HIMARS | Sketchfab CC-BY | 7,472 | 7 | 0.37 MB | 作者注明 "Not my model"，来源待核 |
| `type_j20.glb` | 歼-20 | Sketchfab CC-BY，灰色重着色 | 18,885 | 2 | 0.28 MB | 原模型无纹理，已统一灰蓝金属材质 |
| `type_m60a3.glb` | M60A3 / CM-11 近似 | Sketchfab CC-BY | 9,439 | 2 | 0.19 MB | **CM-11 改用 M60A3 近似**，比 M48A5 更接近（M60A3 炮塔） |
| `type_m777.glb` | M777 超轻型榴弹炮 | Sketchfab CC-BY，归一/减面 | 8,379 | 3 | 0.09 MB | 展开射击状态，含 2 发炮弹；替换原 972 面自建 |
| `type_056a.glb` | 056A 轻护 | 自建细节补强 | 1,660 | 3 | 0.16 MB | 新增烟囱/雷达罩/救生筏/HQ-10/RHIB |

未作为目标类型：
- **H/PJ26 76mm 舰炮**：作为 054A/056A 舰炮细节 donor 保留。

## 2. 集成

- `entityLod.js` 新规则：
  - `055|Renhai|南昌` → `type_055`；`075|两栖攻击|LHD` → `type_075`；
  - `M60A3`、`CM-11` → `type_m60a3`（M48 仍 → `type_m48a5`）；
  - `HIMARS|M142|海马斯` → `type_himars`；
  - `J-20|歼-20` → `type_j20`；`P-3C|Orion` → `type_p3c`；`AH-64|Apache` → `type_ah64e`；
  - `15 式|ZTQ-15` → `type_ztq15`；`空警-500` → `type_kj500`；KJ-2000 保留独立规则。
- `entity_lod_test.js`：192 通过 / 0 失败。
- `CREDITS.md` 39 条；`licenses.json` 39 条，风险 7 条：052c/kj2000/kj500（DCS）、ztd05/phl03（Free Standard）、himars/p3c（来源链待核）。
- `specs.json`、`manifest.json`、两份 `bboxes.json` 均更新到 39 型。

## 3. 量化

| 指标 | P0 后（30 型） | P1 后（39 型） |
|---|---:|---:|
| 文件体积 | 17.48 MB | **21.39 MB** |
| 三角面 | 424,274 | **581,034** |
| primitive | 223 | **263** |
| 台海密集案例 | 506 draws / 937,250 面 | **506 draws / 937,690 面** |

规模增加 9 型（075、KJ-500、15 式、055、P-3C、AH-64E、HIMARS、J-20、M60A3），体积只增加 3.8 MB。台海密集案例使用空警-500×2（`type_kj500`，LOD1 可用）；KJ-2000 不在该案例。

## 4. 验证

- `gltf-transform validate`：39/39 通过。
- `model_budget_test.py`：**ALL PASS (39 assets)**；055 采用 P0 大型舰 35k 的临时上限（33.7k），`--strict` 的 P1 目标 28k 仍待进一步减面。
- `model_license_test.py`：39 条许可，7 条风险。
- Cesium 1.145 + headless Chromium：055、P-3C、AH-64E、HIMARS、J-20、M60A3、075、KJ-500、15 式均成功加载，页面无错误；P-3C 已补 180° yaw 修正。
- 8859 已停止；8848/8858 未重启。

## 5. 已知限制

- 052D 仍用 052C 近似；试做的 052C+平板 AESA 叠加因面板浮空放弃，需要网格级改造或更合适的 052D 资产。
- 空警-500 仍是 Il-76 缩比（涡扇），15 式仍是 99A 缩比（125mm 炮塔）。
- AH-64E 用 AH-64D 近似；M60A3 用 CC-BY 模型；HIMARS/P-3C 来源链需复核。
- 903/磐石/基隆/沱江细节升级未做；056A 已做一轮细节补强。
- 055/052D 的代理炮塔/雷达动画尚未接入；P2 再处理。
- 055 的 bbox 宽 26.27 m，大于公开舰宽约 20 m（+31%），可能是模型外飘/突出物；已记录，建议目视复核。
- P-3C 的机头朝向在审查中发现为 glTF +Z，已在 `MODEL_YAW_FIX_DEG` 登记 180°；055/P-3C 已重新按长度居中、水线/中心原点。

## 6. 下一批

1. 052D 网格级改型（或换资产）；
2. CM-11 真正用 M48H 车体 + M60A3 炮塔拼接（当前整 M60A3 近似）；
3. 903/磐石/基隆/沱江细节；
5. P2：轮系、炮塔、旋翼、雷达、尾流、通信链路；
6. P3：LOD1/实例化与真机 GPU 验收。
