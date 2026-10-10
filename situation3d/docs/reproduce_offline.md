# P2 离线资产复现说明（C1）

> 目的：他人**在离线环境**凭清单与脚本恢复 P2 全量资产并跑通自检；
> **不可复现项被显式声明**，而不是留给对方自己撞。

---

## 1. 为什么需要这份文档

报告 §5.4：**「资产与依赖是最大可复现性风险」**。
下列路径被 `.gitignore` 忽略、**不在版本控制内** ——
换机或清空工作区后，**无法凭仓库恢复**：

| 类别 | 路径 | 文件数 | 体积（实测 2026-10-10） |
| --- | --- | --- | --- |
| tiles | `web/tiles/` | 76,946 | **1384.83 MB** |
| vendor | `web/vendor/` | 396 | 22.46 MB |
| situations | `web/data/situations/` | 2 | 53.62 MB |
| roads | `engine/data/roads/` | 2 | 24.92 MB |
| vector | `engine/data/vector/` | 3 | 0.43 MB |
| intent_model（**可选**） | `engine/data/intent_model.json` | 1 | 0.05 MB |
| **合计** | — | **77,350** | **1486.3 MB** |

> ⚠ **`.gitignore` 里的注释写的是「~10k files, ~200 MB」——实测已是 7 倍。**
> 该注释已过期，**待改**（本轮未改，属仓库规则文件，另行确认）。

---

## 2. 环境前置

| 项 | 要求 | 备注 |
| --- | --- | --- |
| Python | **3.10** | 远程用 `/remote-home/lizilong/py310env.sh` 提供 |
| 磁盘 | **≥ 4 GB 可用** | 归档 + 解出后的资产两份 |
| 网络 | **不需要**（离线即可） | 打包与校验均不联网 |
| 服务 | `python3 server/serve.py --port 8848` | 自检与 `plot_selftest` 需要它在跑 |

---

## 3. 打包（在有资产的一方执行）

```bash
cd <situation3d>
bash scripts/make_offline_bundle.sh dist .
```

产物（`dist/`）：

| 文件 | 说明 |
| --- | --- |
| `offline_assets_manifest.json` | **权威清单**：相对路径 / 字节数 / SHA-256 / 类别 / 是否可选 |
| `offline_assets_manifest.csv` | 同上，CSV（便于 `diff` 与人工核对） |
| `offline_assets.tar.gz` | 资产归档 |
| `offline_assets.tar.gz.sha256` | 归档校验和 |
| `BUNDLE_README.md` | 包内说明（含一行恢复命令） |

**实测耗时**：仅生成清单就 **9分59秒**（77k 小文件，I/O 受限）。

### 包版本号

```
package_version = sha256(所有「必需」条目的 (相对路径, 字节数, sha256) 规范序列)[:16]
```

- **任何文件增删改都会改变版本号**，不依赖人工维护；
- **`intent_model.json` 是可选产物，不参与版本号** —— 否则「缺模型」会把整个包判成另一个版本。

---

## 4. 校验与恢复

```bash
# 只校验，不写盘
python3 scripts/verify_restore_offline_assets.py \
    --manifest dist/offline_assets_manifest.json --target . --mode verify

# 只看计划（秒级，不读哈希）
python3 scripts/verify_restore_offline_assets.py --mode plan

# 只补缺失项（从手边另一份拷贝）
python3 scripts/verify_restore_offline_assets.py \
    --manifest dist/offline_assets_manifest.json --target . \
    --source /path/to/another/copy --mode restore
```

**退出码**：`0` 一致；`1` 有不一致；`2` 用法/路径错误。

**恢复后脚本会重新完整校验一遍**，而不是相信刚才的复制 ——
「复制成功」与「内容正确」是两件事。校验脚本与恢复脚本**分离**，
便于单独停用（C1【风险与回滚】的要求）。

### 实测结果（2026-10-10，远程）

| 步骤 | 结果 |
| --- | --- |
| 全量 verify（77,350 项） | **ok=77350 / missing=0 / hash_mismatch=0**，耗时 5分02秒 |
| 跨工具复核 | 随机 6 项用 `sha256sum` 独立计算，**全部一致** |
| 删除 3 项（覆盖 tiles/vendor/situations 三类）后 verify | **missing=3**，且只报这 3 项 |
| restore 后 | 逐项 `sha256sum` 复核 **3/3 一致** |
| 恢复后全量 verify | 回到 **77350/77350** |

---

## 5. `intent_model.json` 缺失时的行为（C1 步骤 4）

**它是可选产物**，缺失时**规则回落**，不阻塞：

| 场景 | 行为 |
| --- | --- |
| `engine/data/intent_model.json` 存在 | 引擎用学习到的意图模型 |
| **缺失** | 引擎**回落到规则判定**；界面与自检**照常运行** |
| 清单校验 | 标 `optional: true`，缺失**不判失败**（`--strict-optional` 可改为判失败） |

> 回落的具体实现在 `engine/intent.py` 的 `load_ontology()` / 意图判定路径；
> 本说明只声明**契约行为**：缺失不阻塞复现。
> **本轮未修改引擎代码**，故「回落」是既有行为，我只是把它**写进契约并纳入校验口径**。

---

## 6. 启动服务与自检

```bash
# 起服务（若未起）
python3 server/serve.py --port 8848 &

# 效果级自检（图层意图 ↔ 效果）
python3 tools/consistency_selftest.py

# 两案例标绘自检
python3 tools/plot_selftest.py --url "http://127.0.0.1:8848/?case=taiwan_landing"
python3 tools/plot_selftest.py --url "http://127.0.0.1:8848/?case=india_china_west"
```

---

## 7. 不可复现项（**显式声明**）

| 项 | 为什么不可复现 | 现状 |
| --- | --- | --- |
| **遥测瓦片的上游再拉取** | `imagery` 来自第三方在线服务（Esri 等），**离线环境取不到**；且上游影像会随时间更新，**同一 z/x/y 的内容未必可复现** | 只能用归档；**归档即唯一真源** |
| **`web/vendor/`（Cesium）** | 由 `tools/fetch_cesium.sh` 从上游拉取，需外网 | 同左，用归档 |
| **数据集 `situations/*.json`** | 是 `scripts/gen_demo_trajectory.py` 的**随机生成产物**（含随机种子与轨迹），重新生成**不会逐字节相同** | 用归档；重新生成只能得到**等价但不相同**的数据 |
| **GH3D 建筑体块覆盖** | 实测 `gh3d` **只覆盖厦门+福州**，台湾近乎为空 | `web/tiles/manifest.json` 的 `building_blocks` 已切回 `wsf3d`；`coverage_note` 已记录 |
| **闭源引擎相关** | `land_wargame_train_env` 为闭源 whl，且本机**无 `.engine_config`**（MAC 绑定鉴权不通过） | 见 `docs/deployment_auth.md`；**引擎侧指标由平台方验收** |

---

## 8. 资产许可与来源

- `web/tiles/`：影像来自第三方在线地图服务（Esri World Imagery 等），
  地形与建筑高度来自 **DLR EOC GeoService WMS**（见 `web/tiles/manifest.json` 的 `wsf3d.attribution`）；
  `gh3d` 为 3D-GloBFP 逐建筑高度栅格化，**CC BY 4.0**。
- `web/assets/models/`：Kenney Watercraft Kit，**CC0**（见同目录 `License.txt`）。
- **`web/vendor/`（Cesium）** 的许可随其上游分发，见 `tools/fetch_cesium.sh` 与 vendor 内许可文件。
- **再分发前请自行确认各上游的许可与使用条款**；本说明只记录来源，不构成授权。

---

## 9. 本文件的边界

- **未做**：真实跨机传输演练（无第二台目标机）；本轮的打包与恢复都在**同一远程工作区**内完成
  （「源」用的是同机另一份目录，验证的是**校验与恢复逻辑**，不是网络传输）。
- **未改** `.gitignore`（其「~200 MB」注释已过期，见 §1）。
- **未改**引擎代码；§5 的回落行为是既有实现，本轮只做**契约化与校验口径**。
- 归档未做分卷；若目标环境有单文件体积限制，需自行 `split`。
