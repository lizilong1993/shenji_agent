# geo-engine 后续待办清单（2026-10-21）

> 基线：`feat/geo-engine` 最新 `5e9cc3e`（工作树干净）；全量 152 项测试通过。
> 说明：本清单是下一阶段的可执行待办，按依赖排序；涉及 web UI 的条目必须先征得用户确认。

## 当前已完成（不再重复）

P0 基线/规则语义、P2 骨架、P3–P6 引擎核心、在线/离线双路径 envpack、真实 OSM/DEM（经 Clash 反向隧道）、
JEV 适配与规则 vs JEV 的意图/行为评测、建议层 `World.intel`、COT spotting/physics/commander 合并。

## P0（下一阶段必做，建议 2–3 周）

### 1. 补齐 P1 数据管线：PBF / GEBCO / COG
- 现状：Overpass + Terrarium 在线/离线可用；PBF 导入、Copernicus GLO-30、GEBCO 15″ 水深、
  高度回退与 envpack v2 分块已完成（2026-10-21，见 P1_data_pipeline.md §6）；图层仍为 JSON，无 COG/FlatGeobuf。
  高度覆盖 28.1%（目标 80% 未达标，受 GBA/Overture 台湾数据稀疏限制）。
- 交付：
  - `osmium`/.osm.pbf 导入器（离线大范围，替代 Overpass 限流）；
  - Copernicus GLO-30 或现有 20 m DEM 接入；GEBCO 15″ + 近岸 ENC/港口测深的 `water.depth` 栅格；
  - 建筑高度回退（Overture `num_floors×3 m`）、道路 bridge/tunnel/maxheight/maxweight 字段；
  - envpack v2：COG/GeoParquet 或至少按 5 km 分块 JSON + manifest 版本；运行时 `EnvPackEnvironment` 读取 v1/v2。
- 验收：运行时零在线请求 ✓；路网最大连通分量 96.8% ✓；建筑高度面积覆盖 28.1% ✗（待更换高度源）；
  水域/水深无空洞 ✓（20×20 无空值）；在线/离线图层与分块 sha256 一致 ✓。
- 待办：分块按需加载（当前整包读取）、Overture/GBA 高度源改进、ENC/港口测深、COG/GeoParquet。
- 依赖：本机 Clash 隧道（`172.16.10.254:7897` → `ssh -R 27980`）；GDAL/osmium 工具链（服务器已有 `ogr2ogr`）。

### 2. 补齐 P4 规则：间瞄 / 雷场 / 桥隧 / 制导
- 现状：直瞄 CRT/高差/遮蔽/2d6 已实现；桥隧承载、雷场/路障/工事、间瞄计划/校射/散布已完成
  （2026-10-22，见 P4_rules.md：eb20671 / 35165c2 / efb1a7f）；导弹飞行段/制导与 F0/F1/F2 未实现。
- 交付：
  - 间瞄计划/校射/散布：`无校射/格内校射` 表（`rules/data/crt.json` 已含原始表）→ 几何爆炸区/偏移；
  - 雷场/路障/工事：多边形区域 + 通行谓词 + 2d6 伤害/标定表；
  - 桥隧：`bridge_mlc_class/max_weight/clearance/width` 进入通行与路径规划；
  - 制导/导弹：`F0/F1/F2` 语义（P0 已判定不影响直瞄裁决）在飞行段模型中实现；
  - 压制恢复、75 s 状态机、同格 25 s 周期、炮兵不可机动射击等规则文本项。
- 验收：权威表逐项一致 ✓；单发分布偏差 <1 pp（12 万采样，非正式 TOST）✓；场景级对拍无法连 SDK，
  以静态 diff + 单测审计替代（golden replay 仅覆盖 wargame_core）；新增 19 项 P4 单测 + 200 次间瞄可射击循环无非法状态 ✓；
  待办：导弹飞行段/制导、F0/F1/F2 语义、布雷/扫雷工兵动作、工事掩体加成。
- 依赖：`RuleContext` 扩展（爆炸区/区域谓词/制导状态）；`rules/data` 补充偏移/雷场表。

### 3. 性能：300/500 实体 p95 <100 ms
- 现状（2026-10-22）：稳态 300 实体 p95 19.7–46 ms；500 实体 p95 51 ms（宿主机负载≈40）～93–106 ms
  （负载≈50，处于 100 ms 边界）；冷启动首 2–5 tick 仍 0.1–1.7 s；事件哈希与优化前一致。见 PERFORMANCE.md。
- 交付：
  - 传感器空间桶/批量 LOS（NumPy 或预计算格线缓存）；交战候选裁剪；
  - 路径规划缓存/瓦片内 Theta*；必要时 Numba/C++ 热点；
  - 基准脚本输出 p50/p95/内存，纳入回归门槛。
- 验收：300 达标 ✓；500 在负载≈40 达标、负载≈50 处于边界（回归门槛 120 ms 防抖动）；
  待办：安静宿主机复测 500、冷启动预热、候选批量化/整数键优化。
- 依赖：性能基准脚本 `scripts/bench_geo_engine.py`；不改变规则语义。

### 4. P6 案例：台海陆战贯通 + 中印想定
- 现状：台海 1440 tick 海空交战/CZML 可跑；陆战抢滩/下船后战斗仍少；中印只有地图 profile。
- 交付：
  - 登陆舰→两栖车/步兵下船→滩头→内陆夺控的完整链路；蓝方装甲反击；
  - 中印西段战役想定（单位/目标/阶段）补齐并跑通 CZML；
  - 结果对比旧引擎/脚本结局，记录差异与原因。
- 验收：台海案例陆海空全链路有事件、目标归属变化；中印案例可回放；无非法状态。
- 依赖：P1 数据（Anping 高精度路网/建筑）、P4 桥隧/间瞄规则。

## P1（随后 2–4 周）

### 5. JEV 标注数据、GPU 与复测
- 现状：规则基线意图/行为 1.00 vs JEV 0.50（受控合成集）。GPU 复测已完成（2026-10-21，4B v5 单卡 3090：0.42–0.57 s/窗口，意图/行为 0.50/0.25，见 JEV_integration_research.md §7.6），仍仅建议层。
- 交付：
  - 从 replay/campaign 日志切 100–200 个标注窗口（意图/下一行为/提前量）；
  - GPU 部署 openjev（4B v5 优先）或 SGLang；推理服务与引擎解耦、超时回退；
  - 用 `scripts/evaluate_intent.py` 同口径复测 F1/top-3/Brier/ECE/延迟。
- 验收：意图 macro-F1、行为 top-3 达到约定阈值；若不达标，继续保持规则基线，不闭环。
- 依赖：标注人力、模型许可确认（GPU 已可用：155 的 3090 经 CUDA 直用，NVML 故障不影响计算）。

### 6. CZML/前端回放（需用户确认 UI）
- 现状：CZML 快照可用；前端尚未接 `cartographic` 逐帧位置/事件。
- 交付：逐 tick 位置/朝向/事件 CZML；前端解析适配、回放控制、图层开关；不改现有案例数据。
- 验收：Cesium headless 无报错、事件与引擎一致；用户确认 UI 变更保留。
- 依赖：用户对 web UI 的明确授权；前端协议变更评审。

### 7. 合并前复核、推送与交接
- 交付：独立 reviewer 复核本分支；全量测试（含 golden/TW）；合并回 `main`；
  推送 `zililong1993/shenji_agent`；更新交接文档与已知限制。
- 验收：主干测试全绿；8848 不受影响；分支/worktree 清理。
- 依赖：P0 条目完成或明确降级；用户确认合并。

## 建议顺序

1 → 2 → 3（可与 4 并行）→ 4 → 5/6（视 GPU/UI 授权）→ 7。

## 当前阻塞/注意

- GPU：155 的 `nvidia-smi` 因一张故障卡 NVML 不可用，但 CUDA 计算正常（4×3090 可用，单卡按需使用、用完释放）；JEV 4B GPU 复测已完成（§7.6）。
- 网络：服务器直连外网不通，需本机 Clash + SSH 反向隧道（`reference_clash_proxy`）。
- 前端：任何 web UI 改动前必须询问用户，已改需披露并确认保留/回滚。
- 性能：p95 目标未达标，P6 验收前必须给出基准与优化结论。
- 数据许可：OSM ODbL、Overture ODbL、OpenAIP 商用前需法务确认；`.engine_config`/密钥不入库。
