# 评审意见落地方案（4 个方向 · 改动任务单）

> 交付形态：**可执行方案**，不在本任务内完成代码实现（见第 7 节）。
> 本文件与 `NOTES_symbols_and_graphics.md` / `NOTES_term_replacement.md` / `NOTES_audits.md` /
> `NOTES_offline_coverage.md` 同级，遵循同一约定：把"已核实的事实"与"需要拍板的决策"分开写。

## 0. 先说三条会影响方案的核实结果

这三条是写方案前必须查实、且**与评审表述有出入**的地方，直接决定方案怎么落。

**(A) 「预测轨迹」没有现成数据 —— 引擎预测的是"意图"，不是"位置"。**
`engine/intent_predict.py` 的产物是每帧每目标的 `intents[] / intent_probs / primary_intent /
intent_status`（写入数据集 `inference_frames`），**不含未来坐标**。可用的"未来位置"来源只有两个，
且性质不同：

- 每帧目标状态含 `heading` + `speed_kn`（`engine/scenario.py` 帧构造，字段
  `{id, lat, lon, alt, speed_kn, heading, travelled_m, laps, reached_end}`）→ 可做**航向外推**；
- 目标定义含 `waypoints`（`Target.to_dataset_entry()` 已导出）→ 这是**已声明的计划航路**，
  且 `layerRoutes` 已经在画它。

→ **默认**取"计划航路剩余段"作虚线，并标注为**计划/预计**而非"预测"，避免把外推冒充观测；
航向外推作为可配置备选（`PREDICT_MODE = 'planned' | 'kinematic'`）。若你要求的是真正的位置预测，
则必须新增引擎产出（见决策点 1）。

**(B) 「占领区域」在系统里根本不存在 —— 需要新建概念，不是改造现有概念。**
`web/src/plotting.js` 的 `zone` 族只有手工标绘类型：集结地域 / 待机地域 / 防御阵地 / 雷场 /
侦察区域 / 禁飞区 / 海上封锁区。全仓**无** `占领` / `occupied` / `control_zone` 任何实现。
唯一的"区域"数据是 `hotspots`（`{name, bbox, center, span_deg, note, kind}`，
`kind ∈ {strait, island, sea, base, airport, valley, lake, plain}`）。

→ 占领区域只能**新造**，且默认挂靠在 `hotspots` 上（见方向 2）。

**(C) 「活动状态（侦察、协同）」与"不新增未定义状态"存在冲突，但有可满足的解。**
现有**推演状态**是 `frames[i].unit_phase`，内容是**登陆作战阶段**：
`assembly 集结装载 / transit 航渡 / transfer 换乘编波 / assault 抢滩突击 /
consolidate 巩固滩头 / inland 向内陆夺控` —— **里面没有侦察、也没有协同**。
而**意图本体** `engine/data/intent_ontology_v2.json` 里确实有：父类
`RECON_INFORMATION`（侦察与信息）、`SUPPORT_COORDINATION`（支援协同），其下
`RECON` / `SURVEILLANCE` / `TARGET_GUIDANCE` / `ESCORT` / `REINFORCE` 等共 22 个 code，
另有 4 个 status layer（`BACKGROUND` / `UNKNOWN` / `AMBIGUOUS` / `OUT_OF_ONTOLOGY`）。

→ 所以"不新增未定义状态"**可以满足**：用本体既有 code，而非 `unit_phase`。
但两者语义不同（**阶段** vs **活动**），必须二选一或分层。**这是最需要拍板的一点**（见方向 3）。

## 0.1 可复用的现有机制（避免另起通路）

| 机制 | 落点 | 复用于 |
|---|---|---|
| 图层注册表 | `web/src/layerRegistry.js` 的 `LAYERS` 行 + `handler` 或 `kind` sink | 新增任何图层/叠加都走这里，不新起通路 |
| 意图/事实分离 | `desired`（意图）与 `feed`（`ok/pending/failed/hidden/unsupported`） | 占领区域"实时变化"用 `feed` 表达数据可用性 |
| 互斥 | `exclusives`（对称） | 遮罩层与其它叠加的独占关系 |
| 逐帧重算 | `markScene(reason)` + `renderGovernor` | 轨迹/掩码/关系线随 tick 更新 |
| 目标记录 | `state.targets[id] = {def, entity, sensorEntity, trail[], lastTick}` | 轨迹、关系线、活动状态的取数点 |
| 选中 | `state.selected`（**单个 id**）+ `symbology.symbolUrl(icon, aff, {selected})` → `_sel` 变体 | 选中高亮已有基础；掩码需新增 |
| 标绘 | `Plotting.{add, remove, update, focus, setVisible, clearLayer, exportAll, importAll}` | 保留为通用能力；**不再作占领区域来源** |
| 小地图 | `web/src/overview.js`，当前**只画一个** `overview-footprint`（黄框折线） | 掩码需新增图元 |
| 地名 | `web/src/places.js` + 离线 Natural Earth **地名**（城市/地名，非港口/机场） | 方向 4 **不能**直接复用，需新增标注数据 |

## 0.2 共同约束

- 四条方向**相互独立、逐条独立成文**，可分别排期与回滚。
- **默认不改变与四条无关的既有功能**；不新增未在推演中定义的状态。
- 所有新增可视化都必须是**图层行**（可在图层面板单独开关），符合既有"动态图层管理"约定。
- 验收一律走既有自检族，不新造框架：`tools/plot_selftest.py`（前端标绘/图层端到端）、
  `tools/deeplink_selftest.py`（URL 钩子 + 图层意图 + 视图）、`tools/consistency_selftest.py`
  （**意图↔效果**，10 图层 × 2 案例）、引擎侧 `road_selftest.py` / `intent_selftest.py` /
  `intent_set_selftest.py`。
- 数据契约改动**只允许增量字段**，保持既有字段名不变（前端已有读取点不得改名）。

---

## 1. 目标轨迹可视化

### 1.1 问题定位（已核实）

- 轨迹**只有一个来源**：`web/src/app.js` 的 `syncEntities` 内，
  `if (visible && state.layers.tracks && state.tick > rec.lastTick)` 向 `rec.trail` 逐帧追加点，
  驱动 `rec.entity.polyline.show`。
- 因此**没有**"未选中只显示一段"的行为，也**没有**选中/未选中的区分：只要 `layerTracks` 打开，
  一律显示全部已累积轨迹。
- **没有**任何预测轨迹渲染；**没有**掩码（`state.selected` 只切换 `_sel` 符号变体）。

### 1.2 改动对象与范围

| 对象 | 改动 |
|---|---|
| `web/src/app.js` 的 `syncEntities` 轨迹段 | 引入**显示窗口**：未选中时只保留最近 `TRAIL_WINDOW_TICKS` 帧的点，而不是全量 |
| `state.targets[id]` | 增加 `trailFull`（全量）与 `trail`（窗口内）**或**保留全量 + 渲染时切片；默认**保留全量、渲染时切片**（不破坏既有"清空后从第 0 帧重建"的回退逻辑） |
| 选中态轨迹 | 选中目标：历史轨迹**实线**用全量；预测轨迹**虚线**另起一条 polyline（与历史分开两个图元，便于分别设 `dash`） |
| 掩码叠加 | 新增图层行（建议 id `overlay.focusMask`），选中时启用 |
| `web/src/layerRegistry.js` | 新增上述图层行；轨迹保持 `layerTracks` **单一开关**，不新增第二个轨迹开关 |

**默认值（可配置）**

- `TRAIL_WINDOW_TICKS = 120`（≈1 小时推演时间：`TICK_SECONDS=30` × 120）。
- `PREDICT_MODE = 'planned'`（`waypoints` 剩余段）；备选 `'kinematic'`（按 `heading` + `speed_kn`
  外推 `PREDICT_HORIZON_TICKS`，默认 60）。
- 默认点选触发；点选**不**联动目标活动区域/hotspots（按评审"如无特别说明按上述执行"）。

### 1.3 数据流 / 接口影响

- **前端内部**：`frames[i].targets[]` 已有 `lat/lon/alt/heading/speed_kn`；`targets[]` 已有
  `waypoints` → **方向 1 可能完全不需要改引擎**。
- 掩码"高亮选中目标 + 其完整历史轨迹"需要构造**带洞多边形**（Cesium `PolygonGraphics.hierarchy`
  支持外环 + 内环挖空）。**这是 Cesium 能力，不是本仓现有实现**，实现时需确认与现有图元层级、
  深度测试的相容性。
- 若 `PREDICT_MODE='kinematic'`，纯前端计算，无接口变化。
- **接口风险**：若改为引擎产出位置预测，则数据集需新增字段（如 `predicted_frames`），
  且 `build_dataset` 与前端读取点需同步 —— **默认不做**。

### 1.4 交付物与验收

交付物：`app.js` / `layerRegistry.js` 改动；新增图层行；如实现掩码则新增图元构造。

验收（`tools/plot_selftest.py` 增补断言，判据是"**那件事真的发生了**"）：

1. `layerTracks` 打开、未选中 → 轨迹折线顶点数 **≤** 窗口上限；
2. 选中任一目标 → 出现**第二条**虚线 polyline（预测），且历史为实线；
3. 掩码开关打开且选中 → 掩码图元存在且 `show=true`；取消选中 → `show=false`；
4. 用 `tools/consistency_selftest.py` 的同一套判据确认新图层行"意图置上后**场景真的变了**"
   —— **不要只断言 `desired`**（本仓已有多起"意图对了、场景没变"的静默失效）。

---

## 2. 占领区域动态化（去手工标绘）

### 2.1 问题定位（已核实）

- 全仓**无占领/控制区概念**；`zone` 族是手工标绘类型；标绘数据存于 `web/src/situation.js` 的
  `draft.plots`，与推演状态**无任何耦合** —— 与评审描述一致。
- 唯一的区域型数据是 `hotspots`（含 `kind`），且**前端目前完全不读 `hotspot.kind`**（已核对：
  无任何 `h.kind` / `hotspot.kind` 读取点）。
- `web/src/overview.js`（小地图）**只画一个黄框折线** `overview-footprint`，没有任何面/掩码图元
  → 小地图掩码需新增。

### 2.2 改动对象与范围

| 对象 | 改动 |
|---|---|
| `engine/scenario.py` | 新增**占领状态产出**：按 tick 计算每个"活动区域"的归属（red / blue / contested / neutral），归属判据**默认**由推演状态派生 |
| 数据集契约 | 顶层新增增量字段，**默认挂进既有 `frames[i]`**（与 `unit_phase` 同类：逐帧、随帧走，前端无需新增读取通道） |
| `web/src/app.js` | 新增渲染：主视图面图元 + 小地图掩码，随 tick 更新（走 `markScene`） |
| `web/src/overview.js` | 新增"掩码"图元（第二个 entity），复用既有 `footprint` 的更新节拍 |
| `web/src/layerRegistry.js` | 新增图层行（建议 id `situation.occupation`，`group:'TS图层'`） |
| `web/src/plotting.js` / `web/src/situation.js` | **不改代码**；标绘仍是通用能力，只是**不再作为占领区域来源** |

**默认判据（可配置，`OCCUPATION_RULE`）**

- 区域集合 = 当前案例的 `hotspots`（即"目标活动区域"，与评审"掩码应与目标活动区域相关联"对齐）。
- 每个区域每个 tick 的归属：统计该 `bbox` 内 red / blue 目标的数量或**影响力加权**
  （`targets[].influence_km` 已存在于数据集契约）。
- 默认阈值：双方力量比落在 `[1/1.5, 1.5]` 记 `contested`；否则归数量多的一方；**无目标则 `neutral`**。
- 默认**不删**标绘工具入口（它是通用能力），但占领区域**不再读标绘**。

### 2.3 数据流 / 接口影响

- 若挂进 `frames[i]`：前端沿用既有逐帧读取路径，**无新请求、无新事件**。
- 影响面：`engine/scenario.py` 帧构造 + 前端 2 个渲染点（主视图、小地图）。引擎侧需重跑数据集构建
  （`scripts/gen_demo_trajectory.py`），两个案例都要重建 —— 这会影响 `web/data/situations/*.json`
  （该目录不入 git）。
- **接口风险**：`frames[i]` 体积增大（每帧多一个区域表）。默认只存**区域 id → 归属**的紧凑映射，
  不存几何（几何由前端从 `hotspots` 取）。

### 2.4 交付物与验收

交付物：引擎帧构造改动 + 前端两处渲染 + 1 个图层行；重建的数据集。

验收：

1. 引擎侧：扩展 `tools/intent_set_selftest.py` 或 `tools/road_selftest.py` 同级断言 ——
   **同一 tick 的归属可复现**、相邻 tick 不出现"无依据跳变"（跳变必须有目标跨入/跨出该区域）；
2. 前端：`tools/consistency_selftest.py` 纳入该图层行（意图↔效果）；
3. 关键回归：**把标绘全部清空后，占领区域仍随推演变化** —— 这正是"MUST 解耦"的可判定形式；
4. 小地图掩码：断言掩码图元顶点数 > 0 且随 tick 变化。为此**必须给掩码加一个可外部读取的
   debug 出口**（沿用 `web/src/overview.js` 里 `footprintDebug()` 的做法），否则只能靠肉眼。

---

## 3. 群体协作关系表示

### 3.1 问题定位（已核实）

- **无长机/僚机、无编队、无协同**任何实现（全仓零匹配）。
- 空中目标实际有 **12 个**（红 7 / 蓝 5；OOB 共 41 个，台海案例取 31 个），其中战斗机类约 7 个、
  预警机类 5 个 → **现有 OOB 已足够编成"长机-僚机"对，默认不需要新增单位**。
- 空中目标已有 `role` 字段（如"战斗空中巡逻""空中指挥""空中侦察"），但**不是**长/僚机语义。
- 活动状态：见 §0(C)。

### 3.2 改动对象与范围

| 对象 | 改动 |
|---|---|
| `Target.attributes`（`engine/scenario.py`） | **已存在的 `Dict[str, str]`**，已随 `to_dataset_entry()` 导出、前端 `rec.def.attributes` 可读 → **加键即扩展，不动契约**。约定键：`formation_id`、`pair_role ∈ {lead, wingman}`、`paired_with` |
| 数据 | 为 §3.1 的空中目标补齐上述属性（**只补属性，不动航路/速度**，避免改变已验收的推演行为） |
| 活动状态 | **默认**取意图本体的**父类**作为"活动状态"，来源 `inference_frames`，**不新增状态**；`unit_phase`（登陆阶段）保持现状并列显示，两者不互相替代 |
| `web/src/app.js` | 新增关系线渲染：选中时对"关联的敌我目标"画线，线中部叠加文本（复用 `web/src/plotting.js` 已有的**屏幕空间标注**能力 `refreshScreenSpaceDecorations`） |
| `web/src/layerRegistry.js` | 新增图层行（建议 id `situation.relations`） |
| `web/src/intentUi.js` | 活动状态文案复用既有 `intent_names` / `name_zh` 映射，**不新建词表** |

### 3.3 数据流 / 接口影响

- **零契约破坏**：`attributes` 是既有自由字典。
- 引擎侧需重跑数据集（属性进入 `targets[]`）。
- 关系线的"关联"判据需明确，**默认**：① 同 `formation_id` 的配对（我方，实线）；
  ② 双方 `engage_km` / `influence_km` 覆盖交集（敌对，虚线）。**这是默认，必须确认** ——
  评审只说"与其关联的敌我目标"，未定义"关联"。

### 3.4 交付物与验收

交付物：`engine/scenario.py` 属性补齐 + 前端关系线与标注 + 1 个图层行。

验收：

1. 数据集断言：每个 `pair_role='wingman'` 的目标，其 `paired_with` 指向存在的 `lead`，
   且两者 `affiliation` 相同；
2. 前端：选中长机 → 关系线条数 = 该编队成员数 + 被判定为敌对的关联目标数
   （断言**精确数量**，不断言"大于 0"）；
3. 活动状态：断言显示的 code **全部 ∈ 本体 22 个 code ∪ 4 个 status layer**
   （**这条用来兑现"不新增未定义状态"**）；
4. `tools/consistency_selftest.py` 纳入 `situation.relations`。

---

## 4. 特殊场景的地图标注（港口 / 机场）

### 4.1 问题定位（已核实）

- **已有部分机制**：`engine/scenario.py` 的 `_HOTSPOT_BOXES` / `INDIA_CHINA_HOTSPOT_BOXES` 已带
  `kind`，现取值 `strait / island / sea / base / airport / valley / lake / plain`；
  其中**已有一个机场**（`日本 羽田机场`，`kind="airport"`，注释说明它靠热点分级覆盖单独下载
  影像与高程）。
- **但前端完全不读 `kind`** → 现在所有热点渲染无差别，"机场"这个语义白白浪费。
- **无港口**条目；`engine/symbology.py` 的图标集只有 12 个平台类（无港口/机场图标）；
  `web/src/places.js` 是 Natural Earth **地名**，不是设施。

### 4.2 改动对象与范围

| 对象 | 改动 |
|---|---|
| `engine/scenario.py` 两张热点表 | 增补条目：`kind="port"`（台海案例的典型港口）与更多 `kind="airport"`；**沿用既有 `_hotspot()` 构造函数**，不新增结构 |
| 前端热点渲染 | 按 `kind` 区分呈现（默认：`airport` / `port` 显示为**带文字的锚点标注**，其余保持现状） |
| `web/src/layerRegistry.js` | `layerHotspots` 已存在，**默认不新增图层**；若港口/机场需要独立开关，再加一行（建议 id `layerFacilities`），默认**跟随** `layerHotspots` |
| 标注文字 | 复用 `web/src/places.js` 的标注渲染方式或 `plotting.js` 的屏幕空间标注（**以实现时最近者为准**）；地名层 `places.labels` 不动 |

**默认值（可配置）**

- 港口候选：默认取**案例热点表内/邻近**的真实港口（如高雄港、基隆港、厦门港、平潭港 ——
  需在实现时按坐标核实并列入）；**默认不引入外部数据源**。
- 数量：默认每案例 **3–5 个港口 + 2–3 个机场**，与评审"找几个典型"一致。
- 显示层级：默认沿用现有标注的显示层级门槛（**以实现时既有常量为准**）以上显示文字，
  以下只显示锚点。

### 4.3 数据流 / 接口影响

- 纯**数据 + 渲染**改动；`hotspots` 契约不变（`kind` 字段已存在）。
- 引擎侧需重跑数据集；若新增热点超出战区 bbox，需注意**离线瓦片覆盖**（既有机制是"靠热点分级
  覆盖单独下载影像与高程"，见 `_HOTSPOT_BOXES` 中羽田那条注释）→ 新增热点**默认落在既有覆盖
  范围内**；若要出新范围则需同时补瓦片，属**另立工作项**。
- 无新接口。

### 4.4 交付物与验收

交付物：热点表增补 + 前端按 `kind` 渲染 + 数据集重建。

验收：

1. 数据集断言：`hotspots` 中 `kind=='port'` 与 `kind=='airport'` 各 ≥ 默认数量，且每条 bbox
   位于该案例 `THEATRE_BBOXES` 内（或明确列入"需单独覆盖"清单）；
2. 前端：断言按 `kind` 分类后，标注图元数量与数据条数**精确匹配**；
3. 回归：`tools/deeplink_selftest.py` 的 `?case=` 两条 URL 仍需 `pageErrors` 为空，
   且图层数断言随新增行同步更新（该自检当前硬断言 `layers=27` —— **改动图层数时必须同步改它**，
   否则会假失败）。

---

## 5. 横切事项

1. **图层数会被这四条改动**：`web/src/layerRegistry.js` 的行数变化会命中
   `tools/deeplink_selftest.py` 里 `layers=27` 的断言，以及 `tools/consistency_selftest.py`
   的覆盖清单。**每条方向落地时都要同步这两处**，否则自检会假失败（本仓已发生过
   "自检假失败比没有自检更糟"的教训）。
2. **新图层必须走注册表**：既有代码里已修过 4 起"绕过注册表直接改效果 → 被 `apply()` 改回去 →
   静默失效"的缺陷（`?overview=1` / `?places=1` / 鸟瞰图 × 按钮 / `?sensors=1`）。
   新增任何开关**一律** `LayerRegistry.setDesired` + `apply()`，并纳入 `LAYER_PARAM_EXPECT`
   那类断言。
3. **不要只断言 `desired`**：`tools/consistency_selftest.py` 的判据是"意图翻转后**场景/面板**
   真的跟着变"。四条方向的新图层都应纳进去。
4. **每层给一个外部可读的取证出口**（如 `web/src/overview.js` 的 `footprintDebug()`）：
   掩码、关系线、占领区域这三类都只有"画得对不对"这一种直观判断，**没有外部读数就只能靠肉眼**
   —— 这是本仓已经吃过一次的亏。
5. **引擎侧重建数据集后必须重跑**：`road_selftest` / `intent_selftest` / `intent_set_selftest`；
   前端五套全跑。

## 6. 需要拍板的决策点（集中列出）

| # | 决策 | 默认（如无特别说明按此执行） |
|---|---|---|
| 1 | **"预测轨迹"指什么** | 默认 = 已声明的 `waypoints` 剩余段，标注为**计划/预计**；备选 = 航向外推；若要求真正的位置预测，需**新增引擎产出**（另立工作项） |
| 2 | **"活动状态"用哪套词表** | 默认 = **意图本体的 6 个父类**（含 `RECON_INFORMATION` 侦察与信息、`SUPPORT_COORDINATION` 支援协同），**不新增状态**；`unit_phase`（登陆阶段）保持并列显示。若要的是 `unit_phase` 那套，则"侦察/协同"**在现有定义里不存在**，需确认是改本体还是接受阶段语义 |
| 3 | **"关联的敌我目标"如何定义** | 默认 = 编队配对（我方实线）+ 覆盖范围交集（敌方虚线） |
| 4 | **线型语义** | 默认 = 实线我方协同 / 虚线敌对关系（评审只写了"实线和虚线"，方向未指定） |
| 5 | **占领区域判据** | 默认 = 按热点 bbox 内双方数量/影响力加权，比例阈值 1.5，无目标为 `neutral` |
| 6 | **是否保留标绘工具入口** | 默认 = **保留**（通用能力），但**不再作为占领区域来源** |
| 7 | **港口/机场具体清单** | 默认 = 每案例 3–5 港口 + 2–3 机场，落在既有战区内、复用既有热点覆盖；不引入外部数据源 |

## 7. 明确不在本方案范围内

- 不实现任何代码（本任务交付形态为改动任务单）。
- 不改动与四条方向无关的既有功能；不改 `web/src/places.js` 的地名来源；
  不改 `web/src/plotting.js` 的 24 个标绘类型。
- 不为方向 1 新增引擎位置预测（除非决策点 1 选择"要求真正预测"）。
- 不引入外部港口/机场数据源（默认）。
- 不调整 `TICK_SECONDS` / `TICK_COUNT` / 既有航路与速度（方向 3 只补 `attributes`，不动运动学，
  以免让已验收的推演行为变化）。

## 8. 建议实施顺序

方向 **4**（纯数据 + 渲染，风险最低，且能立刻验证"热点 `kind` 通路"）
→ 方向 **1**（前端为主，可完全不碰引擎）
→ 方向 **3**（补 `attributes` + 关系线，零契约破坏）
→ 方向 **2**（唯一需要新增帧字段并重建数据集的一条，放最后）。

每条独立提交、独立跑对应自检；方向 2 合入后**四条一起做一次全量回归**（前端五套 + 引擎三套）。

---

# 新会话开工须知

> 本节是给**没有本次上下文的新会话**看的。上面第 0-8 节是方案本身。
> 开工前**先把第 6 节的 7 个决策点拿到答复**，再按第 8 节顺序动手。

## 9. 环境与工作流（硬约束，先读这一节）

### 9.1 工作方式

- **本机只做写代码与语法检查**；**一切执行、验证、渲染都在远程容器**。
  这是既定约束（"不要在本机跑代码和实验"），不要因为"跑一下更快"而违反。
- 仓库权威在**远程**：主机 `situ155`，仓库 `/remote-home/lizilong/shenji_agent`，
  项目目录 `situation3d`。
- 同步方式：本地改完 → `scp` 到远程同名路径 → 在远程执行。
- git 在远程：`/remote-home/lizilong/.local/bin/git`。
  **本机没有 `.git`**，不要试图在本地提交。

### 9.2 ⚠ 本地副本是**部分镜像**，不是完整仓库

| 本地**有** | 本地**没有**（只在远程） |
|---|---|
| `web/index.html`、`web/src/*`（16 个 js + `style.css`）、`engine/*`、`engine/data/*`、`server/*`、`tools/*`（23 个 `.py`）、`web/data/base.json`、`README.md`、`.gitignore` | `web/assets/symbols/*`（40 个 svg）、`web/vendor/`、`web/tiles/`、`web/poc/`、`NOTES_audits.md`、`NOTES_offline_coverage.md` |

→ 需要读右侧这些文件时，**从远程 `scp` 回来再读**，不要当成"不存在"。
→ `web/data/situations/*.json`（数据集）也在远程，本地方便起见可拉一份看结构。

### 9.3 文件格式：**必须 LF**

- 写文件一律用能保证 LF 的方式；**不要用会产出 CRLF 的批量写行接口**。
- 提交/部署前自查：`tr -cd '\r' < 文件 | wc -c` 必须是 **0**。

### 9.4 重启规则

| 改了什么 | 是否需要重启服务 |
|---|---|
| `web/src/*.js`、`web/index.html`、`web/src/style.css` | **不需要**，刷新页面即可 |
| `server/serve.py`、`.env` | **必须** `sh tools/restart_server.sh 8848` |
| `engine/*.py` | 不涉及服务；但**数据集需重建**（见 9.6） |

### 9.5 提交纪律

- **显式路径**提交，不要 `git add -A`。
- **不要提交**：`web/data/*.json`、`web/data/situations/`、`web/tiles/`、`web/vendor/`、
  `tools/shots/`、`.env`。
- 提交前扫密钥（天地图 key、Cesium ion token）。

### 9.6 数据集重建（方向 2/3/4 会用到）

引擎侧改完后必须重建案例数据集，否则前端读到的还是旧数据：

```
python3 scripts/gen_demo_trajectory.py <case_id>
```

案例 id 由 `GET /api/situations` 列出（固定目录 `web/data/situations/`）。
已知两个案例：`taiwan_landing`（31 目标）、`india_china_west`（10 目标）。
**重建后要重跑引擎三套自检**。

## 10. 自检清单（验收一律走这些，不要另造框架）

位于 `tools/`：

| 自检 | 覆盖 | 何时跑 |
|---|---|---|
| `consistency_selftest.py` | **意图↔效果**：10 图层 × 2 案例，判据是"翻转 `desired` 后场景/面板真的变了" | 任何图层/开关改动 |
| `deeplink_selftest.py` | 16 条 URL + URL 钩子是否真的点亮了对应图层 + 鸟瞰图黄框 | 任何启动链/钩子/视图改动 |
| `plot_selftest.py` | 标绘端到端 | 标绘/图元改动 |
| `road_selftest.py` | 引擎：陆上航路 | 引擎改动 |
| `intent_selftest.py` | 引擎：意图链路 | 引擎改动 |
| `intent_set_selftest.py` | 引擎：意图集合（CES） | 引擎改动 |

本环境实测**六套全跑约 50 分钟**；`deeplink_selftest` 单跑约 15–20 分钟。
它们在远程执行，不要在本机跑。

## 11. 写代码时必须遵守的四条（都是本仓已付过代价的）

1. **新增开关一律走图层注册表**（`LayerRegistry.setDesired` + `apply()`），不要直接调模块 setter。
   本仓已修过 4 起"绕过注册表 → 被 `apply()` 改回去 → **静默失效**"的缺陷。
2. **不要只断言 `desired`**。`consistency_selftest.py` 的判据是"意图翻转后**场景/面板**跟着变"。
   只断言意图，会漏掉"意图对了、场景没变"这一整类。
3. **自检里的假失败与假通过都要当 bug 修**。本仓记录过：假失败会让人去修没坏的东西；
   假通过（空过）更隐蔽。**全绿的报告里少了一条断言，比一条断言失败更难发现。**
4. **生成/改写代码的脚本：先 `ast.parse` 校验、再写盘**，且校验要在**同一份即将写出的内容**上做；
   部署后还要用**独立手段**确认服务上确实是新内容（例如 `curl` 数元素个数、比文件大小）。

## 12. 逐步执行清单（与第 8 节顺序一致）

每一步都是**原子任务**，做完即跑对应自检、独立提交。

### 步骤 A — 方向 4（港口/机场标注），风险最低，先做

- [ ] A1 读 `engine/scenario.py` 的 `_HOTSPOT_BOXES` 与 `INDIA_CHINA_HOTSPOT_BOXES`，确认 `_hotspot()` 构造签名（`name, bbox[w,s,e,n], note, kind`）
- [ ] A2 按坐标核实并增补港口条目（`kind="port"`）与机场条目（`kind="airport"`）；**bbox 必须落在既有战区内**，否则同时要补离线瓦片（属另立工作项）
- [ ] A3 前端：找到热点渲染点（`app.js` 的 `createHotspots` 一族），按 `kind` 分支出"带文字锚点"；此时才需要决定是否新增 `layerFacilities` 图层行
- [ ] A4 重建两个案例数据集
- [ ] A5 验收：数据集断言 `port`/`airport` 条数；前端断言标注图元数**精确等于**数据条数；`deeplink_selftest` 两条 `?case=` 仍 `pageErrors` 为空

### 步骤 B — 方向 1（轨迹窗口 / 选中轨迹 / 掩码）

- [ ] B1 定位 `app.js` 的 `syncEntities` 轨迹段（判据：`state.layers.tracks` 与 `rec.lastTick`）与 `rec.trail` 追加逻辑
- [ ] B2 引入显示窗口（默认 `TRAIL_WINDOW_TICKS = 120`）；**保留全量 `trail`**，渲染时切片，避免破坏"清空后从第 0 帧重建"的回退逻辑
- [ ] B3 选中态：历史实线（全量）+ 预测虚线（两条独立 polyline）
- [ ] B4 掩码：新增图层行 + 带洞多边形图元（外环全图、内环挖空选中目标与轨迹范围）
- [ ] B5 给掩码加**外部可读的 debug 出口**（照 `web/src/overview.js` 的 `footprintDebug()` 样子），否则"画得对不对"只能靠肉眼
- [ ] B6 验收：`plot_selftest` 加 4 条断言（窗口顶点数上界、选中后出现第二条虚线、掩码 `show` 跟随选中、新图层纳入 `consistency_selftest`）

### 步骤 C — 方向 3（长机/僚机 + 活动状态 + 关系线）

- [ ] C1 用 `Target.attributes`（**已存在的 `Dict[str, str]`**）承载 `formation_id` / `pair_role` / `paired_with`；**不动契约**
- [ ] C2 为空中目标补属性。已核实空中目标 12 个（红 7 / 蓝 5），**够编成对，默认不新增单位**；只补属性、不动航路与速度
- [ ] C3 活动状态：默认取 `engine/data/intent_ontology_v2.json` 的 6 个父类，来源 `inference_frames`；`unit_phase` 并列显示
- [ ] C4 关系线渲染 + 线中部文本（复用 `plotting.js` 的 `refreshScreenSpaceDecorations`）
- [ ] C5 验收：`wingman → paired_with` 指向存在的同阵营 `lead`；关系线条数**精确断言**；活动状态 code **全部 ∈ 本体 code ∪ status layer**；纳入 `consistency_selftest`

### 步骤 D — 方向 2（占领区域动态化），放最后（唯一要新增帧字段的）

- [ ] D1 `engine/scenario.py` 帧构造里新增归属表（**挂进既有 `frames[i]`**，与 `unit_phase` 同类，只存 `区域id → 归属` 的紧凑映射，不存几何）
- [ ] D2 归属判据按 `OCCUPATION_RULE` 默认（热点 bbox 内双方数量或 `influence_km` 加权，比例阈值 1.5，无目标 `neutral`）
- [ ] D3 前端主视图面图元 + `overview.js` 小地图掩码图元（第二个 entity，复用 `footprint` 的更新节拍）
- [ ] D4 新增图层行（建议 `situation.occupation`，`group:'TS图层'`）
- [ ] D5 小地图掩码同样要给 debug 出口
- [ ] D6 验收：**把标绘全部清空后占领区域仍随推演变化**（这是"解耦"的可判定形式）；同一 tick 归属可复现；相邻 tick 无"无依据跳变"

### 收尾

- [ ] E1 四条都合入后，跑一次**全量回归**：前端五套 + 引擎三套
- [ ] E2 同步更新 `tools/deeplink_selftest.py` 里 `layers=27` 之类的**硬编码图层数**断言（新增图层行会命中它，不改就是假失败）
- [ ] E3 更新 `tools/consistency_selftest.py` 的覆盖清单（新增图层要进 `SPECS`）