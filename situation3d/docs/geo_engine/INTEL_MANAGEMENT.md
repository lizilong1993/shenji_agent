# 情报管理：人工情报、情报源与多领域扩展

状态：2026-10-26，`feat/intel-management`。本文先记录现有架构分析，再说明本轮实现与多领域扩展方案。

## 1. 现状分析

### 1.1 数据流

```
公开源连接器（USGS/GDACS/FIRMS/ADS-B/EONET/GDELT）
  → Observation（双时间戳、来源、许可、置信度）
  → IntelBundle（案例 bbox/时间窗/源白名单）
  → IntelGateway / IntelScheduler（按 side、future-only、TTL）
  → World.intel_cues / 复盘 / 界面
```

- `intel/model.py` 的 `Observation` 是自动情报的唯一结构；`intel/fusion.py` 做跨源事件聚类；`intel/case.py` 按案例 bbox/域/源/关键词过滤。
- 自动情报的质量字段已有 `confidence`、`license`、`attribution`、`provenance`、`geo_precision`。
- 前端原先没有“情报”界面：只有图层注册表、事件流、目标属性等通用组件；本轮新增统一情报模态框。
- 服务端 `serve.py` 是标准库 HTTP 服务，已有 `/api/*` JSON 路由；`var/intel/` 已 gitignore。

### 1.2 缺口

1. 无人工情报结构：不能录入、编辑、撤销，也没有“人工优先”的冲突处理。
2. 情报源只有代码内静态注册表，不能在界面添加/删除/搜索/停用。
3. 域枚举偏军事情报（hazard/conflict/…），缺气象、新闻、地理、人文、社交媒体、自媒体等。
4. 自动情报与人工情报没有统一视图；自动记录被覆盖后没有保留/追溯。
5. 缺少跨领域关联和关键情报抽取；只有单源置信度。
6. 案例需求只在抓取时静态过滤，不能随任务/案例状态动态调整。

## 2. 本轮实现

### 2.1 统一领域词汇

`intel/taxonomy.py` 定义领域：灾害、冲突、海上、空中、基础设施、气象、新闻、地理、人文社会、社交媒体、自媒体、网络、卫生、其他；并定义情报类型、来源类型（auto/manual/rss/weather/web/social/other）和核实状态。自动源与人工录入共用同一词汇，避免“军方/非军方”两套分类。

### 2.2 人工情报

`intel/manual.py` 的 `ManualIntelStore` 持久化到 `var/intel/manual.json`，字段：

`intel_id`、`case_id`、`title`、`domain`、`intel_type`、`event_time`、`lat/lon`（6 位）、`place`、`source_id`、`source_name`、`content`、`verification`、`priority="manual"`、`status`、`tags`、`created_by`、`created_at`、`updated_at`、`revision`。

- `verification` 默认 `unverified`；人工优先级不等于“已核实事实”。
- 编辑/撤销保留原记录；撤销后自动情报重新成为有效记录。
- 写入原子替换，线程锁；服务端 API 校验必填、时间、经纬度、枚举。

### 2.3 情报源管理

`intel/sources.py` 的 `SourceStore` 持久化到 `var/intel/sources.json`：

- 首次加载从自动连接器注册表种子化内置源（USGS/GDACS/FIRMS/ADS-B/EONET/GDELT + 人工录入）。
- 支持添加自定义源（id、名称、类型、领域、URL、描述、标签、启用状态）、编辑展示字段/启停、删除自定义源；内置源禁止删除、可停用。
- 支持按关键字搜索（ID/名称/描述/标签/领域）和按类型/启用状态过滤；服务端 `GET /api/intel/sources?q=&kind=&enabled=`，前端再做即时过滤。
- 自定义源目前是**来源登记/人工情报归属**，不是自动抓取连接器；URL 只登记，不自动请求。`rss`/`weather` 等 kind 为后续通用连接器预留。

### 2.4 统一列表与优先级

`intel/management.py` 的 `IntelManager.overview(case_id)`：

1. 读取案例的 `var/intel/<case>/bundle.json` 自动观测。
2. 读取人工情报（含撤销记录）。
3. 冲突判定：同领域、同规范类型、时间差 ≤6 h、距离 ≤25 km 的**有效人工情报**覆盖自动情报；自动记录保留，标 `effective=false`、`superseded_by=<manual id>`。`target`/`threat` 不自动匹配所有类型（避免误覆盖）。
4. 输出统一 items（`priority=manual|auto`、`source_name`、`verification`、`effective`、`conflicts`、`record`），人工有效项排前；原始自动 `record` 不删除。
5. 计算跨域链接和关键情报（见下）。

**优先级语义**：人工情报在展示/选用上优先；`verification=unverified` 仍显著标注“未核实，仅作线索”。这不改变 `World` 真值，也不自动提升为 `ScenarioFacts`。

### 2.5 跨领域关联与关键情报抽取

`intel/association.py`：

- `link_items()`：有效/未撤销项两两计算空间距离、时间差、案例关键词重合；输出 `cross_domain`、`co_location`、`corroboration`、`topic` 链接，带分数与理由（距离、时间差、共同词、跨域）。
- `extract_key_items()`：按人工优先、关键词命中、置信度、关联度、时效、案例域聚焦打分，输出 top-N 及理由（“人工情报优先（未核实）”“命中关键词”“与 N 条情报关联”等）。

这是可解释的规则层，不宣称因果关系；后续可替换为可校准模型，但保留字段与理由。

### 2.6 接口与界面

服务端：

- `GET /api/intel/overview?case_id=`
- `GET /api/intel/sources?q=&kind=&enabled=`
- `POST /api/intel/manual`（新增）
- `POST /api/intel/manual/<id>`（编辑，含恢复 `status=active`）
- `POST /api/intel/manual/<id>/revoke`
- `POST /api/intel/sources`、`POST /api/intel/sources/<id>`、`POST /api/intel/sources/<id>/delete`

前端 `web/src/intelUi.js` + `intel.css`，顶栏“情报”按钮打开模态框，三标签页：情报列表（统一列表、筛选、关键情报、关联、编辑/撤销）、人工录入（结构化表单）、情报源管理（搜索/类型/状态过滤、添加、停用、删除）。列表每 5 s 轮询（输入框聚焦或请求进行中时跳过），增删改后立即刷新；深度链接 `?intel=list|manual|sources`。

写接口只在服务端模式为 `test` 时开放（演示模式只读），并要求 `Content-Type: application/json`；演示端仍可读取统一列表。

## 3. 多领域扩展方案

### 3.1 情报源分层

| 层 | 例子 | 接入方式 | 关键约束 |
| --- | --- | --- | --- |
| 官方/权威 | USGS、GDACS、EONET、气象局、海事/民航通告 | 公开 API/GeoJSON/原子订阅 | 许可、更新时间、坐标系 |
| 传感器 | ADS-B、AIS、卫星、雷达 | REST/WebSocket/本地接收 | 覆盖、延迟、欺骗、配额 |
| 媒体/新闻 | GDELT、RSS/Atom、通讯社 | HTTP JSON/RSS | 无坐标、重复、语言、版权 |
| 社交/自媒体 | 公开 RSS、官方账号导出、公开频道 | RSS/官方 API/合规抓取 | 平台条款、PII、真实性 |
| 人文/地理 | OSM、统计、地名、人口、基础设施 | Overpass/开放数据 | 时效、许可、行政边界 |

通用适配器优先级：`rss`（新闻/自媒体，解析 RSS/Atom，按案例关键词和来源国过滤）→ `weather`（Open-Meteo/JMA/NOAA，按案例中心或网格）→ `social`（在条款允许下的官方 API）→ `generic-http`（受 allow-list 约束的 JSON/CSV 模板）。所有连接器必须复用现有 `connectors/http.py` 的 allow-list、大小上限、超时、无重定向和归档；自定义 URL 需额外做私网/回环地址拒绝。

### 3.2 案例驱动的相关性（不是固定规则）

`CaseIntelProfile` 已有 bbox、buffer、时间回看、域/源白名单、关键词、国家。建议新增持久化的 `IntelRequirement`：

```
case_id, task_id, domains[], entities[], topics[], source_ids[],
geo_weight, time_weight, topic_weight, min_confidence, updated_at, revision
```

- 由案例/任务状态生成：任务类型（登陆/巡逻/救灾）改变 domain 权重；目标/路线改变 AOI/实体；阶段改变时间窗。
- 内置 profile 按案例保留相关域子集（全球 profile 才放开全部域），避免“全领域”成为实际不过滤。
- `IntelManager.overview` 用 requirement 做**排序/聚焦**，不替代 bbox 硬过滤；对已定位情报 bbox 为主，对文本情报关键词/国家为主。
- 需求变化时只更新 requirement 并重算 links/key items，不重建已归档 bundle；UI 通过轮询拿到新 `revision`。

### 3.3 跨领域发现

1. **锚点**：地点（地名/坐标/设施）、时间、实体（单位/船/机/机构）、主题（关键词/事件类型）。
2. **候选生成**：同一锚点 ± 半径/时间窗；文本共享实体/关键词；同一来源国的多语言报道。
3. **链接类型**：`corroboration`、`contradiction`、`co_location`、`cross_domain`、`topic`；只记录证据，不推断因果。
4. **排序**：来源可靠度 × 时效 × 地理精度 × 独立来源数 × 需求权重；人工项额外优先级但保留“未核实”。
5. **冲突**：同实体/同事件的分歧保留双方声明，人工优先仅用于“当前选用”，不删自动记录；界面上并列展示“人工 / 自动原始记录”。

### 3.4 关键情报抽取

- 规则层（已实现）：案例关键词/实体命中、域聚焦、时效、置信度、关联度、人工优先级。
- 结构化输出：`id, score, reasons, evidence_ids, verification`；每条都能回到原始记录与来源。
- 可选 LLM 层：只对已过滤的候选做摘要/实体抽取，输出必须带证据 id；LLM 不可用时规则层仍工作；不把 LLM 文本当真值。
- 质量评估：人工标注小样本，计算 precision@k、关键事件召回、来源覆盖；按域分别评估。

### 3.5 动态更新

- 人工增删改：写入 store → `overview` 重算冲突/链接/关键项 → UI 轮询或立即刷新。
- 案例/任务变化：更新 `CaseIntelProfile`/`IntelRequirement` → `IntelGateway.set_case_filter()` 运行时调整；已消费记录保持不变，未消费按新需求重新排序。
- 自动源更新：连接器写入新 bundle/归档 → `revision` 变化 → UI 拉取；保留旧 bundle 以便追溯。

### 3.6 安全与合规

- 自定义源 URL 只允许 HTTPS、公开主机、默认端口；禁止私网/回环/云元数据地址；不把密钥写入 URL/日志。
- 社交/自媒体只接公开、条款允许的接口；不抓取私密群组、不做个人追踪；文本按惰性数据处理，防提示注入。
- 来源许可/署名进入 source store 与每条 item；人工情报保留录入人/时间/修订号。
- 人工优先级不等于事实；`verification` 与 `priority` 分离，分析结果必须显示“未核实”。

## 4. 本轮验证与后续

- 单元测试：人工 store 校验/持久化/修订/撤销；源 store 种子/添加/搜索/过滤/删除；统一 overview 的人工覆盖、原始保留、撤销恢复、跨域不误覆盖；关联与关键项。
- UI 验证：Playwright 打开情报模态框、人工录入、冲突覆盖、撤销、情报源添加/搜索/删除、5 s 轮询。
- 未做（建议下一轮）：通用 RSS/Atom 与 Open-Meteo 连接器、`IntelRequirement` API、LLM 摘要、AIS WebSocket、轨迹生命周期。
