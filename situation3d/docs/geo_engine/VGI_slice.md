# VGI 情报垂直切片（P0 外部连接器 + P1 离线 fixtures）

状态：2026-10-24 起始；`feat/vgi-intel`（融合）、`feat/vgi-gateway-fusion`（网关融合）、`feat/intel-scheduler`（调度器与 World 接线）已合并 main；2026-10-26 `feat/vgi-connectors` 增加 `intel/connectors/`、`scripts/fetch_intel.py` 与原始归档。未接入线上 8848。

## 1. 定位

对应总体报告中的 VGI 情报工作流：把公开数据源（USGS、GDACS、FIRMS 等）转化为“观测（Observation）”，再经可信度评分与情报网关，变成仿真可消费的“未来 cue”。核心原则：

- 情报不是地面真值。网关只产出 `IntelCue`，由仿真侧按观察/提示使用，不直接改写实体状态。
- 只用未来信息。cue 在事件时间到达后才可消费，仿真时间早于事件时不可见，避免“未卜先知”。
- 消费可重放。消费顺序、批次内容、日志哈希均确定，用于 run-manifest 校验。

## 2. 目录

```
situation3d/intel/
  model.py      Observation 数据类、时间解析/序列化
  trust.py      源可靠度、时间衰减、地理精度因子
  normalize.py  USGS / GDACS / FIRMS 归一化器
  fusion.py     跨源时空聚类与置信度融合
  gateway.py    IntelBundle、IntelCue、IntelGateway
  scheduler.py  按 side 的确定性轮询与 run-manifest
  connectors/
    http.py      allow-list HTTPS、超时、大小上限、拒绝重定向
    registry.py  源注册表（host/license/attribution/schedule）
    archive.py   原始响应不可变归档（gzip + SHA-256）
    usgs.py      USGS FDSN 查询 URL + 归一化
    gdacs.py     GDACS 事件列表 URL + 归一化
    firms.py     FIRMS area CSV URL、acq_date/acq_time 合并
    fetch.py     抓取 → 归档 → 归一化 → IntelBundle
situation3d/scripts/
  fetch_intel.py  一次性抓取 CLI（可被 cron 调用）
situation3d/tests/
  test_intel_v1.py
  test_fusion.py
  test_intel_scheduler.py
  test_intel_connectors.py
  fixtures/intel/{usgs,gdacs,firms}.json
```

依赖方向：`gateway -> model, fusion`、`normalize -> model, trust`，无引擎内部依赖，可独立测试。

## 3. Observation 模型

不可变数据类，字段与 P0 契约 `situation3d.intel.v1` 对齐：

| 字段 | 说明 |
| --- | --- |
| `observation_id` | `源:原生ID`，全链路唯一 |
| `source_id` / `source_native_id` | 源标识与源内 ID |
| `domain` / `type` | 领域（hazard/conflict/...）与事件类型 |
| `lat` / `lon` | WGS84，保留完整浮点精度 |
| `event_time` / `ingest_time` | 事件时间与入库时间，ISO 8601 UTC |
| `ingest_seq` | 入库序号，用于同事件时间下的稳定排序 |
| `confidence` | 0–1，见第 4 节 |
| `geo_precision` / `geo_uncertainty_m` / `is_centroid` | 地理精度档位、误差半径、是否质心 |
| `status` | 默认 `active`（预留失效/撤回） |
| `source_url` / `title` / `properties` | 溯源与源特有字段 |

`content_hash()` 只取稳定字段，用于 bundle 哈希；`as_dict()` 用于落盘。

## 4. 可信度

`confidence = 源可靠度 × 时间衰减 × 地理精度因子`：

- 源可靠度：USGS 0.98、GDACS 0.95、FIRMS 0.80、GDELT 0.60，未知源 0.7。
- 时间衰减：按领域半衰期指数衰减（hazard/maritime/aviation 6 h，infrastructure 12 h，conflict/other 24 h，health 72 h），下限 0.2。
- 地理精度：exact 1.0、street 0.95、locality 0.85、admin1 0.70、admin0 0.55、unknown 0.40；质心再乘 0.7。

`components()` 返回各分项与乘积，便于审计与前端展示。当前为启发式初值，后续用历史命中率校准。

## 5. 归一化与 fixtures

`normalize_usgs/gdacs/firms(payload, fetched_at, start_seq=0)` 均为纯函数，输入为已抓取的 JSON，不在函数内发起网络请求。fixtures 为合成数据（`example.invalid` 域名、虚构地名），覆盖点几何、时间戳、震级、告警等级、火点辐亮度等字段。

`normalize(source, payload, fetched_at)` 为统一入口；未知源抛 `ValueError`。

## 6. 跨源融合

`fuse(observations, radius_m=10000, window_s=3600)` 先按 `observation_id` 去重（保留置信度更高者），再按置信度降序处理：每个簇的首个成员即固定锚点（不随成员变化而漂移），domain 相同、规范化类型相同（`EQ`/`earthquake` 等走同义词表），且与锚点的时空距离都在阈值内则加入最近簇，否则新建簇。因此每个成员都在其锚点的半径/时间窗内，不会通过链式相邻把远端事件并进来。`FusedEvent` 用锚点坐标/时间代表事件，保留全部 `member_ids` 与 `sources`，`cluster_id` 为成员 `content_hash` 排序后的 SHA-256 前缀，`confidence` 为 `1-Π(1-ci)`（上限 0.999）。融合前校验经纬度与置信度必须有限且在合法范围。输入顺序不影响结果。

当前网关尚未消费 `FusedEvent`：同一事件融合后仍需决定是作为单条 cue，还是保留各源 cue 并在消费侧去重，留待网关集成时处理。

## 7. 情报网关

`IntelBundle` 按 `(event_ms, 源优先级, observation_id)` 排序，`hash` 为各观测 `content_hash` 的 SHA-256，观测顺序不影响结果。

`IntelGateway.consume(side, sim_time, watermark, lookahead)`：

1. 已消费的不再产出；
2. 事件仿真时间晚于 `sim_time + lookahead` 的跳过（`lookahead` 内预取）；
3. `properties.side` 与请求方不一致的跳过；
4. 事件时间早于 `watermark` 的判为迟到，按 `discard_and_log` 记入 `late` 并标记已消费；
5. 其余转为 `IntelCue`，含 `valid_from/valid_to`（TTL）、置信度、溯源与空 `sensor_model`。

cue 排序键为 `(event_time_ms, source_priority, cue_id)`；批次携带 `log_hash`（cue_id + event_time_ms 的 SHA-256），用于回放一致性校验。

`IntelGateway(bundle, fusion=(radius_m, window_s))` 开启融合模式：每次消费先按 side、`watermark ≤ 成员时间 ≤ horizon` 过滤观测并剔除已消费项，再对剩余观测调用 `fuse()`；每个簇产出一条 `cue_kind="observation"`、`provenance.fused=true` 的 cue，`observation_id` 即 `cluster_id`，`provenance` 保留成员、来源和代表观测。因此 cue 不会包含未来成员，也不会吞掉已过 watermark 的成员；已被消费的成员不会在后续窗口重复融合。`knowledge_time_ms`/`ingest_seq` 取成员最大值，表示整簇信息齐备的时刻。默认 `fusion=None` 保持逐条观测语义。

## 8. 调度器与 World 接入

`IntelScheduler(gateway, sides=(0,1), lookahead_s=300, watermark_lag_s=0)` 每次 `advance(sim_time, lookahead_s=None)` 对每个 side 调一次 `consume`，`horizon = sim_time + lookahead`、`watermark = max(0, sim_time - lag)`，并把 `(side, sim_time, watermark, cue_count, cue_ids, log_hash)` 追加到批次日志。`manifest()` 返回 bundle 哈希/ID、`scenario_start`、调度参数、批次副本、迟到 id 与 cue 总数。

`World.set_intel_gateway(...)` 为可选接线，约定 `world.time_s == 0` 对应 `bundle.scenario_start`（调用方负责对齐，manifest 记录该时间基准）。World 的时间推进统一走 `_advance_clock`：**先**在旧时刻按 `max(lookahead_s, dt)` 预取并缓存 cue，**再**推进时钟，**最后**按 `valid_from` 把到期 cue 发布到 `intel_cues`；`intel_for(side)["cues"]` 只返回 `valid_from <= time_s` 且 `valid_to >= time_s` 的 cue。`step`、`advance_to` 的空闲快进和 `_motion_advance` 都使用同一推进函数，因此步长和推进方式不改变 intel 时序。gateway 的消费集合按 `(side, observation_id)` 记录，side-neutral 观测会对每个侧各送达一次。

`intel_manifest()` 返回 `enabled` 加上述 manifest；未挂载调度器时返回 `{"enabled": False}`，`intel_for` 也不含 `"cues"` 键，保持旧载荷形状。

## 9. 验证

```
cd situation3d
python3 -m unittest tests/test_intel_v1.py tests/test_fusion.py tests/test_intel_scheduler.py tests/test_intel_connectors.py -v
```

39 项测试通过：三源归一化字段、衰减单调性与分项、bundle 哈希与观测顺序无关、future-only 与去重、watermark 迟到策略、日志哈希稳定；同事件跨源合并、异事件不合并、半径/时间窗阈值、融合确定性、簇 id 内容寻址、链式锚点不漂移、重复 id 去重、非有限输入拒绝、融合网关单 cue 去重、关闭融合时逐源保留、融合模式不暴露未来成员、成员级 watermark、融合日志哈希稳定；调度器按 side 轮询、manifest 确定性与隔离、World 只发布到期 cue、多 side 中立观测、窗口内早事件不丢、`advance_to` 同步、未挂载时载荷不变；连接器 URL/allow-list、FIRMS 日期时间合并、归档往返、离线 bundle、契约字段、部分源失败与缺 key 上报。

## 10. 已知限制与下一步

- USGS/GDACS/FIRMS 已有外部连接器；GDELT、AIS/ADS-B 等实时连接器未实现（P2）。
- 融合需显式开启（`fusion=(radius, window)`），默认仍逐条产出 cue；融合以消费窗口内未消费成员为输入，跨窗口到达的同一事件可能分成多条 cue；固定锚点可能欠合并，O(N²) 复杂度需在 10k 级以上观测前引入空间索引。
- 时间基准需调用方保证：`world.time_s=0` 必须对应 `bundle.scenario_start`，当前没有自动校验或 offset 参数，只在 manifest 记录。
- 批次日志每侧每 tick 追加一行，长跑 manifest 无界增长；cue 按 TTL 从 `intel_cues` 淘汰，但 `late_ids` 仍累积。
- 调度器每 tick 全量扫描 bundle（融合模式 O(N²)），10k 级观测前需要时间索引/增量游标。

## 11. 外部连接器（P0，2026-10-26）

`intel/connectors/` 把“抓取—归档—归一化—bundle”放在仿真之外：

- `registry.py`：USGS / GDACS / FIRMS 的 host、license、attribution、调度周期、parser 版本；FIRMS 需要 `FIRMS_MAP_KEY`。
- `http.py`：只允许注册表 host、只允许 HTTPS/443、拒绝 URL 凭据、拒绝重定向、20 s 超时、8 MiB 响应上限；错误信息不包含带 key 的完整 URL。
- `usgs.py`：FDSN `query?format=geojson`，按 bbox/时间窗查询。
- `gdacs.py`：`gdacsapi/api/events/geteventlist/SEARCH`，事件类型 EQ/TC/FL/VO/WF/DR，告警级 Green/Orange/Red；列表接口无 bbox，抓取后本地过滤。
- `firms.py`：`/api/area/csv`，四个 NRT 传感器；把 `acq_date` + `acq_time`（HHMM）合并为 ISO 时间；单个传感器失败不丢弃其他传感器，状态记 `degraded`。
- `fetch.py`：`build_bundle(source_ids, bbox, start, end, ...)` 返回 `FetchResult(bundle, statuses, archives, errors)`；观测按 `(event_time, source_id, observation_id)` 排序后重排 `ingest_seq`；bundle 固定 `domain="hazard"`、bbox/time_window/source_allowlist。
- `archive.py`：`var/intel/archive/<source>/<utc>.<ext>.gz` + 原始字节 SHA-256；`read_archive()` 可回放。
- `scripts/fetch_intel.py`：一次性抓取，写 bundle 与 health JSON；`--allow-partial` 允许部分源失败仍产出 bundle。示例：

```
cd situation3d
python3 scripts/fetch_intel.py --sources usgs,gdacs \
  --bbox 119.0,21.5,123.0,26.5 --hours 24 \
  --out var/intel/bundle.json --archive-root var/intel/archive
```

健康文件记录每个源的 `state`（nominal/degraded/unavailable）、count、bytes、error 和归档路径。仿真侧仍通过 `World.set_intel_gateway()` 消费 bundle；连接器不在 tick 内运行，也不访问网络。`var/intel/` 已加入 `.gitignore`。
