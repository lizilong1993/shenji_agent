# VGI 情报垂直切片（P1，离线 fixtures）

状态：2026-10-24，分支 `feat/vgi-intel`（基于 `4dc9e1d`）。仅新增 `situation3d/intel/`、测试与 fixtures，未修改主引擎代码，未接入线上 8848。

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
situation3d/tests/
  test_intel_v1.py
  test_fusion.py
  fixtures/intel/{usgs,gdacs,firms}.json
```

依赖方向：`gateway -> model`、`normalize -> model, trust`，无引擎内部依赖，可独立测试。

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

## 8. 验证

```
cd situation3d
python3 -m unittest tests/test_intel_v1.py tests/test_fusion.py -v
```

14 项测试通过：三源归一化字段、衰减单调性与分项、bundle 哈希与观测顺序无关、future-only 与去重、watermark 迟到策略、日志哈希稳定；同事件跨源合并、异事件不合并、半径/时间窗阈值、融合确定性、簇 id 内容寻址、链式锚点不漂移、重复 id 去重、非有限输入拒绝。

## 9. 已知限制与下一步

- 仅覆盖三个静态源的离线归一化；GDELT、AIS/ADS-B 等实时连接器未实现（P2）。
- 融合结果尚未进入网关：`fuse()` 可合并同一事件，但 `IntelGateway` 仍逐条产出 cue；需要决定“融合后单 cue”还是“消费侧去重”，并把融合哈希写入 run-manifest。
- 固定锚点避免了链式过合并，但可能欠合并（两个彼此接近、锚点相距较远的簇不会二次合并）；复杂度 O(N²)，10k 级以上观测需要空间索引。
- 未接入调度器：`IntelGateway` 尚未被 `engine`/`sim.world` 调用，需要 run-manifest 记录 bundle 哈希与每批 `log_hash`。
- 置信度参数为经验值，待用历史数据校准。
