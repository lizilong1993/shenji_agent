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
  gateway.py    IntelBundle、IntelCue、IntelGateway
situation3d/tests/
  test_intel_v1.py
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

## 6. 情报网关

`IntelBundle` 按 `(event_ms, 源优先级, observation_id)` 排序，`hash` 为各观测 `content_hash` 的 SHA-256，观测顺序不影响结果。

`IntelGateway.consume(side, sim_time, watermark, lookahead)`：

1. 已消费的不再产出；
2. 事件仿真时间晚于 `sim_time + lookahead` 的跳过（`lookahead` 内预取）；
3. `properties.side` 与请求方不一致的跳过；
4. 事件时间早于 `watermark` 的判为迟到，按 `discard_and_log` 记入 `late` 并标记已消费；
5. 其余转为 `IntelCue`，含 `valid_from/valid_to`（TTL）、置信度、溯源与空 `sensor_model`。

cue 排序键为 `(event_time_ms, source_priority, cue_id)`；批次携带 `log_hash`（cue_id + event_time_ms 的 SHA-256），用于回放一致性校验。

## 7. 验证

```
cd situation3d
python3 -m unittest tests/test_intel_v1.py -v
```

6 项测试通过：三源归一化字段、衰减单调性与分项、bundle 哈希与观测顺序无关、future-only 与去重、watermark 迟到策略、日志哈希稳定。

## 8. 已知限制与下一步

- 仅覆盖三个静态源的离线归一化；GDELT、AIS/ADS-B 等实时连接器未实现（P2）。
- 无跨源去重/融合：同一事件来自 USGS 与 GDACS 时目前会产出两条 cue，需按时空邻近聚类合并（报告中 anchor-proximity 方案）。
- 未接入调度器：`IntelGateway` 尚未被 `engine`/`sim.world` 调用，需要 run-manifest 记录 bundle 哈希与每批 `log_hash`。
- 置信度参数为经验值，待用历史数据校准。
