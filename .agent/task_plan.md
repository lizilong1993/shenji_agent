# Round 1 Task Plan

## 本轮假设

`training/data.py` 当前直接在整条文件名上抓数字做 map-id 推断，会把时间戳和随机串里的数字误判成地图号。把解析面收窄到文件名第二段并继续做 suffix 匹配，能修正 `intent_lstm + situation_gcn` 主线的 bucket 合同，并给后续训练/正式比较提供可信监督证据。

## 计划改动

1. 在 `training/data.py` 增加 `filename_segment_index` 合同，并让 `IntentSequenceDataset` / `StateGraphDataset` 都支持按指定段解析 map id。
2. 把 `configs/supervised_baseline.yaml` 的 `intent` 和 `situation` 都切到 `filename_segment_index: 1`；`situation` 同时改为 `map_id_strategy: suffix`。
3. 给 `tests/test_mainline_contract.py` 增加回归测试，覆盖“随机串数字不能污染 map-id 解析”。
4. 通过轻量验证后，用 `configs/supervised_baseline.yaml` 重训 candidate，再按 100 个 paired 地图协议做 candidate vs strongest 正式比较。

## 本轮成功判据

- `python -m compileall ai training scripts tests` 通过。
- `python -m unittest tests.test_mainline_contract -v` 通过。
- fresh supervised run 的 `slice_report.json` 不再把 state-graph 数据误归到 `simple` 桶。
- `python -m training.selfplay --config configs/selfplay_baseline.yaml --candidate ai/weights --incumbent artifacts/best_agent/weights` 完整跑完 100 个 paired 地图，并且 candidate 的最终汇总胜率高于当前 strongest baseline。

## 本轮失败判据

- 解析仍然出现 map-id 歧义，或 fresh supervised run 依旧产出明显失真的 bucket 分布。
- 训练 / 正式比较无法运行，或 100 paired 协议无法完整落盘。
- candidate 最终汇总胜率未高于 strongest baseline。

## 下一轮入口

- 如果 bucket 合同修正后 self-play 仍无增益，下一轮优先检查 `ai/policy.py` 的 `Move/Shoot` 接入宽度与打分逻辑。
- 如果现有数据集无法从可用元数据中稳定恢复地图真值，则停止并把“缺失显式 map metadata”记为 blocker。
