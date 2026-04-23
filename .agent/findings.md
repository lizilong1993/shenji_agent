# Findings

## 本轮关键发现

1. 当前 archived formal league 结果不能作为 strongest 晋升证据。
   - `experiments/shenji_agent/league_20260422_125405/rating.json` 里 `overall.count = 16`，不是 100 个 paired 地图。
   - 同目录缺少 `map_list.json`，不满足 `program.md` 的正式协议。
2. 当前最关键瓶颈是数据 / map-id 合同，不是模型结构。
   - 最新监督产物 `experiments/shenji_agent/20260422_123838/slice_report.json` 显示：
     - `intent_bucket_counts.val = {'medium': 1024}`
     - `situation_bucket_counts = {'simple': 108}`
   - `intent` 文件名唯一第二段只有 `201033039601` / `201033029601`；如果在整条文件名上做 suffix 匹配，会被随机串中的 `83` 干扰，出现 `[9601, 83]` 歧义。
   - `state graph` 文件名唯一第二段只有 `2110431553` / `2110431453`；当前 `prefix` 策略把全部样本误解成 `map_21/simple`，而第二段 suffix 稳定指向 `map_53/complex`。

## 为什么选这条线

- 这条线满足“单一、高价值、可归因”的条件。
- 它直接解释了最新监督产物里的异常 bucket 分布。
- 它不会引入新的大模型或大范围策略改写，失败时也容易回退。

## 关键证据来源

- `program.md`
- `configs/selfplay_baseline.yaml`
- `configs/selfplay_sota_stop.yaml`
- `configs/supervised_baseline.yaml`
- `training/data.py`
- `training/supervised.py`
- `training/selfplay.py`
- `training/evaluate.py`
- `tests/test_mainline_contract.py`
- `experiments/shenji_agent/20260422_123838/slice_report.json`
- `experiments/shenji_agent/league_20260422_125405/rating.json`
- `experiments/shenji_agent/league_20260422_125405/promotion.json`
- `experiments/shenji_agent/league_20260422_125405/participants.json`

## 已验证 / 未验证

- 已验证：
  - 最新 league archived run 缺少 `map_list.json` 且只汇总了 16 个 paired 项。
  - `intent` / `state graph` 文件名的第二段模式固定，当前整串数字解析会误吸收随机后缀中的数字。
  - `map_53` 属于 `complex`，`map_9601` 属于 `medium`。
- 未验证：
  - 修正合同后 fresh supervised run 是否会把 `state graph` 正确落到 `complex` 桶。
  - 修正合同后 candidate 是否能在正式 100 paired 比较中击败 strongest baseline。
