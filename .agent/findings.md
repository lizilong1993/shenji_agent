# Findings

## Round 1 关键结果

1. Round 1 假设成立。
   - `training/data.py` 改为按文件名第二段解析 map id 后，fresh supervised run `experiments/shenji_agent/20260423_024736/slice_report.json` 显示：
     - `intent_bucket_counts.val = {'medium': 1024}`
     - `situation_bucket_counts = {'complex': 108}`
   - `state graph` 不再被误归到 `simple` 桶。
2. Round 1 正式比较成功。
   - `experiments/shenji_agent/league_20260423_024928`
   - `map_list.json` 含 100 个 ordered paired 项，`duplicates_used = true`
   - `match_results.json` 含 100 个 paired 结果
   - candidate vs bootstrap strongest baseline 的 `overall.point_estimate = 1.0`
   - 因此本轮 candidate 胜过当前 strongest baseline，已接受为新的 strongest snapshot
3. hard-stop 仍未满足。
   - `experiments/shenji_agent/league_20260423_032802`
   - 100 paired 协议有效完成
   - `promotion.json.failures = ['move_path_coverage']`
   - 其余阈值都满足，但 `move_path_coverage = 0.0 < 0.20`

## 当前最关键新瓶颈

下一轮最关键问题已经从数据合同转移为 strongest snapshot / self-play protocol：

- Round 1 成功后 strongest snapshot 已更新到 `artifacts/best_agent/weights`
- 但 hard-stop 窗口的 `participants.json` 仍显示 incumbent profile = `baseline`
- `ai/policy.py` 中 `LearnedPolicy.configure()` 对 `policy_profile == "baseline"` 直接 return，不加载模型
- 这意味着后续 arena 并没有真正拿 strongest learned snapshot 当 incumbent，而是继续拿规则回退当 incumbent

在修正这个协议前，“candidate 是否打赢当前 strongest learned agent”这个核心问题并没有被真实比较。

## 为什么下一轮选 strongest snapshot / incumbent profile 这条线

- 它直接影响用户要求的唯一默认比较规则：candidate 必须打 strongest learned incumbent，而不是规则回退。
- 如果这条线不先修，后续 win-rate 提升都不能严格解释为“打赢当前 strongest agent”。
- 这属于单一、高价值、可归因的 self-play gate / strongest snapshot 管理问题。

## 关键证据来源

- `program.md`
- `configs/selfplay_baseline.yaml`
- `configs/selfplay_sota_stop.yaml`
- `training/data.py`
- `training/supervised.py`
- `training/selfplay.py`
- `training/evaluate.py`
- `ai/policy.py`
- `tests/test_mainline_contract.py`
- `experiments/shenji_agent/20260423_024736/slice_report.json`
- `experiments/shenji_agent/league_20260423_024928/rating.json`
- `experiments/shenji_agent/league_20260423_024928/promotion.json`
- `experiments/shenji_agent/league_20260423_032802/rating.json`
- `experiments/shenji_agent/league_20260423_032802/promotion.json`
- `experiments/shenji_agent/league_20260423_032802/participants.json`

## 已验证 / 未验证

- 已验证：
  - Round 1 数据合同修正后，fresh supervised buckets 合理化。
  - Round 1 candidate 在正式 100 paired 对抗中打赢 bootstrap strongest baseline。
  - hard-stop 当前只差 `move_path_coverage`。
  - strongest snapshot 更新后，incumbent 仍按 `baseline` profile 运行，未真实加载 strongest learned weights。
- 未验证：
  - 修正 incumbent profile / strongest snapshot 合同后，candidate-vs-strongest learned arena 的真实胜率。
  - 在真实 strongest learned incumbent 下，`move_path_coverage = 0.0` 是否仍然存在。
