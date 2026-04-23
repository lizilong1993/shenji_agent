# Progress

## Round 1 Complete

- 时间：2026-04-23 UTC
- 当前循环轮次：1 已完成
- Round 1 commit：`bb26fd52af4aec9029643c8f3666237eb06a1a6d`
- Round 1 假设：按文件名第二段修正 supervised map-id 合同
- Round 1 是否成功：是

## strongest agent

- strongest agent：accepted learned mainline
- strongest commit SHA：`bb26fd52af4aec9029643c8f3666237eb06a1a6d`
- strongest weights snapshot 路径：`/remote-home/lizilong/shenji_agent/artifacts/best_agent/weights`
- strongest weights 已更新来源：`/remote-home/lizilong/shenji_agent/ai/weights`

## 资源检查与同步

- 本轮开始前做了哪些同步：
  - 修复 `git` safe.directory
  - 检查 `origin/main`
  - bootstrap strongest snapshot 到 `artifacts/best_agent/weights`
- 同步了哪些关键资源：
  - strongest weight snapshot
- 哪些资源已确认齐全：
  - `program.md`
  - `configs/`
  - `training/`
  - `scripts/`
  - `tests/`
  - `datasets/`
  - `land_wargame_sdk/Data/Data/`
  - `artifacts/best_agent/weights`
- 哪些资源仍缺失：
  - `AGENTS.md`
  - `references.bib`
- 缺失说明：
  - 上述两项不在工作树、git 历史或 `origin/main`，无可同步上游
- 本轮开始前资源是否齐全：训练与评测所需资源齐全
- 本轮是否做了同步：是

## Fresh verification evidence

- `python -m compileall ai training scripts tests`：通过
- `python -m unittest tests.test_mainline_contract -v`：8 项测试通过
- 数据合同探针：
  - `state_bucket_counts {'complex': 108}`
  - `intent_bucket_counts_sample {'medium': 512}`
- fresh supervised run：
  - 路径：`experiments/shenji_agent/20260423_024736`
  - `situation_bucket_counts = {'complex': 108}`
  - `intent_bucket_counts.val = {'medium': 1024}`

## Round 1 正式比较

- 正式比较目录：`experiments/shenji_agent/league_20260423_024928`
- 本轮地图列表是否重新生成：是
- 地图总数：100
- 是否存在补重复抽样：是
- 实际完成了多少项：100
- 最终 `map_list.json` 路径：`/remote-home/lizilong/shenji_agent/experiments/shenji_agent/league_20260423_024928/map_list.json`
- participant：
  - candidate=`/remote-home/lizilong/shenji_agent/ai/weights`
  - incumbent=`/remote-home/lizilong/shenji_agent/artifacts/best_agent/weights`
- 最终汇总胜率：`1.0000`
- 本轮领先 strongest baseline 的幅度：`+0.5000` 相对中性线；相对 bootstrap strongest baseline 为明确提升
- revert 状态：无需 revert

## hard-stop 检查

- hard-stop 检查目录：`experiments/shenji_agent/league_20260423_032802`
- 本轮地图列表是否重新生成：是
- 地图总数：100
- 是否存在补重复抽样：是
- 实际完成了多少项：100
- 最终 `map_list.json` 路径：`/remote-home/lizilong/shenji_agent/experiments/shenji_agent/league_20260423_032802/map_list.json`
- hard-stop 结果：未通过
- `promotion.json.failures`：`['move_path_coverage']`
- hard-stop 最终汇总胜率：`1.0000`

## 当前距离 stop rule 还差什么

- `move_path_coverage >= 0.20` 目前未满足，当前为 `0.0`
- 两个连续窗口满足 hard-stop 的稳定性条件目前也未满足
- stronger-learned incumbent 的 compare protocol 仍未修正：
  - strongest snapshot 已是 learned weights
  - 但 `participants.json` 里的 incumbent 仍使用 `baseline` profile

## Round 2 入口

- 下一轮准备改的方向：strongest snapshot / incumbent profile 合同
- 原因：不先修正这个协议，后续 candidate-vs-strongest learned 比较不可信

## Round 2 Start

- 时间：2026-04-23 UTC
- 当前循环轮次：2
- strongest agent：accepted learned mainline
- strongest commit SHA：`bb26fd52af4aec9029643c8f3666237eb06a1a6d`
- strongest weights snapshot 路径：`/remote-home/lizilong/shenji_agent/artifacts/best_agent/weights`
- 本轮开始前资源是否齐全：是
- 是否执行了同步：否
- 本轮资源检查结论：代码、配置、数据、strongest snapshot 均齐全，无需同步
- 本轮新增 strongest snapshot 元数据：
  - `artifacts/best_agent/weights/model_manifest.json -> snapshot_role=accepted_learned_mainline`
  - `ai/weights_incumbent_rule/model_manifest.json -> snapshot_role=bootstrap_rule_baseline`
- 本轮轻量验证：
  - `python -m compileall ai training scripts tests`：通过
  - `python -m unittest tests.test_mainline_contract -v`：10 项通过
