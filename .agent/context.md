# Context

- 更新时间：2026-04-23 UTC
- 远程工作根：`/remote-home/lizilong/shenji_agent`
- 远程虚拟环境：`/remote-home/lizilong/venvs/shenji`

## 长期真值与现状

- 当前研究主真值：
  - `program.md`
  - `configs/selfplay_baseline.yaml`
  - `configs/selfplay_sota_stop.yaml`
  - `training/supervised.py`
  - `training/selfplay.py`
  - `training/evaluate.py`
  - `scripts/report_progress.py`
  - 最新 `experiments/shenji_agent/*`
- 当前已接受 strongest code baseline：`bb26fd52af4aec9029643c8f3666237eb06a1a6d`
- 当前 strongest weights snapshot：`/remote-home/lizilong/shenji_agent/artifacts/best_agent/weights`
- strongest bootstrap 来源：`ai/weights_incumbent_rule`

## 上游可用性

- `AGENTS.md`、`references.bib`、`.agent/context.md` 都不在当前工作树、git 历史或 `origin/main` 中。
- 对这三项不要臆造“上游真值来源”；如果后续流程必须依赖其中内容，应直接按 blocker 处理。

## 数据集命名约定

- `intent` 数据文件名形如：
  - `20210103122823291-201033039601-单晨轩-DemoAI-0X81qByG.zip`
  - 唯一第二段值主要为 `201033039601` / `201033029601`
  - 第二段做 suffix 匹配会稳定落到 `map_9601`
- `state graph` 数据文件名形如：
  - `20210730195350197-2110431553-tVVHMGZh.json`
  - 唯一第二段值主要为 `2110431553` / `2110431453`
  - 第二段做 suffix 匹配会稳定落到 `map_53`
- 在整条文件名上抓数字会被时间戳和随机串污染，出现伪 map-id（如 `83`、`84`、`21`）。

## 已知实验结论

- `experiments/shenji_agent/20260422_123838`
  - `intent` 监督桶只显示 `medium`
  - `situation` 监督桶错误落到 `simple`
- `experiments/shenji_agent/league_20260422_125405`
  - `participants.json` 显示 candidate=`ai/weights`，incumbent=`ai/weights_incumbent_rule`
  - 但该 run 只有 16 个 paired 项，且缺少 `map_list.json`
  - 因此它不是 `program.md` 认可的正式 strongest 晋升证据
- `experiments/shenji_agent/20260423_024736`
  - Round 1 fresh supervised run
  - `intent` 仍为 `medium`
  - `situation` 已修正为 `complex`
- `experiments/shenji_agent/league_20260423_024928`
  - Round 1 正式 candidate-vs-bootstrap-strongest 窗口
  - 100 paired 完成，candidate 明确打赢 bootstrap strongest baseline
- `experiments/shenji_agent/league_20260423_032802`
  - hard-stop 检查窗口
  - 100 paired 完成
  - 未通过 stop，唯一失败项是 `move_path_coverage`
  - incumbent 仍以 `baseline` profile 运行 strongest snapshot

## 研究优先级

1. 修正 strongest snapshot / incumbent profile 合同，确保 candidate 真正在打 strongest learned incumbent。
2. 在协议可信后，再追 `move_path_coverage = 0.0` 的根因。
3. 只有满足 hard-stop 的 coverage 与连续窗口条件，才允许停机。
