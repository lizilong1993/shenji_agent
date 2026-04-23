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
- 当前已接受 strongest code baseline：`8c3a509d54d6bd3770b389df3829cbfe5389ee76`
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

## 研究优先级

1. 先修正数据 / map-id 合同，确保监督证据可信。
2. 再用 fresh candidate 按 100 paired 协议与 strongest snapshot 正式比较。
3. 只有正式比较打赢 strongest，才更新 strongest snapshot。
