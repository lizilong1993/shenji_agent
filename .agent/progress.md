# Progress

## Round 1 Start

- 时间：2026-04-23 UTC
- 当前循环轮次：1
- strongest agent：bootstrap baseline（`ai/weights_incumbent_rule`）
- strongest commit SHA：`8c3a509d54d6bd3770b389df3829cbfe5389ee76`
- strongest weights snapshot 路径：`/remote-home/lizilong/shenji_agent/artifacts/best_agent/weights`

## 资源检查与同步

- 已执行同步 / 准备：
  - 修复 `git` safe.directory。
  - 检查并确认远程 origin 可访问。
  - 从 `ai/weights_incumbent_rule` bootstrap `artifacts/best_agent/weights`。
- 已确认齐全：
  - `program.md`
  - `ai/`
  - `training/`
  - `configs/`
  - `scripts/`
  - `tests/`
  - `train_models.py`
  - `run_evaluation.py`
  - `datasets/`
  - `land_wargame_sdk/Data/Data/`
  - strongest weights snapshot
- 仍缺失：
  - `AGENTS.md`
  - `references.bib`
  - `.agent/context.md`（本轮 bootstrap）
- 缺失资源的上游检查结果：
  - `AGENTS.md` / `references.bib` / `.agent/context.md` 不在当前工作树、git 历史或 `origin/main` 中，无法从已知上游真值同步。
- 本轮开始前资源是否齐全：训练/评测所需资源齐全；附加文档资源不齐但无可同步上游。
- 是否执行了同步：是

## strongest baseline bootstrap

- 这是本轮执行的 strongest baseline bootstrap。
- baseline 权重来源：`/remote-home/lizilong/shenji_agent/ai/weights_incumbent_rule`
- bootstrap 后 strongest snapshot：`/remote-home/lizilong/shenji_agent/artifacts/best_agent/weights`

## 当前状态判断

- 最新 archived supervised run：`experiments/shenji_agent/20260422_123838`
- 最新 archived league run：`experiments/shenji_agent/league_20260422_125405`
- 最新 archived league run 不是正式有效证据：
  - `overall.count = 16`
  - `map_list.json` 缺失
- 当前最关键瓶颈：`training/data.py` 的 map-id / bucket 合同会把时间戳和随机串数字误判为地图号。

## 本轮地图列表状态

- 本轮 100 项 paired 地图列表是否已生成并保存：否，尚未进入正式比较阶段
- 最终 `map_list.json` 路径：待本轮正式比较生成

## 当前轮结果

- 当前轮是否成功：进行中
- 本轮最终汇总胜率：待本轮正式比较生成
- 当前距离 `program.md` stop rule 还差什么：
  - 需要一轮符合 100 paired 协议的 candidate vs strongest 正式比较
  - 需要满足 overall / complex / CI lower / exploiter / coverage / bucket regression / consecutive windows 全部 hard-stop 条件
