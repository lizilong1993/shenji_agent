# Round 2 Task Plan

## 本轮假设

Round 1 之后 strongest snapshot 已经更新为 learned weights，但 `training.selfplay` 仍默认把 incumbent 作为 `baseline` profile 运行。`LearnedPolicy.configure()` 在 `policy_profile == "baseline"` 时直接跳过模型加载，这意味着后续 arena 并没有真正拿 `artifacts/best_agent/weights` 里的 strongest learned agent 当 incumbent。先修正 strongest snapshot / incumbent profile 合同，才能继续做可信的 candidate-vs-strongest 研究循环。

## 计划改动

1. 在 strongest snapshot 中引入可落盘的角色元数据，显式区分 `bootstrap_rule_baseline` 与 `accepted_learned_mainline`。
2. 修改 `training.selfplay.py` 的 incumbent profile 选择逻辑：
   - bootstrap rule baseline 继续走 `baseline`
   - accepted learned mainline 必须走 learned incumbent profile，而不是规则回退
3. 增加回归测试，覆盖 strongest snapshot 元数据与 incumbent profile 解析。
4. 轻量验证后重训 candidate，并重新做 100 paired candidate-vs-strongest 正式比较。

## 本轮成功判据

- self-play 正式比较里的 `participants.json` / 诊断信息能证明 incumbent 真正从 `artifacts/best_agent/weights` 加载 strongest learned weights。
- `python -m compileall ai training scripts tests` 通过。
- `python -m unittest tests.test_mainline_contract -v` 通过。
- fresh 100 paired candidate-vs-strongest 正式比较有效落盘，并且 candidate 胜率高于当前 strongest learned snapshot。

## 本轮失败判据

- incumbent 仍以 `baseline` profile 忽略 strongest learned weights。
- 修正协议后 candidate 无法胜过 strongest learned snapshot。
- 正式比较无法完成 100 paired 协议或结果不可判定。

## 下一轮入口

- 如果协议修正后 arena 仍无法满足 hard-stop，下一轮优先追 `move_path_coverage = 0.0` 的根因，扩大真实 move 决策触发面。
- 如果 strongest snapshot 元数据不足以区分 rule baseline 与 learned snapshot，则把 snapshot contract 缺失列为 blocker。
