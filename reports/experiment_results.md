# 实验设计与正式结果

## 1. 实验目标

实验目标是验证候选策略 `CurrentAI` 是否在真实庙算 SDK 后端中，相比规则基线 `IncumbentAI` 具有稳定、可审计、统计上足够显著的闭环推演优势。

实验不以监督训练准确率作为最终结论，而以正式 self-play promotion gate 作为主判据。原因是陆战兵棋智能体的能力必须体现在真实环境闭环中，包括进入战斗阶段、选择有效动作、造成得分变化、避免被固定 exploiter 利用，并在不同地图复杂度上保持稳定。

## 2. 实验环境

| 项 | 内容 |
| --- | --- |
| 仓库路径 | `/remote-home/lizilong/shenji_agent` |
| 正式运行目录 | `experiments/shenji_agent/league_20260501_170204` |
| 环境后端 | `train_env.TrainEnv` |
| mock 后端 | `false` |
| 候选策略 | `CurrentAI` |
| 候选 profile | `challenger` |
| 候选 artifact | `ai/weights` |
| 候选 manifest | `ai/weights/model_manifest.json` |
| 现任基线 | `IncumbentAI` |
| 现任 profile | `baseline` |
| 现任 artifact | `ai/weights_incumbent_rule` |
| 正式配置 | `configs/selfplay_sota_stop.yaml` |
| 完整流水线脚本 | `scripts/run_full_promotion_after_auth.sh` |

## 3. 实验协议

正式自博弈使用 100 个配对地图项。场景来源为：

`land_wargame_sdk/Data/Data/scenarios/*.json`

采样协议是 unique-first random sampling without replacement，不足 100 项时允许补充重复场景。正式运行中唯一场景池为 43 个，因此 100 个配对中包含重复项，但所有唯一场景优先覆盖。

分桶分布如下：

| 地图分桶 | 配对数量 | 胜率点估计 | 95% CI |
| --- | ---: | ---: | --- |
| simple | 52 | 0.9038 | [0.8269, 0.9808] |
| medium | 26 | 0.8462 | [0.7019, 0.9615] |
| complex | 22 | 0.8182 | [0.6364, 0.9545] |
| overall | 100 | 0.8700 | [0.8000, 0.9276] |

## 4. Promotion Gate 阈值

正式 hard-stop 规则要求连续两个评估窗口满足以下门槛：

| 指标 | 阈值 |
| --- | ---: |
| overall point estimate | `>= 0.60` |
| overall CI95 lower | `>= 0.55` |
| complex point estimate | `>= 0.60` |
| complex CI95 lower | `>= 0.55` |
| learned path coverage | `>= 0.30` |
| move path coverage | `>= 0.20` |
| shoot path coverage | `>= 0.20` |
| Elo lower bound | `>= 35.0` |
| max bucket regression | `<= 0.03` |
| max exploiter win rate | `<= 0.45` |
| consecutive windows | `2` |

## 5. 正式结果总览

| 指标 | 正式结果 | 是否通过 |
| --- | ---: | --- |
| promotion accepted | `true` | 是 |
| failure reasons | `[]` | 是 |
| overall point estimate | `0.8700` | 是 |
| overall CI95 lower | `0.8000` | 是 |
| complex point estimate | `0.8182` | 是 |
| complex CI95 lower | `0.6364` | 是 |
| learned path coverage | `0.3728` | 是 |
| move path coverage | `0.3367` | 是 |
| shoot path coverage | `1.0000` | 是 |
| Elo lower bound | `240.8240` | 是 |
| max bucket regression | `0.0` | 是 |
| max exploiter win rate | `0.25` | 是 |
| consecutive windows | `2/2` | 是 |
| battle phase entered | `true` | 是 |
| selected Move/Shoot/Occupy actions | `94682` | 是 |
| auth failure count | `0` | 是 |

结论：候选策略通过正式 promotion gate，满足项目 hard-stop 条件。

## 6. 行为覆盖结果

正式运行中的学习路径覆盖：

| 动作类型 | 总次数 | 学习路径次数 | 覆盖率 |
| --- | ---: | ---: | ---: |
| Move | 23580 | 7939 | 0.3367 |
| Shoot | 1357 | 1357 | 1.0000 |
| Overall learned path | - | - | 0.3728 |

动作阶段诊断统计：

| 指标 | 数值 |
| --- | ---: |
| total turn records | 320000 |
| total turns | 17200 |
| turns with setup/deployment only | 800 |
| turns with Move valid/selected context | 129890 |
| turns with Shoot valid/selected context | 4648 |
| turns with Occupy valid/selected context | 376 |
| selected Move actions | 89018 |
| selected Shoot actions | 5308 |
| selected Occupy actions | 356 |
| first battle turn | 1 |

这些统计说明正式运行不是部署阶段空转，也不是无动作平局。候选策略在战斗阶段产生了大量核心战斗动作。

## 7. Exploiter 鲁棒性

为了避免候选策略只针对单一规则基线过拟合，正式 gate 中加入 exploiter 验证：

| Exploiter | vs candidate win rate | score rate | draw rate | matches |
| --- | ---: | ---: | ---: | ---: |
| aggressive exploiter | 0.00 | 0.375 | 0.75 | 4 |
| occupy exploiter | 0.25 | 0.625 | 0.75 | 4 |

最大 exploiter 胜率为 `0.25`，低于阈值 `0.45`，说明候选策略没有被固定攻击型或夺控型 exploiter 轻易利用。

## 8. Objective Scoring 证据

完整晋升流水线在认证恢复后构建了 objective labels：

| 标签类型 | 数量 |
| --- | ---: |
| score_delta | 2266 |
| distance_to_objective_delta | 78 |
| total | 2344 |

正式运行中，Move 学习评分均值显示 objective score 是综合评分的重要来源：

| Move 评分项 | 均值 |
| --- | ---: |
| intent_bias | -0.8108 |
| situation_score | 0.4117 |
| tactical_score | 0.0 |
| objective_score | 130.4092 |
| combined_score | 219.3561 |

这说明候选策略的行动排序并非只依赖规则 fallback，目标导向评分对动作选择具有实质贡献。

## 9. 认证与真实后端证据

正式运行满足以下可信性条件：

- `env_backend = train_env.TrainEnv`
- `mock_backend = false`
- `auth_status = mac_match_unverified`
- `auth_failure_count = 0`
- `.engine_config` 路径为用户本地 SDK 路径
- bound MAC 与当前 MAC 均为 `02:42:ac:11:00:02`
- `battle_phase_entered = true`

这组证据排除了三类常见伪成功：

1. 使用 mock 环境代替真实 SDK。
2. SDK 认证失败但仍将部署阶段输出当作结果。
3. 未进入战斗阶段，仅用 0-0 或无动作结果完成评估。

## 10. 测试结果

完整流水线报告记录：

`python3.10 -m pytest tests -q`

结果：

`139 passed`

测试覆盖内容包括 SDK 认证诊断、handoff 脚本、完整晋升脚本、阶段/动作诊断、自博弈 battle phase 合同、主线 gate 合同、objective label 构建、objective policy 集成、promotion artifact 合同等。

## 11. 结果解释

候选策略的优势主要体现在三个方面：

1. **总体胜率显著高于基线**：overall point estimate 达到 0.87，CI95 下界为 0.80，远高于 0.55 的停机下界。
2. **复杂地图仍有稳定优势**：complex 分桶胜率 0.8182，CI95 下界 0.6364，说明提升不是由简单地图单独贡献。
3. **学习路径真实参与动作选择**：learned/move/shoot 覆盖率均满足阈值，且 objective scoring 对 Move 评分贡献显著。

## 12. 剩余风险

虽然正式 gate 已通过，后续论文或报告中应诚实说明以下限制：

- 场景池只有 43 个唯一想定，100 配对评估需要重复采样。
- Tactical 模型在正式推理统计中贡献为 0，后续可单独排查加载和评分链路。
- Move 的 empty route fallback 仍较多，尤其在 Maneuver 和 Fly 模式下。
- Exploiter 验证规模较小，每类 exploiter 4 场，适合作为 gate 侧证据，但不足以作为完整鲁棒性研究。
- 当前结果是工程闭环意义上的 SOTA-aligned stop，不应表述为公开排行榜 SOTA。

