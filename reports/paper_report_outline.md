# 论文与项目报告写作提纲

## 1. 推荐题目

可选题目：

1. 面向陆战兵棋推演的学习增强型智能体构建与自博弈验证
2. 基于庙算平台的陆战兵棋智能体闭环评估与 Promotion Gate 方法
3. 融合规则策略与学习模型的陆战兵棋智能体设计、诊断与验证
4. 面向复杂兵棋环境的学习增强决策链路诊断与多地图自博弈评估

## 2. 摘要草稿

本文面向庙算陆战兵棋推演平台，研究复杂陆战兵棋环境中学习增强型智能体的构建、诊断与闭环评估问题。针对兵棋环境动作空间复杂、地图约束强、部署与交战阶段切换明显、监督学习指标难以反映真实推演能力等问题，本文在规则基线智能体基础上引入意图识别、态势图评分和目标导向评分模块，构建了可在 `train_env.TrainEnv` 真实后端中运行的混合决策智能体。为保证实验结论可信，本文设计了包含认证状态、战斗阶段、动作类型、学习路径覆盖、目标进展和 exploiter 鲁棒性的诊断体系，并采用 100 对多地图配对自博弈作为正式 promotion gate。实验结果表明，候选策略在真实 SDK 后端中达到 0.87 的总体胜率点估计，95% 置信区间下界为 0.80；在复杂地图分桶中胜率点估计为 0.8182，置信区间下界为 0.6364；学习路径覆盖率为 0.3728，移动路径覆盖率为 0.3367，射击路径覆盖率为 1.0；最大 exploiter 胜率为 0.25，并连续两个评估窗口通过 gate。结果说明该方法能够在真实兵棋推演闭环中产生有效战斗行动，并相比规则基线取得稳定提升。

## 3. 论文结构建议

### 第 1 章 绪论

建议内容：

- 兵棋推演 AI 的背景和应用价值。
- 陆战兵棋相比棋类/Atari/简单 MARL 环境的复杂性。
- 庙算平台的 SDK 接口特点。
- 为什么仅用监督学习指标不够。
- 本文贡献。

可写贡献点：

1. 构建了一个可在庙算真实 SDK 后端运行的学习增强型智能体。
2. 设计了规则策略与学习模型融合的动作评分机制。
3. 建立了覆盖认证、阶段、动作、目标进展和学习路径的诊断体系。
4. 提出了适用于该项目的 self-play promotion gate，并在 100 对配对地图上通过验证。

### 第 2 章 相关工作

建议分节：

- 兵棋推演与军事仿真智能体。
- 深度强化学习与自博弈。
- AlphaGo Zero、AlphaStar、DeepNash 等 promotion/league 思想。
- 图神经网络在态势建模中的应用。
- 意图识别与轨迹预测。

注意表述：

本项目没有公开 protocol-matched leaderboard，因此不要声称“公开 SOTA”。更准确的说法是“采用 SOTA-aligned 的自博弈晋升验证思想”。

### 第 3 章 平台与问题建模

建议内容：

- `TrainEnv` 与 `Agent` 的接口。
- observation 和 action 的数据结构。
- 地图、想定、算子、城市和阶段。
- 动作空间：Move、Shoot、Occupy 等。
- 任务目标：候选策略相对规则基线的闭环推演优势。

建议图表：

- Agent/TrainEnv 交互流程图。
- observation/action 数据结构表。
- 动作类型表。

### 第 4 章 方法

建议内容：

- 总体架构：TrainEnv、Agent、LearnedPolicy、Route Planner、Diagnostics、Self-play Evaluator。
- 规则策略与学习策略的融合方式。
- 意图识别模型：历史轨迹、多 horizon 预测。
- 态势图模型：108 维特征、图结构评分、normalizer。
- Objective scoring：score_delta 和 distance_to_objective_delta。
- Promotion gate 设计。

建议图表：

- 系统架构图。
- 学习策略评分流程图。
- Promotion gate 判定流程图。

### 第 5 章 工程诊断与实现

建议内容：

- SDK 认证路径问题和修复。
- 动作优先级饥饿问题和修复。
- Move empty route 诊断。
- 训练/推理归一化一致性。
- fail-closed artifacts 设计。

建议图表：

- 诊断 sidecar 列表。
- 修复前后动作覆盖率对比。
- 认证状态证据表。

### 第 6 章 实验设计

建议内容：

- 实验环境和版本。
- 候选策略与基线策略。
- 地图采样协议。
- 指标定义：胜率、CI95、Elo、coverage、exploiter win rate、bucket regression。
- hard-stop 阈值。

建议表格：

- 实验配置表。
- Promotion gate 阈值表。
- 地图分桶统计表。

### 第 7 章 实验结果与分析

建议内容：

- 总体结果。
- 分桶结果。
- 学习路径覆盖。
- Exploiter 鲁棒性。
- Objective scoring 作用。
- 可信性分析。

核心结果表：

| 指标 | 结果 |
| --- | ---: |
| overall point estimate | 0.8700 |
| overall CI95 lower | 0.8000 |
| complex point estimate | 0.8182 |
| complex CI95 lower | 0.6364 |
| learned path coverage | 0.3728 |
| move path coverage | 0.3367 |
| shoot path coverage | 1.0000 |
| Elo lower bound | 240.8240 |
| max exploiter win rate | 0.25 |
| selected Move/Shoot/Occupy actions | 94682 |

### 第 8 章 结论与展望

建议内容：

- 总结项目实现和实验结论。
- 强调真实 SDK 后端、闭环推演和 promotion gate 的可信性。
- 说明限制：场景池规模、tactical 分支贡献、empty route、exploiter 数量。
- 展望：更多想定、更强 exploiter、强化学习/离线 RL、端到端策略优化、多智能体协同。

## 4. 图表清单

建议制作以下图表：

1. 系统总体架构图。
2. Agent/TrainEnv 交互流程图。
3. LearnedPolicy 评分流程图。
4. Promotion gate 流程图。
5. Overall/simple/medium/complex 胜率柱状图，带 CI。
6. Move/Shoot 学习路径覆盖率柱状图。
7. Exploiter win rate 对比图。
8. Objective score 与 combined score 的贡献图。
9. 迭代修复时间线。
10. 正式 self-play artifacts 证据链图。

## 5. 可直接引用的关键句

- 本项目采用真实 `train_env.TrainEnv` 后端完成评估，正式运行中 `mock_backend=false` 且 `auth_failure_count=0`。
- 候选策略在 100 对配对地图上达到 0.87 的总体胜率点估计，95% 置信区间下界为 0.80。
- 在 complex 分桶中，候选策略达到 0.8182 的胜率点估计，95% 置信区间下界为 0.6364，说明提升并非仅来自简单地图。
- 学习路径覆盖率为 0.3728，移动路径覆盖率为 0.3367，射击路径覆盖率为 1.0，说明学习模型真实参与了关键动作选择。
- 最大 exploiter 胜率为 0.25，低于 gate 阈值 0.45，说明候选策略未被固定 exploiter 轻易利用。
- 正式 promotion gate 连续两个窗口通过，满足项目 hard-stop 条件。

## 6. 写作注意事项

- 不要声称“公开 SOTA”，除非后续补充了可比公开排行榜。建议使用“SOTA-aligned promotion gate”或“参考 SOTA 系统评估思想”。
- 明确说明场景池为 43 个唯一想定，100 配对包含重复采样。
- 将 `auth_status=mac_match_unverified` 解释为 MAC 匹配但 SDK 未提供更细粒度公开认证状态；关键证据是 `auth_failure_count=0` 和 `battle_phase_entered=true`。
- Tactical 模型正式推理贡献为 0，应作为限制或后续工作，而不是主要贡献。
- Objective scoring 当前是启发式策略评分，不应描述为已经训练完成的 objective neural model。

