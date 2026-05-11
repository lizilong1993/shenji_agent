# 实验更新日志与问题修复记录

本文件按阶段整理项目从 SDK demo 到正式 promotion gate 通过的主要进展。日期和迭代编号以仓库报告、提交历史和 artifacts 为依据。

## 阶段 1：SDK 接入与基础 Agent

目标：确认庙算 SDK 的运行方式，建立可被 `run_offline_games.py` 加载的 `Agent`。

主要工作：

- 梳理 `TrainEnv.setup`、`Agent.setup`、`Agent.step`、`TrainEnv.step` 的数据流。
- 确认 observation 中的关键字段：`operators`、`valid_actions`、`cities`、`time`、`judge_info`、`scores`。
- 保留规则策略作为 baseline fallback，避免学习模型不可用时 Agent 直接失效。
- 建立地图工具、距离计算、通视和路径规划能力。

阶段结论：

Agent 可以在 SDK 接口内完成基础推演，但此时主要依赖规则策略，尚不能证明学习模型真实影响闭环行为。

## 阶段 2：监督训练与模型发布

目标：构建学习模型训练管线，并把模型发布到运行时权重目录。

主要工作：

- 建立 `training/supervised.py` 训练入口。
- 支持三类模型：意图识别、态势图评分、战术评分。
- 通过 `ai/weights/model_manifest.json` 记录模型文件、训练来源、feature normalizer 和 source commit。
- 将候选策略设置为 `ai/weights`，规则基线设置为 `ai/weights_incumbent_rule`。

关键数据：

- `FEATURE_DIM = 108`
- `history_len = 8`
- `horizons = 5`
- `objective_enabled = true`

阶段结论：

模型可以被发布和加载，但监督学习指标本身不能证明闭环能力，需要 self-play 验证。

## 阶段 3：动作链路诊断

目标：确认学习策略是否真正参与 `Move`、`Shoot` 和 `Occupy` 行为。

发现的问题：

- 动作优先级中 `Occupy` 过早返回，导致 `Move` 和 `Shoot` 学习路径被遮蔽。
- 诊断不足，无法区分“环境没有有效动作”和“Agent 有效动作被策略优先级覆盖”。

修复：

- 调整动作优先级，使 `Shoot` 和 `Move` 进入更靠前的学习评分链路。
- 增加 `phase_action_diagnostics.json` 和 `valid_action_timeline.json`，记录每步有效动作类型与选中动作类型。
- 增加 battle phase 合同测试，防止仅部署阶段的结果被误判。

阶段结论：

动作选择链路具备可观测性，后续可以用 coverage 指标判断学习路径是否真实参与。

## 阶段 4：路径规划与 Move 修复

目标：解决 Move 动作空路径过多的问题。

发现的问题：

- 部分地图/移动模式下 `gen_move_route` 返回空路径。
- `empty_route` 导致学习策略即使选择了 Move 候选，也无法生成有效动作。

修复：

- 增加 Move route diagnostics，记录空路径原因。
- 引入 fallback 策略，允许不同移动模式和备选目标处理。
- 将 route 诊断透传到 self-play artifacts。

正式运行现状：

- Move 总尝试：23580
- 学习路径 Move：7939
- move path coverage：0.3367
- 剩余 fallback 原因主要为 `no_valid_move_candidate`

阶段结论：

Move 路径覆盖已超过 gate 阈值，但 empty route 仍是后续优化方向。

## 阶段 5：态势模型归一化修复

目标：保证态势模型训练和推理使用一致的特征分布。

发现的问题：

- 训练侧计算了 normalizer，但模型训练时未按相同方式应用。
- 推理侧加载 normalizer 后，输入分布与训练分布不一致。
- 模型输出方差极低，动作评分缺乏区分度。

修复：

- 在训练侧实际应用 per-dimension z-score normalizer。
- 将 `situation_feature_normalizer.json` 作为 manifest 管理文件。
- 增加 feature normalizer consistency 测试。

阶段结论：

态势模型输出恢复区分度，训练/推理特征契约得到测试保护。

## 阶段 6：Objective 标签与评分

目标：让策略优化更直接面向得分、夺控和接近目标，而不是只拟合静态标签。

发现的问题：

- 早期 objective labels 高置信得分/夺控/伤害标签不足。
- 静态 graph label 不能充分驱动闭环 score-producing 行为。

修复：

- 新增 `scripts/build_objective_labels.py`，从真实 battle phase artifacts 中抽取 objective labels。
- 引入 `score_delta` 和 `distance_to_objective_delta` 两类目标标签。
- 增加 objective progress diagnostics、action outcome diagnostics 和 objective policy 集成测试。

认证恢复后的 label 结果：

- `score_delta = 2266`
- `distance_to_objective_delta = 78`
- `total = 2344`

正式运行效果：

- Move objective score 均值：130.4092
- Move combined score 均值：219.3561

阶段结论：

Objective scoring 已成为 Move 动作评分的重要信号。

## 阶段 7：SDK 认证阻塞与恢复

目标：确保评估在真实 SDK battle phase 中运行，而不是认证失败状态。

发现的问题：

- self-play smoke 中出现 `did not pass authentication`。
- battle phase 未进入，Move/Shoot/Occupy 选择数为 0。
- 初步 evidence 显示 `/usr/local` 路径下 `.engine_config` 的 MAC 与当前容器 MAC 不一致。

进一步定位：

- SDK 实际读取用户本地路径：
  `/remote-home/lizilong/.local/lib/python3.10/site-packages/train_env/env/authenticate/.engine_config`
- 诊断脚本和 self-play 代码此前偏向 `/usr/local` 路径，导致误判。

修复提交：

- `dab4a84 fix: align auth diagnostics with sdk engine config path`
- `0d448d3 fix: prefer user-local sdk package for selfplay`

验证：

- `auth_environment_ready = True`
- `blocker_reason = None`
- `bound_mac = current_mac = 02:42:ac:11:00:02`
- `auth_failure_count = 0`
- `battle_phase_entered = true`

阶段结论：

认证问题已恢复，正式评估证据来自真实 SDK 后端。

## 阶段 8：正式 Promotion Gate

目标：运行完整晋升流水线并判断主线是否达到 stop-ready。

执行命令：

```bash
bash scripts/run_full_promotion_after_auth.sh
```

流水线内容：

1. SDK auth preflight。
2. Battle smoke。
3. Objective smoke。
4. Objective label 构建。
5. Objective-aware policy scoring 验证。
6. 全量测试。
7. `configs/selfplay_sota_stop.yaml` 正式 self-play。
8. Promotion gate 验证。
9. 最终报告生成。

正式 artifacts：

- `experiments/shenji_agent/league_20260501_170204/promotion.json`
- `experiments/shenji_agent/league_20260501_170204/rating.json`
- `experiments/shenji_agent/league_20260501_170204/summary.md`
- `experiments/shenji_agent/league_20260501_170204/exploitability.json`

结果：

- `promotion_accepted = true`
- `failure_reasons = []`
- `overall_point_estimate = 0.87`
- `overall_ci95_lower = 0.8000`
- `complex_point_estimate = 0.8182`
- `complex_ci95_lower = 0.6364`
- `consecutive_windows = 2/2`

阶段结论：

主线达到 stop-ready，可进入论文/报告整理阶段。

## 当前剩余风险与后续工作

建议后续围绕以下方向继续完善：

1. 扩大 exploiter 验证规模，增强鲁棒性结论。
2. 排查 tactical 模型在正式推理中的加载/评分贡献为 0 的原因。
3. 继续降低 Move empty route 比例，特别是 Maneuver 和 Fly 模式。
4. 增加跨随机种子、跨 map list 的重复正式评估。
5. 将 diagnostic sidecars 转换为论文图表，如动作覆盖时间线、分桶胜率图、objective score 分布图。
6. 若论文需要公开可复现性，整理 SDK 依赖、授权路径和数据许可说明。

