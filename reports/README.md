# Shenji Agent 报告材料索引

本目录汇总用于论文、结题报告和项目汇报的材料。内容按“可直接引用”和“证据索引”组织，核心证据来自真实 `train_env.TrainEnv` 后端的正式自博弈运行：

- 正式运行目录：`experiments/shenji_agent/league_20260501_170204`
- 完整闭环报告：`experiments/shenji_agent/full_promotion_after_auth_report.md`
- 最终完成报告：`experiments/shenji_agent/final_autonomous_completion_report.md`
- 主线停机准则：`program.md`

## 文件说明

- `project_introduction.md`：从研究背景、平台接口、系统架构、算法模块到工程难点的完整项目介绍。
- `experiment_results.md`：实验协议、评价指标、正式结果表格、诊断结论和可信性说明。
- `experiment_update_log.md`：按迭代阶段整理的问题、修复、验证和剩余风险。
- `paper_report_outline.md`：论文/技术报告写作提纲、可引用贡献点、图表建议和摘要草稿。

## 核心结论

候选策略 `CurrentAI` 在真实庙算 SDK 后端完成 100 对形式化配对地图自博弈，promotion gate 通过：

- `promotion_accepted = true`
- 总体胜率点估计：`0.87`
- 总体 95% CI 下界：`0.8000`
- complex 分桶胜率点估计：`0.8182`
- complex 95% CI 下界：`0.6364`
- learned path coverage：`0.3728`
- move path coverage：`0.3367`
- shoot path coverage：`1.0000`
- Elo 优势下界：`240.82`
- 最大 exploiter 胜率：`0.25`
- 连续通过窗口：`2/2`
- 真实后端：`env_backend = train_env.TrainEnv`
- 认证失败次数：`auth_failure_count = 0`
- 战斗阶段进入：`battle_phase_entered = true`
- 选择执行的 Move/Shoot/Occupy 动作：`94682`
- 工程测试：`139 passed`

