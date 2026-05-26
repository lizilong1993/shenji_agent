# 庙算平台陆战兵棋AI开发项目

## 项目简介
基于中科院庙算平台（陆战兵棋推演）的 AI 开发项目，使用强化学习 + self-play 训练智能兵棋 AI。

## 环境

- Python 版本：3.10.12
- 虚拟环境：`/remote-home/lizilong/venvs/shenji`（已激活，包含所有依赖）
- 项目根 PYTHONPATH：本目录（确保可 import `training`, `ai`, `scripts`, `configs`）
- SDK：`land_wargame_train_env==4.1.0`（已安装 whl）
- 硬件：CPU + GPU (CUDA)

## 关键目录

| 目录 | 用途 |
|------|------|
| `ai/` | AI agent 实现（agent.py, policy.py, models.py, features.py） |
| `training/` | 训练流水线（supervised.py, selfplay.py, evaluate.py） |
| `scripts/` | 辅助脚本（诊断、评估、同步等） |
| `configs/` | 训练/自对弈配置（YAML） |
| `land_wargame_sdk/` | SDK 及推演入口（run_offline_games.py） |
| `.agent/` | 研究管理（context.md 主真值、progress.md 进展） |

## 常用命令

```bash
# 激活虚拟环境
source /remote-home/lizilong/venvs/shenji/bin/activate

# 监督训练
python train_models.py --config configs/supervised_score_quality.yaml

# 自对弈
python -m training.selfplay --config configs/selfplay_smoke_learned_engagement.yaml

# 评估
python run_evaluation.py

# SDK 离线推演
python land_wargame_sdk/run_offline_games.py

# AI 审稿
python scripts/run_ai_reviewer.py --config configs/ai_reviewer.yaml
```

## 关键架构

- AI Agent 继承 `BaseAgent`，实现 `setup()` / `step()` / `reset()`
- 推演循环：`TrainEnv.setup → Agent.setup → (Agent.step ↔ TrainEnv.step) → reset`
- 训练 pipeline：`supervised.py`（监督学习）→ `selfplay.py`（自对弈）→ `evaluate.py`（评估）

## 研究管理

- 主真值见 `.agent/context.md`
- 文献见 `references.bib`
- 实验结论见 `experiments/shenji_agent/`
- league 正式证据见 `experiments/shenji_agent/league_*/` sidecars

## 工作流规则

- 复杂任务使用 Claude Code sub-agents 并行执行
- 修改后必须验证（跑测试或执行）
- 无验证证据不声称已修复
- 遵循 AGENTS.md 中的文件治理规则
