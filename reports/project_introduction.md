# 面向庙算陆战兵棋平台的学习增强型智能体项目介绍

## 1. 项目背景

本项目面向庙算陆战兵棋推演平台构建智能体决策系统。庙算平台由中科院自动化研究所开发，提供陆战兵棋推演环境、地图与想定数据、动作裁决规则以及本地 SDK。开发者通过实现 `BaseAgent` 的 `setup`、`step` 和 `reset` 接口，即可将智能体接入 `train_env.TrainEnv` 推演循环。

陆战兵棋智能体和棋类、Atari 或简单多智能体环境相比，具有更强的工程复杂度和评估难度：

- 态势信息复杂：包含算子位置、血量、弹药、载具、城市夺控、通信、裁决信息、阶段信息等。
- 动作空间复杂：核心动作包括 `Move`、`Shoot`、`Occupy`，同时存在上下车、状态切换、进入/退出工事、布雷、引导射击等动作。
- 地图因素复杂：地形、通行成本、通视、高程和六边形距离共同影响行动可行性。
- 闭环评估困难：监督学习指标不能代表真实推演能力，必须在环境闭环中验证策略是否产生有效战斗行为。

因此，本项目的研究重点不是单纯训练一个分类模型，而是建立一条从模型训练、策略接入、行为诊断、自博弈评估到 promotion gate 的完整工程闭环。

## 2. 研究目标

项目目标可以概括为：

> 构建一个能够在庙算陆战兵棋真实 SDK 后端中运行、加载学习模型、产生真实战斗行动，并通过多地图形式化自博弈 promotion gate 的学习增强型智能体。

围绕该目标，项目拆解为以下子目标：

1. 建立可运行的 `Agent` 入口，兼容庙算 SDK 的本地推演接口。
2. 在规则策略基础上接入学习模型，使模型真实影响 `Move` 和 `Shoot` 等关键动作。
3. 建立地图、路径规划和态势特征处理能力，支持不同想定和地图。
4. 建立 self-play 评估流程，用候选策略对比规则基线策略。
5. 建立 fail-closed promotion gate，防止部署阶段平局、认证失败、mock 环境等伪结果被误判为成功。
6. 输出完整、可审计、可复现的实验 artifacts，用于论文和报告撰写。

## 3. 系统总体架构

项目当前由两个并行职责组成：

- 运行时智能体逻辑：在 `TrainEnv` 中实时接收态势并输出动作。
- 模型生产与评估流水线：监督训练、自博弈评估、league gating、artifact 发布和诊断报告。

主要目录职责如下：

| 目录/文件 | 作用 |
| --- | --- |
| `ai/agent.py` | 运行时 Agent 入口，负责动作生成、优先级控制和规则 fallback |
| `ai/policy.py` | 学习策略封装，加载模型并为候选动作打分 |
| `ai/features.py` | 态势、算子和图结构特征提取 |
| `ai/models.py` | PyTorch 模型定义，包括意图、态势和战术模型 |
| `ai/weights/` | 当前候选策略发布权重和 manifest |
| `ai/weights_incumbent_rule/` | 规则基线/现任策略快照 |
| `training/supervised.py` | 监督训练入口，产出模型和训练报告 |
| `training/selfplay.py` | 正式 self-play 与 league 评估入口 |
| `training/evaluate.py` | 评分、置信区间、Elo、promotion gate 逻辑 |
| `scripts/` | 认证检查、objective label 构建、完整晋升流水线等脚本 |
| `configs/` | 监督训练、自博弈、smoke 和 hard-stop 配置 |
| `experiments/shenji_agent/` | 训练、自博弈、诊断和最终报告 artifacts |
| `reports/` | 面向论文和汇报的整理材料 |

## 4. 运行时决策流程

`Agent` 的运行时流程如下：

1. `TrainEnv.setup` 加载想定、地图、通行成本、通视矩阵和玩家信息。
2. `Agent.setup` 接收 `setup_info`，构建地图工具，读取阵营/席位，加载学习模型。
3. 每个推演步，`TrainEnv` 将本方态势 `observation` 传入 `Agent.step`。
4. `Agent.step` 解析 `valid_actions`，识别可控算子，构建候选动作集合。
5. `LearnedPolicy` 对候选 `Move`、`Shoot` 等动作进行意图、态势、目标和规则混合评分。
6. Agent 根据评分与 fallback 规则生成动作列表。
7. `TrainEnv.step(actions)` 执行动作并产生下一帧态势。
8. 诊断模块记录阶段、认证状态、有效动作、选择动作和 objective 进度。

核心设计原则是：学习模型必须进入真实动作选择链路，而不是仅在离线评估中产生指标。

## 5. 学习模型与策略融合

当前候选策略是“规则策略 + 学习模型”的混合策略。学习模型由 `ai/weights/model_manifest.json` 管理，主要包含：

| 模型 | 文件 | 作用 |
| --- | --- | --- |
| 意图识别模型 | `intent_lstm.pt` | 基于敌方历史轨迹预测未来行动倾向 |
| 态势图模型 | `situation_gcn.pt` | 对战场图结构和节点态势进行评分 |
| 特征归一化器 | `situation_feature_normalizer.json` | 保持训练/推理特征分布一致 |
| 战术评分模型 | `tactical_model.pt` | 为候选动作提供战术可行性参考 |
| Objective 评分 | manifest 中 `objective_enabled=true` | 基于得分、距离目标、城市控制等目标函数强化动作排序 |

策略融合不是替换规则系统，而是在规则可行动作空间内重新排序关键动作。这样可以保留规则策略的稳健性，同时让学习模型在高价值决策点上发挥作用。

## 6. 评估思想

项目采用 SOTA-aligned 的 promotion gate，而不是只看监督学习验证集准确率。原因是兵棋智能体必须在闭环推演中表现有效，静态数据指标无法证明策略会真实行动、进入战斗阶段或获得优势。

正式 stop rule 记录在 `program.md`，核心要求包括：

- 候选策略和现任策略必须来自不同 artifact 目录。
- 正式评估必须执行 100 个配对地图项。
- 总体和 complex 分桶都必须达到胜率点估计和置信区间下界要求。
- Elo 等效优势下界必须为正且超过阈值。
- exploiters 不能以过高胜率击败候选策略。
- 不能出现地图分桶退化。
- learned path、move path、shoot path 覆盖率必须达到下限。
- 以上条件必须连续两个窗口通过。

该评估思想借鉴了 AlphaGo Zero 的自博弈晋升机制和 AlphaStar 的 league/exploiter 思路，但针对庙算闭环控制任务重新定义了指标和 artifacts。

## 7. 工程难点与解决方案

### 7.1 SDK 认证路径问题

早期运行中，系统多次停留在部署阶段，并出现 SDK 认证失败。最初诊断认为是容器 MAC 与 `.engine_config` 绑定 MAC 不匹配。后续定位发现，诊断脚本和 self-play 模块读取了错误的 `/usr/local` SDK 路径，而真实认证器使用的是用户本地路径：

`/remote-home/lizilong/.local/lib/python3.10/site-packages/train_env/env/authenticate/.engine_config`

修正路径解析后，preflight 确认 MAC 匹配，正式运行中 `auth_failure_count=0`。

### 7.2 动作优先级饥饿

早期动作优先级中过早选择 `Occupy`，导致 `Move` 和 `Shoot` 学习路径被遮蔽，覆盖率接近零。修复后，核心战斗动作优先进入学习评分链路，最终正式运行中：

- `selected_move_actions = 89018`
- `selected_shoot_actions = 5308`
- `selected_occupy_actions = 356`

### 7.3 特征归一化不一致

态势模型训练和推理曾存在归一化不一致，导致模型输出方差极低，动作区分度不足。修复后，训练侧和推理侧统一使用 per-dimension z-score normalizer，态势模型输出恢复区分度。

### 7.4 Objective 目标评分

单纯 graph-level 标签不足以驱动得分产生行为，因此项目引入 objective label 构建和启发式 objective scoring。正式运行中 objective score 是 Move 综合评分的主要贡献项，说明目标导向评分对行动选择产生了实质影响。

## 8. 当前项目状态

截至正式报告记录，主线已完成：

- 真实 SDK 后端可运行。
- 候选策略模型 manifest 可加载。
- 自博弈流水线可执行。
- promotion gate 通过。
- 工程测试通过。
- 项目已具备论文和技术报告撰写所需的证据链。

剩余工作主要是研究写作、图表整理、进一步消融实验和泛化验证，而不是主线工程阻塞。

