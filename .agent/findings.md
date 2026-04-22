# 科研路线审计 Findings

## 任务范围与证据面

- 本轮任务目标是“科研路线审计 + 后续优化路线设计”，不修改业务代码，只把分析结果落盘到 `.agent/`。
- 项目根 `AGENTS.md` 与 `.agent/context.md` 在本地工作树中均不存在；本轮执行约束来自用户贴出的项目 `AGENTS.md`。
- 已执行 `bootstrap-task`，`C:\Users\lizilong\.codex\.agent\task_bootstrap.json` 记录状态为 `bootstrapped/succeeded`。
- 已交叉核对的证据面包括：
  - 文档与历史：`README.md`、`ALGORITHM_OPTIMIZATION_PLAN.md`、根目录旧 `findings.md` / `task_plan.md` / `progress.md`
  - 代码：`ai/agent.py`、`ai/policy.py`、`ai/models.py`、`ai/features.py`、`training/common.py`、`training/data.py`、`training/maps.py`、`training/supervised.py`、`training/selfplay.py`、`training/evaluate.py`、`scripts/report_progress.py`
  - 配置：`configs/supervised_baseline.yaml`、`configs/selfplay_baseline.yaml`
  - 实验产物：`experiments/shenji_agent/20260327_141522/*`、`experiments/shenji_agent/league_20260327_141724/*`、`league/league_state.json`、`ai/weights/model_manifest.json`
  - 数据证据：`datasets/意图识别研究数据集WargameData_mini01/*`、`datasets/[重庆大学] 分队级陆战兵棋态势表征图数据集 WG-StateGraph/*`、`datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv`

## 一句话结论

当前项目的真实科研主线不是“端到端策略学习”或“用更大模型替换规则 AI”，而是：

- 以 `ai/agent.py` 的规则决策链为主干；
- 用三条 supervised 线训练局部 surrogate model；
- 再把其中的 `intent + tactical` 以局部打分器形式挂接到 `Shoot` / `Move` 两类动作；
- 最后用 self-play / league 脚手架尝试做晋升评测。

但这条路线目前还没有形成可信的科研闭环。当前第一瓶颈更像是“任务定义/标签正确性 + 评估闭环 + 模型接入宽度”，而不是“模型容量不够”。

## 当前项目真正的研究问题是什么

从代码、配置、实验产物三类证据交叉看，当前项目真正试图回答的问题是：

- 能否用敌方短期运动预测（intent）和局部动作/目标点评分（tactical）改善规则 AI 的局部决策？
- 这些局部 surrogate signal 能否在 self-play league 中转化为更高的 paired score，并最终晋升为新 champion？
- `situation` 线是否能提供更高层的战场态势价值信号，为后续策略优化服务？

但当前真正被验证到的，只是“是否能离线训练出这些模型并把一部分权重接入 agent”。还没有证据证明它们提升了最终对抗能力。

## 当前代码和实验实际上在验证什么科学假设

### 1. intent 线

- 训练目标：`training/data.py` 从 `tank_enemy` 字段抽取 8 帧历史，预测未来 5 个时间窗口的 7 类方向 one-hot；`training/supervised.py` 用 `MultiHorizonIntentionModel` 训练并汇报 `val_immediate_accuracy` 与 `h1-h5` 准确率。
- 接入方式：`ai/policy.py` 用 `intent_model.pt` 预测敌方下一步方向，只在 `choose_city()` 中生成敌方下一格预测位置，用作 `threat_penalty`。
- 实验事实：`experiments/shenji_agent/20260327_141522/metrics.json` 中 `intent.val_immediate_accuracy=0.515625`，远期 horizon 准确率上升到 `0.6572-0.7324`。
- 科学含义：当前 intent 线验证的是“敌方短期离散方向是否可预测”，而不是“识别高层战术意图”。

### 2. tactical 线

- 训练目标：`training/supervised.py` 用 `TacticalScorer` 对 32 维动作特征做二分类。
- 接入方式：`ai/policy.py` 在 `choose_shoot_candidate()` 和 `choose_city()` 中把 tactical score 与规则项相加，改变射击目标与目标点排序。
- 实验事实：`metrics.json` 给出的 `tactical.val_accuracy=0.5307`。
- 关键问题：`training/data.py` 选择标签列的逻辑是“找英文 `label`，否则退回最后一列”。而战术 CSV 的真实标签列是第 4 列 `样本标签`，最后一列其实是 `整体移动方向`。因此当前 tactical 训练指标大概率并不对应原始战术优劣标签。

### 3. situation 线

- 训练目标：`training/supervised.py` 用 `GraphSituationModel` 从 `encoded_nodes-features/*.csv` 预测 `label.csv` 中的 0/1 胜负标签。
- 实验事实：`metrics.json` 中 `situation.val_accuracy=0.4091`，样本量只有 108。
- 接入事实：`ai/policy.py` 根本不加载 `situation_model.pt`；`ai/agent.py` 虽在构造函数里尝试加载旧格式 `situation_gcn.pth`，但后续决策逻辑没有任何地方调用 `self.situation_model`。
- 科学含义：situation 线目前只是“离线训练过一个图分类器”，还不是 agent 决策链的一部分。

### 4. self-play / league 线

- 设计意图：`training/selfplay.py` 试图在不同地图 bucket 上跑 paired matches，用 `training/evaluate.py` 决定 candidate 是否通过 promotion gate。
- 实际产物：`experiments/shenji_agent/league_20260327_141724/match_results.json` 里 64 场比赛全部 `paired_score=0.5`，且四个 reward 全都是 `1000.0`。
- 直接原因：该次运行默认 `--candidate ai/weights --incumbent ai/weights`，challenger 与 incumbent 使用的是同一套产物目录。
- 科学含义：当前 latest league 产物只能证明“同一模型打自己得到 0.5”，不能证明 supervised 提升了对抗能力。

## 当前系统到底是什么路线

当前系统最准确的描述是：

- `规则基线 + learned policy 局部挂接 + supervised surrogate tasks + self-play/league 脚手架`

而不是：

- 纯规则系统
- 端到端强化学习策略
- 已完成闭环的 self-play league 主线

原因是：

- `ai/agent.py` 的动作主链、优先级、绝大多数动作生成仍是规则/随机启发式；
- learned policy 只覆盖 `Shoot` 与 `Move` 两个动作分支；
- `situation` 线没有进入决策；
- self-play 晋升链最新一次实验不是有效 challenger-vs-incumbent 对比。

## `intent` / `tactical` / `situation` 三条 supervised 线分别在学什么、产出什么、如何接入

| 线 | 在学什么 | 训练产物 | 接入位置 | 当前作用范围 |
| --- | --- | --- | --- | --- |
| `intent` | 敌方坦克未来 5 个时间窗的离散方向 | `intent_model.pt` | `ai/policy.py::predict_location()` | 仅影响 `Move` 的威胁惩罚 |
| `tactical` | 局部动作/目标点 32 维特征的二分类分数 | `tactical_model.pt` | `choose_shoot_candidate()` / `choose_city()` | 仅影响射击目标与目标点排序 |
| `situation` | 战局图的 0/1 胜负标签 | `situation_model.pt` | 未接入 `LearnedPolicy`；旧 `.pth` 也未参与决策 | 当前对 agent 决策无直接作用 |

## self-play / league 的晋升门槛是否已经构成有效闭环

结论：没有。

### 已落地部分

- 有 league 状态文件：`league/league_state.json`
- 有 promotion gate 配置：`configs/selfplay_baseline.yaml`
- 有 self-play 评测产物：`rating.json`、`promotion.json`、`exploitability.json`、`league_state.json`

### 关键断裂

- 最新一次 self-play 不是 candidate-vs-incumbent 的有效对比，而是同一产物目录对打。
- `promotion_decision()` 只检查 `overall/complex` 的 point estimate 和 CI，下列配置项没有真正进入 acceptance 逻辑：
  - `elo_delta_lower_min`
  - `max_bucket_regression`
  - `max_exploiter_win_rate`
- exploitability 虽被计算，但只在 `scripts/report_progress.py` 里参与“breakthrough/stall/none”分类，不参与正式晋升判定。
- 因此当前 league 更像“评测脚手架 + 状态落盘”，还不是完整的科研闭环。

## supervised 指标提升能否证明对抗能力提升

结论：不能。

### 证据断裂点

- 离线 supervised 指标与最终 paired score 之间没有做过 baseline / ablation 映射实验。
- 当前 latest league 运行并未比较不同模型版本。
- `intent` / `tactical` 只改变局部子决策，不等于改变主策略主链。
- `situation` 根本没有接入 agent。
- bucket 指标本身不可信：
  - intent 数据的文件名被当前 `infer_map_id_from_scenario_id()` 全部推断成 `9601`，所以训练报告里 intent 只出现 `medium` bucket；
  - state graph 标签文件被同样的字符串启发式几乎全部误推到 `map_id 21/19`，导致 `slice_report.json` 里几乎只看到 `simple` bucket。

## `README.md` 与 `ALGORITHM_OPTIMIZATION_PLAN.md` 哪些已落地，哪些过时

### 已落地

- 有 supervised 训练入口、self-play 评测入口、权重发布到 `ai/weights/`
- 有 learned policy 局部挂接
- 有 league 状态与晋升结果落盘

### 部分落地

- “数据驱动 + 规则混合”这个大方向已开始
- `intent` / `tactical` 两条线已训练并局部接入
- `situation` 线只训练了离线模型，没有接入主链

### 仅规划或已明显过时

- `README.md` 仍基本停留在“SDK 与 Docker 使用说明”，没有描述当前 `shenji_agent` 路线
- `ALGORITHM_OPTIMIZATION_PLAN.md` 中的 HGT、Conv-LSTM、TensorRT、Kafka/Flink、对抗训练、异常检测、人机实战部署等都没有对应代码/配置/实验闭环
- 文档里把当前系统描述成“从规则驱动向数据驱动转型的关键阶段”是对的，但实现深度远小于文档中宣称的路线宽度

## 已落地 / 部分落地 / 仅规划

### 已落地

- `training/supervised.py` 三任务训练、权重保存、`metrics.json` / `slice_report.json` / `summary.md` 输出
- `ai/policy.py` 读取 `model_manifest.json` 并加载 `intent_model.pt` / `tactical_model.pt`
- `training/selfplay.py` 执行 paired matches，并生成 `rating.json` / `promotion.json` / `exploitability.json`
- `scripts/report_progress.py` 能根据实验目录输出 `none/breakthrough/stall`

### 部分落地

- `situation` 训练完成但未接入
- league gate 有配置但 acceptance 逻辑不完整
- map bucket 评测存在形式，但当前实现对现有文件名体系基本失真
- exploiters 已定义，但 exploitability 没有进入 promotion acceptance

### 仅规划

- 更大模型、更复杂图模型、流式架构、量化部署、对抗训练、异常检测、人机实战验证
- “完整替换规则模块”的目标

## 代码、配置、实验产物是否一致

| 主题 | 代码 | 配置 | 实验 | 一致性判断 |
| --- | --- | --- | --- | --- |
| supervised 三线存在 | 有 | 有 | 有 | 一致 |
| intent/tactical 已接入 agent | 有 | 间接体现 | 无直接 ablation 证据 | 代码有、实验未证成 |
| situation 已服务决策 | 训练代码有 | 有 | 有离线指标 | 不一致，未接入主链 |
| league 晋升闭环有效 | 有脚手架 | 有 gate | latest run 退化为同模型对打 | 不一致 |
| bucket 评测有效 | 有实现 | 有 map sampling | latest report 极度偏斜 | 不一致 |
| README/优化计划反映现状 | 否 | 否 | 否 | 已过时 |

## 当前最关键的科研问题列表

### 1. tactical 任务的标签语义是错的

- 证据来源：
  - `training/data.py` 把标签列设为“英文 `label` 否则最后一列”
  - `Wargame_Feature_Dataset.csv` 的真实标签列是 `样本标签`，最后一列是 `整体移动方向`
- 影响：
  - 当前 `tactical.val_accuracy=0.5307` 不能被解释为“战术动作好坏预测准确率”
  - 这条线的科研结论目前不成立

### 2. 当前最核心的证据断裂是“surrogate 指标 -> 对抗能力”

- 证据来源：
  - supervised 产物只有局部准确率/桶准确率
  - latest league 运行没有比较不同模型版本
- 影响：
  - 目前无法回答“supervised / situation / tactical / self-play 这些线是否真正服务于最终对抗能力提升”
  - 任何“模型有效”的说法都还缺少终局证据

### 3. 模型接入主链过窄，agent 仍主要靠规则跑

- 证据来源：
  - `ai/agent.py` 只在 `_generate_action()` 的 `Shoot` / `Move` 分支调用 `LearnedPolicy`
  - 动作优先级、`Occupy`、`GuideShoot`、`JMPlan`、`GetOn/GetOff`、`ChangeState` 等仍是规则/随机逻辑
- 影响：
  - 即便局部模型有增益，也容易被规则主链淹没
  - 当前路线更像“局部决策打分器”，不是“学到作战策略”

### 4. situation 线没有进入科研闭环

- 证据来源：
  - `training/supervised.py` 训练 `situation_model.pt`
  - `ai/policy.py` 不加载 `situation_model.pt`
  - `ai/agent.py` 中旧 `.pth` 模型也不参与动作生成
- 影响：
  - 当前 `situation` 实验对最终能力没有可验证贡献
  - 这条线若继续推进，必须先定义它如何影响决策

### 5. league 晋升逻辑不完整，latest run 不能支撑路线判断

- 证据来源：
  - `training/selfplay.py` 默认 `candidate == incumbent == ai/weights`
  - latest `match_results.json` 全部 `paired_score=0.5`
  - `promotion_decision()` 未使用 exploitability/regression/elo gate
- 影响：
  - 当前 league 更像“有输出文件”而不是“有科研判据”
  - 晋升失败不等于路线无效，晋升成功也还不够严格

### 6. map bucket 评测失真，跨地图泛化判断不可靠

- 证据来源：
  - `infer_map_id_from_scenario_id()` 依赖文件名字符串包含关系
  - intent 数据全被推成 `9601 -> medium`
  - state graph 数据几乎全被推成 `21/19 -> simple`
- 影响：
  - 当前 `slice_report.json` 不能支撑“复杂地图更强/更弱”的科研判断

## 对“当前第一瓶颈是什么”的结论

按照当前证据，优先级从高到低是：

1. `任务定义问题 + 标签问题`
2. `评估问题 + 训练闭环问题`
3. `模型接入与决策脱耦问题`
4. `模型本身问题`

也就是说，当前最值得优先优化的不是“换更大模型”，而是先把“训练的到底是什么、评估的到底是不是最终目标、模型到底有没有进入主决策链”这三件事钉实。

## 审计结论

- 当前真实主线是“规则主干上的局部 learned policy 实验”，不是完整的 self-play 主线。
- `intent` 与 `tactical` 目前都更像 surrogate task；其中 `tactical` 还存在标签语义错误。
- `situation` 线没有真正服务于最终决策。
- latest self-play 产物不能证明 supervised 指标提升带来对抗能力提升。
- 下一阶段最值得做的科研优化，不是扩大模型规模，而是先修正标签/评估/晋升闭环，并做 baseline-vs-ablated-vs-learned 的因果验证。
