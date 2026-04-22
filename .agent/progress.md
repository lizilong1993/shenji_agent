# Progress Log

## 本次任务摘要

- 任务类型：科研路线审计 + 2-4 周优化路线设计
- 工作范围：只读扫描 + 文档落盘
- 代码实现：未改业务代码，未做训练、远程评测或大规模运行

## 规则与上下文状态

- 本地项目根 `AGENTS.md`：未发现文件
- 本地 `.agent/context.md`：未发现文件
- 执行方式：按用户贴出的项目 `AGENTS.md` 规则执行
- bootstrap：
  - 已执行 `C:\Users\lizilong\.codex\automations\weekly-skill-trigger-optimization\codex-runtime-ops.ps1 -Action bootstrap-task -BootstrapChain using-superpowers`
  - `C:\Users\lizilong\.codex\.agent\task_bootstrap.json` 当前记录为 `bootstrapped`

## 本次实际扫描的文件与产物

### 文档与历史参考

- `README.md`
- `ALGORITHM_OPTIMIZATION_PLAN.md`
- 根目录旧 `findings.md`
- 根目录旧 `task_plan.md`
- 根目录旧 `progress.md`

### 代码

- `ai/agent.py`
- `ai/policy.py`
- `ai/models.py`
- `ai/features.py`
- `training/common.py`
- `training/data.py`
- `training/maps.py`
- `training/supervised.py`
- `training/selfplay.py`
- `training/evaluate.py`
- `scripts/report_progress.py`

### 配置

- `configs/supervised_baseline.yaml`
- `configs/selfplay_baseline.yaml`

### 实验与权重产物

- `experiments/shenji_agent/20260327_141522/metrics.json`
- `experiments/shenji_agent/20260327_141522/summary.md`
- `experiments/shenji_agent/20260327_141522/slice_report.json`
- `experiments/shenji_agent/league_20260327_141724/rating.json`
- `experiments/shenji_agent/league_20260327_141724/promotion.json`
- `experiments/shenji_agent/league_20260327_141724/exploitability.json`
- `experiments/shenji_agent/league_20260327_141724/league_state.json`
- `experiments/shenji_agent/league_20260327_141724/match_results.json`
- `league/league_state.json`
- `ai/weights/model_manifest.json`

### 数据与说明文件

- `datasets/意图识别研究数据集WargameData_mini01/train.json`
- `datasets/意图识别研究数据集WargameData_mini01/test.json`
- `datasets/意图识别研究数据集WargameData_mini01/WargameData_mini01生成及使用说明.md`
- `datasets/[重庆大学] 分队级陆战兵棋态势表征图数据集 WG-StateGraph/label.csv`
- `datasets/[重庆大学] 分队级陆战兵棋态势表征图数据集 WG-StateGraph/Readme.md`
- `datasets/陆战兵棋态势特征数据集/Wargame_Feature_Dataset.csv`

## 关键证据来自哪里

- 当前真实训练路线：
  - `training/supervised.py`
  - `training/data.py`
  - `configs/supervised_baseline.yaml`
- 当前模型真实接入点：
  - `ai/agent.py`
  - `ai/policy.py`
- self-play / league 是否构成闭环：
  - `training/selfplay.py`
  - `training/evaluate.py`
  - `experiments/shenji_agent/league_20260327_141724/*`
- latest 状态是否有突破：
  - `scripts/report_progress.py`
  - 直接运行 `python scripts/report_progress.py --experiments-root experiments/shenji_agent`
- 数据/标签语义：
  - `WargameData_mini01生成及使用说明.md`
  - `WG-StateGraph/Readme.md`
  - `Wargame_Feature_Dataset.csv`
  - `training/data.py`

## 已验证判断

- 已验证：当前项目真实主线是“规则 agent + 局部 learned policy + supervised surrogate + self-play 脚手架”，不是端到端策略学习。
- 已验证：`intent` 预测敌方短期方向，`tactical` 排序射击/目标点，`situation` 只做离线图分类。
- 已验证：`ai/policy.py` 只加载 `intent_model.pt` 和 `tactical_model.pt`，不加载 `situation_model.pt`。
- 已验证：`ai/agent.py` 中 learned policy 只影响 `Shoot` 与 `Move` 两个动作分支。
- 已验证：latest league 运行中 `candidate == incumbent == ai/weights`，64 场比赛全部 `paired_score=0.5`。
- 已验证：`report_progress.py` 对当前实验根输出 `event=none`。
- 已验证：`tactical` 训练代码把最后一列 `整体移动方向` 当成标签，而不是第 4 列 `样本标签`。
- 已验证：intent/state graph 的 bucket 映射对现有文件名体系明显失真。

## 未验证或本轮未做的内容

- 未验证：修正 tactical 标签后，当前路线是否会显著提升 paired score。
- 未验证：`situation` 若接入主链，是否会带来任何正收益。
- 未验证：更大模型、多任务训练、重型基础设施是否必要。
- 未验证：远程服务器上是否存在更新的实验产物。本轮按用户要求未做远程训练、远程评测或远程同步。

## 当前阻塞点

- 阻塞 1：现有离线 tactical 指标不可信，因为标签语义错误。
- 阻塞 2：现有 bucket 报告不可信，因为 map 推断几乎退化。
- 阻塞 3：现有 latest league 产物不构成有效 challenger-vs-incumbent 对比。
- 阻塞 4：`situation` 线没有接入主决策链，导致其科研价值无法判断。

## 本次是否只做分析、未做实现

- 是。本次只做审计、证据核对和方案落盘。
- 未修改业务代码、训练脚本、配置逻辑或远程环境。

## 本次收尾动作

- 已新建 `.agent/` 工作区并写入：
  - `.agent/findings.md`
  - `.agent/task_plan.md`
  - `.agent/plans/research_route_optimization.md`
  - `.agent/progress.md`
- 待执行：
  - 校验文档已写入磁盘
  - 运行 `complete-task` 记录 workflow candidate
  - 执行一次 `git commit` 形成提交闭环
