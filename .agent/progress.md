# Progress

## 本次任务摘要

- 任务类型：实现 `LSTM + GCN` 主线收敛、修正主线错误、补远程运行清单
- 本次范围：代码实现 + 本地轻量验证 + `.agent/*` 落盘
- 未做内容：远程训练、远程评测、大规模运行

## 规则与上下文状态

- 项目根 `AGENTS.md`：本地未发现
- `.agent/context.md`：本地未发现
- bootstrap：已执行
  - 命令：`C:\Users\lizilong\.codex\automations\weekly-skill-trigger-optimization\codex-runtime-ops.ps1 -Action bootstrap-task -BootstrapChain using-superpowers`
- 当前任务启动状态：`bootstrapped`

## 本次实际扫描与修改

### 读取过的关键文件

- `ai/agent.py`
- `ai/policy.py`
- `ai/models.py`
- `ai/features.py`
- `training/common.py`
- `training/data.py`
- `training/evaluate.py`
- `training/selfplay.py`
- `training/supervised.py`
- `scripts/report_progress.py`
- `configs/supervised_baseline.yaml`
- `configs/selfplay_baseline.yaml`
- `ai/weights/model_manifest.json`
- `.agent/findings.md`
- `.agent/task_plan.md`
- `.agent/plans/research_route_optimization.md`

### 本次实际改动文件

- `ai/agent.py`
- `ai/features.py`
- `ai/policy.py`
- `training/common.py`
- `training/data.py`
- `training/evaluate.py`
- `training/selfplay.py`
- `training/supervised.py`
- `configs/supervised_baseline.yaml`
- `configs/supervised_tactical_ablation.yaml`
- `configs/selfplay_baseline.yaml`
- `configs/selfplay_minimal.yaml`
- `run_evaluation.py`
- `scripts/report_progress.py`
- `ai/weights/model_manifest.json`
- `tests/__init__.py`
- `tests/test_mainline_contract.py`
- `.agent/findings.md`
- `.agent/task_plan.md`
- `.agent/progress.md`
- `.agent/plans/research_route_optimization.md`

## 关键实现结果

- 已移除 `ai/agent.py` 中旧的直载模型逻辑。
- 已把 `ai/policy.py` 改为：
  - 加载 `intent_lstm` 与 `situation_gcn`
  - 在 `Move` / `Shoot` 上使用 `intent bias + GCN score`
  - 输出 coverage / trace 统计
- 已把默认训练主线改成 `intent + situation`
- 已把 tactical 改成独立消融线
- 已修正 self-play：
  - 默认拒绝 `candidate == incumbent`
  - 输出 `participants.json`
  - 输出 learned-path coverage
- 已修正 progress 脚本在空目录时报错

## Fresh verification evidence

### 已完成

1. 语法验证
   - 命令：
     - `C:\Users\lizilong\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall ai training scripts tests`
   - 结果：通过

2. 轻量回归测试
   - 命令：
     - `C:\Users\lizilong\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.test_mainline_contract -v`
   - 结果：4 项测试全部通过
   - 覆盖：
     - tactical 显式标签列
     - self-play 同路径保护
     - coverage 汇总逻辑
     - `report_progress.py` 空目录返回 `none`

3. 权重契约检查
   - 结果：
     - `ai/weights/model_manifest.json` 已切到 `intent_lstm` / `situation_gcn`
     - 兼容旧文件仍保留在目录中，但默认 contract 已更新
   - 注意：
     - `ai/weights/*.pt` 与 `ai/weights/model_manifest.json` 被 `.gitignore` 排除，不会进入本次 git 提交
     - 远程同步时必须显式带上整个 `ai/weights/` 目录

### 未完成

- 未完成：本机真实 `torch` 训练 / 推理验证
- 原因：工作区自带 Python 缺少 `torch`，而系统 Python 初始化失败
- 影响：本机无法证明新主线在真实模型加载下已经跑通，只能完成语法和轻量行为验证

## 当前阻塞点

### 阻塞 1

- 本机无可用 `torch` 运行时
- 影响：无法本地完成训练与 self-play 真跑
- 最直接下一步：按 `.agent/task_plan.md` 中远程清单在 `/remote-home/lizilong/shenji_agent` 执行

### 阻塞 2

- 当前工作树存在若干与本任务无关的已删除文件
- 影响：收尾提交时不能无差别 `git add -A`
- 处理原则：只提交本次相关文件，不动这些外部改动

## 本次是否只做分析、未做实现

- 否
- 本次已完成真实代码实现与本地轻量验证

## 收尾动作

- `.agent/*` 已更新
- 远程操作清单已写入 `.agent/task_plan.md`
- 已完成：
  - `complete-task`
  - workflow candidate 已记录到 `C:\Users\lizilong\.codex\.agent\workflow_candidates.jsonl`
  - fingerprint: `shenji-lstm-gcn-mainline`
- Git closeout：
  - 将只 stage 本次相关文件
  - `ai/weights/*` 继续通过文件同步进入远程，不进入本次提交
