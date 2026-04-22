# Task Plan

## 本轮执行状态

### 已完成

1. 主线收敛
   - 研究目标：默认在线主线只保留 `LSTM + GCN`
   - 科学假设：当前第一瓶颈不是更大模型，而是主线分裂与模型接入不一致
   - 现在为什么值得优先做：不先收敛主线，后续任何评估都不可信
   - 涉及模块：`ai/agent.py`、`ai/policy.py`、`training/supervised.py`、`configs/supervised_baseline.yaml`
   - 最小可执行改动：移除旧直载，默认只发布 `intent_lstm.pt + situation_gcn.pt`
   - 成功标准：默认配置下 tactical 不进入在线主链
   - 失败信号：仍存在 `intent_model/tactical_model/situation_model` 混合在线加载
   - 验证方法：本地语法检查 + manifest 回读
   - 风险与前置条件：真实训练产物仍需远程重跑验证

2. 数据契约修正
   - 研究目标：让 tactical 标签与 bucket 映射可解释
   - 科学假设：错误标签与错误 bucket 会掩盖真实算法效果
   - 现在为什么值得优先做：这是当前最确定的错误源
   - 涉及模块：`training/data.py`、`configs/supervised_baseline.yaml`
   - 最小可执行改动：显式 `label_col=样本标签`；intent/situation 使用显式 map id strategy
   - 成功标准：训练代码不再依赖“最后一列标签”和模糊 substring bucket 推断
   - 失败信号：远程重跑后 bucket 仍退化到单桶
   - 验证方法：轻量单测 + 远程重跑 `metrics.json/slice_report.json`
   - 风险与前置条件：数据文件名策略仍依赖远程真实样本确认

3. 评估闭环修正
   - 研究目标：让 self-play 至少能证明“candidate 与 incumbent 不同，模型是否真实参与决策”
   - 科学假设：当前评估退化主要来自 protocol 错误而不是模型本体
   - 现在为什么值得优先做：不先修正 protocol，paired score 没有解释力
   - 涉及模块：`training/selfplay.py`、`training/evaluate.py`、`configs/selfplay_baseline.yaml`
   - 最小可执行改动：同路径保护、profile 显式区分、coverage 统计
   - 成功标准：`rating.json` 可读到 `learned_path_coverage`
   - 失败信号：remote 运行后仍是全 0.5 且 coverage 近零
   - 验证方法：本地轻量单测 + 远程 `selfplay_minimal.yaml`
   - 风险与前置条件：远程必须有可用 torch 环境和完整数据

## 接下来 2-4 周优先级

### P0

- P0.1 远程重跑 `supervised_baseline`
  - 研究目标：验证新 manifest 与 bucket/label 契约是否真实生效
  - 科学假设：修正后 `metrics.json` 和 `slice_report.json` 会比旧产物更可解释
  - 现在为什么值得优先做：这是所有后续实验的基线
  - 涉及模块：`training/supervised.py`、`configs/supervised_baseline.yaml`
  - 最小可执行改动：无新增代码，只做远程训练与产物回读
  - 成功标准：产物目录包含 `intent_lstm.pt`、`situation_gcn.pt`，manifest 只指向这两者
  - 失败信号：训练产物仍沿用旧命名或 tactical 混入默认发布
  - 验证方法：检查 `experiments/shenji_agent/<timestamp>/metrics.json` 和 `ai/weights/model_manifest.json`
  - 风险与前置条件：远程环境需安装 `torch`

- P0.2 远程跑 `selfplay_minimal`
  - 研究目标：得到第一版 baseline-vs-learned 实证
  - 科学假设：如果接入有效，`learned_path_coverage > 0` 且 paired score 不应退化
  - 现在为什么值得优先做：这是从 surrogate 指标走向最终对抗能力的第一步
  - 涉及模块：`training/selfplay.py`、`configs/selfplay_minimal.yaml`
  - 最小可执行改动：远程使用 `ai/weights_incumbent_rule` 与 `ai/weights`
  - 成功标准：`rating.json` 中 `learned_path_coverage > 0` 且 paired score 非退化
  - 失败信号：coverage 接近 0 或 candidate/incumbent 又混为同一路径
  - 验证方法：读取 `rating.json`、`promotion.json`、`participants.json`
  - 风险与前置条件：必须先做远程基线快照

### P1

- P1.1 tactical 消融重跑
  - 研究目标：确认修正标签后的 tactical 是否仍值得保留
  - 科学假设：如果修正标签后 tactical 仍无增益，它应该继续留在消融线而不是回主线
  - 现在为什么值得优先做：避免把无效任务重新混回主线
  - 涉及模块：`configs/supervised_tactical_ablation.yaml`
  - 最小可执行改动：独立训练 + 与 baseline 对照
  - 成功标准：对抗或局部决策上出现明确增益
  - 失败信号：指标改善但 paired score 无提升
  - 验证方法：`supervised_tactical_ablation` + 同 protocol self-play
  - 风险与前置条件：P0 先完成

- P1.2 扩展 learned 接入宽度
  - 研究目标：确认当前瓶颈是否转移为“接入太窄”
  - 科学假设：若 `Move/Shoot` 两条链有用，扩展到更多关键动作可能继续带来收益
  - 现在为什么值得优先做：只有在 P0 证明主线有效后，这一步才有价值
  - 涉及模块：`ai/agent.py`、`ai/policy.py`
  - 最小可执行改动：优先扩到高频关键动作，而不是重写整套 agent
  - 成功标准：coverage 增加且 paired score 继续上升
  - 失败信号：coverage 增加但 paired score 不动
  - 验证方法：新增 ablation profile
  - 风险与前置条件：P0 必须先完成

## 远程操作清单

### 1. 环境准备

```bash
cd /remote-home/lizilong/shenji_agent
if [ ! -d /remote-home/lizilong/venvs/shenji ]; then
  python3 -m venv /remote-home/lizilong/venvs/shenji
fi
source /remote-home/lizilong/venvs/shenji/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2. 同步代码与数据

- 代码：`ai/ training/ configs/ scripts/ tests/ train_models.py run_evaluation.py README.md .agent/`
- 数据：`datasets/` 与 `land_wargame_sdk/Data/Data/`

### 3. 远程基线快照

```bash
cd /remote-home/lizilong/shenji_agent
rm -rf ai/weights_incumbent_rule
cp -r ai/weights ai/weights_incumbent_rule
```

### 4. 远程静态烟雾检查

```bash
source /remote-home/lizilong/venvs/shenji/bin/activate
cd /remote-home/lizilong/shenji_agent
python -m compileall ai training scripts tests
python train_models.py --config configs/supervised_baseline.yaml --help
python -m training.selfplay --config configs/selfplay_baseline.yaml --help
```

### 5. 监督训练

```bash
source /remote-home/lizilong/venvs/shenji/bin/activate
cd /remote-home/lizilong/shenji_agent
python train_models.py --config configs/supervised_baseline.yaml
```

重点检查：

- `experiments/shenji_agent/<timestamp>/metrics.json`
- `experiments/shenji_agent/<timestamp>/slice_report.json`
- `experiments/shenji_agent/<timestamp>/summary.md`
- `ai/weights/model_manifest.json`

### 6. 最小对抗验证

```bash
source /remote-home/lizilong/venvs/shenji/bin/activate
cd /remote-home/lizilong/shenji_agent
python -m training.selfplay --config configs/selfplay_minimal.yaml --candidate ai/weights --incumbent ai/weights_incumbent_rule
```

重点检查：

- `experiments/shenji_agent/league_<timestamp>/rating.json`
- `experiments/shenji_agent/league_<timestamp>/promotion.json`
- `experiments/shenji_agent/league_<timestamp>/participants.json`
- `experiments/shenji_agent/league_<timestamp>/exploitability.json`

### 7. tactical 消融

```bash
source /remote-home/lizilong/venvs/shenji/bin/activate
cd /remote-home/lizilong/shenji_agent
python train_models.py --config configs/supervised_tactical_ablation.yaml
```

### 8. 结果回传

至少回传：

- 最新 `metrics.json`
- 最新 `slice_report.json`
- 最新 `rating.json`
- 最新 `promotion.json`
- 最新 `summary.md`
