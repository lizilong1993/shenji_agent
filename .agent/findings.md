# Findings

## 当前真实主线

当前仓库的已实现主线已经从“规则主干 + `intent+tactical` 局部挂接 + 离线 `situation` 未接入”的分裂状态，收敛为：

- `ai/agent.py` 统一通过 `LearnedPolicy` 接入 learned 路径；
- `ai/policy.py` 的在线主线是 `intent_lstm + situation_gcn`；
- `training/supervised.py` 的默认发布主线是 `intent_lstm.pt + situation_gcn.pt`；
- `tactical` 退为独立消融线，不再进入默认在线主链；
- `training/selfplay.py` 默认要求 `candidate` 与 `incumbent` 是不同产物目录，并输出 learned-path coverage 证据。

## 已落地 / 部分落地 / 仅规划

### 已落地

- 删除了 `ai/agent.py` 中旧的 `IntentionLSTM` / `SituationGCN` 直载逻辑。
- `LearnedPolicy` 现在加载 `ai/weights/model_manifest.json` 中的 `intent_lstm` 与 `situation_gcn`。
- `Move` / `Shoot` 现在由 `LSTM intent bias + GCN situation score` 联合排序。
- `training/data.py` 对 tactical 标签改为显式读取 `样本标签`。
- `training/data.py` 对 intent/state graph 的 map bucket 改为显式策略：
  - intent: `suffix`
  - situation/state graph: `prefix`
- `configs/supervised_baseline.yaml` 默认关闭 `tactical.enabled`。
- 新增 `configs/supervised_tactical_ablation.yaml`，只作为 tactical 兼容/消融入口。
- `training/selfplay.py` 新增同路径保护和 `learned_path_coverage / move_path_coverage / shoot_path_coverage` 落盘。
- 新增 `configs/selfplay_minimal.yaml` 作为远程最小对抗验证配置。
- `scripts/report_progress.py` 修复为空目录/缺失目录返回 `event=none`，不再直接 `SystemExit`。

### 部分落地

- 默认 `ai/weights/` manifest 已切到 `intent_lstm.pt + situation_gcn.pt`，但旧权重文件仍保留在目录中作兼容。
- promotion gate 现在已有 learned coverage 证据，但更严格的 `bucket regression / exploiter gate / elo gate` 仍未完全进入正式 acceptance 逻辑。
- `run_evaluation.py` 已改成显式 profile 触发，但它仍是调试脚本，不是正式结论入口。

### 仅规划

- 更重的 backbone、多任务共享编码器、扩大决策接入范围、修复更完整的 league gate，仍属于后续优化路线，不是本次实现范围。

## 代码 / 配置 / 运行产物的一致性

### 现在已经一致的部分

- 代码主线：`ai/policy.py` 与 `training/supervised.py` 都以 `intent_lstm + situation_gcn` 为默认在线主线。
- 默认配置：`configs/supervised_baseline.yaml` 不再把 `tactical` 作为默认发布主线。
- 评估协议：`configs/selfplay_baseline.yaml` 和 `training/selfplay.py` 都明确区分 `candidate_profile=challenger` 与 `incumbent_profile=baseline`。

### 仍需远程实训验证的部分

- 当前仓库历史实验目录 `experiments/shenji_agent/*` 仍是旧产物命名，不代表新主线已经跑通。
- 本机没有可用 `torch` 运行时，无法在本地验证真实模型加载、监督训练和对抗运行，只完成了语法与轻量 CLI/脚本验证。

## 当前最关键科研问题

### 1. surrogate 任务的正确性

- tactical 现在改成显式 `样本标签`，但修正后是否提升最终对抗能力，仍未验证。
- intent/state graph 的 map bucket 现在有显式策略，但需要远程重跑确认样本分布是否合理。

### 2. 模型是否真正进入决策闭环

- 代码层面已经接入：
  - `Shoot`: 目标血量反事实更新后的 GCN 态势分 + intent 威胁偏置
  - `Move`: 候选目标点反事实更新后的 GCN 态势分 + intent 风险偏置
- 但是否在真实对抗中形成有效优势，仍要靠远程 `selfplay_minimal.yaml` 验证。

### 3. 评估闭环是否能支撑科研结论

- 同路径 self-play 退化问题已修复。
- 现在可以回答“模型有没有真的参与动作选择”，因为 `rating.json` 将包含 coverage 证据。
- 但更强的晋升门仍需后续补全。

## 关键证据来源

- `ai/agent.py`
- `ai/policy.py`
- `training/data.py`
- `training/supervised.py`
- `training/selfplay.py`
- `training/evaluate.py`
- `configs/supervised_baseline.yaml`
- `configs/supervised_tactical_ablation.yaml`
- `configs/selfplay_baseline.yaml`
- `configs/selfplay_minimal.yaml`
- `scripts/report_progress.py`
- `ai/weights/model_manifest.json`
- `tests/test_mainline_contract.py`

## 本机已验证 / 未验证

### 已验证

- 语法通过：`python -m compileall ai training scripts tests`
- 轻量测试通过：
  - tactical 标签列契约
  - self-play 同路径保护
  - progress 空目录返回 `none`
  - coverage 汇总逻辑

### 未验证

- 未验证：本机真实 `torch` 模型加载
- 未验证：监督训练能否顺利产出新 manifest
- 未验证：`LSTM+GCN` 对 `weights_incumbent_rule` 的 paired score 是否为正
- 未验证：remote 数据/环境是否与本地预期一致
