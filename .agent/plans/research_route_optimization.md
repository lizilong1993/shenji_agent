# Research Route Optimization

## Baseline 定义

### B0 纯规则 baseline

- profile：`baseline`
- artifacts：`ai/weights_incumbent_rule`
- 作用：作为 `LSTM+GCN` 的零点参照

### B1 当前主线 baseline

- profile：`challenger`
- artifacts：`ai/weights`
- contract：`intent_lstm.pt + situation_gcn.pt`
- 作用：验证默认主线是否已经真实影响对抗结果

### B2 tactical 消融

- 训练配置：`configs/supervised_tactical_ablation.yaml`
- 在线地位：不进入默认主线
- 作用：只回答“修正标签后的 tactical 还有没有价值”

## 优先实验矩阵

| ID | 成本 | 实验 | 验证假设 | 关键指标 | 通过判据 | 不通过判据 |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | 低 | `baseline vs baseline` | 协议稳定，无系统偏置 | paired score, CI | score 接近 0.5 | 明显偏离 0.5 |
| E1 | 低 | `supervised_baseline` 远程重跑 | 新 contract 与 bucket/label 修正生效 | `metrics.json`, `slice_report.json`, manifest | 只发布 `intent_lstm + situation_gcn` | tactical 混入默认发布 |
| E2 | 低 | `challenger vs baseline` | `LSTM+GCN` 已真实进入决策链 | `paired_score`, `learned_path_coverage`, `participants.json` | coverage > 0 且 paired score 非退化 | coverage≈0 或再度自对弈 |
| E3 | 中 | `tactical_ablation vs baseline` | 修正标签后的 tactical 是否仍有独立增益 | paired score, move/shoot divergence | tactical 版本带来稳定增益 | paired score 不动或变差 |
| E4 | 中 | 扩大 learned 接入宽度 | 当前瓶颈是否转为“接入太窄” | coverage, paired score | coverage 与 score 同向提升 | coverage 升高但 score 不动 |

## 每个实验读哪些指标

### 监督训练

- `metrics.json`
  - `intent.val_immediate_accuracy`
  - `intent.bucket_counts`
  - `situation.bucket_counts`
  - `tactical.schema.label_column`（仅 tactical 消融）
- `slice_report.json`
  - `intent_bucket_accuracy`
  - `intent_bucket_counts`
  - `situation_bucket_accuracy`
  - `situation_bucket_counts`
- `ai/weights/model_manifest.json`
  - 是否只保留 `intent_lstm` 与 `situation_gcn`

### 对抗评测

- `rating.json`
  - `overall.point_estimate`
  - `overall.ci95`
  - `policy.learned_path_coverage`
  - `policy.move_path_coverage`
  - `policy.shoot_path_coverage`
- `promotion.json`
  - `accepted`
  - `failures`
  - `learned_path_coverage`
- `participants.json`
  - candidate/incumbent 是否真的不同
- `exploitability.json`
  - exploiter 对 candidate 的压制程度

## 通过 / 不通过统一判据

### 通过

- `participants.json` 证明不是自对弈
- `learned_path_coverage > 0`
- `Move` 或 `Shoot` 至少一类 coverage 明显非零
- paired score 不再退化为全 0.5

### 不通过

- manifest 仍混入旧主线
- tactical 又被混回默认在线加载
- candidate/incumbent 仍是同一路径
- learned coverage 接近 0，说明模型没有进入真实决策

## 先做哪些 ablation

1. `B0 vs B0`
2. `B1 vs B0`
3. `B2 vs B0`

原因：

- 信息量最大
- 成本最低
- 能最快判断当前最值得继续投资源的是主线接入、tactical 消融，还是重新定义任务

## 哪些实验后做

- 扩大 learned 接入宽度
- 更严格的 promotion gate
- situation 更复杂用途
- 更重 backbone

原因：

- 只有在 `B1 vs B0` 已经证明主线有效时，后面这些优化才有意义
