# 研究路线优化实验设计矩阵

## 实验设计原则

- 先修正证据链，再追求更高模型复杂度。
- 所有实验优先回答“是否提升最终对抗能力”，而不是只回答“离线准确率是否更高”。
- 默认沿现有 `规则主干 + supervised surrogate + self-play` 路线优化，不重起一套新体系。

## Baseline 定义

### B0 纯规则基线

- 定义：
  - `ai/agent.py` 使用 `baseline` profile，不加载 learned policy 影响决策。
- 用途：
  - 作为最终 paired self-play 的零点参照。

### B1 当前 learned overlay 基线

- 定义：
  - 使用当前 `ai/weights/` 中 manifest 指向的 `intent_model.pt + tactical_model.pt`
  - `policy_profile=challenger`
- 用途：
  - 衡量“现有已落地路线”相对于纯规则 baseline 的真实增益。

### B2 当前 latest league 退化基线

- 定义：
  - `candidate == incumbent == ai/weights`
- 用途：
  - 只作为 sanity check，验证“同模型对打应接近 0.5”；不作为科研结论。

## 低成本高信息量实验

| ID | 优先级 | 实验 | 验证假设 | 关键指标 | 通过判据 | 不通过判据 |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | P0 | `baseline vs baseline` | 评测协议稳定、无系统性偏置 | overall paired score, CI, reward pattern | paired score 约等于 0.5，reward 无异常漂移 | 明显偏离 0.5，说明评测协议本身不稳 |
| E1 | P0 | `tactical wrong-label vs correct-label` | 当前 tactical 结果受标签语义错误污染 | offline acc, 校验集 confusion, paired score | 修正标签后离线指标更可解释，且至少不弱于错误标签版本的 paired score | 修正标签后完全无效，说明这条任务可能本身不对 |
| E2 | P0 | `old bucket mapping vs explicit mapping` | 当前 bucket 报告失真 | bucket sample counts, slice_report, complex/medium/simple 指标 | 三个 bucket 的样本分布合理，slice_report 不再退化 | 仍无法形成有效 bucket 划分 |
| E3 | P0 | `learned overlay vs pure baseline` | 现有 intent+tactical 至少能改变局部对抗结果 | paired score, CI, action divergence | overall point estimate > 0.52 且动作差异明显 | 仍接近 0.5 或动作差异极小 |

## 后续主线实验

| ID | 优先级 | 实验 | 验证假设 | 关键指标 | 通过判据 | 不通过判据 |
| --- | --- | --- | --- | --- | --- | --- |
| E4 | P1 | `tactical-only vs tactical+intent` | intent threat penalty 对目标点选择有增益 | complex bucket paired score, city-choice divergence, exploitability | `tactical+intent` 在 complex bucket 明显优于 `tactical-only` | 二者无差异，说明 intent 当前接入点太弱或 surrogate 不对齐 |
| E5 | P1 | `current overlay vs widened decision-chain overlay` | 当前瓶颈是模型接入宽度不足 | paired score, key-action coverage, exploitability | widened 版本在不增加 exploitability 的前提下提升 paired score | 接入更广后仍无收益，说明主要问题不在接入宽度 |
| E6 | P1 | `league gate old vs repaired gate` | exploitability/regression/elo 应进入正式晋升门 | promotion.json gate 明细, accepted/rejected consistency | 每次拒绝/接受都能被具体 gate 解释 | gate 仍然无法区分明显好坏模型 |
| E7 | P2 | `no-situation vs situation-risk-gating` | situation 信号只有进主链才有价值 | complex bucket paired score, exploitability, action shift | situation 版本至少改善一项核心指标且不显著伤害其他指标 | 接入后无收益或副作用明显，则暂停 situation 线 |

## 每个实验要读哪些指标

### 离线监督指标

- `intent`
  - `val_immediate_accuracy`
  - `h1-h5` 准确率
  - bucket 样本分布与 bucket accuracy
- `tactical`
  - 正确标签下的准确率
  - 正负样本分布
  - 若条件允许，补充 PR/AUC 或 top-k ranking hit
- `situation`
  - 准确率
  - 各 bucket 样本分布
  - 与最终对抗结果的相关性

### 对抗评测指标

- `rating.json`
  - overall point estimate
  - overall CI95 lower
  - complex bucket point estimate / CI95 lower
  - count
- `promotion.json`
  - accepted
  - failures
  - gate 细项
- `exploitability.json`
  - 每个 exploiter 对 candidate 的 win rate
- 新增建议指标
  - action divergence ratio
  - city-choice divergence
  - shoot-target divergence
  - artifact provenance

## 通过 / 不通过的统一判据

### 可进入下一阶段

- 离线指标语义明确，且 bucket 划分可信。
- `baseline vs baseline` 稳定在 0.5 附近。
- `learned vs baseline` 出现非退化 paired score 差异。
- 任何晋升判断都能说明 candidate/incumbent 分别是谁，以及为什么过/不过。

### 不应继续往更重模型推进

- tactical 标签仍不确定。
- bucket 映射仍靠不住。
- learned overlay 进入 agent 后几乎不改变动作。
- self-play 依旧无法区分不同产物。

## 推荐实验顺序

1. `E0 baseline vs baseline`
2. `E1 tactical wrong-label vs correct-label`
3. `E2 old bucket mapping vs explicit mapping`
4. `E3 learned overlay vs pure baseline`
5. `E4 tactical-only vs tactical+intent`
6. `E5 current overlay vs widened decision-chain overlay`
7. `E6 repaired league gate replay`
8. `E7 situation-risk-gating`

## 哪些实验现在就做，哪些后做

### 现在就做

- `E0`
- `E1`
- `E2`
- `E3`

原因：

- 成本低；
- 信息量最大；
- 直接决定当前路线是否值得继续；
- 能最快回答“现在该优化什么，而不是优化别的”。

### 放到后续再做

- `E4`
- `E5`
- `E6`
- `E7`

原因：

- 它们依赖前面四个实验先把标签、评测、baseline 定义清楚；
- 如果前四个实验都过不了，后续实验的科研意义会大打折扣。

## 当前建议的 stop/go 规则

- 如果 `E1-E3` 之后，correct-label learned overlay 仍无法稳定优于纯规则 baseline：
  - 暂停继续扩模型；
  - 回到 surrogate task 定义，重做 tactical/situation 与最终目标的对齐。
- 如果 `E1-E3` 之后，learned overlay 已能稳定优于 baseline：
  - 优先做 `E4-E6`，把 intent 贡献和 league gate 完整化；
  - `situation` 只作为后续候选，不抢当前主线。
