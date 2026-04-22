# 未来 2-4 周科研优化路线

## 总体判断

未来 2-4 周不建议把主要精力投向更大模型、更多模块或重型基础设施。当前最优路线应沿现有 `规则 agent + supervised surrogate + self-play` 方向继续推进，但先把科研闭环修正到“可解释、可验证、可比较”。

## P0

### P0.1 修正监督任务语义与离线评估可信度

- 研究目标：
  - 让 `intent / tactical / situation` 三条 supervised 线的指标真正可解释。
- 科学假设：
  - 当前主瓶颈不是模型容量，而是任务定义与标签/评估语义错误；一旦修正，现有路线的好坏才有资格被判断。
- 现在为什么值得优先做：
  - `tactical` 当前极可能在学错标签；
  - bucket 评测当前基本失真；
  - 不先修正这些问题，后续所有模型比较都不可信。
- 涉及模块：
  - `training/data.py`
  - `training/maps.py`
  - `training/supervised.py`
  - `training/cache/map_catalog.json`
  - 数据目录下的说明文件与样本元数据
- 最小可执行改动：
  - tactical 任务显式指定 `样本标签` 为标签列；
  - 为 intent/state graph 增加显式 map/scenario 映射，不再靠文件名子串猜 bucket；
  - 在 supervised summary 中同时输出“标签来源”和“bucket 样本分布”。
- 成功标准：
  - tactical 标签来源与数据说明一致；
  - intent/state graph 的 bucket 分布不再退化为单桶或伪单桶；
  - `metrics.json` / `slice_report.json` 能反映真实样本分布。
- 失败信号：
  - 修正后依然无法稳定定位标签或地图归属；
  - 修正后发现当前数据根本不足以支撑 bucket 级结论。
- 验证方法：
  - 重跑一轮小规模 supervised baseline；
  - 对比修正前后 `metrics.json`、`slice_report.json`、bucket 计数。
- 风险与前置条件：
  - 需要先确认数据说明中的字段语义是否覆盖所有样本；
  - 可能需要为文件名补充显式 scenario/map 索引表。

### P0.2 建立“baseline vs learned”最小对抗证据链

- 研究目标：
  - 用最小成本回答“当前 learned policy 到底有没有提升对抗表现”。
- 科学假设：
  - 如果现有 intent/tactical 真有价值，那么在与纯规则 baseline 的 paired 对战中，至少应出现非退化的 paired score 差异和动作选择差异。
- 现在为什么值得优先做：
  - latest league 运行是同一产物目录对打，不能回答任何科研问题；
  - 这是从 surrogate 指标走向最终能力的第一条必经证据链。
- 涉及模块：
  - `training/selfplay.py`
  - `training/evaluate.py`
  - `ai/agent.py`
  - `ai/policy.py`
  - `configs/selfplay_baseline.yaml`
- 最小可执行改动：
  - 让 candidate/incumbent 显式指向不同产物；
  - 固定三个评测对象：`baseline`、`tactical+intent challenger`、`ablation`
  - 落盘每场比赛使用的 artifact 路径、policy profile、动作差异摘要。
- 成功标准：
  - baseline-vs-baseline 近似 0.5，作为 sanity check；
  - learned-vs-baseline 不再是退化的全 0.5；
  - 产物能清楚说明“谁打谁”。
- 失败信号：
  - 即使 candidate/incumbent 分离后，paired score 仍长期等于 0.5；
  - learned 与 baseline 的动作选择几乎无差异。
- 验证方法：
  - 运行 baseline-vs-baseline、learned-vs-baseline、ablation-vs-baseline 三组 64 paired matches；
  - 读取 `rating.json`、`promotion.json`、`exploitability.json` 和动作差异日志。
- 风险与前置条件：
  - 需要确保 baseline profile 真正关闭 learned overlay；
  - 需要给实验产物补足 provenance 字段，避免再次出现同目录对打。

## P1

### P1.1 把模型接入从“局部排序器”扩展到“可观测的决策子链”

- 研究目标：
  - 明确 learned policy 在 agent 中究竟影响了哪些决策，并让影响范围足够大到能改变对抗结果。
- 科学假设：
  - 当前瓶颈更偏向“模型接入与决策脱耦”；不是模型完全没学到，而是进入主链的带宽太小。
- 现在为什么值得优先做：
  - 当前只有 `Shoot` / `Move` 用了 learned policy；
  - 绝大多数动作仍由规则/随机逻辑主导。
- 涉及模块：
  - `ai/agent.py`
  - `ai/policy.py`
  - 可能新增轻量日志/诊断模块
- 最小可执行改动：
  - 先增加动作来源标记：`rule` / `tactical` / `intent-assisted`
  - 优先扩展到与最终得分最相关的几个高频子决策，而不是全面重写 agent
  - 增加“动作差异率”“关键动作覆盖率”的统计输出。
- 成功标准：
  - learned profile 与 baseline profile 在关键动作上的差异率显著提高；
  - 动作差异能在 paired score 上体现方向一致的变化。
- 失败信号：
  - 接入扩展后动作差异增加，但胜率仍无改进；
  - 说明 surrogate task 与最终目标不对齐，需要回到任务定义层重做。
- 验证方法：
  - 做 `tactical-only`、`tactical+intent`、`expanded-policy` 三组 ablation；
  - 对比动作差异率、paired score、complex bucket 表现。
- 风险与前置条件：
  - 需要先完成 P0.2 的 baseline 证据链；
  - 扩展接入时必须避免把规则安全约束一起破坏。

### P1.2 修复 league 晋升判据，让它真正服务科研判断

- 研究目标：
  - 把 league 从“有输出文件”升级为“有科研约束的晋升门”。
- 科学假设：
  - 只有当 exploitability、bucket regression、paired score 共同受控时，晋升结果才有科研解释力。
- 现在为什么值得优先做：
  - 当前 gate 配置与代码不一致；
  - exploitability 虽已计算，却未进入正式 acceptance。
- 涉及模块：
  - `training/evaluate.py`
  - `training/selfplay.py`
  - `scripts/report_progress.py`
  - `configs/selfplay_baseline.yaml`
- 最小可执行改动：
  - 在 `promotion_decision()` 中接入 `max_exploiter_win_rate`、`max_bucket_regression`、`elo_delta_lower_min`
  - 区分“评测摘要事件”和“正式晋升接受条件”
  - 在 `promotion.json` 中输出每个 gate 的通过/失败详情。
- 成功标准：
  - gate 代码与配置一一对应；
  - `promotion.json` 能解释为什么被拒绝或接受。
- 失败信号：
  - gate 逻辑复杂化后仍无法区分明显好坏模型；
  - 说明上游评测信号不足。
- 验证方法：
  - 用 baseline / learned / exploiter 三组历史或新跑结果回放 decision；
  - 校验 `promotion.json` 的失败项与配置一致。
- 风险与前置条件：
  - 需要先保证 self-play 比较的是不同产物；
  - 需要补 bucket regression 的统计口径。

## P2

### P2.1 重新定义 `situation` 线，只在它能影响决策时继续投入

- 研究目标：
  - 判断 `situation` 线是否应该继续保留，以及它应该以什么形式进入 agent。
- 科学假设：
  - 当前 `situation` 不是“模型太弱”，而是“任务定义与接入点都不清楚”；先定义用途，再决定是否继续训练。
- 现在为什么值得优先做：
  - 当前 `situation_model.pt` 不在主链；
  - 继续单纯提高 0/1 图分类准确率，科研价值有限。
- 涉及模块：
  - `training/data.py`
  - `training/supervised.py`
  - `ai/policy.py`
  - `ai/agent.py`
- 最小可执行改动：
  - 先提出两个清晰用途并做对照：
    - 作为“目标点/交战区域风险分”
    - 作为“是否推进/固守”的上层 gating signal
  - 若两种用途都无收益，暂停该线。
- 成功标准：
  - situation 信号被明确接入到一个可解释决策点；
  - 对 complex maps 或 exploitability 至少有一项稳定改善。
- 失败信号：
  - 接入后无收益甚至降低对抗能力；
  - 说明该任务当前不是优先主线。
- 验证方法：
  - 做 `no-situation` vs `situation-risk-gating` vs `situation-city-ranking` 对照实验。
- 风险与前置条件：
  - 必须先完成 P0、P1，否则看不清 situation 的真实贡献。

### P2.2 仅在闭环可信后，再考虑更强模型或多任务训练

- 研究目标：
  - 在已有科研闭环可信的前提下，评估是否值得做更强 backbone、多任务训练或更复杂表征学习。
- 科学假设：
  - 一旦标签、评估、接入三件事打通，模型升级才可能带来真实收益。
- 现在为什么排到 P2：
  - 当前还没有证据表明模型容量是主瓶颈；
  - 提前上更重模型会放大不可信评估的问题。
- 涉及模块：
  - `ai/models.py`
  - `training/supervised.py`
  - 可能新增实验配置
- 最小可执行改动：
  - 在闭环可信后，再比较更强 intent backbone 或多任务共享编码器；
  - 不做与当前主线脱钩的重型系统改造。
- 成功标准：
  - 升级模型后对抗指标优于已修正 baseline；
  - 同时不放大 exploitability 或 bucket regression。
- 失败信号：
  - 离线指标提升但 paired score 不动；
  - 说明仍然不是模型层问题。
- 验证方法：
  - 与 P1 最佳方案做同 protocol 对比。
- 风险与前置条件：
  - 必须先通过 P0/P1 的验证门。

## 推荐执行顺序

1. 先做 `P0.1`，修正 tactical 标签与 bucket 评测。
2. 紧接着做 `P0.2`，建立 baseline-vs-learned 对抗证据链。
3. 如果 learned overlay 仍几乎不改变结果，再做 `P1.1` 扩展接入宽度。
4. 并行或随后做 `P1.2`，让晋升门真正反映科研约束。
5. `P2` 只在前四步完成后再启动。

## 当前最值得优先推进的 1-2 件事

- 第一优先：修正 `tactical` 标签与 bucket 评测，让离线指标重新可解释。
- 第二优先：做 baseline-vs-learned-vs-ablation 的 paired self-play，对“supervised 是否提升最终对抗能力”给出第一版实证回答。
