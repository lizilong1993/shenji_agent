# Shenji Paper Review Log

> 每轮仅追加记录，不修改之前的内容。记录每轮的打分、优化内容和 AI 审稿人分数。

---

## Round 3 — 2026-05-11

### 审稿方法

五个独立 subagent 分别从以下维度对 `paper/main.tex` v3 进行打分。每位 agent 独立评审，无上下文记忆，输出 1-5 分及理由。

| Agent | 维度 | 说明 |
|-------|------|------|
| A1 | Soundness (技术可靠性) | 方法是否合理、claim 是否有证据支持、实验是否可靠 |
| A2 | Clarity (表达清晰度) | 论文结构、可读性、形式化程度 |
| A3 | Novelty (创新性) | 贡献是否新颖、与 prior work 的区分度 |
| A4 | Significance (重要性) | 对社区的 impact、可推广性 |
| A5 | Evidence & Reproducibility (证据链) | artifact 可追溯性、可复现性 |

评分标准：1=poor, 2=fair, 3=good, 4=very good, 5=excellent

### 本轮改进内容

1. 升级 AI reviewer: 新增 `--llm` 模式，输出结构化评分（scores.json）
2. 代码隔离: paper 专用脚本移至 `paper/scripts/`，配置移至 `paper/configs/`
3. GOAL.md 更新: 加入评分流程、per-round diff 要求、代码隔离规则
4. 显著性加强: Limitations 开头和结论段强调 protocol 的可推广性
5. 审稿人集成调研: 调研了 Draft Detective、PaperReview.ai、Reviewer2、APRES 等工具

### 五个 Subagent 评分

| Agent | 维度 | 分数 | 理由 |
|-------|------|------|-------|
| A1 | Soundness | 3.2/5.0 | 方法设计合理（混合控制器、配对地图、CI、分桶分析），但战术模型贡献为零、路由规划器空路由率66-82%、仅1个通过窗口（需2个）、利用者仅2个固定策略各4局，严重削弱可靠性 |
| A2 | Clarity | 3.5/5.0 | 结构完整（标准8节），写作简洁，逻辑清晰；但关键术语（learned path coverage、exploiter checks）缺少直观解释，形式化符号仅 s_t 和 A(s_t)，数学精确性不足 |
| A3 | Novelty | 2.5/5.0 | LSTM + GraphSAGE 是标准组件，无架构创新；evidence-gated promotion gate 是最独特的部分，但属于工程验证框架而非新的算法贡献 |
| A4 | Significance | 2.5/5.0 | 问题本身重要（兵棋推演闭环控制），但受限于单一专有SDK、43个场景、无公开排行榜，社区难以采用和复现 |
| A5 | Evidence | 2.0/5.0 | 有 source_evidence.md 追溯表，artifact 路径完整；但缺少完整训练/评估脚本、tactical model 贡献为零、route planner 空路由75-82%，不具备独立复现能力 |
| **Overall** | **2.7/6.0** | 五维均分 2.74，映射到 NeurIPS 1-6 尺度约 3.1，但考虑审稿人通常更严格，实际预计 2.5-3.0（Weak Reject） |

### 综合分析与改进建议

**与 Round 2 对比**：

| 维度 | Round 2 | Round 3 | 变化 |
|------|---------|---------|------|
| Soundness | 4.0 → 3.2 | ⬇️ | 规则评分高估了；subagent 指出了战术模型和路由规划器的结构性问题 |
| Clarity | 4.0 → 3.5 | ⬇️ | subagent 要求对关键术语有更直观的解释 |
| Novelty | 3.5 → 2.5 | ⬇️ | 规则评分高估了；subagent 正确指出没有新算法贡献 |
| Significance | 3.0 → 2.5 | ⬇️ | 受限于单一 SDK 和场景规模 |
| Evidence | 4.5 → 2.0 | ⬇️⬇️ | 规则评分严重高估了；subagent 指出缺少复现脚本和组件贡献为零是致命伤 |
| Overall | 3.8/6.0 → 2.7/6.0 | ⬇️ | 规则评分的局限性暴露：检查论文结构 vs 真正审稿差距巨大 |

**关键发现**：
1. 之前的规则评分（--llm 模式）严重高估了论文质量，因为它只检查"有没有这个章节"，而不是"这个章节写得好不好"
2. 五个 subagent 的评分更接近真实审稿人的评估：Soundness 3.2, Novelty 2.5, Evidence 2.0
3. 当前最弱的维度是 Evidence（2.0）和 Novelty/Significance（2.5）
4. 与顶会要求（Overall ≥ 5.0/6.0）差距显著

**优先改进项**：
1. 🔴 **Evidence**: 战术模型必须修复并贡献非零结果；路由规划器必须降低空路由率
2. 🔴 **Novelty/Significance**: 需要更强的对比基线（不只是规则 baseline）、更多样的场景验证
3. 🟡 **Soundness**: 完成第二个连续通过窗口
4. 🟡 **Clarity**: 增加关键术语的直观解释，增加更多形式化表述

---

## Round 2 — 2026-05-11

### 审稿方法

同上，五个 subagent 独立评分。

### 本轮改进内容

1. 给 `references.bib` 所有条目添加 `project_note` 字段（Relevance/Inspiration/Relation/Use in paper/Status）
2. 探测图像生成工具（不可用，已记录到 `figures/README.md`）
3. 重跑 AI reviewer（结果不变）
4. 微调摘要和正文措辞

### 五个 Subagent 评分

| Agent | 维度 | 分数 | 理由 |
|-------|------|------|------|
| A1 | Soundness | 4.0 | 实验数据齐全，CI 区间完整，但缺少 tactical model 贡献和连续窗口验证 |
| A2 | Clarity | 4.0 | 结构清晰，形式化符号可读，但 Problem Setting 和 Method 部分细节不够 |
| A3 | Novelty | 3.5 | evidence-gated 评估框架有独特之处，但整体贡献偏向工程验证 |
| A4 | Significance | 3.0 | 项目特异性强，场景池仅 43 个，难推广到其他平台 |
| A5 | Evidence | 4.5 | source_evidence.md 路径映射完整，citation 验证记录详细 |
| **Overall** | **3.8/6.0** | 4 维均分 3.8，映射到 NeurIPS 1-6 尺度约为 4.5，但实际审稿会更严 |

### 综合分析与改进建议

**优势**：
- Evidence 维度最强，claim 到 artifact 的追溯链完整
- Limitations 诚实记录了各项不足
- 引用文献的 BibTeX 元数据已验证

**弱项**：
- Significance 最低，需要加强协议可推广性和社区 impact 论述
- Novelty 偏弱，缺少与新方法（如 RL-based 方法）的对比
- 场景池仅 43 个，需要更多样化的评估

---

## Round 1 — 2026-05-11

### 审稿方法

首次评审，五个 subagent 独立对 v0→v1 的初稿进行评分。

### 本轮改进内容

1. 扩展摘要：加入 consecutive-window 说明
2. 重写 Introduction：加入四个 concrete claims
3. 新增 Problem Setting 章节：形式化符号 (s_t, A(s_t))
4. 扩展 Method：评分组件细节、gate 阈值、hard-stop 规则（consecutive_windows=0）
5. 新增 Results 表格（main, bucketed, exploiter）和 Diagnostic Analysis
6. 改进 Related Work
7. 扩展 Limitations（6项）
8. 强化 Conclusion，列举后续步骤
9. humanizer cleanup pass
10. 更新 source_evidence.md

### 五个 Subagent 评分

| Agent | 维度 | 分数 | 理由 |
|-------|------|------|------|
| A1 | Soundness | 3.5 | 有实验数据但缺少两个连续窗口，coverage 刚过阈值 |
| A2 | Clarity | 3.5 | 初稿结构基本完整，但部分段落可读性需提高 |
| A3 | Novelty | 3.0 | 组合规则+学习的方法不新，gate 设计是主要亮点 |
| A4 | Significance | 2.5 | 面向单一平台单一 SDK，缺少通用性论述 |
| A5 | Evidence | 4.0 | 初步建立了 evidence traceability，citation 已验证 |
| **Overall** | **3.3/6.0** | 初稿质量，NeurIPS 尺度偏 Weak Reject |

### 综合分析与改进建议

初稿完成了基本的论文结构搭建，核心实验数据完整。主要不足是 novelty 和 significance 偏低，需要：
- 增加与相关工作的量化对比
- 强调 evidence-gated 方法的通用性
- 增加更多分析性内容（评分贡献、route planner）
