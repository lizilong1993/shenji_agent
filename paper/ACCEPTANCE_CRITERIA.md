# Shenji Paper Acceptance Criteria

> 各顶会审稿标准参考与 Shenji 论文的目标分数。
> 最后更新: 2026-05-11

---

## 1. 顶会审稿维度对比

| 维度 | NeurIPS | ICML | ICLR | 我们的评分 |
|------|---------|------|------|-----------|
| Soundness / Quality | Quality (1-4) | Soundness (1-4) | Soundness (1-5) | 1-5 |
| Clarity / Presentation | Clarity (1-4) | Presentation (1-4) | Presentation (1-5) | 1-5 |
| Significance | Significance (1-4) | Significance (1-4) | Contribution (1-5) | 1-5 |
| Originality / Novelty | Originality (1-4) | Originality (1-4) | — | 1-5 |
| Evidence | 无单独维度 | 无单独维度 | 无单独维度 | 1-5 (我们自定义) |
| **Overall** | **1-6** | **1-6** | **1-10** | **1-6** |

### 各会议评分细则

**NeurIPS** (源自 OpenReview 数据):
- 4 维度: Quality, Clarity, Significance, Originality (各 1-4)
- Overall: 1-6 (1=Strong Reject, 6=Strong Accept)
- 置信度: 1-5
- 典型接收论文 Overall ≥ 5 (Accept) 或 ≥ 4 (Weak Accept)
- 接收率约 25-30%

**ICML** (源自官方审稿表格):
- 4 维度: Soundness, Presentation, Significance, Originality (各 1-4)
- Overall: 1-6 (6=Strong Accept, 5=Accept, 4=Weak Accept, 3=Weak Reject, 2=Reject, 1=Strong Reject)
- "Accept" 标签从 5 开始，4 是 "Weak Accept — use sparingly"
- 置信度: 1-5
- 接收率约 25%

**ICLR** (源自 OpenReview):
- 3 维度: Soundness, Presentation, Contribution (各 1-5, OpenReview)
- Overall: 1-10
- 典型接收论文 Overall ≥ 6 (OpenReview 10 分制)
- 接收率约 27% (2026: 5355/13763)

---

## 2. Shenji 论文目标分数

| 维度 | 当前 (Round 3) | 🌙 目标 (NeurIPS 投稿) | ☀️ 目标 (ICLR 投稿) | 差距 |
|------|---------------|----------------------|-------------------|------|
| Soundness | **3.2/5.0** | ≥ 4.0/5.0 | ≥ 4.0/5.0 | -0.8 |
| Clarity | **3.5/5.0** | ≥ 4.0/5.0 | ≥ 4.5/5.0 | -0.5 |
| Novelty | **2.5/5.0** | ≥ 3.5/5.0 | ≥ 4.0/5.0 | -1.0 |
| Significance | **2.5/5.0** | ≥ 3.5/5.0 | ≥ 3.5/5.0 | -1.0 |
| Evidence | **2.0/5.0** | ≥ 4.5/5.0 | ≥ 4.5/5.0 | -2.5 |
| **Overall (1-6)** | **2.7/6.0** | **≥ 5.0/6.0 (Accept)** | **≥ 5.0/6.0** | **-2.3** |

### 各维度达到顶会标准的具体要求

#### Soundness ≥ 4.0/5.0
- [ ] 实验基于真实 SDK 后端 (已完成: mock_backend=false)
- [ ] 100 对配对地图 (已完成)
- [ ] 置信区间报告 (已完成: 含 CI95)
- [ ] 分桶分析 (已完成: simple/medium/complex)
- [ ] Exploiter 鲁棒性检查 (已完成: max 0.25)
- [ ] **待完成**: 两个连续通过窗口 (当前 consecutive_windows=0)
- [ ] **待完成**: 与更多 baseline 方法的对比
- [ ] **待完成**: tactical model 贡献修复

#### Clarity ≥ 4.0/5.0
- [ ] 论文结构完整 (已完成: 8 节)
- [ ] 形式化符号 (已完成: s_t, A(s_t))
- [ ] 图表 (进行中: LaTeX 编译正常, 需要 publication-quality 图片)
- [ ] 表格 (已完成: 4 张表格)
- [ ] **待完成**: 图表质量提升
- [ ] **待完成**: Problem Setting 和方法部分细节扩写

#### Novelty ≥ 3.5/5.0
- [ ] evidence-gated 评估协议 (已完成)
- [ ] fail-closed promotion gate (已完成)
- [ ] 评分组件分析 (已完成: intent/situation/objective 贡献)
- [ ] **待完成**: 与至少 2 个替代方法的量化对比
- [ ] **待完成**: 更强的 novelty framing ("第一篇在 Miaosuan 上做 fail-closed 评估的工作")

#### Significance ≥ 3.5/5.0
- [ ] 明确的应用场景 (已完成: 陆战兵棋推演)
- [ ] 协议可推广性论述 (已加强)
- [ ] **待完成**: 更多场景/地图/任务类型的验证
- [ ] **待完成**: 开源评估工具包发布

#### Evidence ≥ 4.5/5.0
- [ ] source_evidence.md 完整追溯 (已完成)
- [ ] 所有引用有 verified BibTeX (已完成)
- [ ] project_note 字段 (已完成)
- [ ] 可复现性说明 (已完成)
- [ ] **待完成**: 完整训练/评测脚本可访问

---

## 3. 与已有 AI 审稿工具的差距

| 审稿工具 | 特点 | 能否集成到本流程 |
|----------|------|----------------|
| **Draft Detective** | LangGraph 多 agent 审稿, 需 PostgreSQL + API key | 有条件集成 (需启动 Docker) |
| **PaperReview.ai** (Stanford) | 7 维度评分, 1-10 尺度, 线性回归校准 | 仅 Web 服务, 无 API |
| **Reviewer2** | Longformer 生成审稿意见 | 可本地运行 (需 Java + PyTorch), 但模型较旧 |
| **OpenReviewer** | 预测 accept/reject | 论文级别, 不适合迭代改进 |
| **APRES** | 发现 rubric 并自动修改论文 | 无公开代码 |

### 当前审稿流程的局限性

1. 当前的 `--llm` 评分基于确定性规则（检查论文结构信号），不是语义理解
2. 要实现真正的语义评分，最可行路径：
   - **使用现有的 Draft Detective**（已安装，需启动 Docker + PostgreSQL）
   - **或直接用 Claude 的 API 进行审稿**
3. 本文件中的 subagent 评分由人工独立 agent 完成，是当前最接近真实审稿的替代方案

---

## 4. 决策阈值

### 可否投稿的判断

| 条件 | NeurIPS | ICML | ICLR |
|------|---------|------|------|
| 最小 Overall | ≥ 4.0/6.0 | ≥ 4.0/6.0 | ≥ 4.0/6.0 |
| 推荐 Overall | ≥ 5.0/6.0 | ≥ 5.0/6.0 | ≥ 5.0/6.0 |
| 最低单维度 | ≥ 3.0/5.0 | ≥ 3.0/5.0 | ≥ 3.0/5.0 |
| 最低 Evidence | ≥ 4.0/5.0 | ≥ 4.0/5.0 | ≥ 4.0/5.0 |

### 可否停机的硬条件 (来自 program.md + GOAL.md)

- [ ] 两个连续通过窗口 (当前: ❌ 仅 1 个)
- [ ] Overall win-rate ≥ 0.60, CI95 lower > 0.55 (当前: ✅)
- [ ] Complex bucket 同样达标 (当前: ✅)
- [ ] Max exploiter win rate ≤ 0.45 (当前: ✅ 0.25)
- [ ] 各 coverage 达标 (当前: ✅)
- [ ] 无 unsupported claim (当前: ✅ 0 个)
- [ ] Overall 审稿分数 ≥ 4.0/6.0 (当前: 待评估)
