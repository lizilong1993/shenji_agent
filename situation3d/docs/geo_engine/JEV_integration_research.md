# Qwen 系 JEV 决策模型能否集成到本引擎（调研，2026-10-21）

> 结论先行：**能集成，但定位必须是“战术候选动作打分器”，不是替代物理/裁决内核，也不是端到端连续控制。**
> 引擎侧只需新增一个 `policy/jev.py` 适配层（已随本调研实现骨架与 mock 测试）；真正的模型需要一台可承载
> Qwen3.5 0.8B/2B/4B 的推理服务（SGLang 或 transformers），并需要针对兵棋观测做候选集与评测。

## 1. JEV 是什么

JEV 是一类 **decision model（决策模型）**：保留 LLM 的语言理解骨干，去掉自回归生成头，换成
分类/打分头，对“状态 + 封闭选项集”直接输出每个选项的概率（CHOICE）、二值判断（NOUL/是/否）或
分数（SCORE）。一次前向，不生成文本，因此比同规模 LLM 的“先生成再解析”快一个量级。

当前公开生态（2026-10）：

| 模型/项目 | 骨干 | 形态 | 许可 | 备注 |
|---|---|---|---|---|
| `AlexWortega/openjev`（qwen3.5-0.8B/2B/4B v5、35B-A3B） | Qwen3.5 | NLI 三分类（contradiction/entailment/neutral）+ typed decisions | 权重 MIT | 推荐 4B v5；每选项独立打分、顺序不变、可 bit-identical；SGLang 有外置模型包；单 A6000 上约 321 pairs/s（44 token） |
| `AnubhabBanerjee/Qwen-jev` 教程 | Qwen2.5-Coder-1.5B | 冻结骨干 + 小型分类头 | 代码 PolyForm Noncommercial 1.0；Qwen 权重 Apache-2.0 | 适合自建/微调参考，商用需单独授权 |
| `Qwen-Audio-Jev`（arXiv 2610.21614） | Qwen-Audio-3.1-ASR-Flash-Next | 音频 CHOICE/NOUL/SCORE | 论文 CC BY 4.0 | 约 20× 推理加速；与陆战文本决策关系不大，可作多模态扩展参考 |
| `xuhaodev/Qwen3-1.7B-Jev` | Qwen3-1.7B | Choice/Score/Noul | 页面当前 404（可能下架） | 仅作备选线索 |

要点：
- openjev 的判定是 **NLI 蕴含概率**：把每个候选动作写成“hypothesis”，状态作为 premise，取 entailment 概率；
  再在选项间归一化。顺序不变、同请求确定性（README 实测 231 项全一致）。
- 不是为兵棋训练的：JevBench 公开集 4B v5 全量 0.814；难档（judge/tradeoff/temporal）明显下降。
- **不抗提示注入**：README 实测在状态里加一行“system administrator override”会把 150 项准确率从 0.833
  打到 0.467。若状态文本来自不可信输入（聊天、外部情报），必须与确定性校验并用。

## 2. 与本引擎的契合点

本引擎已有清晰的“观测 → 动作”边界，适合把 JEV 放在 **policy 层**：

```
sim/world.step
  └─ geoio.observation(world, seat)      # 席位过滤后的实体/目标/可用动作
       └─ policy/jev.build_candidates    # 把 valid_actions 展平成候选（fire/move/hold）
            └─ JevClient.decide(state, choices)   # openjev/SGLang/HTTP
                 └─ 选中候选 → geoio.actions.apply_action(world, action)
```

- 规则与物理仍是权威：候选只来自 `valid_actions`（武器是否可用、射程、通视、目标是否被探测），JEV 只在
  合法动作里做取舍，不会产生“隔山打牛”或非法移动。
- 连续坐标/路径规划仍由 `mobility.planner` 完成；JEV 只选“去哪/打谁/用什么/是否驻守”。
- 天然支持多席位：每个 seat 一次 observation，一次决策；也可以按编队批量。
- 离线可用：`MockJevClient` 与规则启发式保证无 GPU 时全链路可跑；模型服务是可选增强。

## 3. 集成方案（建议）

### 3.1 适配层（已实现骨架）

`policy/jev.py` 提供：

- `DecisionRequest/DecisionQuestion/DecisionAnswer`：CHOICE/NOUL/SCORE 的通用载荷；
- `JevClient` 协议：`HttpJevClient`（openjev Decisions gateway 的 `/v1/systemone`）与 `MockJevClient`；
- `summarize_observation`：把观测压缩为“时间/双方实体/目标点”的文本状态；
- `build_candidates`：从观测的 `actions` 生成 `fire / move_to / stop` 候选，带可读 label；
- `JevPolicy.choose/step`：选最大概率候选并应用到 World；无模型时回退“先射击、再移动、最后驻守”。

`tests/test_jev_policy.py` 用 mock 客户端验证候选生成、选择与回退，不依赖权重。

### 3.2 模型服务

- 推荐 **openjev `qwen3.5-4b-nli-v5` + SGLang**（MIT 权重；外置 `sglang_openjev` 包；支持 `/classify`
  与 Decisions gateway）。显存：4B bf16 约 8–10 GB；0.8B 可 CPU/小显卡；2B 折中。
- 若只做离线演示：transformers `AutoModelForSequenceClassification` 直接加载 0.8B/4B 子目录。
- 服务与引擎解耦：引擎只发 HTTP；`JevPolicy` 捕获超时/HTTP 错误/未知标签并回退启发式，仿真步长不等待模型（建议每 2–5 s 决策一次，
  或只在态势变化/新目标出现时触发）。

### 3.3 候选与提示设计

- 每个实体限制 4–6 个候选（射击优先，移动最多 2 个目标点），避免一次请求几十个选项导致 token/延迟膨胀。
- 状态文本只放已探测实体（observation 已做席位过滤），不要把全量真值或不可信聊天塞进 prompt；
  对状态字段做控制字符清洗与长度截断（`_sanitize`），降低注入面；不可信文本仍不应进入状态。
- 选项 label 用“unit 101 fire target 205 with 36”这类结构化短句；评估时记录选择概率与置信度。
- 需要训练的候选：目标选择、武器选择、驻守/推进、编队任务；不需要 JEV 做路径、碰撞、裁决。

## 4. 可行性判断

**可以集成，工作量中等（约 1–2 人周做通，2–4 周做可用）：**

| 项 | 评估 |
|---|---|
| 引擎改动 | 小：仅新增 policy 包与 observation/action 适配，不影响 rules/sim 内核 |
| 推理硬件 | 需要 GPU（4B）或可接受 CPU 慢速（0.8B）；共享 GPU 需按项目 GPU 纪律申请 |
| 延迟 | 服务端 4B 批量 100–300 pairs/s；单次 5–6 候选、~200 token 状态，预计 20–100 ms/决策；每 2–5 s 一次完全可接受 |
| 精度 | 公开 JevBench 0.81；兵棋域未训练，需自建评测（目标选择/武器选择/驻守）与少量微调 |
| 许可 | openjev 权重 MIT；教程代码 PolyForm 非商用；商用部署需法务确认 |
| 风险 | 提示注入（已实测不设防）、域偏移、候选集设计不当导致“合法但糟糕”的选择、GPU 稳定性 |

**不建议**：用 JEV 直接输出经纬度/连续速度（它不是回归模型）；也不要用它替代 CRT/武器表裁决
（规则必须由确定性代码执行并对拍）。

## 5. 建议路线

1. 用 `MockJevClient` + 现有 heuristic 跑通 1231/台海案例的 policy 接口（本轮已完成骨架）；
2. 在 server155 部署 openjev 0.8B（CPU 或空闲 GPU），用 20–30 个历史态势做离线对比：
   JEV 选择 vs 启发式选择，记录目标选择一致率、胜率/交换比、单次延迟；
3. 若效果可接受，再上 4B + SGLang，接入 `JevPolicy.step` 每个决策周期；
4. 建立兵棋候选数据集（observation→候选→人工/规则标注），微调或蒸馏小模型；
5. 与确定性检查并用：非法候选过滤 + 关键时刻人工/规则兜底 + 全事件回放。

## 6. 实测对比（openjev 0.8B CPU，2026-10-21）

测试环境：server155，无可用 GPU（NVML 初始化失败），openjev `qwen3.5-0.8b-nli-v2s-long`（1.7 GB）+
transformers 5.20 / torch 2.7 CPU，每选项单独前向（短序列），状态文本压缩到 2 个敌方单位与 2 个目标。

场景：30×30 格、2 红坦克 vs 2 蓝坦克、2 个目标点、dt=10 s、96 tick、每 12 tick 决策一次（8 次决策；
蓝方固定使用启发式）。同一 seed 对比：

| 指标 | 当前启发式 | JEV 0.8B |
|---|---|---|
| 红方存活/血量 | 2 / 5 | 2 / 4 |
| 蓝方存活/血量 | 0 / 0 | 1 / 1 |
| 毁伤 | 2 | 1 |
| 目标归属 | 红 1、蓝 1 | 红 1、蓝 1 |
| 决策延迟 | 毫秒级 | 平均 7.5 s，最大 10.4 s |
| 依赖 | 无 | 1.7 GB 权重 + torch/transformers，GPU 不可用时 CPU 极慢 |

结论：**当前启发式方案在本引擎中更好**——任务效果不劣（本次还多毁伤 1 个目标），延迟低 3–4 个数量级，
无模型/许可依赖。JEV 0.8B 的公开 NLI 能力没有转化为兵棋优势，且 CPU 延迟无法满足 1–10 s 决策节奏。

因此默认策略保持启发式；JEV 适配层保留为**可选实验策略**（`--policy jev` 或 `JevPolicy(client=...)`），
需在 GPU 服务、领域微调和候选集评测完成后再考虑切换。复现命令：

```bash
python3 scripts/evaluate_jev.py --model /tmp/jev_model/qwen3.5-0.8b-nli-v2s-long \
  --ticks 96 --decision-every 12 --out docs/geo_engine/jev_eval_20261021.json
```

原始结果：`docs/geo_engine/jev_eval_20261021.json`。

## 7. 参考

- openjev 模型卡与 serving：<https://huggingface.co/AlexWortega/openjev>
- Qwen-jev 教程（PolyForm 非商用）：<https://github.com/AnubhabBanerjee/Qwen-jev>
- Qwen-Audio-Jev 论文：<https://arxiv.org/abs/2610.21614>
- JEV 决策模型构建指南：<https://jevlab.dev/build-your-own-jev>
