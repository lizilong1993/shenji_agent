# 智能体接口契约对照表（I03 步骤 4）

> 目的：把本仓实现的五方法契约与 `31006 5.3` / 附录 `A.3` 要求的接口**逐项对齐**，
> 差异写在这里，而不是留在代码注释里。

## 0. 一处必须先说明的限制（**不要跳过**）

**本仓没有 31006 的原文抄录件。** 本条契约的字段名来自任务清单
`outputs/兵棋推演9项团体标准对标任务清单_20261010.md` 的 §I03 条目与上级报告摘要，
**不是**从标准正文逐字核对得来。

因此下表的「标准项」列**只能确认"标准要求存在这五个方法"**，
**不能确认**每个方法的**参数名与取值域**是否与标准逐字一致。

- 若拿到 31006 原文，应把「标准签名」列替换为**逐字抄录**，并重新核对每个参数；
- 在此之前，**不要**在任何符合性声明里写「签名与 31006 逐项一致」。
- 这与 `I05` 遇到的是同一类问题（该条的前置也要求先有原文抄录）。

---

## 1. 五方法对照表（`31006 5.3` / 附录 `A.3`）

| # | 标准项 | 本仓实现 | 转发/占位 | 现有签名（实测） | 差异与说明 |
| --- | --- | --- | --- | --- | --- |
| 1 | `setup` | `AgentAPI.setup` → `Agent.setup` | **转发** | `setup(setup_info: Dict[str, Any]) -> Any` | 参数名 `setup_info` 取自本仓既有实现；**是否与标准同名未核** |
| 2 | `reset` | `AgentAPI.reset` → `Agent.reset` | **转发** | `reset() -> Any` | 无参，无差异 |
| 3 | `step` | `AgentAPI.step` → `Agent.step` | **转发** | `step(observation: Dict[str, Any]) -> List[Dict[str, Any]]` | 返回**平台动作 dict 列表**；列表元素字段与 `ai/action_safety.py` 的 `ACTION_SPECS` 同源 |
| 4 | `load_llm_model` | `AgentAPI.load_llm_model` → `LLMAdapter.load_llm_model` | **占位** | `load_llm_model(*args, **kwargs) -> Dict[str, Any]` | 标准未给签名细节；本仓返回**结构化状态**而非抛异常（见 §3） |
| 5 | `parse_natural_language_command` | `AgentAPI.parse_natural_language_command` → `LLMAdapter.parse_natural_language_command` | **占位（规则版）** | `parse_natural_language_command(text: str, *args, **kwargs) -> Dict[str, Any]` | 见 §4 的返回结构与降级语义 |

**旁证（`31007` 附录 C 智能体必备函数）**：`__init__` / `save` / `load` ——
本适配层一并提供，`save` / `load` 同样**只转发**。

### 1.1 为什么 `load_llm_model` / `parse_natural_language_command` 用 `*args, **kwargs`

标准侧只说了方法要存在，**未规定参数名**。若自行固定成 `load_llm_model(model_path)`，
将来标准或平台给出不同签名时，调用方会 `TypeError`。
故适配层**吸收任意参数**再转给 `LLMAdapter`（后者有明确签名），
把「签名不确定」这件事收敛在一处，而不是扩散到契约表面。

---

## 2. 转发方法的**行为契约**

| 性质 | 约定 | 由哪条测试守护 |
| --- | --- | --- |
| **只转发** | `setup`/`reset`/`step` 的方法体只有一行 `return`，不实现任何决策 | `test_setup_reset_step_forward_to_wrapped_agent` |
| **不改写返回** | `step` 的结果**原样透出**；`None` → `[]`，**不伪造成功** | `test_step_returns_agent_result_unchanged`、`test_step_never_invents_actions_when_agent_returns_none` |
| **不导入主链路** | 适配层**不得**导入 `ai.policy`（会形成第二个决策入口） | `test_adapter_does_not_import_policy_main_chain` |
| **可整体禁用** | `enabled=False` 时**不触碰**被包装的智能体 | `test_disabled_adapter_does_not_touch_agent` |
| **调用可审计** | `forwarded` 计数暴露每个方法被真实调用的次数 | `test_forwarded_counters_expose_use` |

> 最后一条不是形式主义：若没有转发计数，就无法区分
> 「适配层把活干了」与「适配层只是转发了」——而这两者在策略漂移风险上完全不同。

---

## 3. `load_llm_model()` 的状态契约（判据 3）

**任何情况下都不抛异常**，只返回结构化状态。返回字段固定为五个：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `status` | `str` | `disabled` / `unavailable` / `loaded` |
| `loaded` | `bool` | 是否真的可做模型推理 |
| `model_path` | `str \| None` | 传入或配置的路径 |
| `reason` | `str` | **可读原因**（必须非空） |
| `capabilities` | `dict` | `{natural_language, model_inference}` |

**调用方无需 `try/except` 即可判断** —— 这是判据 3 的字面要求。

### 3.1 状态取值与触发条件

| 状态 | 触发 | 说明 |
| --- | --- | --- |
| `disabled` | `enabled=False`（**默认**） | Q6 未关闭，不允许悄悄启用真实模型 |
| `unavailable` | 未给 `model_path` / 路径不存在 / **占位实现不加载权重** | 已启用但拿不到可用模型 |
| `loaded` | 真实权重加载成功 | **当前实现永不返回**（占位） |

> **不得谎报 `loaded=True`**：文件存在时本占位实现仍返回 `unavailable`，
> 因为"文件在"不等于"模型可推理"。有测试断言这一点
> （`test_load_llm_model_does_not_claim_loaded_without_inference`）。

### 3.2 为什么默认 `enabled=False`

`I03` 前置条件写明：「Q6 未关闭前按报告默认处理，**只做接口与占位实现，不引入真实 LLM 依赖**」。
默认关掉是**遵守该前置**，不是保守。

---

## 4. `parse_natural_language_command()` 的返回契约

返回字段（`I03` 步骤 3 点名要求）：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `intent` | `str \| None` | 解析出的意图；未命中为 `None` |
| `target` | `Any \| None` | 目标对象 |
| `params` | `dict` | 动作参数 |
| `confidence` | `float` | `0.0` ~ `1.0` |
| `degraded` | `bool` | **是否降级为规则解析**（占位实现恒为 `True`） |
| `action` | `dict \| None` | **可直接放入动作列表的动作**，与 `step` 接受的格式一致 |
| `method` | `str` | 解析方法（`rule`） |
| `text` | `str` | 原文（便于追溯） |
| `reason` | `str` | 可读原因 |

### 4.1 与 `step` 的格式一致性（硬要求）

`action` 的键**只能是平台动作字段**：`actor` / `type` / `obj_id` / `target_obj_id`。
字段名与 `ai/action_safety.py::ACTION_SPECS` **同源**（该字典取自
`docs/wargame_ref/actions_reference.txt`，平台自有文档）。

有测试断言 `set(action) ⊆ {actor, type, obj_id, target_obj_id}`
—— **自造字段会被拒**，因为它们无法被 `step` 消费。

### 4.2 降级与"不编造"原则

| 情形 | 行为 |
| --- | --- |
| 命中意图且信息齐 | 返回 `action`，`confidence=0.5`，`degraded=True` |
| 未命中任何意图 | `intent=None`、`action=None`、**`confidence=0.0`** |
| 攻击/夺控**缺目标** | `action=None`、`confidence=0.0`，并说明缺什么 |
| 机动**缺路径** | `action=None`、`confidence=0.0` |
| `text` 为 `None` | 按未命中处理，**不抛异常** |

**置信度不随文本长度或关键词个数虚增**：规则命中 ≠ 理解，故给固定 `0.5`；
未命中给 `0.0`。**宁可低报，不可高报** —— 调用方会据它决定是否信任解析结果。

---

## 5. 未做的部分（如实列出）

| 项 | 状态 |
| --- | --- |
| 与 `31006` 原文**逐字**核对签名 | **未做**（无原文抄录件，见 §0） |
| 真实 LLM 权重加载与推理 | **未做**（Q6 未关闭；接口已就绪，替换 `llm_adapter` 实现即可） |
| `parse_natural_language_command` 的**模型**解析 | **未做**（现为规则占位，`degraded` 恒 `True` 就是如实标注） |
| 在 `TrainEnv` 侧**实盘**联调 | **未做**（需闭源引擎，本环境鉴权不通过） |

**故本契约表可作为"接口形状已定、降级语义明确"的依据，
但不可作为"LLM 能力已交付"的依据。**
