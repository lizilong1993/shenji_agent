# 智能体接口契约对照表（I03 步骤 4）

> 目的：把本仓实现的五方法契约与 `31006 5.3` / 附录 `A.3` 要求的接口**逐项对齐**，
> 差异写在这里，而不是留在代码注释里。

## 0. 标准原文是**有的** —— 我先前的说法是错的（已订正）

> **第五十一轮订正。** 本文首版 §0 写「本仓没有 31006 的原文抄录件」，
> 并据此把「与标准逐字核对签名」列为**做不到**。
> **那是错的**：原文就在工作区里 ——
> `.scratch/std/s31006.txt`（**56,588 B**，2026-10-09 提取），
> 含 `5.3` 与附录 `A.3.1`–`A.3.5` 的规范签名与参数表（表 A.20–A.23）。
> 任务清单 §0 也点名 `.scratch/std/*` 为原文来源。
>
> **教训**：把"我没找到"写成"不存在"，会把一个**可完成项**降级成"做不到"，
> 并让后续所有判断都建立在这个错误前提上。**独立评审 B 直接推翻了这个前提。**

### 0.1 附录 A.3 的**逐字**签名（原文行号）

| 条目 | 原文行 | 形式 |
| --- | --- | --- |
| A.3.1 | `s31006.txt:820` | `BaseAgent::setup(setup_info: Dict) -> None` |
| A.3.2 | `:838` | `BaseAgent::step(observation: Dict) -> List[Dict]` |
| A.3.3 | `:864` | `BaseAgent::reset() -> Boolean` |
| A.3.4 | `:877` | `BaseAgent::load_llm_model(model_path: str, config: Dict) -> Boolean` |
| A.3.5 | `:906` | `BaseAgent::parse_natural_language_command(command: str) -> List[Dict]` |

输出结构（表 A.23，`:923`–`:932`）：`actionList: List[Dict]`，每个 dict 含
`unit_id`(string) / `action_type`(string) / `target`(Dict) / `actionP`(Dict)。

`load_llm_model` 的 `config` 键见表 A.20：`inference_precision` / `max_context_length` /
`device` / `batch_size`。

`reset` 的输出参数名是 `resetStatus`（`:789`、`:872`）。

## 1. 五方法对照表（`31006 5.3` / 附录 `A.3`）

| # | 标准项 | 本仓实现 | 转发/占位 | 现有签名（实测） | 差异与说明 |
| --- | --- | --- | --- | --- | --- |
| 1 | `setup` | `AgentAPI.setup` → `Agent.setup` | **转发** | `setup(setup_info: Dict[str, Any]) -> Any` | 参数名 `setup_info` 取自本仓既有实现；**是否与标准同名未核** |
| 2 | `reset` | `AgentAPI.reset` → `Agent.reset` | **转发** | `reset() -> Any` | 无参，无差异 |
| 3 | `step` | `AgentAPI.step` → `Agent.step` | **转发** | `step(observation: Dict[str, Any]) -> List[Dict[str, Any]]` | 返回**平台动作 dict 列表**；列表元素字段与 `ai/action_safety.py` 的 `ACTION_SPECS` 同源 |
| 4 | `load_llm_model` | `AgentAPI.load_llm_model` → `LLMAdapter.load_llm_model` | **占位** | `load_llm_model(model_path: Any = None, config: Any = None) -> bool` | 标准未给签名细节；本仓返回**结构化状态**而非抛异常（见 §3） |
| 5 | `parse_natural_language_command` | `AgentAPI.parse_natural_language_command` → `LLMAdapter.parse_natural_language_command` | **占位（规则版）** | `parse_natural_language_command(command: Any = None, *, unit_id=None, target_obj_id=None, weapon_id=None, move_path=None, platform_shape=True) -> List[Dict[str, Any]]` | 见 §4 的返回结构与降级语义 |

### 1.2 A.3 符合性（**第五十四轮已达成**）

第五十一轮曾在此记录"五项里三项不符"。**本轮已按原文实现**，现逐项复核：

| 方法 | 标准要求（原文行） | 本仓实现 | 结论 |
| --- | --- | --- | --- |
| `setup` | `(setup_info: Dict) -> None`（`:820`） | 转发 `Agent.setup`，返回 `None` | **符合** |
| `step` | `(observation: Dict) -> List[Dict]`（`:838`） | 同 | **符合** |
| `reset` | `() -> Boolean`（`:864`） | 返回 `bool`（禁用 → `False`；未抛异常 → `True`） | **符合** |
| `load_llm_model` | `(model_path: str, config: Dict) -> Boolean`（`:877`） | 参数即 `(model_path, config)`；返回 `bool` | **符合** |
| `parse_natural_language_command` | `(command: str) -> List[Dict]`（`:906`），元素含 `unit_id`/`action_type`/`target`/`actionP`（表 A.23 `:923`） | 返回 `List[Dict]`，元素含四标准键（类型也对：`str`/`str`/`Dict`/`Dict`） | **符合** |

**判据不再是"看着像"，而是可执行的断言**：
`tests/test_i03_agent_contract.py::test_appendix_a3_signatures_are_now_implemented`
逐项检查返回类型与四键类型。

### 1.2b 两处刻意的设计（**不是偏离**）

1. **`parse_natural_language_command` 允许额外的关键字参数**
   （`unit_id` / `target_obj_id` / `weapon_id` / `move_path`）。标准只规定 `command`；
   但"从自然语言里抽出动作参数"不在占位实现的能力范围内（Q6 未关闭）。
   故：**标准的调用方式 `f(command)` 完全可用**，同时保留把参数给全的能力 ——
   否则「打击」永远产不出合法动作（`ACTION_SPECS[2]` 要求 `weapon_id`）。
   **额外参数是可选关键字，不影响标准兼容性。**

2. **平台形状与标准形状并存，可用 `platform_shape=False` 取纯标准形状**。
   默认 `True` 时元素**同时**带平台动作字段（可直接喂 `step`）；
   `False` 时只给四标准键。**两种形状下标准四键取值完全一致**（有测试断言）。
   这样既满足标准侧，也不牺牲本仓主链路的可用性 —— 且**不互相冒充**。

3. **`load_llm_model` 只返回布尔**，细节移到 `AgentAPI.last_llm_status`
   （含按表 A.20 校验后的 `config` 四键与缺失项）。**字段不减，只换出口。**

**旁证（`31007` 附录 C 智能体必备函数）**：`__init__` / `save` / `load` ——
本适配层一并提供，`save` / `load` 同样**只转发**。

### 1.1 `*args, **kwargs` 的**真实**理由（原先的理由是错的）

首版这里写「标准未规定参数名，故用 `*args, **kwargs` 吸收」。
**那句是失实的** —— 标准**规定了**参数名与类型（见 §0.1）。

现在的理由只剩一条（仍然成立）：适配层转发给 `Agent`/`LLMAdapter` 时
**不固定自己的参数名**，使将来按标准形状改造时**不必改调用方**。
但**这不改变下面 §1.2 的结论**：当前签名**不符合** A.3。

---

## 2. 转发方法的**行为契约**

| 性质 | 约定 | 由哪条测试守护 |
| --- | --- | --- |
| **只转发** | `setup`/`reset`/`step` **不实现任何决策逻辑** | `test_setup_reset_step_forward_to_wrapped_agent` |
| **不改写返回** | `step` 的结果**原样透出**；仅 `None` → `[]`（`Agent.step` 的既有约定）；**其余非 list 不抹平** | `test_step_returns_agent_result_unchanged`、`test_step_passes_through_non_list_instead_of_disguising_it` |
| **无反向渗入** | **主链路**（`ai/policy.py`、`training/`）**不得**引用 `ai.adapters` | `test_main_chain_does_not_import_the_adapter` |
| **导入无副作用** | 只 `import ai.adapters` **不得构造任何策略对象**（第二个决策入口的可检测形态） | `test_importing_adapter_builds_no_policy_object` |
| **可整体禁用** | `enabled=False` 时**不触碰**被包装的智能体 | `test_disabled_adapter_does_not_touch_agent` |
| **调用可审计** | `forwarded` 计数暴露每个方法被真实调用的次数 | `test_forwarded_counters_expose_use` |

> 转发计数不是形式主义：没有它就无法区分「适配层把活干了」与「只是转发了」——
> 这两者在策略漂移风险上完全不同。计数覆盖五方法**与附录 C 的 `save`/`load`**。

**关于"不导入主链路"的方向性更正（留档）**：首版写成"适配层不得导入 `ai.policy`"，
并用**扫源码字符串**检查 —— 那是**平凡可满足**的，而且**实质上做不到**：
转发必须拿到 `Agent`，而 `ai/agent.py` 自己就 `from .policy import LearnedPolicy`，
构造 Agent 必然构造一个策略对象。

**方向搞反了并不要紧**：转发本来就需要 `Agent`。真正要防的是**反向**——
主链路去导入适配层，那才多出一条决策入口。故守卫改为断言**方向**，
并加一条**行为**守卫（只导入适配层不得构造策略对象）。

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

### 3.1b `enabled=False` 的语义（同为禁用）

`AgentAPI(enabled=False)` 时 **LLM 两个方法也返回 `disabled`** ——
首版只在适配层拦住前三个方法，LLM 方法仍照常转发，"整层禁用"只禁了一半。

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

判据**不是"键名看起来对"，而是"能过本仓自己的校验器"**：

```
ai.action_safety.validate_action(action, mode="strict")["ok"] is True
```

字段要求**单一来源** = `ai.action_safety.ACTION_SPECS`（该字典取自平台自有文档
`docs/wargame_ref/actions_reference.txt`）。

> ⚠ **这里踩过一个坑，留档**：首版手写了一份"哪个意图需要哪些字段"的规则，
> 并断言 `set(action) ⊆ {actor, type, obj_id, target_obj_id}` ——
> **那条断言把 `weapon_id` 也禁止了**，而 `ACTION_SPECS[2]` 把 `weapon_id` 列为**必需**。
> 于是测试反过来**锁死了"产出非法动作"这个缺陷**：
> 「打击」产出的动作过不了本仓 strict 校验（`missing_required_field:weapon_id`）。
>
> 独立评审用**本仓自己的校验器**把它抓出来的。
> 现改为：字段要求一律查 `ACTION_SPECS`，判据一律用 `validate_action(strict)`，
> **不再手写第二份规则**（两份规则必然漂移）。

自造字段仍会被拒 —— 但拒它的理由不再是"不在白名单里"，而是"`step` 消费不了 / 校验器不认"。

### 4.2 降级与"不编造"原则

| 情形 | 行为 |
| --- | --- |
| 命中意图且**必需字段齐备** | 返回 `action`，`confidence=0.5`，`degraded=True` |
| 未命中任何意图 | `intent=None`、`action=None`、**`confidence=0.0`** |
| **含否定词**（不要/不得/禁止/取消/别/do not/cancel/ceasefire/abort…） | `intent=None`、`action=None`、`confidence=0.0` |
| 命中意图但**缺必需字段**（按 `ACTION_SPECS` 判定） | `action=None`、`confidence=0.0`，并在 `reason` 里列出缺哪些 |
| 无法取得 `ACTION_SPECS` | `action=None`、`confidence=0.0`（**保守拒绝**，不猜） |
| `text` 为 `None` 或非字符串 | 按未命中处理，**不抛异常** |

**"缺字段就不产出"是一条硬规则**：首版对"缺 `obj_id` 的打击"仍给 `confidence=0.5`，
产生了**"看起来能用、其实会被拒"的中间态**。调用方据置信度决定是否信任，
故**宁可 0.0 并说明原因**。

**置信度不随文本长度或关键词个数虚增**：规则命中 ≠ 理解，故给固定 `0.5`；
未命中给 `0.0`。**宁可低报，不可高报** —— 调用方会据它决定是否信任解析结果。

---

## 5. 未做的部分（如实列出）

| 项 | 状态 |
| --- | --- |
| 与 `31006` 原文**逐字**核对签名 | **已做**（§0.1 逐字摘录 + §1.2 差距表）；**三项不符** |
| 经闭源引擎的 TrainEnv 侧实盘联调 | **未做**（鉴权不通过） |
| 首版六类缺陷的整改 | **已完成**（见 §6） |
| **按 A.3 改造签名** | **已完成**（第五十四轮，见 §1.2） |
| 真实 LLM 权重加载与推理 | **未做**（Q6 未关闭；接口已就绪，替换 `llm_adapter` 实现即可） |
| `parse_natural_language_command` 的**模型**解析 | **未做**（现为规则占位，`degraded` 恒 `True` 就是如实标注） |
| 在 `TrainEnv` 侧**实盘**联调 | **未做**（需闭源引擎，本环境鉴权不通过） |

**故本契约表可作为"接口形状已定、降级语义明确"的依据，
但不可作为"LLM 能力已交付"的依据。**

---

## 6. 独立评审整改记录（第五十一轮）

评审 A 用**本仓自己的校验器与运行时**逐条实测，确认 11 项缺陷。以下是处置：

| # | 缺陷（评审实测） | 处置 |
| --- | --- | --- |
| 1 | 「打击」产出缺 `weapon_id`，**过不了本仓 strict 校验** | 字段要求改查 `ACTION_SPECS`；有测试断言产出必过 strict |
| 2 | 缺 `obj_id` 仍给 `confidence=0.5` | 缺必需字段 → `action=None`、`confidence=0.0` |
| 3 | **否定句被忽略**（「不要打击目标」→ 攻击指令） | 加否定词检测，命中即不产出动作（11 条参数化用例） |
| 4 | 机动**永远不可达**（恒真守卫 + 无法传路径） | 加 `move_path` 形参；机动可达且过校验 |
| 5 | 夺控**过约束**（type 5 只需 `obj_id`） | 改按 `ACTION_SPECS[5].required` 判定 |
| 6 | 「停止」→ type **333 结束部署**（不可逆） | 只认「停止机动」→ type 10 |
| 7 | `load_llm_model` 对非路径对象**抛 `TypeError`** | 先做类型收窄；6 类非路径输入均不抛 |
| 8 | "不导入主链路"守卫**平凡可满足**且方向搞反 | 改断言**方向**（主链路不得引用适配层）+ **行为**守卫（只导入不得构造策略对象） |
| 9 | `step` 把**任何**非 list 抹平成 `[]` | 仅 `None` → `[]`；其余原样透出 |
| 10 | `enabled=False` **只禁用了前三个方法** | LLM 两方法也返回 `disabled` |
| 11 | `forwarded` 只记五方法，`save`/`load` 不可审计 | 审计项扩到七项 |

**测试从 25 条增至 54 条**，新增的全部是**负向/边界**用例。

### 6.1 本轮最值得留档的一条

缺陷 1、2、5 是**同一个根因**：我手写了一份"哪个意图需要哪些字段"的规则，
而本仓 `ai/action_safety.py` 里**已经有权威的 `ACTION_SPECS`**。
**两份规则必然漂移** —— 评审正是用本仓自己的校验器把我这份抓出来的。

更糟的是，我的测试断言 `set(action) ⊆ {actor,type,obj_id,target_obj_id}`
**反过来把缺陷锁死了**：它禁止了必需的 `weapon_id`，
于是"产出非法动作"在测试里看起来是**正确行为**。

> **教训**：当仓库里已有权威规范（`ACTION_SPECS`）时，
> 不要在别处再写一份并用自己的副本做断言 ——
> **要用权威规范去校验，否则测试会把错误固化。**
