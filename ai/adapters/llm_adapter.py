"""LLM 与自然语言命令的**占位适配**（I03 步骤 2/3）。

为什么是占位
------------
Q6（「LLM 与多模态接口是否为必须交付」）**未关闭**。I03 的前置条件写明：
「Q6 未关闭前按报告默认处理，**只做接口与占位实现，不引入真实 LLM 依赖**」。

故本文件**不 import torch/transformers**，也不下载任何模型。
将来 Q6 若裁定需要真实 LLM，只替换本文件的实现，**接口与调用方不动**。

## 独立评审整改（第五十一轮）

首版有多处**实质缺陷**（评审 A 逐条实测确认），本版修的是这些：

| # | 缺陷 | 修法 |
| --- | --- | --- |
| 1 | 产出的动作**过不了本仓自己的 strict 校验**（打击缺 `weapon_id`） | **不再手写字段名**：改为按 `ai.action_safety.ACTION_SPECS` 的 `required` **逐项检查**，缺什么就返回 `None` 并说明 |
| 2 | 结构不可用却给 `confidence=0.5`（缺 `obj_id` 也报 0.5） | 结构不完整 → `action=None`、`confidence=0.0`。**绝不给"看起来能用其实不能用"的中间置信度** |
| 3 | **否定句被忽略**：「不要打击目标」→ 攻击指令 | 加否定词检测；命中即 `intent=None`、`confidence=0.0` |
| 4 | 机动**永远不可达**（`if "move_path" not in params` 恒真，且无法传入路径） | 加 `move_path` 形参；路径缺省时明说"未提供路径" |
| 5 | 夺控**过约束**（type 5 只需 `obj_id`，却要求 `target_obj_id`） | 改为按 `ACTION_SPECS[type].required` 判定，不再自定义规则 |
| 6 | 「停止」映射到 type **333 结束部署**（不可逆），而非 type 10 停止机动 | 改为明确短语（「停止机动」）→ type 10；不再用裸「停止」 |
| 7 | `load_llm_model` 对非路径对象**抛 TypeError**，违背判据 3 | 路径取值先做类型校验，**任何输入都不抛异常** |

> 缺陷 1、2、5 其实是**同一个根因**：我手写了一份"哪个意图需要哪些字段"的规则，
> 而本仓 `ai/action_safety.py` 里**已经有权威的 `ACTION_SPECS`**。
> 两份规则必然漂移 —— 评审正是用本仓自己的校验器把我这份抓出来的。
> 现改为**单一来源**：字段要求一律查 `ACTION_SPECS`。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# 「未启用」的状态取值。用固定字符串常量而不是散落的字面量。
STATUS_DISABLED = "disabled"
STATUS_UNAVAILABLE = "unavailable"
STATUS_LOADED = "loaded"

PARSE_METHOD_RULE = "rule"

# 否定词：命中即认为命令被否定，**不产出动作**。
# 这是**保守**取向：漏判一个否定句会下发真实攻击指令，代价远高于把肯定句判成否定。
_NEGATION_MARKERS = ("不要", "不得", "禁止", "取消", "别", "停止攻击", "no ", "not ", "don't",
                     "do not", "cancel", "ceasefire", "abort")

# (意图, 平台动作 type, 中文关键词, 英文关键词)
# ⚠ type 必须与 `docs/wargame_ref/actions_reference.txt` 一致。
# 「停止」**不映射到 333**：333 是「结束部署」（不可逆的阶段动作），
# 而口语「停止」通常指停下机动。故只认**明确短语**「停止机动」→ type 10。
_INTENT_KEYWORDS: List[Dict[str, Any]] = [
    {"intent": "move", "type": 1, "zh": ("机动", "移动", "前进", "开进"), "en": ("move", "advance")},
    {"intent": "attack", "type": 2, "zh": ("打击", "攻击", "进攻", "射击"), "en": ("attack", "strike", "fire")},
    {"intent": "occupy", "type": 5, "zh": ("夺控", "占领", "夺取"), "en": ("occupy", "seize")},
    {"intent": "stop_move", "type": 10, "zh": ("停止机动", "停止移动"), "en": ("stop moving", "halt")},
]


def _action_spec_required(action_type: int) -> Optional[List[str]]:
    """查 `ai.action_safety.ACTION_SPECS` 拿该动作类型的必需字段。

    **单一来源**：不在这里另写一份"哪个意图需要哪些字段"——
    首版就是那样写的，于是与 `ACTION_SPECS` 漂移（打击漏 `weapon_id`），
    被本仓自己的 strict 校验抓出来。
    """
    try:
        from ai.action_safety import ACTION_SPECS  # noqa: PLC0415

        spec = ACTION_SPECS.get(int(action_type))
        if spec is None:
            return None
        return list(spec.get("required", []))
    except Exception:  # noqa: BLE001 - 取不到规范时不阻断解析，但下方会拒绝产出
        return None


def _missing_fields(action: Dict[str, Any]) -> Optional[List[str]]:
    """返回动作缺失的必需字段；**返回 None 表示无法判定规范**（保守拒绝）。"""
    req = _action_spec_required(int(action.get("type", -1)))
    if req is None:
        return None
    return [f for f in req if action.get(f) is None]


class LLMAdapter:
    """LLM 加载与自然语言命令解析的占位实现。

    Parameters
    ----------
    model_path:
        本地权重路径。**任何取值都不会报错**，只降级为结构化状态。
    enabled:
        `False` 时 `load_llm_model()` 直接返回 `disabled`。
    """

    def __init__(self, model_path: Optional[str] = None, *, enabled: bool = False) -> None:
        # 默认 `enabled=False`：Q6 未关闭，**不允许**悄悄启用真实模型。
        self.model_path = model_path
        self.enabled = bool(enabled)
        self._loaded = False
        self.load_calls = 0
        self.parse_calls = 0

    # ── 五方法契约之四 ────────────────────────────────────────────────────

    def load_llm_model(self, model_path: Any = None) -> Dict[str, Any]:
        """加载本地 LLM 权重。**任何输入都不抛异常**，只返回结构化状态。

        ⚠ 判据 3 的原文是「**不抛异常**」。首版对非路径对象
        （`object()` / `[1]` / `{...}`）会在 `os.path.isfile` 处抛 `TypeError` ——
        评审实测出来的。故此处**先做类型校验**再交给 `os.path`。
        """
        self.load_calls += 1
        path = model_path if model_path is not None else self.model_path

        if not self.enabled:
            return self._status(STATUS_DISABLED, None,
                                "LLM 未启用（Q6 未关闭，按 I03 前置条件只做占位实现）")
        if not path:
            return self._status(STATUS_UNAVAILABLE, None, "未提供 model_path")

        # 类型收窄：只有 str / bytes / os.PathLike / int(fd) 才可能是路径。
        import os  # noqa: PLC0415

        if not isinstance(path, (str, bytes, os.PathLike, int)):
            return self._status(STATUS_UNAVAILABLE, None,
                                "model_path 类型非法（%s），应为路径字符串" % type(path).__name__)
        try:
            is_file = os.path.isfile(path)
        except (TypeError, ValueError, OSError) as exc:
            # 例如含 NUL 字节、超长路径 —— 一律降级，不抛出
            return self._status(STATUS_UNAVAILABLE, path,
                                "model_path 不可用：%s: %s" % (type(exc).__name__, exc))
        if not is_file:
            return self._status(STATUS_UNAVAILABLE, path, "模型文件不存在: %s" % (path,))

        # 文件在，但本占位实现**不加载权重** —— 如实说明，不假装 loaded=True。
        self._loaded = False
        return self._status(STATUS_UNAVAILABLE, path,
                            "占位实现不加载权重（Q6 未关闭）；接口已就绪，替换 llm_adapter 实现即可")

    @staticmethod
    def _status(status: str, path: Any, reason: str) -> Dict[str, Any]:
        return {
            "status": status,
            "loaded": False,
            "model_path": path if isinstance(path, (str, type(None))) else str(path),
            "reason": reason,
            "capabilities": {"natural_language": True, "model_inference": False},
        }

    # ── 五方法契约之五 ────────────────────────────────────────────────────

    def parse_natural_language_command(self, text: Any, *, actor: int = 0,
                                       obj_id: Optional[Any] = None,
                                       target_obj_id: Optional[Any] = None,
                                       weapon_id: Optional[Any] = None,
                                       move_path: Optional[Any] = None,
                                       ) -> Dict[str, Any]:
        """把自然语言命令解析成**与 `step` 输出同构且能过 strict 校验**的动作。

        返回字段（I03 步骤 3 点名要求）：
        `intent` / `target` / `params` / `confidence` / `degraded`
        外加 `action`（可直接放入动作列表）与 `method`。

        **三条不可让步的性质**（评审抓出来的）：

        1. **产出的动作必须过本仓自己的 `validate_action(strict)`** ——
           字段要求一律查 `ACTION_SPECS`，不手写；
        2. **结构不可用就不给中间置信度** —— 缺字段即 `action=None`、`confidence=0.0`；
        3. **否定句不产出动作** —— 「不要打击目标」必须解析为"不执行"。
        """
        self.parse_calls += 1
        raw = "" if text is None else str(text)
        norm = raw.strip().lower()

        # ① 否定句先行：宁可漏判肯定句，不可误判否定句
        for marker in _NEGATION_MARKERS:
            if marker in raw or marker in norm:
                return {"intent": None, "target": None, "params": {}, "confidence": 0.0,
                        "degraded": True, "method": PARSE_METHOD_RULE, "action": None,
                        "text": raw, "reason": "命令含否定词 %r，不产出动作" % marker}

        # ② 关键词匹配（长词优先，避免「停止机动」被「机动」抢走）
        matched = None
        best_len = 0
        for spec in _INTENT_KEYWORDS:
            for kw in list(spec["zh"]) + list(spec["en"]):
                if (kw in raw or kw in norm) and len(kw) > best_len:
                    matched, best_len = spec, len(kw)
        if matched is None:
            return {"intent": None, "target": None, "params": {}, "confidence": 0.0,
                    "degraded": True, "method": PARSE_METHOD_RULE, "action": None,
                    "text": raw, "reason": "规则未命中任何已知意图关键词"}

        intent = matched["intent"]
        atype = int(matched["type"])
        params: Dict[str, Any] = {"type": atype}

        # ③ 组装候选动作。**字段名一律取自平台动作字段**，不造新字段。
        action: Dict[str, Any] = {"actor": actor, "type": atype}
        if obj_id is not None:
            action["obj_id"] = obj_id
        if target_obj_id is not None:
            action["target_obj_id"] = target_obj_id
        if weapon_id is not None:
            action["weapon_id"] = weapon_id
        if move_path is not None:
            action["move_path"] = move_path
            params["move_path"] = move_path

        # ④ **权威校验**：缺什么必需字段就拒绝产出（单一来源 = ACTION_SPECS）
        missing = _missing_fields(action)
        if missing is None:
            return {"intent": intent, "target": None, "params": params, "confidence": 0.0,
                    "degraded": True, "method": PARSE_METHOD_RULE, "action": None,
                    "text": raw,
                    "reason": "无法取得动作类型 %d 的字段规范（ai.action_safety 不可用），保守拒绝"
                              % atype}
        if missing:
            return {"intent": intent, "target": target_obj_id if target_obj_id is not None else obj_id,
                    "params": params, "confidence": 0.0, "degraded": True,
                    "method": PARSE_METHOD_RULE, "action": None, "text": raw,
                    "reason": "命令语义可识别，但缺少必需字段 %s，故不产出动作"
                              "（宁可拒绝，也不下发会被拒的动作）" % missing}

        # ⑤ 只有**结构完整**才给置信度。固定 0.5：规则命中 ≠ 理解，不虚增。
        return {
            "intent": intent,
            "target": target_obj_id if target_obj_id is not None else obj_id,
            "params": params,
            "confidence": 0.5,
            "degraded": True,
            "method": PARSE_METHOD_RULE,
            "action": action,
            "text": raw,
            "reason": "规则解析命中意图 %s（无模型推理，故 degraded=True）" % intent,
        }

    def debug(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_path": self.model_path,
            "loaded": self._loaded,
            "load_calls": self.load_calls,
            "parse_calls": self.parse_calls,
            "intents": [s["intent"] for s in _INTENT_KEYWORDS],
            "method": PARSE_METHOD_RULE,
        }
