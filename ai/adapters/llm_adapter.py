"""LLM 与自然语言命令的**占位适配**（I03 步骤 2/3）。

为什么是占位
------------
Q6（「LLM 与多模态接口是否为必须交付」）**未关闭**。I03 的前置条件写明：
「Q6 未关闭前按报告默认处理，**只做接口与占位实现，不引入真实 LLM 依赖**」。

故本文件**不 import torch/transformers**，也不下载任何模型。
它的价值在于把**接口形状与降级语义**先定下来：
将来 Q6 若裁定需要真实 LLM，只替换本文件的实现，**接口与调用方不动**
（这正是【风险与回滚】写的「保留接口只替换 llm_adapter 实现」）。

两条硬要求
----------
1. `load_llm_model()` 无模型或未启用时**返回结构化状态，不抛异常**（判据 3）；
2. `parse_natural_language_command()` 的输出必须**与 `step` 接受的动作格式一致** ——
   故它产出的是可直接放进动作列表的 dict（字段名取自平台动作文档，
   与本仓 `ai/action_safety.py` 的 ACTION_SPECS 同源）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# 「未启用」的状态取值。**用固定字符串常量**而不是散落的字面量，
# 便于调用方判断，也便于将来新增状态（如 `loading`）时不破坏既有比较。
STATUS_DISABLED = "disabled"
STATUS_UNAVAILABLE = "unavailable"
STATUS_LOADED = "loaded"

# 降级解析用不到外部模型，故默认即「规则解析」。
PARSE_METHOD_RULE = "rule"

_INTENT_KEYWORDS: List[Dict[str, Any]] = [
    # (意图, 平台动作 type, 中文关键词, 英文关键词)
    {"intent": "move",      "type": 1,  "zh": ("机动", "移动", "前进", "开进"), "en": ("move", "advance")},
    {"intent": "attack",    "type": 2,  "zh": ("打击", "攻击", "进攻", "射击"), "en": ("attack", "strike", "fire")},
    {"intent": "occupy",    "type": 5,  "zh": ("夺控", "占领", "夺取"),         "en": ("occupy", "seize")},
    {"intent": "stop",      "type": 333, "zh": ("停止", "结束部署"),            "en": ("stop", "halt")},
]


class LLMAdapter:
    """LLM 加载与自然语言命令解析的占位实现。

    Parameters
    ----------
    model_path:
        本地权重路径。**不存在的路径不会报错**，而是降级为结构化 `unavailable`。
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

    def load_llm_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """加载本地 LLM 权重。**任何情况下都不抛异常**，只返回结构化状态。

        返回字段固定为 `status / loaded / model_path / reason / capabilities`，
        使调用方无需 try/except 即可判断（判据 3）。
        """
        self.load_calls += 1
        path = model_path if model_path is not None else self.model_path

        if not self.enabled:
            return {
                "status": STATUS_DISABLED, "loaded": False, "model_path": path,
                "reason": "LLM 未启用（Q6 未关闭，按 I03 前置条件只做占位实现）",
                "capabilities": {"natural_language": True, "model_inference": False},
            }
        if not path:
            return {
                "status": STATUS_UNAVAILABLE, "loaded": False, "model_path": None,
                "reason": "未提供 model_path",
                "capabilities": {"natural_language": True, "model_inference": False},
            }
        import os  # noqa: PLC0415 - 只在需要时导入

        if not os.path.isfile(path):
            return {
                "status": STATUS_UNAVAILABLE, "loaded": False, "model_path": path,
                "reason": "模型文件不存在: %s" % path,
                "capabilities": {"natural_language": True, "model_inference": False},
            }
        # 走到这里说明文件在，但本占位实现**不加载权重** —— 如实说明，
        # 而不是假装 loaded=True（那会让调用方以为模型推理可用）。
        self._loaded = False
        return {
            "status": STATUS_UNAVAILABLE, "loaded": False, "model_path": path,
            "reason": "占位实现不加载权重（Q6 未关闭）；接口已就绪，替换 llm_adapter 实现即可",
            "capabilities": {"natural_language": True, "model_inference": False},
        }

    # ── 五方法契约之五 ────────────────────────────────────────────────────

    def parse_natural_language_command(self, text: str, *, actor: int = 0,
                                       obj_id: Optional[Any] = None,
                                       target_obj_id: Optional[Any] = None) -> Dict[str, Any]:
        """把自然语言命令解析成**与 `step` 输出同构**的动作。

        返回字段（I03 步骤 3 点名要求）：
        `intent` / `target` / `params` / `confidence` / `degraded`
        外加 `action`（可直接放入动作列表）与 `method`。

        **降级语义**：本占位实现不做模型推理，故 `degraded` 恒为 `True`、
        `method` 恒为 `rule`。**不谎报高置信度**：规则命中给中间值，
        未命中给 `0.0` 且 `action=None`。
        """
        self.parse_calls += 1
        raw = "" if text is None else str(text)
        norm = raw.strip().lower()

        matched: Optional[Dict[str, Any]] = None
        for spec in _INTENT_KEYWORDS:
            for kw in spec["zh"]:
                if kw in raw:
                    matched = spec
                    break
            if matched:
                break
            for kw in spec["en"]:
                if kw in norm:
                    matched = spec
                    break
            if matched:
                break

        if matched is None:
            return {
                "intent": None, "target": None, "params": {}, "confidence": 0.0,
                "degraded": True, "method": PARSE_METHOD_RULE, "action": None,
                "text": raw,
                "reason": "规则未命中任何已知意图关键词",
            }

        intent = matched["intent"]
        atype = matched["type"]
        params: Dict[str, Any] = {"type": atype}

        # 动作字段名必须与平台动作文档一致（见 ai/action_safety.py 的 ACTION_SPECS）。
        if intent == "move":
            # 机动需要 move_path；调用方未给路径时不能编一个 —— 缺字段即返回不可用动作。
            if "move_path" not in params:
                return {
                    "intent": intent, "target": obj_id, "params": params,
                    "confidence": 0.0, "degraded": True, "method": PARSE_METHOD_RULE,
                    "action": None, "text": raw,
                    "reason": "机动命令需 move_path，自然语言中未提供路径",
                }
        if intent in ("attack", "occupy") and target_obj_id is None:
            return {
                "intent": intent, "target": None, "params": params,
                "confidence": 0.0, "degraded": True, "method": PARSE_METHOD_RULE,
                "action": None, "text": raw,
                "reason": "%s 命令需目标对象，未解析出目标" % intent,
            }

        action: Dict[str, Any] = {"actor": actor, "type": atype}
        if obj_id is not None:
            action["obj_id"] = obj_id
        if target_obj_id is not None:
            action["target_obj_id"] = target_obj_id

        # 置信度：规则命中 ≠ 理解。给固定中间值，**不随文本长度或关键词数虚增**。
        confidence = 0.5
        return {
            "intent": intent,
            "target": target_obj_id if target_obj_id is not None else obj_id,
            "params": params,
            "confidence": confidence,
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
