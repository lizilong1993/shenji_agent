"""31006 五方法契约的**薄适配层**（I03 步骤 1）。

标准依据
--------
`31006 5.3` 与附录 `A.3`：智能体侧需提供
`setup` / `reset` / `step` / `load_llm_model` / `parse_natural_language_command`。
`31007` 附录 C 另要求 `__init__` / `save` / `load`（本层一并转发，作旁证覆盖）。

本层的职责边界（**必须守住**）
------------------------------
**只转发，不复制策略逻辑。**

⚠ 首版头部写"三个方法体各只有一行 `return`" —— 那是**失实**的：
`step` 里有一处 `None -> []` 兼容（评审指出）。准确表述是
"**不实现任何决策逻辑**"；`None` 的兼容是既有 `Agent.step` 的约定，不是新决策。

【风险与回滚】点名的头号风险是「适配层反向渗入主链路造成策略漂移」。
故本文件**不实现任何决策逻辑**：`setup`/`reset`/`step` 三个方法体各只有一行 `return`，
直接交给既有 `Agent` 实例。若将来有人想在这里"顺手改一下动作"，那是在主链路上
开第二个决策入口 —— 本层的存在意义恰恰是**避免**这件事。

可整体禁用
----------
`AgentAPI(..., enabled=False)` 时 `step()` 返回 `[]` 且不触碰被包装的智能体，
满足【风险与回滚】的「适配层可用配置整体禁用」。
"""
from __future__ import annotations

import time
from pathlib import Path

from typing import Any, Dict, List, Optional

from ai.agent import Agent
from ai.adapters.llm_adapter import LLMAdapter

# 五方法契约（31006 5.3 / 附录 A.3）+ 附录 C 三方法，用于自检与文档对照。
FIVE_METHOD_CONTRACT = (
    "setup",
    "reset",
    "step",
    "load_llm_model",
    "parse_natural_language_command",
)
APPENDIX_C_CONTRACT = ("__init__", "save", "load")


class AgentAPI:
    """把 `Agent` 包装成 31006 可调用的五方法接口。

    Parameters
    ----------
    agent:
        既有 `Agent` 实例。**必须由调用方构造**，本层不替他决定 seed/日志等级；
        传 `None` 时惰性新建（方便标准侧按契约直接 `AgentAPI()`）。
    enabled:
        `False` 时整层禁用（`step()` 恒返回 `[]`），用于回滚与对照实验。
    llm_adapter:
        可注入，便于测试与将来替换为真实 LLM 实现（只换实现，不换接口）。
    """

    def __init__(self, agent: Optional[Agent] = None, *, enabled: bool = True,
                 llm_adapter: Optional[LLMAdapter] = None) -> None:
        self.enabled = bool(enabled)
        self._agent: Optional[Agent] = agent
        self.llm = llm_adapter if llm_adapter is not None else LLMAdapter()
        # 覆盖五方法**与**附录 C 的 save/load —— 文档自称"调用可审计 … 每个方法"，
        # 首版只记五方法，save/load 无法审计（评审指出）。
        # `load_llm_model` 只返回布尔（标准要求），细节从这里取
        self.last_llm_status: Dict[str, Any] = {}
        self.last_parse: Dict[str, Any] = {}
        self.forwarded: Dict[str, int] = {
            name: 0 for name in (FIVE_METHOD_CONTRACT + ("save", "load"))}

    # -- 内部 ---------------------------------------------------------------

    def _require_agent(self) -> Agent:
        """惰性构造。**不吞异常**：构造失败要显式暴露，不能假装智能体就绪。"""
        if self._agent is None:
            self._agent = Agent(seed=0)
        return self._agent

    @property
    def agent(self) -> Optional[Agent]:
        """被包装的智能体（只读）。**不提供 setter** —— 换智能体应新建适配层。"""
        return self._agent

    def contract(self) -> Dict[str, Any]:
        """契约自检：哪些方法在、哪些是转发、哪些是占位。"""
        return {
            "five_methods": {
                name: {
                    "present": hasattr(self, name),
                    "kind": ("forward" if name in ("setup", "reset", "step")
                             else "placeholder" if name == "load_llm_model"
                             else "rule_based_placeholder" if name == "parse_natural_language_command"
                             else "unknown"),
                }
                for name in FIVE_METHOD_CONTRACT
            },
            "appendix_c": {
                name: hasattr(self, name) for name in APPENDIX_C_CONTRACT
                if name != "__init__"
            },
            "enabled": self.enabled,
            "forwarded": dict(self.forwarded),
        }

    # -- 五方法契约（前三个：**纯转发**）--------------------------------------

    def setup(self, setup_info: Dict[str, Any]) -> Any:
        """转发到 `Agent.setup()`。本层不解读 `setup_info`。"""
        self.forwarded["setup"] += 1
        if not self.enabled:
            return None
        return self._require_agent().setup(setup_info)

    def reset(self) -> bool:
        """重置智能体状态，返回 **Boolean**（`31006` A.3.3 / 表 A.14 `resetStatus`）。

        原文 `BaseAgent::reset()->Boolean`，输出参数
        `resetStatus: Boolean 是否成功重置智能体状态`。`Agent.reset()` 返回 `None`（成功）
        或抛异常，故本层布尔语义：被禁用 → `False`；未抛异常 → `True`；抛异常 → `False`
        （契约要求返回布尔，**不再向上抛**）。
        """
        self.forwarded["reset"] += 1
        if not self.enabled:
            return False
        try:
            self._require_agent().reset()
        except Exception:  # noqa: BLE001 - 契约要求返回 Boolean，不抛
            return False
        return True

    def step(self, observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """转发到 `Agent.step()`。**禁用时返回空列表**，不去动被包装的智能体。"""
        self.forwarded["step"] += 1
        if not self.enabled:
            return []
        # 透传：`Agent.step` 正常返回 list；其异常路径自己返回 []。
        # 首版把**任何**非 list 都强转成 []，于是类型 bug 被伪装成"没有动作"
        # （评审指出）。现在只对 `None` 做兼容（那是 `Agent.step` 的既有约定），
        # 其余非 list 原样透出 —— 让上游看见真实的类型问题，而不是被我们抹平。
        result = self._require_agent().step(observation)
        if result is None:
            return []
        return result

    # -- 五方法契约（后两个：占位实现，见 llm_adapter）------------------------

    def load_llm_model(self, model_path: Any = None, config: Any = None) -> bool:
        """加载大模型并配置参数，返回 **Boolean**（`31006` A.3.4 / 表 A.20–A.21）。

        原文 `BaseAgent::load_llm_model(model_path: str, config: Dict)->Boolean`，
        输出参数 `loadStatus: Boolean 是否成功加载大模型`。

        `config` 四键（表 A.20）：`inference_precision` / `max_context_length` /
        `device` / `batch_size`。**占位实现不加载真实权重，故恒返回 `False`**；
        细节记入 `self.last_llm_status`，使只拿到布尔的调用方仍可诊断。
        """
        self.forwarded["load_llm_model"] += 1
        if not self.enabled:
            self.last_llm_status = {"status": "disabled", "loaded": False,
                                    "reason": "适配层已禁用（AgentAPI(enabled=False)）",
                                    "config": self._normalise_llm_config(config)}
            return False
        status = self.llm.load_llm_model(model_path)
        status["config"] = self._normalise_llm_config(config)
        status["loaded"] = False          # 不谎报加载成功
        self.last_llm_status = status
        return False

    @staticmethod
    def _normalise_llm_config(config: Any) -> Dict[str, Any]:
        """按表 A.20 校验 `config` 四键；**记下缺失与类型问题，不猜默认值**。"""
        keys = ("inference_precision", "max_context_length", "device", "batch_size")
        if config is None:
            return {"provided": False, "keys": {}, "problems": ["未提供 config"]}
        if not isinstance(config, dict):
            return {"provided": True, "keys": {},
                    "problems": ["config 应为 Dict，实际 %s" % type(config).__name__]}
        out: Dict[str, Any] = {"provided": True, "keys": {}, "problems": []}
        for k in keys:
            if k in config:
                out["keys"][k] = config[k]
            else:
                out["problems"].append("缺 config[%r]" % k)
        return out

    def parse_natural_language_command(self, command: Any = None, *,
                                       unit_id: Any = None,
                                       target_obj_id: Any = None,
                                       weapon_id: Any = None,
                                       move_path: Any = None,
                                       platform_shape: bool = True,
                                       ) -> List[Dict[str, Any]]:
        """把自然语言指令解析为**结构化动作指令队列**，返回 `List[Dict]`。

        `31006` A.3.5 / 表 A.22–A.23：形式
        `parse_natural_language_command(command: str) -> List[Dict]`；
        输出 `actionList: List[Dict]`，元素含
        `unit_id`(str) / `action_type`(str) / `target`(Dict) / `actionP`(Dict)。

        **与标准的关系（不牺牲能力）**：标准只规定一个 `command` 参数；
        但"从自然语言里抽出动作参数"不在占位实现的能力范围内（Q6 未关闭），
        故把 `unit_id` / `target_obj_id` / `weapon_id` / `move_path` 作为**可选关键字**
        并列传入 —— 既满足 `f(command)` 的调用方式，又保留把参数给全的能力。

        `platform_shape=True`（默认）时元素**同时**带平台动作字段（可直接喂 `step`）；
        `False` 时只给标准四键，得到纯净的 A.3.5 形状。

        四键取值口径（**如实映射，不造语义**）：
        `unit_id` = `str(unit_id)`；`action_type` = **意图名**（标准举例 `"move"`/`"attack"`）；
        `target` = `{"obj_id": ...}`；`actionP` = `{}`（规则解析给不出参数，**不编造**）。
        """
        self.forwarded["parse_natural_language_command"] += 1
        if not self.enabled:
            return []
        parsed = self.llm.parse_natural_language_command(
            command, obj_id=unit_id, target_obj_id=target_obj_id,
            weapon_id=weapon_id, move_path=move_path)
        self.last_parse = parsed
        act = parsed.get("action")
        if not act:
            return []                                # 空队列，不是"含空元素的队列"
        el: Dict[str, Any] = {
            "unit_id": str(act.get("obj_id")) if act.get("obj_id") is not None else None,
            "action_type": parsed.get("intent"),
            "target": ({"obj_id": act["target_obj_id"]}
                       if act.get("target_obj_id") is not None else {}),
            "actionP": {},
        }
        if platform_shape:
            el.update(act)                           # 叠加平台字段，**不改标准四键**
        return [el]

    def save(self, path: str) -> Any:
        self.forwarded["save"] += 1
        if not self.enabled:
            return None
        return self._require_agent().save(path)

    def load(self, path: str) -> Any:
        self.forwarded["load"] += 1
        if not self.enabled:
            return None
        return self._require_agent().load(path)

    # ── 多模态态势接口（I04 步骤 1/2/3）────────────────────────────────────

    def get_multimodal_situation(self, frames: Any = None, *,
                                 vector: Any = None,
                                 image: Any = None,
                                 targets: Any = None,
                                 hotspot_names: Any = (),
                                 case_id: Optional[str] = None,
                                 case_name: Optional[str] = None,
                                 start_utc: Optional[str] = None,
                                 tick_seconds: Optional[float] = None,
                                 observation: Any = None,
                                 ) -> Dict[str, Any]:
        """返回 `{text, image, vector, meta}` 多模态态势四件套。

        **任一模态失败置空并标 `degraded=true`，不做隐式替代**（`I04`【风险与回滚】）。

        Parameters
        ----------
        frames:
            态势帧序列（`build_brief` 的输入）。缺省时文本模态置空。
        vector:
            108 维特征。**不传时**会尝试由 `observation` 经 P3 `ai.features` 生成；
            两者都没有则 `vector=[]`。
        image:
            简化态势图**引用**（路径字符串或含 `path` 的 dict）。
            **接口内不做渲染** —— 渲染是调用方的事（步骤 3）。
        observation:
            P3 的 observation；从中取算子 + 帧级标量生成 108 维。

        Returns
        -------
        dict
            `text:str` / `image:dict` / `vector:list[float]` / `meta:dict`
        """
        self.forwarded.setdefault("get_multimodal_situation", 0)
        self.forwarded["get_multimodal_situation"] += 1

        degraded: List[str] = []
        meta: Dict[str, Any] = {
            "schema_version": MULTIMODAL_SCHEMA_VERSION,
            "interface_version": MULTIMODAL_SCHEMA_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "ai.adapters.AgentAPI",
            "modalities_available": [],
            "degraded": False,
        }
        if case_id is not None:
            meta["case_id"] = case_id

        # ── 文本模态：复用 P2 brief.py，不另写叙述逻辑（步骤 2）──────────
        text = ""
        brief_mod = _p2_brief_module()
        if brief_mod is None:
            degraded.append("text: P2 engine/brief.py 不可用")
        elif not frames:
            degraded.append("text: 未提供 frames")
        else:
            try:
                brief = brief_mod.build_brief(
                    frames, targets=targets, hotspot_names=hotspot_names,
                    case_id=case_id, case_name=case_name,
                    start_utc=start_utc, tick_seconds=tick_seconds)
                text = brief_mod.render_text(brief) or ""
                meta["brief_headline"] = brief.get("headline")
                meta["brief_frame_count"] = brief.get("frame_count")
                if not text.strip():
                    degraded.append("text: brief 渲染为空")
            except Exception as exc:  # noqa: BLE001 - 单模态失败不得拖垮接口
                degraded.append("text: %s: %s" % (type(exc).__name__, exc))

        # ── 图像模态：只返回引用与尺寸，**不渲染**（步骤 3）────────────
        image_ref, image_err = _image_ref(image)
        if image_err:
            degraded.append("image: %s" % image_err)

        # ── 向量模态：108 维。不截断、不补齐、不换另一套语义 ────────────
        vec = vector
        if vec is None and observation is not None:
            made, err = _vector_from_observation(observation)
            if err:
                degraded.append("vector: %s" % err)
            else:
                vec = made
        vals, vec_err = _p3_vector(vec, EXPECTED_VECTOR_DIM)
        if vec_err:
            degraded.append("vector: %s" % vec_err)
        else:
            meta["vector_feature_dim"] = len(vals)
            meta["vector_source"] = "ai.features.operator_to_feature (P3)"
            # Q10 未关闭：**分别**登记两套 schema，不合并、不假装同义
            names = _p2_feature_names()
            meta["p2_vector_schema"] = {
                "module": "situation3d/engine/intent_features.py",
                "schema_version": "ifv2",
                "feature_names_available": bool(names),
                "feature_dim": len(names) if names else None,
                "note": ("与 P3 的 vector **语义不同**（见 docs/data_dictionary_mapping.md）；"
                         "本接口不做数值合并，也不在两套之间做隐式替代"),
            }

        if text.strip():
            meta["modalities_available"].append("text")
        if image_ref:
            meta["modalities_available"].append("image")
        if vals:
            meta["modalities_available"].append("vector")
        meta["degraded"] = bool(degraded)
        if degraded:
            meta["degraded_reasons"] = degraded

        return {"text": text, "image": image_ref, "vector": vals, "meta": meta}


# ── 多模态接口的取值辅助（I04）────────────────────────────────────────────

EXPECTED_VECTOR_DIM = 108
MULTIMODAL_SCHEMA_VERSION = "mm/1"


def _p2_brief_module():
    """惰性导入 P2 的 `engine/brief.py`。

    **惰性**：`situation3d/` 不在 P3 常规 import 路径上，而适配层必须在
    **没有 P2 引擎**的环境里也能被导入（否则 `I03` 的契约会一起崩）。
    导入失败返回 `None`，交调用方置空并标 `degraded`。
    """
    import sys

    root = Path(__file__).resolve().parents[2] / "situation3d"
    if not (root / "engine" / "brief.py").is_file():
        return None
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)
    try:
        from engine import brief as brief_mod

        return brief_mod
    except Exception:  # noqa: BLE001
        return None


def _p2_feature_names():
    """P2 `ifv2` 的 108 个名字（拿不到返回 `None`）。仅用于**登记**两套 schema 不同。"""
    try:
        from engine import intent_features as if_mod

        return list(if_mod.feature_names())
    except Exception:  # noqa: BLE001
        return None


def _vector_from_observation(observation):
    """由 P3 observation 生成 108 维（复用 `ai.features`，**不另写特征逻辑**）。"""
    if not isinstance(observation, dict):
        return None, "observation 不是 dict"
    try:
        from ai import features as feat_mod

        ops = observation.get("operators") or []
        op = None
        for o in ops:
            if isinstance(o, dict) and o.get("color") is not None:
                op = o
                break
        if op is None and ops:
            op = ops[0] if isinstance(ops[0], dict) else None
        if op is None:
            return None, "observation 里没有可用算子"
        return feat_mod.operator_to_feature(op, frame=observation), None
    except Exception as exc:  # noqa: BLE001
        return None, "%s: %s" % (type(exc).__name__, exc)


def _p3_vector(vector, feature_dim):
    """规整 108 维向量；长度不符则**拒绝**（不截断、不补齐 —— 那是隐式替代）。"""
    if vector is None:
        return [], "未提供 vector（P3 侧应由 ai.features.operator_to_feature 生成）"
    try:
        vals = [float(x) for x in vector]
    except (TypeError, ValueError) as exc:
        return [], "vector 不是数值序列: %s: %s" % (type(exc).__name__, exc)
    if len(vals) != feature_dim:
        return [], "vector 长度 %d != FEATURE_DIM %d（不截断也不补齐）" % (len(vals), feature_dim)
    return vals, None


def _image_ref(image):
    """规整图像模态为**引用 + 尺寸**（步骤 3：接口内不做渲染阻塞）。"""
    if image is None:
        return {}, "未提供 image 引用"
    if isinstance(image, str):
        return {"path": image}, None
    if isinstance(image, dict):
        path = image.get("path")
        if not path:
            return {}, "image 缺 path"
        out = {"path": str(path)}
        for k in ("width", "height", "format"):
            if image.get(k) is not None:
                out[k] = image[k]
        return out, None
    return {}, "image 类型非法（应为路径字符串或含 path 的 dict）"
