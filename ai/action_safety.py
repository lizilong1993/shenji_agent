"""动作 schema 校验（I05 步骤 3）：双校验 —— 前置条件 + 后果约束。

依据
----
字段结构**不是从标准正文臆测的**，而是取自平台自带文档
`docs/wargame_ref/actions_reference.txt`（《庙算·陆战指挥官》动作章），
逐条解析出 **42 个动作 type** 及其附加字段。这样做的理由：
"庙算陆战"与 31007 附录 B（空战格式）**字段名与结构不同**，
按附录 B 硬套会得到**错误的语义** —— 这正是 I05【风险与回滚】点明的头号风险。

两类校验
--------
- **前置条件校验**（`check_preconditions`）：动作**自身是否成形**——
  必填字段在不在、类型对不对、取值范围合不合法。这一层**不依赖环境状态**，可离线跑。
- **后果约束校验**（`check_consequences`）：动作**与环境是否相容**——
  算子是否可控、目标是否存在、席位是否匹配。这一层需要 `observation`。

两者的差别很重要：**前者拦"畸形动作"，后者拦"合法但此刻不该发的动作"**。

三种模式（I05【风险与回滚】要求「先告警不拦截，后逐步收紧」）
--------------------------------------------------------
| 模式 | 行为 |
| --- | --- |
| `off` | 不做任何检查（等价于本模块不存在） |
| `warn` | **默认**：检查并记录问题，但**不改判定**、不拦动作 |
| `strict` | 有问题即拒绝（返回 `ok=False`） |

默认 `warn` 是**有意选择**：校验规则若过严会拦掉合法战术动作，
而合法动作被拦的代价远高于放过一个畸形动作。收紧需要证据，不是靠改默认值。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ── 动作字典：由 docs/wargame_ref/actions_reference.txt 解析而来 ──────────────
# 每条 = (必需字段, 取值范围约束)。actor 与 type 由公共规则统一要求，不在此重复。
# 取值约束形如 {"字段": (最小值, 最大值)}。
ACTION_SPECS: Dict[int, Dict[str, Any]] = {
    1:   {"required": ["obj_id", "move_path"]},
    2:   {"required": ["obj_id", "target_obj_id", "weapon_id"]},
    3:   {"required": ["obj_id", "target_obj_id"]},
    4:   {"required": ["obj_id", "target_obj_id"]},
    5:   {"required": ["obj_id"]},
    # 状态之间互斥，算子在某一时刻只能处于一种状态（平台文档 Note）
    6:   {"required": ["obj_id", "target_state"], "ranges": {"target_state": (0, 5)}},
    7:   {"required": ["obj_id"]},
    8:   {"required": ["obj_id", "jm_pos", "weapon_id"]},
    9:   {"required": ["obj_id", "target_obj_id", "weapon_id", "guided_obj_id"]},
    10:  {"required": ["obj_id"]},
    11:  {"required": ["obj_id"]},
    12:  {"required": ["obj_id"]},
    13:  {"required": ["obj_id"]},
    14:  {"required": ["obj_id"]},
    15:  {"required": ["obj_id", "target_obj_id"]},
    16:  {"required": ["obj_id", "target_altitude"]},
    17:  {"required": ["obj_id"]},
    18:  {"required": ["obj_id", "target_obj_id"]},
    19:  {"required": ["obj_id", "target_obj_id"]},
    20:  {"required": ["obj_id", "target_pos"]},
    # 干扰方向必须是 0~5（平台文档 Field 表）
    21:  {"required": ["obj_id", "direction"], "ranges": {"direction": (0, 5)}},
    22:  {"required": []},
    23:  {"required": []},
    24:  {"required": []},
    25:  {"required": ["obj_id"]},
    27:  {"required": ["obj_id"]},
    28:  {"required": ["obj_id"]},
    29:  {"required": ["obj_id"]},
    30:  {"required": ["obj_id"]},
    31:  {"required": ["obj_id"]},
    32:  {"required": ["obj_id"]},
    33:  {"required": []},
    # 营长专属
    100: {"required": ["info"]},
    202: {"required": ["msg_id"]},
    204: {"required": ["to_all", "msg_body"]},
    205: {"required": ["msg_body"]},
    207: {"required": ["seat", "hex", "start_time", "end_time", "unit_ids", "route"]},
    208: {"required": ["seat", "hex", "start_time", "end_time", "unit_ids", "route"]},
    209: {"required": ["seat", "hex", "radius", "start_time", "end_time", "unit_ids", "route"]},
    210: {"required": ["seat", "hex", "start_time", "end_time", "unit_ids", "route"]},
    # 赛前部署族
    303: {"required": ["obj_id", "target_obj_id"]},
    304: {"required": ["obj_id", "target_obj_id"]},
    314: {"required": ["obj_id"]},
    315: {"required": ["obj_id", "target_obj_id"]},
    # 平台文档标注「此动作正在开发中」
    316: {"required": ["obj_id", "target_altitude"], "experimental": True},
    # 结束部署
    333: {"required": []},
    # 导演动作：只有导演可以下达
    401: {"required": ["target_obj_id"], "director_only": True},
    402: {"required": ["target_pos"], "director_only": True},
    403: {"required": ["target_pos"], "director_only": True},
    404: {"required": ["sub_type", "color", "hex"], "director_only": True},
}

# 字段值的合法范围（跨动作通用）
FIELD_VALUE_RANGES: Dict[str, Tuple[int, int]] = {
    "direction": (0, 5),
    "target_state": (0, 5),
    "color": (0, 1),        # 0红, 1蓝
    "to_all": (0, 1),       # 0发给队友, 1发给全部
}

VALID_MODES = ("off", "warn", "strict")


def _as_int(value: Any) -> Optional[int]:
    """把动作类型归一为 int；不可转换时返回 None。

    平台文档里 `type` 一律是数字，但线上报文可能以字符串到达，
    故做兼容转换，而不是直接判失败。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def check_preconditions(action: Any) -> List[str]:
    """前置条件校验：动作**自身是否成形**。返回问题列表（空 = 无问题）。"""
    problems: List[str] = []
    if not isinstance(action, dict):
        return ["action_not_a_dict:%s" % type(action).__name__]

    # 公共规则：任何动作都必须有 actor 与 type
    if "actor" not in action:
        problems.append("missing_actor")
    atype = _as_int(action.get("type"))
    if atype is None:
        problems.append("missing_or_invalid_type")
        return problems          # 类型都读不出，后面的检查没有意义

    spec = ACTION_SPECS.get(atype)
    if spec is None:
        # **未知类型不判失败**：平台会加新动作，未知不等于非法。
        # 记一条问题供告警，是否拦截交给模式决定。
        problems.append("unknown_action_type:%d" % atype)
        return problems

    for field in spec.get("required", []):
        if field not in action or action[field] is None:
            problems.append("missing_required_field:%s" % field)

    # 取值范围
    ranges = dict(FIELD_VALUE_RANGES)
    ranges.update(spec.get("ranges", {}))
    for field, (lo, hi) in ranges.items():
        if field in action and action[field] is not None:
            v = action[field]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                problems.append("field_not_numeric:%s" % field)
            elif not (lo <= v <= hi):
                problems.append("field_out_of_range:%s=%s(expect %d..%d)" % (field, v, lo, hi))

    # 空列表通常等于"没做这件事"，属畸形而非合法
    for field in ("move_path", "unit_ids", "route"):
        if field in action and isinstance(action[field], (list, tuple)) and len(action[field]) == 0:
            problems.append("empty_list_field:%s" % field)
    return problems


def check_consequences(action: Dict[str, Any],
                       observation: Optional[Dict[str, Any]] = None,
                       *,
                       controllable_ops: Optional[Any] = None) -> List[str]:
    """后果约束校验：动作**与环境是否相容**。返回问题列表。

    `observation` 缺省时**不报问题**（无环境可比），而不是报"全部非法" ——
    离线场景下不该把"查不了"变成"不合法"。
    """
    problems: List[str] = []
    if not isinstance(action, dict):
        return ["action_not_a_dict"]
    atype = _as_int(action.get("type"))
    if atype is None:
        return problems

    spec = ACTION_SPECS.get(atype) or {}
    obj_id = action.get("obj_id")

    # 算子可控性：有 obj_id 的动作才检查
    if obj_id is not None and controllable_ops is not None:
        try:
            if obj_id not in controllable_ops:
                problems.append("operator_not_controllable:%s" % obj_id)
        except TypeError:
            pass

    # 目标存在性：有 target_obj_id 的动作才检查
    target = action.get("target_obj_id")
    if target is not None and observation is not None:
        known = set()
        for key in ("operators", "passengers"):
            for op in (observation.get(key) or []):
                if isinstance(op, dict) and "obj_id" in op:
                    known.add(op["obj_id"])
        if known and target not in known:
            problems.append("target_obj_not_present:%s" % target)

    # 导演动作：只有导演可以下达
    if spec.get("director_only") and observation is not None:
        role = observation.get("role")
        if role is not None and str(role) not in ("5", "director", "Director"):
            problems.append("director_action_by_non_director")

    return problems


def validate_action(action: Any, observation: Optional[Dict[str, Any]] = None, *,
                    mode: str = "warn",
                    controllable_ops: Optional[Any] = None,
                    allowlist: Optional[Any] = None,
                    ) -> Dict[str, Any]:
    """双校验入口。返回 `{ok, mode, problems, action_type}`。

    - `ok=False` 只在 `mode="strict"` 且有实质问题时出现；
    - `mode="warn"` 下**始终 `ok=True`**，问题放在 `problems` 里供记录；
    - `allowlist` 命中时**直接放行**（I05 要求的「白名单旁路」）。
    """
    if mode not in VALID_MODES:
        return {"ok": False, "mode": mode, "problems": ["invalid_mode:%s" % mode], "action_type": None}
    if mode == "off":
        return {"ok": True, "mode": mode, "problems": [], "action_type": None}

    atype = _as_int(action.get("type")) if isinstance(action, dict) else None
    if allowlist is not None and atype in set(allowlist):
        return {"ok": True, "mode": mode, "problems": [], "action_type": atype, "allowlisted": True}

    problems = check_preconditions(action)
    problems += check_consequences(action, observation, controllable_ops=controllable_ops)

    ok = True if mode == "warn" else (len(problems) == 0)
    return {"ok": ok, "mode": mode, "problems": problems, "action_type": atype}


def validate_actions(actions: Any, observation: Optional[Dict[str, Any]] = None, *,
                     mode: str = "warn", controllable_ops: Optional[Any] = None,
                     allowlist: Optional[Any] = None) -> Dict[str, Any]:
    """批量校验（`TrainEnv.step` 的入参就是一个动作列表）。"""
    if not isinstance(actions, (list, tuple)):
        return {"ok": False, "mode": mode, "rejected": [], "problems": ["actions_not_a_list"],
                "checked": 0}
    rejected: List[Dict[str, Any]] = []
    all_problems: List[str] = []
    for act in actions:
        r = validate_action(act, observation, mode=mode,
                            controllable_ops=controllable_ops, allowlist=allowlist)
        if r["problems"]:
            all_problems.extend(r["problems"])
        if not r["ok"]:
            rejected.append({"action": act, "problems": r["problems"]})
    return {"ok": not rejected, "mode": mode, "checked": len(actions),
            "rejected": rejected, "problems": all_problems}


def supported_action_types() -> List[int]:
    return sorted(ACTION_SPECS)
