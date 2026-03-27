from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

FEATURE_DIM = 108
TACTICAL_FEATURE_DIM = 32

NUMERIC_FIELDS = [
    "obj_id",
    "color",
    "type",
    "sub_type",
    "basic_speed",
    "armor",
    "A1",
    "stack",
    "guide_ability",
    "value",
    "move_state",
    "cur_hex",
    "cur_pos",
    "speed",
    "move_to_stop_remain_time",
    "can_to_move",
    "flag_force_stop",
    "stop",
    "blood",
    "max_blood",
    "tire",
    "tire_accumulate_time",
    "keep",
    "keep_remain_time",
    "on_board",
    "car",
    "launcher",
    "lose_control",
    "alive_remain_time",
    "get_on_remain_time",
    "get_off_remain_time",
    "change_state_remain_time",
    "target_state",
    "weapon_cool_time",
    "weapon_unfold_time",
    "weapon_unfold_state",
    "C2",
    "C3",
]


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return float(sum(_as_float(v) for v in value.values()))
    if isinstance(value, (list, tuple, set)):
        return float(len(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _scale(field: str, value: float) -> float:
    if field in {"obj_id", "cur_hex", "car", "launcher"}:
        return value / 10000.0
    if field in {"blood", "max_blood", "value"}:
        return value / 100.0
    if field.endswith("_time") or field == "weapon_cool_time":
        return value / 100.0
    if field == "basic_speed" or field == "speed":
        return value / 20.0
    return value


def find_operator(frame: Dict[str, Any], operator_id: int) -> Optional[Dict[str, Any]]:
    for operator in frame.get("operators", []):
        if operator.get("obj_id") == operator_id:
            return operator
    return None


def operator_to_feature(
    operator: Optional[Dict[str, Any]],
    *,
    frame: Optional[Dict[str, Any]] = None,
    fallback_pos: Optional[int] = None,
) -> np.ndarray:
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)
    op = operator or {}
    for idx, field in enumerate(NUMERIC_FIELDS):
        value = _scale(field, _as_float(op.get(field)))
        if field == "cur_hex" and not value and fallback_pos is not None:
            value = fallback_pos / 10000.0
        vector[idx] = value

    remain_bullet = op.get("remain_bullet_nums", {})
    vector[40] = _as_float(remain_bullet)
    vector[41] = _as_float(op.get("carry_weapon_ids"))
    vector[42] = _as_float(op.get("valid_passenger_types"))
    vector[43] = _as_float(op.get("max_passenger_nums"))
    vector[44] = _as_float(op.get("observe_distance"))
    vector[45] = _as_float(op.get("passenger_ids"))
    vector[46] = _as_float(op.get("launch_ids"))
    vector[47] = _as_float(op.get("see_enemy_bop_ids"))

    if frame is not None:
        time_info = frame.get("time", {})
        scores = frame.get("scores", {})
        cities = frame.get("cities", [])
        vector[48] = _scale("cur_step", _as_float(time_info.get("cur_step")))
        vector[49] = _as_float(time_info.get("tick"))
        vector[50] = _scale("red_total", _as_float(scores.get("red_total")))
        vector[51] = _scale("blue_total", _as_float(scores.get("blue_total")))
        vector[52] = _scale("red_win", _as_float(scores.get("red_win")))
        vector[53] = _scale("blue_win", _as_float(scores.get("blue_win")))
        vector[54] = float(len(frame.get("operators", []))) / 64.0
        vector[55] = float(len(cities)) / 16.0
        vector[56] = float(len(frame.get("valid_actions", {}))) / 64.0
        vector[57] = float(len(frame.get("communication", []))) / 16.0
        vector[58] = float(len(frame.get("tank_enemy", []))) / 16.0
        if cities:
            vector[59] = max(_as_float(city.get("value")) for city in cities) / 100.0

    return vector


def action_feature_vector(
    operator: Dict[str, Any],
    candidate: Dict[str, Any],
    *,
    target_operator: Optional[Dict[str, Any]] = None,
    city: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    vector = np.zeros(TACTICAL_FEATURE_DIM, dtype=np.float32)
    vector[0] = _scale("blood", _as_float(operator.get("blood")))
    vector[1] = _scale("speed", _as_float(operator.get("speed")))
    vector[2] = _scale("cur_hex", _as_float(operator.get("cur_hex")))
    vector[3] = _scale("damage", _as_float(candidate.get("damage")))
    vector[4] = _as_float(candidate.get("hit_prob"))
    vector[5] = _scale("attack_level", _as_float(candidate.get("attack_level")))
    vector[6] = _scale("target_blood", _as_float(candidate.get("target_blood")))
    vector[7] = _scale("target_obj_id", _as_float(candidate.get("target_obj_id")))
    vector[8] = _scale("weapon_id", _as_float(candidate.get("weapon_id")))
    vector[9] = float(len(candidate.get("move_path", []))) / 20.0
    if target_operator:
        vector[10] = _scale("blood", _as_float(target_operator.get("blood")))
        vector[11] = _scale("type", _as_float(target_operator.get("type")))
        vector[12] = _scale("sub_type", _as_float(target_operator.get("sub_type")))
        vector[13] = _scale("cur_hex", _as_float(target_operator.get("cur_hex")))
        vector[14] = _scale("speed", _as_float(target_operator.get("speed")))
    if city:
        vector[15] = _scale("value", _as_float(city.get("value")))
        vector[16] = _scale("cur_hex", _as_float(city.get("coord")))
        vector[17] = _as_float(city.get("flag"))
    return vector


def mean_feature(features: Iterable[np.ndarray]) -> np.ndarray:
    items = list(features)
    if not items:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    return np.mean(np.stack(items), axis=0)
