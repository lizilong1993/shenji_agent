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


GRAPH_FEATURE_FIELDS_73 = [
    ("op_id", "obj_id"),
    ("color", "color"),
    ("basic_speed", "basic_speed"),
    ("A1", "A1"),
    ("stack", "stack"),
    ("guide_ability", "guide_ability"),
    ("value", "value"),
    ("cur_hex", "cur_hex"),
    ("cur_pos", "cur_pos"),
    ("speed", "speed"),
    ("flag_force_stop", "flag_force_stop"),
    ("stop", "stop"),
    ("blood", "blood"),
    ("tire_accumulate_time", "tire_accumulate_time"),
    ("keep", "keep"),
    ("on_board", "on_board"),
    ("alive_remain_time", "alive_remain_time"),
    ("weapon_unfold_state", "weapon_unfold_state"),
]


def operator_to_graph_feature(
    operator: Optional[Dict[str, Any]],
    *,
    frame: Optional[Dict[str, Any]] = None,
    fallback_pos: Optional[int] = None,
) -> np.ndarray:
    """Convert an operator to the 73-dim feature vector matching the StateGraph CSV format.

    This matches the column order used in WG-StateGraph encoded_nodes-features/*.csv
    training data, without scaling. Values are raw as in the training CSVs.
    """
    vector = np.zeros(FEATURE_DIM, dtype=np.float32)
    op = operator or {}

    for idx, (csv_col, op_field) in enumerate(GRAPH_FEATURE_FIELDS_73):
        raw = _as_float(op.get(op_field))
        if csv_col in ("cur_hex", "op_id") and raw == 0 and fallback_pos is not None:
            raw = float(fallback_pos)
        vector[idx] = raw

    vector[18] = 0.0  # stationary_count (not available from observation)
    vector[19] = 0.0  # altitude
    vector[20] = 0.0  # changing_altitude
    vector[21] = 0.0  # changing_altitude_remain_time

    # wid_* columns [22-35]: weapons carried by specific weapon ID
    weapon_ids = op.get("carry_weapon_ids")
    if isinstance(weapon_ids, list):
        known_wids = [37, 43, 84, 56, 73, 29, 35, 71, 74, 36, 54, 69, 76, 72]
        for j, wid in enumerate(known_wids):
            vector[22 + j] = 1.0 if wid in weapon_ids else 0.0

    # re_bullet_* columns [36-39]: remaining bullets by weapon ID
    remain_bullet = op.get("remain_bullet_nums", {})
    if isinstance(remain_bullet, dict):
        known_re_bullets = [37, 43, 84, 56]
        for j, wid in enumerate(known_re_bullets):
            vector[36 + j] = float(remain_bullet.get(str(wid), 0))

    # be_seen_bullet_* columns [40-43]: bullet types visible
    see_enemy = op.get("see_enemy_bop_ids", [])
    if isinstance(see_enemy, list):
        known_bullets = [0, 100, 101, 102]
        for j, bid in enumerate(known_bullets):
            vector[40 + j] = 1.0 if bid in see_enemy else 0.0

    # type_* one-hot [44-46]
    op_type = int(_as_float(op.get("type")))
    vector[44 + min(op_type, 2)] = 1.0

    # name_* one-hot [47-53] — operator name is typically the type/subtype string
    vector[47] = 1.0  # default name_0

    # subtype_* one-hot [54-59]
    op_subtype = int(_as_float(op.get("sub_type")))
    vector[54 + min(op_subtype, 5)] = 1.0

    # armor_* binned [60-64]
    armor_val = _as_float(op.get("armor"))
    armor_bin = min(int(armor_val / 50), 4)
    vector[60 + armor_bin] = 1.0

    # move_state_* one-hot [65-68]
    move_state = int(_as_float(op.get("move_state")))
    vector[65 + min(move_state, 3)] = 1.0

    # tire_* one-hot [69-70]
    tire_val = _as_float(op.get("tire"))
    tire_bin = 0 if tire_val < 50 else 1
    vector[69 + min(tire_bin, 1)] = 1.0

    # Derived count fields [71-73]
    move_path = op.get("move_path")
    vector[70] = float(len(move_path)) if isinstance(move_path, (list, tuple)) else 0.0
    passenger_ids = op.get("passenger_ids")
    vector[71] = float(len(passenger_ids)) if isinstance(passenger_ids, (list, tuple)) else 0.0
    see_enemy_bop = op.get("see_enemy_bop_ids")
    vector[72] = float(len(see_enemy_bop)) if isinstance(see_enemy_bop, (list, tuple)) else 0.0

    return vector


def mean_feature(features: Iterable[np.ndarray]) -> np.ndarray:
    items = list(features)
    if not items:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    return np.mean(np.stack(items), axis=0)


def build_feature_similarity_graph(features: np.ndarray, top_k: int = 4) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape={features.shape}")
    num_nodes = features.shape[0]
    adjacency = np.eye(num_nodes, dtype=np.float32)
    if num_nodes <= 1:
        return adjacency
    similarity = features @ features.T
    np.fill_diagonal(similarity, 0.0)
    neighbor_count = min(max(top_k, 1), num_nodes - 1)
    for row_idx in range(num_nodes):
        neighbor_idx = np.argsort(similarity[row_idx])[-neighbor_count:]
        adjacency[row_idx, neighbor_idx] = 1.0
    return adjacency
