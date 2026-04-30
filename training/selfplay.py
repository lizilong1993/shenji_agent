from __future__ import annotations

import argparse
import base64
from collections import Counter
import importlib
import json
import logging
import os
import pickle
import pwd
import random
import sys
import types
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .common import dump_yaml, ensure_dir, git_head_sha, git_show_text, load_config, now_tag, repo_root, save_json, set_seed
from .evaluate import promotion_decision, summarize_matches
from .maps import (
    MapRecord,
    build_map_catalog,
    infer_map_id_from_scenario_id,
    infer_map_id_from_scenario_with_fallback,
    map_bucket_lookup,
)

RED, BLUE = 0, 1
NORMAL_BATTLE_ACTIONS = {"Move", "Shoot", "Occupy"}
DEPLOYMENT_ACTION_TYPES = {"303", "304", "314", "315", "316", "333"}
ENGINE_CONFIG_RELATIVE_PATH = Path(
    ".local/lib/python3.10/site-packages/train_env/env/authenticate/.engine_config"
)
_SNAPSHOT_ROLE_TO_PROFILE = {
    "bootstrap_rule_baseline": "baseline",
    "accepted_learned_mainline": "incumbent",
}
_TRAIN_ENV_CLASS: type[Any] | None = None
_TRAIN_ENV_BACKEND: str | None = None
_AGENT_CLASS_CACHE: Dict[str, type[Any]] = {}
_AGENT_RUNTIME_LOG_LEVEL = logging.CRITICAL


def _configure_torch_cpu_inference_threads(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Dict[str, int] | None:
    try:
        torch = import_module("torch")
    except ImportError:
        return None

    if not hasattr(torch, "inference_mode") and hasattr(torch, "no_grad"):
        torch.inference_mode = torch.no_grad  # type: ignore[attr-defined]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return {
        "num_threads": int(torch.get_num_threads()),
        "num_interop_threads": int(torch.get_num_interop_threads()),
    }


def _resolve_train_env_class(
    import_module: Callable[[str], Any] = importlib.import_module,
) -> Tuple[type[Any], str]:
    for module_name, class_name in (
        ("land_wargame_train_env", "TrainEnv"),
        ("train_env", "TrainEnv"),
    ):
        try:
            module = import_module(module_name)
        except ImportError:
            continue
        env_class = getattr(module, class_name, None)
        if env_class is not None:
            return env_class, f"{module_name}.{class_name}"

    mock_module = import_module("mock_train_env")
    return mock_module.MockTrainEnv, "mock_train_env.MockTrainEnv"


def _train_env_runtime() -> Tuple[type[Any], str]:
    global _TRAIN_ENV_CLASS, _TRAIN_ENV_BACKEND
    if _TRAIN_ENV_CLASS is None or _TRAIN_ENV_BACKEND is None:
        _TRAIN_ENV_CLASS, _TRAIN_ENV_BACKEND = _resolve_train_env_class()
    return _TRAIN_ENV_CLASS, _TRAIN_ENV_BACKEND


_BASIC_MAP_FIELD_ALIASES = {
    "neighbors": ("nb", None),
    "roads": ("rd", [0, 0, 0, 0, 0, 0]),
    "rivers": ("rv", [0, 0, 0, 0, 0, 0]),
    "cond": ("cd", 0),
    "elev": ("e", None),
}


def _normalize_basic_map_schema(basic_data: object, *, map_id: int, basic_path: Path) -> Dict[str, object]:
    if not isinstance(basic_data, dict):
        raise ValueError(f"map_id={map_id} basic.json schema error at {basic_path}: expected top-level object")

    map_data = basic_data.get("map_data")
    if not isinstance(map_data, list):
        raise ValueError(f"map_id={map_id} basic.json schema error at {basic_path}: expected map_data to be a 2D list")

    normalized_rows: List[List[Dict[str, object]]] = []
    for row_idx, row in enumerate(map_data):
        if not isinstance(row, list):
            raise ValueError(
                f"map_id={map_id} basic.json schema error at {basic_path}: expected row {row_idx} to be a list"
            )
        normalized_row: List[Dict[str, object]] = []
        for col_idx, cell in enumerate(row):
            if not isinstance(cell, dict):
                raise ValueError(
                    f"map_id={map_id} basic.json schema error at {basic_path}: expected cell[{row_idx}][{col_idx}] to be an object"
                )
            normalized_cell = dict(cell)
            missing_fields: List[str] = []
            for field_name, (legacy_field_name, default_value) in _BASIC_MAP_FIELD_ALIASES.items():
                value = normalized_cell.get(field_name)
                if value is None:
                    value = normalized_cell.get(legacy_field_name)
                if value is None and default_value is not None:
                    value = list(default_value) if isinstance(default_value, list) else default_value
                if value is None:
                    missing_fields.append(field_name)
                    continue
                normalized_cell[field_name] = value
            if missing_fields:
                missing_text = ", ".join(missing_fields)
                raise ValueError(
                    f"map_id={map_id} basic.json schema error at {basic_path}: cell[{row_idx}][{col_idx}] missing fields: {missing_text}"
                )
            normalized_row.append(normalized_cell)
        normalized_rows.append(normalized_row)

    normalized_basic_data = dict(basic_data)
    normalized_basic_data["map_data"] = normalized_rows
    return normalized_basic_data


def _load_case_data(scenario_path: Path, map_id: int) -> Dict[str, object]:
    data_root = repo_root() / "land_wargame_sdk" / "Data" / "Data"
    with scenario_path.open("r", encoding="utf-8") as fh:
        scenario_data = json.load(fh)
    basic_path = data_root / "maps" / f"map_{map_id}" / "basic.json"

    with basic_path.open("r", encoding="utf-8") as fh:
        basic_data = _normalize_basic_map_schema(json.load(fh), map_id=map_id, basic_path=basic_path)
    with (data_root / "maps" / f"map_{map_id}" / "cost.pickle").open("rb") as fh:
        cost_data = pickle.load(fh)
    see_path = data_root / "maps" / f"map_{map_id}" / f"{map_id}see.npz"
    see_data = np.load(see_path)["data"] if see_path.exists() else None
    return {
        "scenario_data": scenario_data,
        "basic_data": basic_data,
        "cost_data": cost_data,
        "see_data": see_data,
    }


def _side_snapshot(observation: Dict[str, Any], faction: int) -> Dict[str, object]:
    cities = observation.get("cities", [])
    operators = observation.get("operators", [])
    controlled_value = 0.0
    controlled_count = 0
    neutral_count = 0
    enemy_count = 0
    for city in cities:
        if not isinstance(city, dict):
            continue
        flag = city.get("flag")
        if flag == faction:
            controlled_count += 1
            controlled_value += float(city.get("value", 0.0) or 0.0)
        elif flag in (0, 1):
            enemy_count += 1
        else:
            neutral_count += 1
    friendly_blood = 0.0
    enemy_blood = 0.0
    friendly_alive = 0
    enemy_alive = 0
    for operator in operators:
        if not isinstance(operator, dict):
            continue
        blood = float(operator.get("blood", 0.0) or 0.0)
        alive = blood > 0.0
        if operator.get("color") == faction:
            friendly_blood += blood
            friendly_alive += int(alive)
        else:
            enemy_blood += blood
            enemy_alive += int(alive)
    return {
        "score": float(observation.get("reward", 0.0) or 0.0),
        "city_control_count": int(controlled_count),
        "city_value_controlled": float(controlled_value),
        "neutral_city_count": int(neutral_count),
        "enemy_city_count": int(enemy_count),
        "friendly_blood": float(friendly_blood),
        "enemy_blood": float(enemy_blood),
        "friendly_alive": int(friendly_alive),
        "enemy_alive": int(enemy_alive),
    }


def _snapshot_delta(before: Dict[str, object], after: Dict[str, object]) -> Dict[str, object]:
    return {
        "score_before": float(before.get("score", 0.0)),
        "score_after": float(after.get("score", 0.0)),
        "score_delta": float(after.get("score", 0.0)) - float(before.get("score", 0.0)),
        "city_value_controlled_before": float(before.get("city_value_controlled", 0.0)),
        "city_value_controlled_after": float(after.get("city_value_controlled", 0.0)),
        "city_value_delta": float(after.get("city_value_controlled", 0.0))
        - float(before.get("city_value_controlled", 0.0)),
        "damage_dealt": max(0.0, float(before.get("enemy_blood", 0.0)) - float(after.get("enemy_blood", 0.0))),
        "damage_taken": max(0.0, float(before.get("friendly_blood", 0.0)) - float(after.get("friendly_blood", 0.0))),
        "kill_dealt": max(0, int(before.get("enemy_alive", 0)) - int(after.get("enemy_alive", 0))),
        "kill_taken": max(0, int(before.get("friendly_alive", 0)) - int(after.get("friendly_alive", 0))),
    }


def _action_side(action: Dict[str, Any]) -> int | None:
    actor = action.get("actor")
    if actor == 1:
        return RED
    if actor == 11:
        return BLUE
    return None


def _action_type_name(action: Dict[str, Any]) -> str:
    action_type = action.get("type")
    try:
        from ai.agent import ACTION_TYPE_NAMES

        return str(ACTION_TYPE_NAMES.get(int(action_type), action_type))
    except Exception:
        return str(action_type)


def _city_owner_by_coord(observation: Dict[str, Any]) -> Dict[int, int]:
    owners: Dict[int, int] = {}
    for city in observation.get("cities", []):
        if not isinstance(city, dict) or city.get("coord") is None:
            continue
        owners[int(city["coord"])] = int(city.get("flag", -99))
    return owners


def _operator_by_id(observation: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {
        int(operator["obj_id"]): operator
        for operator in observation.get("operators", [])
        if isinstance(operator, dict) and operator.get("obj_id") is not None
    }


def _current_mac_addresses() -> List[str]:
    addresses: List[str] = []
    for path in sorted(Path("/sys/class/net").glob("*/address")):
        try:
            address = path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if address and address != "00:00:00:00:00:00":
            addresses.append(address)
    return addresses


def _repo_owner_home() -> Path | None:
    try:
        owner = pwd.getpwuid(repo_root().stat().st_uid)
    except (KeyError, OSError):
        return None
    return Path(owner.pw_dir)


def _default_engine_config_path() -> Path:
    env_path = os.environ.get("SDK_ENGINE_CONFIG")
    if env_path:
        return Path(env_path).expanduser()

    home_path = Path.home() / ENGINE_CONFIG_RELATIVE_PATH
    if home_path.exists():
        return home_path

    owner_home = _repo_owner_home()
    if owner_home and owner_home != Path.home():
        owner_path = owner_home / ENGINE_CONFIG_RELATIVE_PATH
        if owner_path.exists():
            return owner_path

    return home_path


def _decode_engine_config(config_path: Path | None = None) -> Dict[str, object]:
    path = config_path or _default_engine_config_path()
    if not path.exists():
        return {"path": str(path), "exists": False, "bound_mac": None, "timestamp": None}
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    decoded = ""
    try:
        decoded = base64.b64decode(raw).decode("utf-8", errors="replace")
    except Exception:
        return {"path": str(path), "exists": True, "bound_mac": None, "timestamp": None, "decode_error": True}
    parts = decoded.split(",", 1)
    return {
        "path": str(path),
        "exists": True,
        "bound_mac": parts[0].lower() if parts and parts[0] else None,
        "timestamp": parts[1] if len(parts) > 1 else None,
    }


def _sdk_auth_status(env_backend: str) -> Dict[str, object]:
    if "train_env" not in env_backend:
        return {"status": "not_applicable", "env_backend": env_backend}
    config = _decode_engine_config()
    current_macs = _current_mac_addresses()
    bound_mac = config.get("bound_mac")
    if bound_mac and bound_mac not in current_macs:
        status = "failed_mac_mismatch"
    elif bound_mac and bound_mac in current_macs:
        status = "mac_match_unverified"
    elif not config.get("exists"):
        status = "engine_config_missing"
    else:
        status = "unknown"
    return {
        "status": status,
        "env_backend": env_backend,
        "engine_config": config,
        "current_macs": current_macs,
    }


def _valid_action_type_counts(observation: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for action_map in (observation.get("valid_actions") or {}).values():
        if not isinstance(action_map, dict):
            continue
        for action_type in action_map:
            action_name = _action_type_name({"type": action_type})
            counts[action_name] = counts.get(action_name, 0) + 1
    return dict(sorted(counts.items()))


def _phase_name(observation: Dict[str, Any]) -> str:
    stage = observation.get("time", {}).get("stage")
    if stage == 0:
        return "env_config"
    if stage == 1:
        return "deployment"
    if stage == 2:
        return "battle"
    return "unknown"


def _score_from_observation(observation: Dict[str, Any], side: int) -> float:
    reward = observation.get("reward")
    if reward is not None:
        return float(reward or 0.0)
    scores = observation.get("scores", {})
    if isinstance(scores, dict):
        key = "red_total" if side == RED else "blue_total"
        if key in scores:
            return float(scores.get(key, 0.0) or 0.0)
    return 0.0


def _selected_action_counts(actions: List[Dict[str, Any]], seat: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for action in actions:
        if not isinstance(action, dict) or action.get("actor") != seat:
            continue
        name = _action_type_name(action)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _selected_action_source(selected_counts: Dict[str, int], phase: str) -> str:
    if not selected_counts:
        return "none"
    if phase == "deployment" and set(selected_counts).issubset(DEPLOYMENT_ACTION_TYPES):
        return "deployment_stage_branch"
    return "agent_valid_action_policy"


def _no_battle_action_reason(
    *,
    phase: str,
    valid_counts: Dict[str, int],
    selected_counts: Dict[str, int],
    auth_status: str,
) -> str | None:
    battle_valid = sum(int(valid_counts.get(name, 0)) for name in NORMAL_BATTLE_ACTIONS)
    battle_selected = sum(int(selected_counts.get(name, 0)) for name in NORMAL_BATTLE_ACTIONS)
    if phase != "battle" and auth_status.startswith("failed"):
        return "auth_failed_or_unverified_stage_stuck"
    if phase != "battle":
        return "deployment_or_setup_phase"
    if battle_valid <= 0:
        return "no_move_shoot_occupy_valid_actions"
    if battle_selected <= 0:
        return "no_move_shoot_occupy_selected_actions"
    return None


def _build_phase_action_records(
    *,
    match_id: str,
    turn: int,
    before_state: List[Dict[str, Any]],
    after_state: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
    done: bool,
    env_backend: str,
    auth_status: Dict[str, object],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    auth_state = str(auth_status.get("status", "unknown"))
    for side, seat in ((RED, 1), (BLUE, 11)):
        before_obs = before_state[side]
        after_obs = after_state[side]
        valid_counts = _valid_action_type_counts(before_obs)
        selected_counts = _selected_action_counts(actions, seat)
        phase = _phase_name(before_obs)
        stage = before_obs.get("time", {}).get("stage")
        after_stage = after_obs.get("time", {}).get("stage")
        setup_count = sum(int(selected_counts.get(name, 0)) for name in DEPLOYMENT_ACTION_TYPES)
        selected_type = ",".join(f"{name}:{count}" for name, count in selected_counts.items()) or None
        records.append(
            {
                "match_id": match_id,
                "turn": int(turn),
                "side": int(side),
                "env_backend": env_backend,
                "auth_status": auth_state,
                "phase": phase,
                "stage": stage,
                "step_type": "battle" if phase == "battle" else "setup_or_deployment",
                "observation_keys": sorted(str(key) for key in before_obs.keys()),
                "valid_action_count": int(sum(valid_counts.values())),
                "valid_action_types": sorted(valid_counts),
                "valid_action_type_counts": valid_counts,
                "setup_action_count": int(setup_count),
                "deployment_action_count": int(setup_count),
                "move_action_count": int(valid_counts.get("Move", 0)),
                "shoot_action_count": int(valid_counts.get("Shoot", 0)),
                "occupy_action_count": int(valid_counts.get("Occupy", 0)),
                "other_action_count": int(
                    sum(count for name, count in valid_counts.items() if name not in NORMAL_BATTLE_ACTIONS)
                ),
                "selected_action_type": selected_type,
                "selected_action_type_counts": selected_counts,
                "selected_action_source": _selected_action_source(selected_counts, phase),
                "action_payload_valid": bool(selected_counts),
                "env_step_returned_done": bool(done),
                "env_step_returned_score": _score_from_observation(after_obs, side),
                "score_before": _score_from_observation(before_obs, side),
                "score_after": _score_from_observation(after_obs, side),
                "phase_transition": f"{stage}->{after_stage}",
                "no_battle_action_reason": _no_battle_action_reason(
                    phase=phase,
                    valid_counts=valid_counts,
                    selected_counts=selected_counts,
                    auth_status=auth_state,
                ),
            }
        )
    return records


def _build_step_objective_records(
    *,
    match_id: str,
    turn: int,
    before_state: List[Dict[str, Any]],
    after_state: List[Dict[str, Any]],
    actions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    before_red = _side_snapshot(before_state[RED], RED)
    before_blue = _side_snapshot(before_state[BLUE], BLUE)
    after_red = _side_snapshot(after_state[RED], RED)
    after_blue = _side_snapshot(after_state[BLUE], BLUE)
    red_delta = _snapshot_delta(before_red, after_red)
    blue_delta = _snapshot_delta(before_blue, after_blue)
    progress_record = {
        "match_id": match_id,
        "turn": int(turn),
        "red": {"side": RED, **red_delta, **after_red},
        "blue": {"side": BLUE, **blue_delta, **after_blue},
    }
    before_owners = _city_owner_by_coord(before_state[RED])
    after_owners = _city_owner_by_coord(after_state[RED])
    before_ops = _operator_by_id(before_state[RED])
    outcome_records: List[Dict[str, object]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        side = _action_side(action)
        side_delta = red_delta if side == RED else blue_delta if side == BLUE else {}
        obj_id = action.get("obj_id")
        operator = before_ops.get(int(obj_id)) if obj_id is not None else None
        current_hex = int(operator.get("cur_hex")) if operator and operator.get("cur_hex") is not None else None
        target_hex = None
        if action.get("move_path"):
            target_hex = int(action["move_path"][-1])
        elif action.get("jm_pos") is not None:
            target_hex = int(action["jm_pos"])
        elif current_hex is not None:
            target_hex = current_hex
        owner_before = before_owners.get(target_hex) if target_hex is not None else None
        owner_after = after_owners.get(target_hex) if target_hex is not None else None
        action_name = _action_type_name(action)
        outcome_records.append(
            {
                "match_id": match_id,
                "turn": int(turn),
                "side": side,
                "operator_id": obj_id,
                "operator_type": operator.get("type") if operator else None,
                "current_hex": current_hex,
                "action_type": action_name,
                "target_hex": target_hex,
                "target_city_id": target_hex if target_hex in before_owners else None,
                "target_owner_before": owner_before,
                "target_owner_after": owner_after,
                "occupy_available": action_name == "Occupy",
                "occupy_selected": action_name == "Occupy",
                "occupy_success": action_name == "Occupy" and owner_before != owner_after and owner_after == side,
                **side_delta,
            }
        )
    city_timeline = {
        "match_id": match_id,
        "turn": int(turn),
        "red_city_control_count": after_red["city_control_count"],
        "blue_city_control_count": after_blue["city_control_count"],
        "red_city_value_controlled": after_red["city_value_controlled"],
        "blue_city_value_controlled": after_blue["city_value_controlled"],
        "neutral_city_count": after_red["neutral_city_count"],
    }
    return [progress_record], outcome_records, city_timeline


def _run_single(
    scenario_path: Path,
    map_id: int,
    red_profile: Dict[str, Any],
    blue_profile: Dict[str, Any],
    max_steps: int,
    heartbeat: Callable[[int], None] | None = None,
    match_id: str | None = None,
) -> Tuple[float, float, Dict[str, object], Dict[str, object]]:
    env_class, env_backend = _train_env_runtime()
    env = env_class()
    auth_status = _sdk_auth_status(env_backend)
    shared = _load_case_data(scenario_path, map_id)
    player_info = [
        {"seat": 1, "faction": 0, "role": 1, "user_name": red_profile["user_name"], "user_id": 1},
        {"seat": 11, "faction": 1, "role": 1, "user_name": blue_profile["user_name"], "user_id": 11},
    ]
    env_info = {**shared, "player_info": player_info}
    state = env.setup(env_info)
    red = red_profile["agent_class"](log_level=_AGENT_RUNTIME_LOG_LEVEL)
    blue = blue_profile["agent_class"](log_level=_AGENT_RUNTIME_LOG_LEVEL)
    red.setup({**shared, "seat": 1, "faction": 0, "role": 1, "user_name": red_profile["user_name"], "policy_profile": red_profile["profile"], "policy_artifacts_dir": red_profile["artifacts"], "state": state})
    blue.setup({**shared, "seat": 11, "faction": 1, "role": 1, "user_name": blue_profile["user_name"], "policy_profile": blue_profile["profile"], "policy_artifacts_dir": blue_profile["artifacts"], "state": state})
    done = False
    steps = 0
    objective_progress_turns: List[Dict[str, object]] = []
    action_outcomes: List[Dict[str, object]] = []
    city_control_timeline: List[Dict[str, object]] = []
    phase_action_records: List[Dict[str, object]] = []
    diagnostic_match_id = match_id or scenario_path.name
    while not done and steps < max_steps:
        before_state = deepcopy(state)
        actions = red.step(state[RED]) + blue.step(state[BLUE])
        state, done = env.step(actions)
        phase_action_records.extend(
            _build_phase_action_records(
                match_id=diagnostic_match_id,
                turn=steps,
                before_state=before_state,
                after_state=state,
                actions=actions,
                done=done,
                env_backend=env_backend,
                auth_status=auth_status,
            )
        )
        progress, outcomes, city_timeline = _build_step_objective_records(
            match_id=diagnostic_match_id,
            turn=steps,
            before_state=before_state,
            after_state=state,
            actions=actions,
        )
        objective_progress_turns.extend(progress)
        action_outcomes.extend(outcomes)
        city_control_timeline.append(city_timeline)
        steps += 1
        if heartbeat is not None:
            heartbeat(steps)
    red_score = float(state[RED].get("reward", 0.0))
    blue_score = float(state[BLUE].get("reward", 0.0))
    red_stats = red.get_policy_stats()
    blue_stats = blue.get_policy_stats()
    for stats in (red_stats, blue_stats):
        if stats:
            stats["objective_progress_turns"] = objective_progress_turns
            stats["action_outcomes"] = action_outcomes
            stats["city_control_timeline"] = city_control_timeline
            stats["phase_action_records"] = phase_action_records
            stats["auth_status"] = auth_status
    return red_score, blue_score, red_stats, blue_stats


def _paired_score(candidate_a: float, opponent_a: float, candidate_b: float, opponent_b: float) -> float:
    def outcome(left: float, right: float) -> float:
        if left > right:
            return 1.0
        if left < right:
            return 0.0
        return 0.5

    return (outcome(candidate_a, opponent_a) + outcome(candidate_b, opponent_b)) / 2.0


def _collect_scenario_candidates(
    scenario_root: Path,
    bucket_lookup: Dict[int, str],
    records: Iterable[MapRecord] | None = None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    record_list = list(records) if records is not None else None
    for scenario_path in sorted(scenario_root.glob("*.json")):
        scenario_id = scenario_path.stem
        if record_list is None:
            map_id = infer_map_id_from_scenario_id(scenario_id, bucket_lookup.keys())
        else:
            with scenario_path.open("r", encoding="utf-8") as fh:
                scenario_data = json.load(fh)
            map_id = infer_map_id_from_scenario_with_fallback(scenario_id, scenario_data, record_list)
        if map_id is None:
            continue
        candidates.append(
            {
                "scenario": scenario_path.name,
                "map_id": int(map_id),
                "bucket": str(bucket_lookup.get(map_id, "medium")),
            }
        )
    return candidates


def _build_paired_map_list(scenario_candidates: List[Dict[str, Any]], paired_matches: int) -> Dict[str, Any]:
    if paired_matches <= 0:
        raise SystemExit("paired_matches must be positive for self-play evaluation.")
    if not scenario_candidates:
        raise SystemExit("No valid scenarios found under land_wargame_sdk/Data/Data/scenarios for self-play evaluation.")

    unique_prefix = random.sample(scenario_candidates, k=min(paired_matches, len(scenario_candidates)))
    ordered_items = list(unique_prefix)
    duplicates_used = paired_matches > len(unique_prefix)
    while len(ordered_items) < paired_matches:
        ordered_items.append(random.choice(scenario_candidates))

    return {
        "sampling_protocol": "unique_first_then_replacement",
        "scenario_source": "land_wargame_sdk/Data/Data/scenarios/*.json",
        "requested_pairs": paired_matches,
        "candidate_pool_size": len(scenario_candidates),
        "unique_item_count_before_replacement": min(paired_matches, len(scenario_candidates)),
        "duplicates_used": duplicates_used,
        "items": [
            {
                "scenario": str(item["scenario"]),
                "map_id": int(item["map_id"]),
                "bucket": str(item["bucket"]),
                "order_index": order_index,
            }
            for order_index, item in enumerate(ordered_items)
        ],
    }


def _bootstrap_league(league_path: Path) -> Dict[str, object]:
    if league_path.exists():
        return json.loads(league_path.read_text(encoding="utf-8"))
    state = {
        "champion": "ai/weights",
        "historical_mains": [],
        "exploiters": ["aggressive_exploiter", "occupy_exploiter"],
        "specialists": [],
        "consecutive_windows": 0,
    }
    save_json(state, league_path)
    return state


def _load_artifact_manifest(artifacts: str) -> Dict[str, Any]:
    manifest_path = Path(artifacts) / "model_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_agent_class_from_commit(commit_sha: str) -> type[Any]:
    cached = _AGENT_CLASS_CACHE.get(commit_sha)
    if cached is not None:
        return cached

    package_name = f"_selfplay_snapshot_{commit_sha[:12]}"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = []  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package

    for module_name in ("features", "models", "policy", "agent"):
        qualified_name = f"{package_name}.{module_name}"
        if qualified_name in sys.modules:
            continue
        if module_name == "models":
            live_models = importlib.import_module("ai.models")
            module = types.ModuleType(qualified_name)
            module.__file__ = getattr(live_models, "__file__", f"live:ai.models")
            module.__package__ = package_name
            module.__dict__.update(live_models.__dict__)
            sys.modules[qualified_name] = module
            continue
        source = git_show_text(f"{commit_sha}:ai/{module_name}.py", cwd=repo_root())
        module = types.ModuleType(qualified_name)
        module.__file__ = f"{commit_sha}:ai/{module_name}.py"
        module.__package__ = package_name
        sys.modules[qualified_name] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)

    agent_class = getattr(sys.modules[f"{package_name}.agent"], "Agent")
    _AGENT_CLASS_CACHE[commit_sha] = agent_class
    return agent_class


def _resolve_source_commit_sha(artifacts: str, fallback_commit_sha: str) -> str:
    manifest = _load_artifact_manifest(artifacts)
    commit_sha = str(manifest.get("source_commit_sha", "")).strip()
    return commit_sha or str(fallback_commit_sha)


def _build_participant_spec(artifacts: str, profile: str, fallback_commit_sha: str, user_name: str) -> Dict[str, Any]:
    source_commit_sha = _resolve_source_commit_sha(artifacts, fallback_commit_sha)
    return {
        "artifacts": artifacts,
        "profile": profile,
        "user_name": user_name,
        "source_commit_sha": source_commit_sha,
        "code_origin": f"git:{source_commit_sha}",
        "agent_class": _load_agent_class_from_commit(source_commit_sha),
    }


def _heartbeat_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _progress_current_fields(current_match: Dict[str, object] | None) -> Dict[str, object]:
    if current_match is None:
        return {
            "current_order_index": None,
            "current_map_id": None,
            "current_scenario": None,
            "current_bucket": None,
            "current_leg_index": None,
            "leg_steps": None,
        }
    return {
        "current_order_index": current_match.get("order_index"),
        "current_map_id": current_match.get("map_id"),
        "current_scenario": current_match.get("scenario"),
        "current_bucket": current_match.get("bucket"),
        "current_leg_index": current_match.get("current_leg_index"),
        "leg_steps": current_match.get("leg_steps"),
    }


def _build_inflight_match(
    *,
    map_item: Dict[str, object],
    current_leg_index: int,
    leg_steps: int,
    candidate_artifacts: str,
    opponent_artifacts: str,
    candidate: Dict[str, Any],
    incumbent: Dict[str, Any],
    candidate_profile: str,
    incumbent_profile: str,
    first_leg: Dict[str, float] | None = None,
    second_leg: Dict[str, float] | None = None,
) -> Dict[str, object]:
    return {
        "status": "in_progress",
        "scenario": str(map_item["scenario"]),
        "bucket": str(map_item["bucket"]),
        "map_id": int(map_item["map_id"]),
        "order_index": int(map_item["order_index"]),
        "current_leg_index": int(current_leg_index),
        "leg_steps": int(leg_steps),
        "first_leg": first_leg,
        "second_leg": second_leg,
        "candidate_artifacts": candidate_artifacts,
        "opponent_artifacts": opponent_artifacts,
        "candidate_source_commit_sha": candidate["source_commit_sha"],
        "opponent_source_commit_sha": incumbent["source_commit_sha"],
        "candidate_code_origin": candidate["code_origin"],
        "opponent_code_origin": incumbent["code_origin"],
        "candidate_profile": candidate_profile,
        "opponent_profile": incumbent_profile,
    }


def _save_progress_snapshot(
    run_dir: Path,
    *,
    env_backend: str,
    matches: List[Dict[str, object]],
    total_pairs: int,
    candidate: Dict[str, Any],
    incumbent: Dict[str, Any],
    phase: str = "formal_compare",
    current_match: Dict[str, object] | None = None,
    status: str | None = None,
    error: Dict[str, object] | None = None,
) -> None:
    heartbeat_ts = _heartbeat_timestamp()
    progress_payload = {
        "env_backend": env_backend,
        "phase": phase,
        "completed_pairs": len(matches),
        "total_pairs": total_pairs,
        "candidate_source_commit_sha": candidate["source_commit_sha"],
        "incumbent_source_commit_sha": incumbent["source_commit_sha"],
        "heartbeat_ts": heartbeat_ts,
        **_progress_current_fields(current_match),
    }
    partial_payload = {
        "env_backend": env_backend,
        "phase": phase,
        "completed_pairs": len(matches),
        "total_pairs": total_pairs,
        "heartbeat_ts": heartbeat_ts,
        "matches": matches,
        "current_match": current_match,
    }
    if status is not None:
        progress_payload["status"] = status
        partial_payload["status"] = status
    if error is not None:
        progress_payload["error"] = error
        partial_payload["error"] = error
    save_json(progress_payload, run_dir / "progress.json")
    save_json(partial_payload, run_dir / "partial_match_results.json")


def _write_participants_sidecar(
    run_dir: Path,
    *,
    env_backend: str,
    candidate_artifacts: str,
    incumbent_artifacts: str,
    candidate_profile: str,
    incumbent_profile: str,
    candidate: Dict[str, Any],
    incumbent: Dict[str, Any],
) -> None:
    save_json(
        {
            "env_backend": env_backend,
            "candidate": _participant_payload(
                candidate_artifacts,
                candidate_profile,
                candidate["source_commit_sha"],
                candidate["code_origin"],
                user_name=str(candidate.get("user_name", "")),
            ),
            "incumbent": _participant_payload(
                incumbent_artifacts,
                incumbent_profile,
                incumbent["source_commit_sha"],
                incumbent["code_origin"],
                user_name=str(incumbent.get("user_name", "")),
            ),
        },
        run_dir / "participants.json",
    )


def _compare_abort_status(exc: BaseException) -> str:
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted"
    return "failed"


def _compare_abort_error(exc: BaseException, *, status: str) -> Dict[str, object]:
    message = str(exc).strip()
    if not message:
        message = f"compare {status}"
    return {
        "type": type(exc).__name__,
        "message": message,
        "raised_at": _heartbeat_timestamp(),
    }


def _finalize_current_match(
    current_match: Dict[str, object] | None,
    *,
    status: str,
    error: Dict[str, object],
) -> Dict[str, object] | None:
    if current_match is None:
        return None
    finalized = dict(current_match)
    finalized["status"] = status
    finalized["ended_at"] = error["raised_at"]
    finalized["error_type"] = error["type"]
    finalized["error_message"] = error["message"]
    return finalized


def _write_compare_abort_sidecars(
    run_dir: Path,
    *,
    env_backend: str,
    matches: List[Dict[str, object]],
    total_pairs: int,
    candidate_artifacts: str,
    incumbent_artifacts: str,
    candidate_profile: str,
    incumbent_profile: str,
    candidate: Dict[str, Any],
    incumbent: Dict[str, Any],
    current_match: Dict[str, object] | None,
    exc: BaseException,
) -> None:
    status = _compare_abort_status(exc)
    phase = f"formal_compare_{status}"
    error = _compare_abort_error(exc, status=status)
    terminal_current_match = _finalize_current_match(current_match, status=status, error=error)
    abort_payload = {
        "env_backend": env_backend,
        "phase": phase,
        "status": status,
        "completed_pairs": len(matches),
        "total_pairs": total_pairs,
        "matches": matches,
        "current_match": terminal_current_match,
        "error": error,
    }

    _save_progress_snapshot(
        run_dir,
        env_backend=env_backend,
        matches=matches,
        total_pairs=total_pairs,
        candidate=candidate,
        incumbent=incumbent,
        phase=phase,
        current_match=terminal_current_match,
        status=status,
        error=error,
    )
    _write_participants_sidecar(
        run_dir,
        env_backend=env_backend,
        candidate_artifacts=candidate_artifacts,
        incumbent_artifacts=incumbent_artifacts,
        candidate_profile=candidate_profile,
        incumbent_profile=incumbent_profile,
        candidate=candidate,
        incumbent=incumbent,
    )
    save_json(abort_payload, run_dir / "match_results.json")
    save_json(
        {
            "env_backend": env_backend,
            "phase": phase,
            "status": status,
            "completed_pairs": len(matches),
            "total_pairs": total_pairs,
            "error": error,
        },
        run_dir / "rating.json",
    )
    save_json(
        {
            "env_backend": env_backend,
            "phase": phase,
            "status": status,
            "accepted": False,
            "completed_pairs": len(matches),
            "total_pairs": total_pairs,
            "failures": [f"compare_{status}"],
            "error": error,
        },
        run_dir / "promotion.json",
    )
    save_json(
        {
            "env_backend": env_backend,
            "phase": phase,
            "status": status,
            "completed_pairs": len(matches),
            "total_pairs": total_pairs,
            "error": error,
        },
        run_dir / "exploitability.json",
    )

    summary_lines = [
        "# Shenji Self-Play Summary",
        "",
        f"- status: `{status}`",
        f"- phase: `{phase}`",
        f"- env_backend: `{env_backend}`",
        f"- completed_pairs: `{len(matches)}`",
        f"- total_pairs: `{total_pairs}`",
        f"- candidate_source_commit_sha: `{candidate['source_commit_sha']}`",
        f"- incumbent_source_commit_sha: `{incumbent['source_commit_sha']}`",
        f"- error_type: `{error['type']}`",
        f"- error_message: `{error['message']}`",
    ]
    if terminal_current_match is not None:
        summary_lines.extend(
            [
                f"- current_order_index: `{terminal_current_match.get('order_index')}`",
                f"- current_scenario: `{terminal_current_match.get('scenario')}`",
                f"- current_leg_index: `{terminal_current_match.get('current_leg_index')}`",
            ]
        )
    (run_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


def _resolve_incumbent_profile(artifacts: str, requested_profile: str | None, fallback_profile: str) -> str:
    if requested_profile:
        return str(requested_profile)
    manifest = _load_artifact_manifest(artifacts)
    snapshot_role = str(manifest.get("snapshot_role", "")).strip()
    return _SNAPSHOT_ROLE_TO_PROFILE.get(snapshot_role, str(fallback_profile))


def _participant_payload(
    artifacts: str,
    profile: str,
    source_commit_sha: str,
    code_origin: str,
    *,
    user_name: str | None = None,
) -> Dict[str, Any]:
    manifest = _load_artifact_manifest(artifacts)
    payload: Dict[str, Any] = {
        "artifacts": _resolved_path_string(artifacts),
        "profile": profile,
        "source_commit_sha": source_commit_sha,
        "code_origin": code_origin,
        "manifest_path": str((Path(artifacts) / "model_manifest.json").resolve()),
        "manifest_loaded": bool(manifest),
    }
    if user_name is not None:
        payload["user_name"] = user_name
    if manifest:
        payload["snapshot_role"] = manifest.get("snapshot_role")
        payload["manifest_generated_at"] = manifest.get("generated_at")
        payload["managed_files"] = manifest.get("managed_files", [])
    return payload


def _resolved_path_string(value: str) -> str:
    return str(Path(value).resolve())


def _manifest_path_string(artifacts: str) -> str:
    return str((Path(artifacts) / "model_manifest.json").resolve())


def _resolve_formal_compare_artifacts(
    config: Dict[str, object],
    *,
    candidate_arg: str | None,
    incumbent_arg: str | None,
) -> Tuple[str, str]:
    candidate = str(candidate_arg or config.get("candidate_artifacts") or "ai/weights")
    incumbent = str(incumbent_arg or config.get("incumbent_artifacts") or "ai/weights_incumbent_rule")
    return candidate, incumbent


def _policy_fallback_reasons(stats: Dict[str, object], *, profile: str, manifest_loaded: bool) -> List[str]:
    reasons = {
        str(trace.get("reason"))
        for trace in stats.get("trace_samples", [])
        if isinstance(trace, dict) and trace.get("reason")
    }
    intent_loaded = bool(stats.get("intent_model_loaded_count"))
    situation_loaded = bool(stats.get("situation_model_loaded_count"))
    if profile == "baseline":
        reasons.add("baseline_profile")
    elif not manifest_loaded:
        reasons.add("manifest_missing")
    elif not intent_loaded or not situation_loaded:
        reasons.add("policy_models_unavailable")
    return sorted(reasons)


def _build_match_policy_diagnostics(
    *,
    artifacts: str,
    participant: Dict[str, Any],
    profile: str,
    stats: Dict[str, object],
) -> Dict[str, object]:
    manifest = _load_artifact_manifest(artifacts)
    managed_files = manifest.get("managed_files", []) if isinstance(manifest, dict) else []
    intent_model_loaded = bool(stats.get("intent_model_loaded_count"))
    situation_model_loaded = bool(stats.get("situation_model_loaded_count"))
    learned_policy_requested = profile != "baseline"
    learned_policy_enabled = learned_policy_requested and intent_model_loaded and situation_model_loaded
    fallback_reasons = _policy_fallback_reasons(stats, profile=profile, manifest_loaded=bool(manifest))
    tactical_model_name = manifest.get("tactical_model") if isinstance(manifest, dict) else None
    tactical_model_path = Path(artifacts) / str(tactical_model_name or "tactical_model.pt")
    tactical_model_present = tactical_model_path.exists() or (
        isinstance(managed_files, list) and str(tactical_model_path.name) in managed_files
    )
    return {
        "user_name": participant["user_name"],
        "artifact_dir": _resolved_path_string(artifacts),
        "profile": profile,
        "manifest_path": _manifest_path_string(artifacts),
        "manifest_loaded": bool(manifest),
        "manifest_generated_at": manifest.get("generated_at") if isinstance(manifest, dict) else None,
        "manifest_source_commit_sha": str(manifest.get("source_commit_sha", "")).strip() if isinstance(manifest, dict) else "",
        "snapshot_role": manifest.get("snapshot_role") if isinstance(manifest, dict) else None,
        "managed_files": managed_files if isinstance(managed_files, list) else [],
        "intent_model_loaded": intent_model_loaded,
        "situation_model_loaded": situation_model_loaded,
        "tactical_model_present": tactical_model_present,
        "tactical_model_loaded": bool(stats.get("tactical_model_loaded_count", 0) > 0),
        "learned_policy_requested": learned_policy_requested,
        "learned_policy_enabled": learned_policy_enabled,
        "fallback_reason": fallback_reasons,
        "learned_path_coverage": float(stats.get("learned_path_coverage", 0.0)),
        "move_path_coverage": float(stats.get("move_path_coverage", 0.0)),
        "shoot_path_coverage": float(stats.get("shoot_path_coverage", 0.0)),
        "inference_devices": list(stats.get("inference_devices", [])),
        "loaded_from_dirs": list(stats.get("loaded_from_dirs", [])),
    }


def _merge_policy_stats(*payloads: Dict[str, object]) -> Dict[str, object]:
    merged = {
        "move": {"total": 0, "learned": 0},
        "shoot": {"total": 0, "learned": 0},
    }
    score_sums = {
        "move": {"samples": 0, "intent_bias": 0.0, "situation_score": 0.0, "tactical_score": 0.0, "objective_score": 0.0, "combined_score": 0.0},
        "shoot": {"samples": 0, "intent_bias": 0.0, "situation_score": 0.0, "tactical_score": 0.0, "objective_score": 0.0, "combined_score": 0.0},
    }
    traces: List[Dict[str, object]] = []
    policy_profiles = set()
    inference_devices = set()
    loaded_from_dirs = set()
    intent_model_loaded_count = 0
    situation_model_loaded_count = 0
    tactical_model_loaded_count = 0
    valid_action_type_counts: Dict[str, int] = {}
    generated_action_type_counts: Dict[str, int] = {}
    move_route_samples: List[Dict[str, object]] = []
    objective_progress_turns: List[Dict[str, object]] = []
    action_outcomes: List[Dict[str, object]] = []
    city_control_timeline: List[Dict[str, object]] = []
    phase_action_records: List[Dict[str, object]] = []
    auth_statuses: List[Dict[str, object]] = []
    move_route_agg_merged = {
        "total_attempts": 0,
        "empty_route_count": 0,
        "reachable_city_total": 0,
        "by_move_type": {},
        "route_failure_reasons": {},
    }
    for payload in payloads:
        counts = payload.get("counts", {}) if isinstance(payload, dict) else {}
        score_means = payload.get("score_means", {}) if isinstance(payload, dict) else {}
        for action_key in ("move", "shoot"):
            action_counts = counts.get(action_key, {}) if isinstance(counts, dict) else {}
            merged[action_key]["total"] += int(action_counts.get("total", 0))
            merged[action_key]["learned"] += int(action_counts.get("learned", 0))
            action_scores = score_means.get(action_key, {}) if isinstance(score_means, dict) else {}
            samples = int(action_scores.get("samples", 0))
            score_sums[action_key]["samples"] += samples
            for metric in ("intent_bias", "situation_score", "tactical_score", "objective_score", "combined_score"):
                score_sums[action_key][metric] += float(action_scores.get(metric, 0.0)) * samples
        if isinstance(payload, dict):
            traces.extend(payload.get("trace_samples", [])[:6])
            profile = payload.get("policy_profile")
            if profile:
                policy_profiles.add(str(profile))
            device = payload.get("device")
            if device:
                inference_devices.add(str(device))
            loaded_from = payload.get("loaded_from")
            if loaded_from:
                loaded_from_dirs.add(str(loaded_from))
            intent_model_loaded_count += int(bool(payload.get("intent_model_loaded")))
            situation_model_loaded_count += int(bool(payload.get("situation_model_loaded")))
            tactical_model_loaded_count += int(bool(payload.get("tactical_model_loaded")))
            for key, count in payload.get("valid_action_type_counts", {}).items():
                valid_action_type_counts[str(key)] = valid_action_type_counts.get(str(key), 0) + int(count)
            for key, count in payload.get("generated_action_type_counts", {}).items():
                generated_action_type_counts[str(key)] = generated_action_type_counts.get(str(key), 0) + int(count)
            route_agg = payload.get("move_route_aggregate", {})
            if isinstance(route_agg, dict):
                move_route_agg_merged["total_attempts"] += int(route_agg.get("total_attempts", 0))
                move_route_agg_merged["empty_route_count"] += int(route_agg.get("empty_route_count", 0))
                move_route_agg_merged["reachable_city_total"] += int(route_agg.get("reachable_city_total", 0))
                for mt, mt_data in route_agg.get("by_move_type", {}).items():
                    existing = move_route_agg_merged["by_move_type"].setdefault(str(mt), {"total": 0, "empty": 0})
                    if isinstance(mt_data, dict):
                        existing["total"] += int(mt_data.get("total", 0))
                        existing["empty"] += int(mt_data.get("empty", 0))
                for reason, count in route_agg.get("route_failure_reasons", {}).items():
                    r = str(reason)
                    move_route_agg_merged["route_failure_reasons"][r] = move_route_agg_merged["route_failure_reasons"].get(r, 0) + int(count)
            route_samples = payload.get("move_route_samples", [])
            if isinstance(route_samples, list):
                move_route_samples.extend(route_samples)
            progress_turns = payload.get("objective_progress_turns", [])
            if isinstance(progress_turns, list):
                objective_progress_turns.extend(progress_turns)
            outcomes = payload.get("action_outcomes", [])
            if isinstance(outcomes, list):
                action_outcomes.extend(outcomes)
            timeline = payload.get("city_control_timeline", [])
            if isinstance(timeline, list):
                city_control_timeline.extend(timeline)
            phase_records = payload.get("phase_action_records", [])
            if isinstance(phase_records, list):
                phase_action_records.extend(phase_records)
            auth_status = payload.get("auth_status")
            if isinstance(auth_status, dict):
                auth_statuses.append(auth_status)

    def coverage(total: int, learned: int) -> float:
        if total <= 0:
            return 0.0
        return float(learned / total)

    move_total = merged["move"]["total"]
    move_learned = merged["move"]["learned"]
    shoot_total = merged["shoot"]["total"]
    shoot_learned = merged["shoot"]["learned"]
    score_means = {}
    for action_key, values in score_sums.items():
        samples = int(values["samples"])
        score_means[action_key] = {
            "samples": samples,
            "intent_bias": float(values["intent_bias"] / samples) if samples else 0.0,
                "situation_score": float(values["situation_score"] / samples) if samples else 0.0,
                "tactical_score": float(values["tactical_score"] / samples) if samples else 0.0,
                "objective_score": float(values["objective_score"] / samples) if samples else 0.0,
                "combined_score": float(values["combined_score"] / samples) if samples else 0.0,
            }

    return {
        "counts": merged,
        "learned_path_coverage": coverage(move_total + shoot_total, move_learned + shoot_learned),
        "move_path_coverage": coverage(move_total, move_learned),
        "shoot_path_coverage": coverage(shoot_total, shoot_learned),
        "score_means": score_means,
        "policy_profiles": sorted(policy_profiles),
        "inference_devices": sorted(inference_devices),
        "loaded_from_dirs": sorted(loaded_from_dirs),
        "intent_model_loaded_count": intent_model_loaded_count,
        "situation_model_loaded_count": situation_model_loaded_count,
        "tactical_model_loaded_count": tactical_model_loaded_count,
        "valid_action_type_counts": dict(sorted(valid_action_type_counts.items())),
        "generated_action_type_counts": dict(sorted(generated_action_type_counts.items())),
        "trace_samples": traces[:12],
        "move_route_aggregate": move_route_agg_merged,
        "move_route_samples": move_route_samples[:50],
        "objective_progress_turns": objective_progress_turns,
        "action_outcomes": action_outcomes,
        "city_control_timeline": city_control_timeline,
        "phase_action_records": phase_action_records,
        "auth_statuses": auth_statuses,
    }


def _collect_objective_sidecars(matches: List[Dict[str, object]]) -> Dict[str, object]:
    progress_turns: List[Dict[str, object]] = []
    action_outcomes: List[Dict[str, object]] = []
    city_timeline: List[Dict[str, object]] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        for stats_key in ("candidate_policy_stats", "opponent_policy_stats"):
            stats = match.get(stats_key, {})
            if not isinstance(stats, dict):
                continue
            progress_turns.extend(stats.get("objective_progress_turns", []))
            action_outcomes.extend(stats.get("action_outcomes", []))
            city_timeline.extend(stats.get("city_control_timeline", []))
    occupy_opportunities = sum(1 for item in action_outcomes if item.get("occupy_available"))
    occupy_successes = sum(1 for item in action_outcomes if item.get("occupy_success"))
    return {
        "objective_progress": {
            "turns": progress_turns,
            "summary": {
                "turn_records": len(progress_turns),
                "score_delta_records": sum(
                    1
                    for item in progress_turns
                    for side_key in ("red", "blue")
                    if float(item.get(side_key, {}).get("score_delta", 0.0)) != 0.0
                ),
                "damage_records": sum(
                    1
                    for item in progress_turns
                    for side_key in ("red", "blue")
                    if float(item.get(side_key, {}).get("damage_dealt", 0.0)) > 0.0
                ),
            },
        },
        "action_outcomes": {
            "actions": action_outcomes,
            "summary": {
                "action_records": len(action_outcomes),
                "occupy_opportunities": int(occupy_opportunities),
                "occupy_successes": int(occupy_successes),
                "score_delta_actions": sum(1 for item in action_outcomes if float(item.get("score_delta", 0.0)) != 0.0),
                "damage_actions": sum(1 for item in action_outcomes if float(item.get("damage_dealt", 0.0)) > 0.0),
            },
        },
        "city_control": {
            "turns": city_timeline,
            "summary": {
                "turn_records": len(city_timeline),
                "city_value_change_records": sum(
                    1
                    for prev, cur in zip(city_timeline, city_timeline[1:])
                    if prev.get("red_city_value_controlled") != cur.get("red_city_value_controlled")
                    or prev.get("blue_city_value_controlled") != cur.get("blue_city_value_controlled")
                ),
            },
        },
    }


def _strip_objective_records(stats: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    return {
        "objective_progress_turns": list(stats.pop("objective_progress_turns", [])),
        "action_outcomes": list(stats.pop("action_outcomes", [])),
        "city_control_timeline": list(stats.pop("city_control_timeline", [])),
        "phase_action_records": list(stats.pop("phase_action_records", [])),
        "auth_statuses": list(stats.pop("auth_statuses", [])),
    }


def _build_objective_sidecars_from_records(
    progress_turns: List[Dict[str, object]],
    action_outcomes: List[Dict[str, object]],
    city_timeline: List[Dict[str, object]],
) -> Dict[str, object]:
    return _collect_objective_sidecars(
        [
            {
                "candidate_policy_stats": {
                    "objective_progress_turns": progress_turns,
                    "action_outcomes": action_outcomes,
                    "city_control_timeline": city_timeline,
                }
            }
        ]
    )


def _summarize_phase_action_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    reason_counts = Counter(
        str(record.get("no_battle_action_reason"))
        for record in records
        if record.get("no_battle_action_reason")
    )
    auth_failure_count = sum(1 for record in records if str(record.get("auth_status", "")).startswith("failed"))
    first_battle_turn = None
    for record in records:
        if record.get("phase") == "battle":
            first_battle_turn = int(record.get("turn", 0))
            break
    selected_move = sum(
        int(record.get("selected_action_type_counts", {}).get("Move", 0))
        for record in records
        if isinstance(record.get("selected_action_type_counts"), dict)
    )
    selected_shoot = sum(
        int(record.get("selected_action_type_counts", {}).get("Shoot", 0))
        for record in records
        if isinstance(record.get("selected_action_type_counts"), dict)
    )
    selected_occupy = sum(
        int(record.get("selected_action_type_counts", {}).get("Occupy", 0))
        for record in records
        if isinstance(record.get("selected_action_type_counts"), dict)
    )
    return {
        "total_turn_records": len(records),
        "total_turns": len({(record.get("match_id"), record.get("turn")) for record in records}),
        "turns_with_setup_deployment_only": sum(1 for record in records if record.get("phase") != "battle"),
        "turns_with_move": sum(1 for record in records if int(record.get("move_action_count", 0)) > 0),
        "turns_with_shoot": sum(1 for record in records if int(record.get("shoot_action_count", 0)) > 0),
        "turns_with_occupy": sum(1 for record in records if int(record.get("occupy_action_count", 0)) > 0),
        "selected_move_actions": int(selected_move),
        "selected_shoot_actions": int(selected_shoot),
        "selected_occupy_actions": int(selected_occupy),
        "battle_phase_entered": first_battle_turn is not None,
        "first_battle_turn": first_battle_turn,
        "auth_failure_count": int(auth_failure_count),
        "no_battle_action_reason_distribution": dict(sorted(reason_counts.items())),
    }


def _build_phase_action_sidecars(records: List[Dict[str, object]]) -> Dict[str, object]:
    summary = _summarize_phase_action_records(records)
    timeline = [
        {
            "match_id": record.get("match_id"),
            "turn": record.get("turn"),
            "side": record.get("side"),
            "phase": record.get("phase"),
            "stage": record.get("stage"),
            "phase_transition": record.get("phase_transition"),
            "valid_action_type_counts": record.get("valid_action_type_counts", {}),
            "selected_action_type_counts": record.get("selected_action_type_counts", {}),
            "no_battle_action_reason": record.get("no_battle_action_reason"),
        }
        for record in records
    ]
    return {
        "phase_action": {"turns": records, "summary": summary},
        "valid_action_timeline": {"turns": timeline, "summary": summary},
    }


def _apply_battle_phase_gate(decision: Dict[str, object], phase_summary: Dict[str, object]) -> Dict[str, object]:
    gated = dict(decision)
    failures = list(gated.get("failures", []))
    if not bool(phase_summary.get("battle_phase_entered")):
        failures.append("battle_phase_not_entered")
    selected_total = (
        int(phase_summary.get("selected_move_actions", 0))
        + int(phase_summary.get("selected_shoot_actions", 0))
        + int(phase_summary.get("selected_occupy_actions", 0))
    )
    if selected_total <= 0:
        failures.append("no_move_shoot_occupy_selected_actions")
    if failures:
        deduped = list(dict.fromkeys(str(failure) for failure in failures))
        gated["failures"] = deduped
        gated["accepted"] = False
    gated["battle_phase_entered"] = bool(phase_summary.get("battle_phase_entered"))
    gated["selected_move_shoot_occupy_actions"] = int(selected_total)
    return gated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Shenji self-play league evaluation.")
    parser.add_argument("--config", default="configs/selfplay_baseline.yaml")
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--incumbent", default=None)
    parser.add_argument("--candidate-profile", default=None)
    parser.add_argument("--incumbent-profile", default=None)
    parser.add_argument("--allow-self-match-smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(repo_root() / args.config)
    candidate_arg, incumbent_arg = _resolve_formal_compare_artifacts(
        config,
        candidate_arg=args.candidate,
        incumbent_arg=args.incumbent,
    )
    if not args.allow_self_match_smoke and _resolved_path_string(candidate_arg) == _resolved_path_string(incumbent_arg):
        raise SystemExit("candidate and incumbent must point to different artifact directories unless --allow-self-match-smoke is set.")

    _configure_torch_cpu_inference_threads()
    set_seed(int(config["seed"]))
    run_dir = ensure_dir(repo_root() / config["output_root"] / f"league_{now_tag()}")
    league_dir = ensure_dir(repo_root() / config["league_dir"])
    league_state_path = league_dir / "league_state.json"
    league_state = _bootstrap_league(league_state_path)
    current_head_sha = git_head_sha(repo_root())
    candidate_profile = str(args.candidate_profile or config.get("candidate_profile", "challenger"))
    configured_incumbent_profile = config.get("incumbent_profile")
    incumbent_profile = _resolve_incumbent_profile(
        incumbent_arg,
        args.incumbent_profile if args.incumbent_profile is not None else configured_incumbent_profile,
        "baseline",
    )
    candidate_participant = _build_participant_spec(candidate_arg, candidate_profile, current_head_sha, "CurrentAI")
    incumbent_participant = _build_participant_spec(incumbent_arg, incumbent_profile, current_head_sha, "IncumbentAI")
    candidate_artifacts = _resolved_path_string(candidate_arg)
    opponent_artifacts = _resolved_path_string(incumbent_arg)
    _, env_backend = _train_env_runtime()
    if env_backend == "mock_train_env.MockTrainEnv":
        raise SystemExit(
            "Formal self-play evaluation requires a real TrainEnv backend; resolved mock_train_env.MockTrainEnv instead."
        )

    maps_root = repo_root() / "land_wargame_sdk" / "Data" / "Data" / "maps"
    scenario_root = repo_root() / "land_wargame_sdk" / "Data" / "Data" / "scenarios"
    records = build_map_catalog(maps_root, repo_root() / "training" / "cache" / "map_catalog.json")
    bucket_lookup = map_bucket_lookup(records)
    scenario_candidates = _collect_scenario_candidates(scenario_root, bucket_lookup, records)
    paired_map_list = _build_paired_map_list(scenario_candidates, int(config["paired_matches"]))
    save_json(paired_map_list, run_dir / "map_list.json")
    scenarios_by_bucket: Dict[str, List[Tuple[Path, int]]] = {"simple": [], "medium": [], "complex": []}
    for item in scenario_candidates:
        scenario_path = scenario_root / str(item["scenario"])
        scenarios_by_bucket[str(item["bucket"])].append((scenario_path, int(item["map_id"])))

    matches: List[Dict[str, object]] = []
    objective_progress_records: List[Dict[str, object]] = []
    action_outcome_records: List[Dict[str, object]] = []
    city_control_records: List[Dict[str, object]] = []
    phase_action_records: List[Dict[str, object]] = []
    auth_status_records: List[Dict[str, object]] = []
    total_pairs = len(paired_map_list["items"])
    current_match_state: Dict[str, object] | None = None
    _save_progress_snapshot(
        run_dir,
        env_backend=env_backend,
        matches=matches,
        total_pairs=total_pairs,
        candidate=candidate_participant,
        incumbent=incumbent_participant,
    )
    try:
        for map_item in paired_map_list["items"]:
            scenario_path = scenario_root / str(map_item["scenario"])
            map_id = int(map_item["map_id"])
            bucket = str(map_item["bucket"])
            order_index = int(map_item["order_index"])
            first_leg_result: Dict[str, float] | None = None

            def emit_pair_heartbeat(
                current_leg_index: int,
                *,
                leg_steps: int,
                first_leg: Dict[str, float] | None = None,
                second_leg: Dict[str, float] | None = None,
            ) -> None:
                nonlocal current_match_state
                current_match_state = _build_inflight_match(
                    map_item=map_item,
                    current_leg_index=current_leg_index,
                    leg_steps=leg_steps,
                    candidate_artifacts=candidate_artifacts,
                    opponent_artifacts=opponent_artifacts,
                    candidate=candidate_participant,
                    incumbent=incumbent_participant,
                    candidate_profile=candidate_profile,
                    incumbent_profile=incumbent_profile,
                    first_leg=first_leg,
                    second_leg=second_leg,
                )
                _save_progress_snapshot(
                    run_dir,
                    env_backend=env_backend,
                    matches=matches,
                    total_pairs=total_pairs,
                    candidate=candidate_participant,
                    incumbent=incumbent_participant,
                    current_match=current_match_state,
                )

            emit_pair_heartbeat(0, leg_steps=0)
            red_score, blue_score, red_stats, blue_stats = _run_single(
                scenario_path,
                map_id,
                candidate_participant,
                incumbent_participant,
                int(config["max_steps"]),
                heartbeat=lambda leg_steps: emit_pair_heartbeat(0, leg_steps=leg_steps),
            )
            first_leg_result = {"candidate": red_score, "opponent": blue_score}
            emit_pair_heartbeat(1, leg_steps=0, first_leg=first_leg_result)
            blue_red_score, blue_blue_score, blue_leg_red_stats, blue_leg_blue_stats = _run_single(
                scenario_path,
                map_id,
                incumbent_participant,
                candidate_participant,
                int(config["max_steps"]),
                heartbeat=lambda leg_steps: emit_pair_heartbeat(1, leg_steps=leg_steps, first_leg=first_leg_result),
            )
            candidate_stats = _merge_policy_stats(red_stats, blue_leg_blue_stats)
            opponent_stats = _merge_policy_stats(blue_stats, blue_leg_red_stats)
            for stats in (candidate_stats, opponent_stats):
                objective_records = _strip_objective_records(stats)
                objective_progress_records.extend(objective_records["objective_progress_turns"])
                action_outcome_records.extend(objective_records["action_outcomes"])
                city_control_records.extend(objective_records["city_control_timeline"])
                phase_action_records.extend(objective_records["phase_action_records"])
                auth_status_records.extend(objective_records["auth_statuses"])
            candidate_diagnostics = _build_match_policy_diagnostics(
                artifacts=candidate_arg,
                participant=candidate_participant,
                profile=candidate_profile,
                stats=candidate_stats,
            )
            opponent_diagnostics = _build_match_policy_diagnostics(
                artifacts=incumbent_arg,
                participant=incumbent_participant,
                profile=incumbent_profile,
                stats=opponent_stats,
            )
            matches.append(
                {
                    "scenario": scenario_path.name,
                    "bucket": bucket,
                    "map_id": map_id,
                    "order_index": order_index,
                    "paired_score": _paired_score(red_score, blue_score, blue_blue_score, blue_red_score),
                    "first_leg": first_leg_result,
                    "second_leg": {"candidate": blue_blue_score, "opponent": blue_red_score},
                    "candidate_artifacts": candidate_artifacts,
                    "opponent_artifacts": opponent_artifacts,
                    "candidate_source_commit_sha": candidate_participant["source_commit_sha"],
                    "opponent_source_commit_sha": incumbent_participant["source_commit_sha"],
                    "candidate_code_origin": candidate_participant["code_origin"],
                    "opponent_code_origin": incumbent_participant["code_origin"],
                    "candidate_user_name": candidate_participant["user_name"],
                    "opponent_user_name": incumbent_participant["user_name"],
                    "candidate_profile": candidate_profile,
                    "opponent_profile": incumbent_profile,
                    "candidate_policy_stats": candidate_stats,
                    "candidate_policy_diagnostics": candidate_diagnostics,
                    "opponent_policy_stats": opponent_stats,
                    "opponent_policy_diagnostics": opponent_diagnostics,
                }
            )
            current_match_state = None
            _save_progress_snapshot(
                run_dir,
                env_backend=env_backend,
                matches=matches,
                total_pairs=total_pairs,
                candidate=candidate_participant,
                incumbent=incumbent_participant,
            )
    except KeyboardInterrupt as exc:
        _write_compare_abort_sidecars(
            run_dir,
            env_backend=env_backend,
            matches=matches,
            total_pairs=total_pairs,
            candidate_artifacts=candidate_arg,
            incumbent_artifacts=incumbent_arg,
            candidate_profile=candidate_profile,
            incumbent_profile=incumbent_profile,
            candidate=candidate_participant,
            incumbent=incumbent_participant,
            current_match=current_match_state,
            exc=exc,
        )
        raise
    except Exception as exc:
        _write_compare_abort_sidecars(
            run_dir,
            env_backend=env_backend,
            matches=matches,
            total_pairs=total_pairs,
            candidate_artifacts=candidate_arg,
            incumbent_artifacts=incumbent_arg,
            candidate_profile=candidate_profile,
            incumbent_profile=incumbent_profile,
            candidate=candidate_participant,
            incumbent=incumbent_participant,
            current_match=current_match_state,
            exc=exc,
        )
        raise

    summary = summarize_matches(matches)

    exploiter_results = {}
    for profile in league_state["exploiters"]:
        exploiter_participant = _build_participant_spec(incumbent_arg, profile, current_head_sha, "ExploiterAI")
        profile_scores = []
        profile_coverages = []
        for scenario_path, map_id in scenarios_by_bucket.get("complex", [])[:4]:
            score_a, score_b, candidate_stats, _ = _run_single(
                scenario_path,
                map_id,
                candidate_participant,
                exploiter_participant,
                int(config["max_steps"]),
            )
            profile_scores.append(1.0 if score_b > score_a else 0.0)
            profile_coverages.append(float(candidate_stats.get("learned_path_coverage", 0.0)))
        exploiter_results[profile] = {
            "win_rate_vs_candidate": float(np.mean(profile_scores)) if profile_scores else 0.0,
            "matches": len(profile_scores),
            "candidate_learned_path_coverage": float(np.mean(profile_coverages)) if profile_coverages else 0.0,
        }

    phase_action_sidecars = _build_phase_action_sidecars(phase_action_records)
    phase_summary = phase_action_sidecars["phase_action"]["summary"]
    decision = _apply_battle_phase_gate(
        promotion_decision(summary, config["promotion_gate"], exploiter_results),
        phase_summary,
    )

    if decision["accepted"]:
        league_state["consecutive_windows"] = int(league_state.get("consecutive_windows", 0)) + 1
    else:
        league_state["consecutive_windows"] = 0
    if league_state["consecutive_windows"] >= int(config["promotion_gate"]["consecutive_windows"]):
        league_state["historical_mains"] = [league_state.get("champion", incumbent_arg)] + list(league_state.get("historical_mains", []))[:7]
        league_state["champion"] = candidate_arg
        league_state["consecutive_windows"] = 0

    objective_sidecars = _build_objective_sidecars_from_records(
        objective_progress_records,
        action_outcome_records,
        city_control_records,
    )
    save_json({"env_backend": env_backend, "matches": matches}, run_dir / "match_results.json")
    save_json(objective_sidecars["objective_progress"], run_dir / "objective_progress_diagnostics.json")
    save_json(objective_sidecars["action_outcomes"], run_dir / "action_outcome_diagnostics.json")
    save_json(objective_sidecars["city_control"], run_dir / "city_control_timeline.json")
    save_json(phase_action_sidecars["phase_action"], run_dir / "phase_action_diagnostics.json")
    save_json(phase_action_sidecars["valid_action_timeline"], run_dir / "valid_action_timeline.json")
    save_json({"auth_statuses": auth_status_records}, run_dir / "auth_status_diagnostics.json")
    save_json(summary, run_dir / "rating.json")
    save_json(decision, run_dir / "promotion.json")
    save_json(exploiter_results, run_dir / "exploitability.json")
    _write_participants_sidecar(
        run_dir,
        env_backend=env_backend,
        candidate_artifacts=candidate_arg,
        incumbent_artifacts=incumbent_arg,
        candidate_profile=candidate_profile,
        incumbent_profile=incumbent_profile,
        candidate=candidate_participant,
        incumbent=incumbent_participant,
    )
    dump_yaml(config, run_dir / "config.yaml")
    save_json(league_state, league_state_path)
    save_json(league_state, run_dir / "league_state.json")
    (run_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Shenji Self-Play Summary",
                "",
                f"- env_backend: `{env_backend}`",
                f"- paired_map_items: `{len(paired_map_list['items'])}`",
                f"- map_list_duplicates_used: `{paired_map_list['duplicates_used']}`",
                f"- candidate_source_commit_sha: `{candidate_participant['source_commit_sha']}`",
                f"- incumbent_source_commit_sha: `{incumbent_participant['source_commit_sha']}`",
                f"- candidate_user_name: `{candidate_participant['user_name']}`",
                f"- candidate_profile: `{candidate_profile}`",
                f"- candidate_artifacts: `{candidate_artifacts}`",
                f"- candidate_manifest_path: `{_manifest_path_string(candidate_arg)}`",
                f"- candidate_manifest_loaded: `{bool(_load_artifact_manifest(candidate_arg))}`",
                f"- incumbent_user_name: `{incumbent_participant['user_name']}`",
                f"- incumbent_profile: `{incumbent_profile}`",
                f"- incumbent_artifacts: `{opponent_artifacts}`",
                f"- incumbent_manifest_path: `{_manifest_path_string(incumbent_arg)}`",
                f"- incumbent_manifest_loaded: `{bool(_load_artifact_manifest(incumbent_arg))}`",
                f"- overall_point_estimate: `{summary['overall']['point_estimate']:.4f}`",
                f"- overall_ci95_lower: `{summary['overall']['ci95'][0]:.4f}`",
                f"- complex_point_estimate: `{summary['buckets'].get('complex', {}).get('point_estimate', 0.0):.4f}`",
                f"- complex_ci95_lower: `{summary['buckets'].get('complex', {}).get('ci95', [0.0, 0.0])[0]:.4f}`",
                f"- learned_path_coverage: `{summary['policy'].get('learned_path_coverage', 0.0):.4f}`",
                f"- move_path_coverage: `{summary['policy'].get('move_path_coverage', 0.0):.4f}`",
                f"- shoot_path_coverage: `{summary['policy'].get('shoot_path_coverage', 0.0):.4f}`",
                f"- candidate_learned_policy_enabled_matches: `{summary['policy'].get('learned_policy_enabled_match_count', 0)}`",
                f"- candidate_fallback_reasons: `{summary['policy'].get('fallback_reasons', [])}`",
                f"- elo_delta_lower_bound: `{decision.get('elo_delta_lower_bound', 0.0):.4f}`",
                f"- max_exploiter_win_rate: `{decision.get('max_exploiter_win_rate', 0.0):.4f}`",
                f"- promotion_accepted: `{decision['accepted']}`",
                f"- promotion_failures: `{decision.get('failures', [])}`",
                f"- battle_phase_entered: `{phase_summary.get('battle_phase_entered')}`",
                f"- first_battle_turn: `{phase_summary.get('first_battle_turn')}`",
                f"- turns_with_setup_deployment_only: `{phase_summary.get('turns_with_setup_deployment_only')}`",
                f"- turns_with_move: `{phase_summary.get('turns_with_move')}`",
                f"- turns_with_shoot: `{phase_summary.get('turns_with_shoot')}`",
                f"- turns_with_occupy: `{phase_summary.get('turns_with_occupy')}`",
                f"- auth_failure_count: `{phase_summary.get('auth_failure_count')}`",
                f"- no_battle_action_reason_distribution: `{phase_summary.get('no_battle_action_reason_distribution')}`",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
