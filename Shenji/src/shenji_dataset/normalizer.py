from __future__ import annotations

import copy
import json
from collections import OrderedDict
from typing import Any

from .ontology import load_ontology
from .validation import validate_intent_segment


CANONICAL_TABLES = (
    "episodes",
    "entities",
    "states",
    "observations",
    "actions",
    "events",
    "relations",
    "intent_segments",
)

ACTION_NAMES = {
    1: "MOVE",
    2: "SHOOT",
    3: "EMBARK",
    4: "DISEMBARK",
    5: "SEIZE",
    6: "CHANGE_STATE",
    7: "REMOVE_SUPPRESSION",
    8: "INDIRECT_FIRE",
    9: "GUIDED_FIRE",
    10: "STOP",
    11: "WEAPON_LOCK",
    12: "WEAPON_DEPLOY",
    13: "CANCEL_INDIRECT_FIRE",
    14: "DISAGGREGATE",
    15: "AGGREGATE",
    16: "CHANGE_ALTITUDE",
    17: "ENABLE_CORRECTION_RADAR",
    18: "ENTER_FORTIFICATION",
    19: "EXIT_FORTIFICATION",
    20: "LAY_MINES",
    100: "CONFIGURE_GROUPING",
    102: "CONFIGURE_ROLE",
    200: "MISSION_TASK",
    201: "OPERATION_DIRECTION",
    303: "DEPLOY_EMBARK",
    304: "DEPLOY_DISEMBARK",
    314: "DEPLOY_DISAGGREGATE",
    315: "DEPLOY_AGGREGATE",
    316: "DEPLOY_CHANGE_ALTITUDE",
    333: "END_DEPLOYMENT",
    401: "DIRECTOR_ACTION_401",
    402: "DIRECTOR_ACTION_402",
    403: "DIRECTOR_ACTION_403",
    404: "DIRECTOR_ACTION_404",
}

SIDE_NAMES = {0: "red", 1: "blue", -1: "neutral"}


def _identifier(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _identifier_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [str(value)]
    return [str(item) for item in value]


def _side_name(value: Any) -> str:
    if value in SIDE_NAMES:
        return SIDE_NAMES[value]
    if isinstance(value, str) and value.lower() in {"red", "blue", "neutral"}:
        return value.lower()
    return f"unknown:{value}"


def _step(observation: dict[str, Any], fallback: int) -> int:
    time_value = observation.get("time", {})
    if isinstance(time_value, dict) and "cur_step" in time_value:
        return int(time_value["cur_step"])
    if "cur_step" in observation:
        return int(observation["cur_step"])
    return fallback


def _payload(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _state_row(episode_id: str, step: int, operator: dict[str, Any]) -> dict[str, Any]:
    entity_id = _identifier(operator.get("obj_id"))
    if entity_id is None:
        raise ValueError("every operator must contain obj_id")
    return {
        "episode_id": episode_id,
        "step": step,
        "entity_id": entity_id,
        "side": _side_name(operator.get("color")),
        "unit_type_id": _identifier(operator.get("type")),
        "subtype_id": _identifier(operator.get("sub_type")),
        "hex_id": _identifier(operator.get("cur_hex")),
        "longitude": operator.get("longitude", operator.get("lon", operator.get("x"))),
        "latitude": operator.get("latitude", operator.get("lat", operator.get("y"))),
        "altitude": operator.get("altitude"),
        "speed": operator.get("speed"),
        "health": operator.get("blood", operator.get("health")),
        "fatigue": operator.get("fatigue"),
        "suppression": operator.get("suppression"),
        "move_state": operator.get("move_state"),
        "payload_json": _payload(operator),
    }


def _normalize_entities(
    episode_id: str, truth_frames: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entities: dict[str, dict[str, Any]] = {}
    states: list[dict[str, Any]] = []
    for fallback_step, frame in enumerate(truth_frames):
        step = _step(frame, fallback_step)
        for operator in frame.get("operators", []):
            state = _state_row(episode_id, step, operator)
            states.append(state)
            entity_id = state["entity_id"]
            if entity_id not in entities:
                entities[entity_id] = {
                    "episode_id": episode_id,
                    "entity_id": entity_id,
                    "side": state["side"],
                    "unit_type_id": state["unit_type_id"],
                    "subtype_id": state["subtype_id"],
                    "first_step": step,
                    "last_step": step,
                }
            else:
                entities[entity_id]["last_step"] = max(entities[entity_id]["last_step"], step)
    entity_rows = [entities[key] for key in sorted(entities)]
    states.sort(key=lambda row: (row["step"], row["entity_id"]))
    return entity_rows, states


def _normalize_observations(
    episode_id: str, observation_streams: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for observer_side in ("truth", "red", "blue"):
        frames = observation_streams.get(observer_side)
        if not isinstance(frames, list):
            raise ValueError(f"observations.{observer_side} must be a list")
        for fallback_step, frame in enumerate(frames):
            step = _step(frame, fallback_step)
            for operator in frame.get("operators", []):
                state = _state_row(episode_id, step, operator)
                rows.append(
                    {
                        "episode_id": episode_id,
                        "step": step,
                        "observer_side": observer_side,
                        "entity_id": state["entity_id"],
                        "observed_side": state["side"],
                        "hex_id": state["hex_id"],
                        "visible": True,
                        "payload_json": state["payload_json"],
                    }
                )
    rows.sort(key=lambda row: (row["step"], row["observer_side"], row["entity_id"]))
    return rows


def _normalize_actions(
    episode_id: str, raw_actions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    actions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for index, raw_action in enumerate(raw_actions):
        action_type = int(raw_action.get("type", -999))
        action_name = ACTION_NAMES.get(action_type, f"UNKNOWN_ACTION_{action_type}")
        step = int(raw_action.get("cur_step", raw_action.get("step", 0)))
        action_id = f"{episode_id}:action:{index:06d}"
        entity_id = _identifier(raw_action.get("obj_id"))
        target_entity_id = _identifier(raw_action.get("target_obj_id"))
        target_hex_id = _identifier(raw_action.get("jm_pos", raw_action.get("target_pos")))
        action_row = {
            "episode_id": episode_id,
            "step": step,
            "action_id": action_id,
            "actor_id": _identifier(raw_action.get("actor")),
            "entity_id": entity_id,
            "action_type_id": str(action_type),
            "action_name": action_name,
            "target_entity_id": target_entity_id,
            "target_hex_id": target_hex_id,
            "weapon_id": _identifier(raw_action.get("weapon_id")),
            "path": _identifier_list(raw_action.get("move_path", raw_action.get("route"))),
            "payload_json": _payload(raw_action),
        }
        actions.append(action_row)
        events.append(
            {
                "episode_id": episode_id,
                "step": step,
                "event_id": action_id,
                "event_type": action_name,
                "subject_id": entity_id,
                "object_id": target_entity_id,
                "location_id": target_hex_id,
                "source": "action",
                "payload_json": action_row["payload_json"],
            }
        )

        relation_type: str | None = None
        relation_object: str | None = None
        relation_target: str | None = target_entity_id
        if action_type == 9:
            relation_type = "GUIDES"
            relation_object = _identifier(raw_action.get("guided_obj_id"))
        elif action_type in {15, 315}:
            relation_type = "AGGREGATES_WITH"
            relation_object = target_entity_id
        elif action_type in {3, 303}:
            relation_type = "EMBARKS_ON"
            relation_object = target_entity_id
        elif action_type in {4, 304}:
            relation_type = "DISEMBARKS_FROM"
            relation_object = target_entity_id
        if relation_type is not None and entity_id is not None and relation_object is not None:
            relations.append(
                {
                    "episode_id": episode_id,
                    "step": step,
                    "relation_id": f"{episode_id}:relation:{len(relations):06d}",
                    "relation_type": relation_type,
                    "subject_id": entity_id,
                    "object_id": relation_object,
                    "target_id": relation_target,
                    "source_event_id": action_id,
                }
            )
    return actions, events, relations


def _adjudication_events(
    episode_id: str, truth_frames: list[dict[str, Any]], start_index: int
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for fallback_step, frame in enumerate(truth_frames):
        step = _step(frame, fallback_step)
        judge_info = frame.get("judge_info", [])
        if isinstance(judge_info, dict):
            judge_info = [judge_info]
        for item in judge_info or []:
            if not isinstance(item, dict):
                continue
            events.append(
                {
                    "episode_id": episode_id,
                    "step": step,
                    "event_id": f"{episode_id}:judge:{start_index + len(events):06d}",
                    "event_type": "ADJUDICATION",
                    "subject_id": _identifier(
                        item.get("attack_obj_id", item.get("source_obj_id"))
                    ),
                    "object_id": _identifier(
                        item.get("target_obj_id", item.get("target_id"))
                    ),
                    "location_id": _identifier(item.get("target_pos")),
                    "source": "judge_info",
                    "payload_json": _payload(item),
                }
            )
    return events


def _normalize_intents(
    episode_id: str, intent_plan: dict[str, Any] | None
) -> list[dict[str, Any]]:
    if intent_plan is None:
        return []
    segments = intent_plan.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("intent_plan.segments must be a list")
    ontology = load_ontology()
    rows: list[dict[str, Any]] = []
    for raw_segment in segments:
        segment = copy.deepcopy(raw_segment)
        segment["episode_id"] = episode_id
        segment["segment_id"] = str(segment.get("segment_id"))
        segment["subject_id"] = str(segment.get("subject_id"))
        segment["target_id"] = _identifier(segment.get("target_id"))
        segment["secondary_intents"] = list(segment.get("secondary_intents", []))
        segment["evidence_refs"] = [str(item) for item in segment.get("evidence_refs", [])]
        validation_view = {key: value for key, value in segment.items() if key != "episode_id"}
        errors = validate_intent_segment(validation_view, ontology)
        if errors:
            raise ValueError(
                f"invalid intent segment {segment['segment_id']}: {'; '.join(errors)}"
            )
        rows.append(segment)
    rows.sort(key=lambda row: (row["start_step"], row["subject_id"], row["segment_id"]))
    return rows


def normalize_episode(
    raw_episode: dict[str, Any], intent_plan: dict[str, Any] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Normalize one raw Shenji episode into the v1 canonical table set."""

    metadata = raw_episode.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("raw episode metadata must be an object")
    if metadata.get("dataset", "Shenji") != "Shenji":
        raise ValueError("dataset must be Shenji")
    episode_id = _identifier(metadata.get("episode_id"))
    if episode_id is None:
        raise ValueError("metadata.episode_id is required")
    observation_streams = raw_episode.get("observations")
    if not isinstance(observation_streams, dict):
        raise ValueError("raw episode observations must be an object")
    truth_frames = observation_streams.get("truth")
    if not isinstance(truth_frames, list):
        raise ValueError("observations.truth must be a list")
    raw_actions = raw_episode.get("actions", [])
    if not isinstance(raw_actions, list):
        raise ValueError("raw episode actions must be a list")

    entities, states = _normalize_entities(episode_id, truth_frames)
    observations = _normalize_observations(episode_id, observation_streams)
    actions, events, relations = _normalize_actions(episode_id, raw_actions)
    events.extend(_adjudication_events(episode_id, truth_frames, len(events)))
    intents = _normalize_intents(episode_id, intent_plan)
    episode_row = {
        "episode_id": episode_id,
        "dataset": "Shenji",
        "schema_version": metadata.get("schema_version", "shenji_schema_v1"),
        "simulation_backend": metadata.get("simulation_backend", "MiaoSuan"),
        "scenario_id": _identifier(metadata.get("scenario_id")),
        "map_id": _identifier(metadata.get("map_id")),
        "seed": metadata.get("seed"),
        "policy_red": metadata.get("policy_red"),
        "policy_blue": metadata.get("policy_blue"),
        "num_truth_frames": len(truth_frames),
        "metadata_json": _payload(metadata),
    }
    return OrderedDict(
        (
            ("episodes", [episode_row]),
            ("entities", entities),
            ("states", states),
            ("observations", observations),
            ("actions", actions),
            ("events", events),
            ("relations", relations),
            ("intent_segments", intents),
        )
    )
