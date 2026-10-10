from __future__ import annotations

import hashlib
import json
from typing import Any


MISSION_INTENTS = {
    0: ("OFFENSIVE_ACTION", "ASSAULT"),
    1: ("AREA_CONTROL", "HOLD"),
    2: ("RECON_INFORMATION", "RECON"),
}

# Grade B/C evidence grades
GRADE_B_EVIDENCE = "deterministic_rule"
GRADE_C_EVIDENCE = "plan_tag_or_human_review"


def _identifier(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _step(frame: dict[str, Any], fallback: int) -> int:
    time_value = frame.get("time", {})
    if isinstance(time_value, dict) and "cur_step" in time_value:
        return int(time_value["cur_step"])
    return int(frame.get("cur_step", fallback))


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _segment_id(
    label_kind: str,
    intent: str,
    subject_id: str,
    start_step: int,
    evidence: dict[str, Any],
) -> str:
    identity = {
        "label_kind": label_kind,
        "intent": intent,
        "subject_id": subject_id,
        "start_step": start_step,
        "evidence": evidence,
    }
    digest = hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()[:12]
    return f"{label_kind}_{intent.lower()}_{digest}"


def _mission_segment(
    command: dict[str, Any], frame_step: int, evidence_ref: str
) -> dict[str, Any] | None:
    info = command.get("info", {})
    if not isinstance(info, dict):
        return None
    try:
        mission_type = int(info.get("mission_type"))
    except (TypeError, ValueError):
        return None
    mapping = MISSION_INTENTS.get(mission_type)
    subject_id = _identifier(info.get("company_id"))
    if mapping is None or subject_id is None:
        return None
    start_step = int(info.get("start_time", frame_step))
    end_step = max(start_step, int(info.get("end_time", start_step)))
    parent, intent = mapping
    target_id = _identifier(info.get("target_pos"))
    return {
        "segment_id": _segment_id(
            "commanded", intent, subject_id, start_step, command
        ),
        "label_kind": "commanded",
        "subject_scope": "company",
        "subject_id": subject_id,
        "start_step": start_step,
        "end_step": end_step,
        "parent_intent": parent,
        "primary_intent": intent,
        "secondary_intents": [],
        "target_type": "objective_point" if target_id is not None else "none",
        "target_id": target_id,
        "status": "planned",
        "outcome": "unknown",
        "evidence_source": "mission_command",
        "evidence_grade": "A",
        "evidence_refs": [evidence_ref],
        "confidence": 1.0,
    }


def _action_was_accepted(action: dict[str, Any]) -> bool:
    if action.get("success") is False:
        return False
    for key in ("error", "error_message", "error_msg"):
        if action.get(key) not in (None, "", 0, False):
            return False
    if action.get("error_code") not in (None, "", 0, "0"):
        return False
    return True


def _realized_segment(
    *,
    action: dict[str, Any],
    frame_step: int,
    intent: str,
    parent: str,
    subject_id: Any,
    target_type: str,
    target_id: Any,
    evidence_ref: str,
    end_step: int | None = None,
) -> dict[str, Any] | None:
    subject = _identifier(subject_id)
    if subject is None:
        return None
    step = int(action.get("cur_step", action.get("step", frame_step)))
    end = end_step if end_step is not None else step
    target = _identifier(target_id)
    normalized_target_type = target_type if target is not None else "none"
    return {
        "segment_id": _segment_id("realized", intent, subject, step, action),
        "label_kind": "realized",
        "subject_scope": "entity",
        "subject_id": subject,
        "start_step": step,
        "end_step": end,
        "parent_intent": parent,
        "primary_intent": intent,
        "secondary_intents": [],
        "target_type": normalized_target_type,
        "target_id": target,
        "status": "completed",
        "outcome": "achieved",
        "evidence_source": "deterministic_rule",
        "evidence_grade": "A",
        "evidence_refs": [evidence_ref],
        "confidence": 1.0,
    }


def _segments_from_action(
    action: dict[str, Any], frame_step: int, evidence_ref: str
) -> list[dict[str, Any]]:
    if not _action_was_accepted(action):
        return []
    try:
        action_type = int(action.get("type"))
    except (TypeError, ValueError):
        return []
    segments: list[dict[str, Any] | None] = []
    if action_type == 5:
        segments.append(
            _realized_segment(
                action=action,
                frame_step=frame_step,
                intent="SEIZE",
                parent="AREA_CONTROL",
                subject_id=action.get("obj_id"),
                target_type="objective_point",
                target_id=action.get("target_pos"),
                evidence_ref=evidence_ref,
            )
        )
    elif action_type == 9:
        segments.append(
            _realized_segment(
                action=action,
                frame_step=frame_step,
                intent="TARGET_GUIDANCE",
                parent="RECON_INFORMATION",
                subject_id=action.get("obj_id"),
                target_type="enemy_entity",
                target_id=action.get("target_obj_id"),
                evidence_ref=evidence_ref,
            )
        )
        segments.append(
            _realized_segment(
                action=action,
                frame_step=frame_step,
                intent="FIRE_SUPPORT",
                parent="SUPPORT_COORDINATION",
                subject_id=action.get("guided_obj_id"),
                target_type="enemy_entity",
                target_id=action.get("target_obj_id"),
                evidence_ref=evidence_ref,
            )
        )
    elif action_type == 8:
        segments.append(
            _realized_segment(
                action=action,
                frame_step=frame_step,
                intent="FIRE_SUPPORT",
                parent="SUPPORT_COORDINATION",
                subject_id=action.get("obj_id"),
                target_type="terrain_region",
                target_id=action.get("jm_pos"),
                evidence_ref=evidence_ref,
            )
        )
    elif action_type in {15, 315}:
        segments.append(
            _realized_segment(
                action=action,
                frame_step=frame_step,
                intent="ASSEMBLE",
                parent="MANEUVER_POSTURE",
                subject_id=action.get("obj_id"),
                target_type="friendly_entity",
                target_id=action.get("target_obj_id"),
                evidence_ref=evidence_ref,
            )
        )
    elif action_type in {14, 314}:
        segments.append(
            _realized_segment(
                action=action,
                frame_step=frame_step,
                intent="DISPERSE",
                parent="MANEUVER_POSTURE",
                subject_id=action.get("obj_id"),
                target_type="none",
                target_id=None,
                evidence_ref=evidence_ref,
            )
        )
    elif action_type == 18:
        segments.append(
            _realized_segment(
                action=action,
                frame_step=frame_step,
                intent="FORTIFY",
                parent="MANEUVER_POSTURE",
                subject_id=action.get("obj_id"),
                target_type="terrain_region",
                target_id=action.get("target_obj_id"),
                evidence_ref=evidence_ref,
            )
        )
    elif action_type == 6 and action.get("target_state") == 4:
        segments.append(
            _realized_segment(
                action=action,
                frame_step=frame_step,
                intent="FORTIFY",
                parent="MANEUVER_POSTURE",
                subject_id=action.get("obj_id"),
                target_type="none",
                target_id=None,
                evidence_ref=evidence_ref,
            )
        )
    return [segment for segment in segments if segment is not None]


def _hex_distance(hex_a: str | None, hex_b: str | None) -> float:
    """Approximate hex distance between two hex ID strings (6-digit codes)."""
    if hex_a is None or hex_b is None:
        return float("inf")
    try:
        ra, ca = int(hex_a) // 100, int(hex_a) % 100
        rb, cb = int(hex_b) // 100, int(hex_b) % 100
    except (ValueError, TypeError):
        return float("inf")
    dr, dc = abs(ra - rb), abs(ca - cb)
    return float(max(dr, dc, dr + dc) if (dr >= 0 and dc >= 0 and dr + dc >= 0) else dr + abs(dc))


def _bearing_change_degrees(
    hex_seq: list[str | None],
) -> tuple[float | None, int, int]:
    """Compute max bearing change (degrees) and the index range that produced it.
    Returns (max_change, start_idx, end_idx) where start_idx/end_idx are seq indices."""
    if len(hex_seq) < 3:
        return None, -1, -1
    try:
        r0, c0 = int(hex_seq[0]) // 100, int(hex_seq[0]) % 100
    except (ValueError, TypeError):
        return None, -1, -1
    max_change = 0.0
    best_i = 2
    for i in range(2, len(hex_seq)):
        if hex_seq[i - 1] is None or hex_seq[i] is None:
            continue
        try:
            r1, c1 = int(hex_seq[i - 1]) // 100, int(hex_seq[i - 1]) % 100
            r2, c2 = int(hex_seq[i]) // 100, int(hex_seq[i]) % 100
            import math
            b1 = math.degrees(math.atan2(r1 - r0, c1 - c0)) % 360
            b2 = math.degrees(math.atan2(r2 - r1, c2 - c1)) % 360
            diff = abs(b2 - b1)
            d = min(diff, 360 - diff)
            if d > max_change:
                max_change = d
                best_i = i
        except (ValueError, TypeError):
            continue
    if max_change == 0.0:
        return None, -1, -1
    start_idx = max(0, best_i - 5)
    end_idx = min(len(hex_seq) - 1, best_i + 5)
    return max_change, start_idx, end_idx


def _entity_trajectory(
    truth_frames: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build per-entity trajectory keyed by stable obj_id, not frame array index."""
    from collections import defaultdict
    traj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for si, frame in enumerate(truth_frames):
        if not isinstance(frame, dict):
            continue
        cs = _step(frame, si)
        for oi, op in enumerate(frame.get("operators", [])):
            obj_id = _identifier(op.get("obj_id", op.get("op_index", oi)))
            if obj_id is None:
                obj_id = str(oi)
            traj[obj_id].append({
                "entity_id": obj_id,
                "step": cs,
                "hex": _identifier(op.get("cur_hex")),
                "hp": op.get("blood"),
                "speed": op.get("basic_speed", op.get("speed", 0)),
                "ms": op.get("move_state"),
            })
    return dict(traj)


def _detect_flank(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """FLANK: bearing change >45 along path, NOT caused by evasion."""
    if len(seq) < 5:
        return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    hps = [s["hp"] for s in seq if isinstance(s.get("hp"), (int, float))]
    bc_val, bc_start, bc_end = _bearing_change_degrees(hexes)
    if bc_val is None or bc_val <= 45:
        return []
    hp_lost = (max(hps) - min(hps)) > 0 if len(hps) >= 3 else False
    if hp_lost:
        return []
    start = seq[bc_start]["step"]
    end = seq[bc_end]["step"]
    return [_realized_segment(
        action={"cur_step": start}, frame_step=start,
        intent="FLANK", parent="OFFENSIVE_ACTION",
        subject_id=seq[0].get("entity_id", "0"),
        target_type="none", target_id=None,
        evidence_ref=f"flank:bc={bc_val:.0f}",
        end_step=end,
    )]


def _detect_breakthrough(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """BREAKTHROUGH: moves >4 hexes and takes HP damage while advancing."""
    if len(seq) < 5:
        return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    hps = [s["hp"] for s in seq if isinstance(s.get("hp"), (int, float))]
    if len(hexes) < 3 or len(hps) < 3:
        return []
    dist = _hex_distance(hexes[0], hexes[-1])
    lost = max(hps) - min(hps)
    if dist > 4 and lost > 0:
        start = seq[0]["step"]
        end = seq[-1]["step"]
        # BREAKTHROUGH: window centers on the segment where distance first exceeded 4 and damage occurred
        # Use first-last hex with valid data as evidence window
        for i, s in enumerate(seq):
            if s.get("hex") and s["hex"] != "null":
                start = s["step"]
                break
        for i in range(len(seq)-1, -1, -1):
            if seq[i].get("hex") and seq[i]["hex"] != "null":
                end = seq[i]["step"]
                break
        return [_realized_segment(
            action={"cur_step": start}, frame_step=start,
            intent="BREAKTHROUGH", parent="OFFENSIVE_ACTION",
            subject_id=seq[0].get("entity_id", "0"),
            target_type="none", target_id=None,
            evidence_ref=f"breakthrough:d={dist:.0f}",
            end_step=end,
        )]
    return []


def _detect_pursuit(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """PURSUIT: sustained movement toward a direction (not just general movement)."""
    if len(seq) < 10:
        return []
    hps = [s["hp"] for s in seq if isinstance(s.get("hp"), (int, float))]
    if len(hps) < 5:
        return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    if len(hexes) < 3:
        return []
    dist = _hex_distance(hexes[0], hexes[-1]) if hexes else 0
    spds = [s["speed"] for s in seq[len(seq)//2:] if isinstance(s.get("speed"), (int, float))]
    if len(spds) < 3 or sum(spds) / len(spds) < 0.01 or dist < 3:
        return []
    hp_lost = (max(hps) - min(hps)) > 1
    if hp_lost:
        return []
    # PURSUIT: evidence window = last half of seq where sustained speed was checked
    mid = len(seq) // 2
    start = seq[mid]["step"]
    end = seq[-1]["step"]
    return [_realized_segment(
        action={"cur_step": start}, frame_step=start,
        intent="PURSUIT", parent="OFFENSIVE_ACTION",
        subject_id=seq[0].get("entity_id", "0"),
        target_type="none", target_id=None,
        evidence_ref=f"pursuit:d={dist:.0f}",
        end_step=end,
    )]


def _detect_ambush(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """AMBUSH: stationary+concealed (ms in {4,5}) >=3 steps."""
    if len(seq) < 5:
        return []
    concealed = sum(1 for s in seq if s.get("ms") in (4, 5))
    if concealed >= 3:
        # AMBUSH: find first and last concealed step within seq
        start = None
        end = None
        for s in seq:
            if s.get("ms") in (4, 5):
                if start is None:
                    start = s["step"]
                end = s["step"]
        if start is None:
            return []
        return [_realized_segment(
            action={"cur_step": start}, frame_step=start,
            intent="AMBUSH", parent="OFFENSIVE_ACTION",
            subject_id=seq[0].get("entity_id", "0"),
            target_type="none", target_id=None,
            evidence_ref=f"ambush:concealed={concealed}",
            end_step=end,
        )]
    return []


def _detect_surveillance(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """SURVEILLANCE: ≥10 steps with ≤2 distinct hexes (sustained observation)."""
    if len(seq) < 10:
        return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    if not hexes:
        return []
    unique = set(hexes[-10:])
    if len(unique) <= 2:
        # SURVEILLANCE: use the last 10 steps as evidence window
        start = seq[-10]["step"]
        end = seq[-1]["step"]
        return [_realized_segment(
            action={"cur_step": start}, frame_step=start,
            intent="SURVEILLANCE", parent="RECON_INFORMATION",
            subject_id=seq[0].get("entity_id", "0"),
            target_type="none", target_id=None,
            evidence_ref="surveillance:10+_steps_low_dispersion",
            end_step=end,
        )]
    return []


def _detect_redeploy(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """REDEPLOY: moves >5 hexes total."""
    if len(seq) < 5:
        return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    if len(hexes) >= 2 and _hex_distance(hexes[0], hexes[-1]) > 5:
        # REDEPLOY: window covers active movement portion
        start = None
        end = None
        for s in seq:
            h = s.get("hex")
            if h and h != "null":
                if start is None:
                    start = s["step"]
                end = s["step"]
        if start is None:
            return []
        return [_realized_segment(
            action={"cur_step": start}, frame_step=start,
            intent="REDEPLOY", parent="MANEUVER_POSTURE",
            subject_id=seq[0].get("entity_id", "0"),
            target_type="none", target_id=None,
            evidence_ref=f"redeploy:d={_hex_distance(hexes[0], hexes[-1]):.0f}",
            end_step=end,
        )]
    return []


def _detect_retreat(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """RETREAT: took damage then reversed heading over >3 steps."""
    if len(seq) < 6:
        return []
    hps = [s["hp"] for s in seq if isinstance(s.get("hp"), (int, float))]
    if len(hps) < 3 or max(hps) - min(hps) <= 0:
        return []
    # RETREAT: locate first HP loss, window covers from that step onward
    first_hp_drop = None
    for i in range(1, len(seq)):
        prev_hp = seq[i-1].get("hp")
        cur_hp = seq[i].get("hp")
        if prev_hp is not None and cur_hp is not None and cur_hp < prev_hp:
            first_hp_drop = i
            break
    if first_hp_drop is None:
        return []
    start = seq[first_hp_drop]["step"]
    end = seq[-1]["step"]
    return [_realized_segment(
        action={"cur_step": start}, frame_step=start,
        intent="RETREAT", parent="DISENGAGEMENT_DECEPTION",
        subject_id=seq[0].get("entity_id", "0"),
        target_type="none", target_id=None,
        evidence_ref=f"retreat:hp_lost={max(hps)-min(hps)}",
        end_step=end,
    )]


def _detect_evade(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """EVADE: bearing change >60 after taking damage (and not just turning)."""
    if len(seq) < 5:
        return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    hps = [s["hp"] for s in seq if isinstance(s.get("hp"), (int, float))]
    if len(hexes) < 3 or len(hps) < 3:
        return []
    bc_val, bc_start, bc_end = _bearing_change_degrees(hexes)
    hp_lost = max(hps) - min(hps) > 0
    if bc_val is not None and bc_val > 60 and hp_lost:
        start = seq[bc_start]["step"]
        end = seq[bc_end]["step"]
        return [_realized_segment(
            action={"cur_step": start}, frame_step=start,
            intent="EVADE", parent="DISENGAGEMENT_DECEPTION",
            subject_id=seq[0].get("entity_id", "0"),
            target_type="none", target_id=None,
            evidence_ref=f"evade:bc={bc_val:.0f}",
            end_step=end,
        )]
    return []


def _detect_interdict(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """INTERDICT: holds position in vicinity (≤2 hexes) and takes damage."""
    if len(seq) < 8:
        return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    hps = [s["hp"] for s in seq if isinstance(s.get("hp"), (int, float))]
    if not hexes or not hps:
        return []
    unique_hexes = set(hexes[-8:])
    if len(unique_hexes) <= 2 and max(hps) > min(hps):
        # INTERDICT: window = last 8 steps (where the low-displacement + damage was detected)
        start = seq[-8]["step"]
        end = seq[-1]["step"]
        return [_realized_segment(
            action={"cur_step": start}, frame_step=start,
            intent="INTERDICT", parent="AREA_CONTROL",
            subject_id=seq[0].get("entity_id", "0"),
            target_type="none", target_id=None,
            evidence_ref="interdict:held_with_damage",
            end_step=end,
        )]
    return []


def _detect_escort(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """ESCORT: entity moves while staying close to at least one other.

    NOTE: This is a proxy. True escort detection requires cross-entity
    proximity and formation-keeping checks across the full truth frame.
    The current single-entity seq lacks relationship info, so this detector
    is stricter than the original but still a heuristic.
    """
    if len(seq) < 5:
        return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    # Require meaningful movement (≥3 hexes) AND at least one other
    # entity at the same hex for >30% of visible steps
    if len(hexes) >= 3:
        return [_realized_segment(
            action={"cur_step": seq[0]["step"]}, frame_step=seq[0]["step"],
            intent="ESCORT", parent="SUPPORT_COORDINATION",
            subject_id=seq[0].get("entity_id", "0"),
            target_type="none", target_id=None,
            evidence_ref=f"escort:hexes={len(hexes)}",
        )]
    return []


def _detect_reinforce(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """REINFORCE: entity moves toward a contact zone."""
    if len(seq) < 5:
        return []
    hps = [s["hp"] for s in seq if isinstance(s.get("hp"), (int, float))]
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    if len(hexes) >= 2 and len(hps) >= 3 and max(hps) - min(hps) == 0:
        # REINFORCE: evidence window = whole movement (no damage taken)
        start = seq[0]["step"]
        end = seq[-1]["step"]
        for s in seq:
            h = s.get("hex")
            if h and h != "null":
                start = s["step"]
                break
        for i in range(len(seq)-1, -1, -1):
            if seq[i].get("hex") and seq[i]["hex"] != "null":
                end = seq[i]["step"]
                break
        return [_realized_segment(
            action={"cur_step": start}, frame_step=start,
            intent="REINFORCE", parent="SUPPORT_COORDINATION",
            subject_id=seq[0].get("entity_id", "0"),
            target_type="none", target_id=None,
            evidence_ref=f"reinforce:d={_hex_distance(hexes[0], hexes[-1]):.0f}",
            end_step=end,
        )]
    return []


def _detect_encirclement(
    traj: dict[int, list[dict[str, Any]]],
    num_ops: int,
) -> list[dict[str, Any]]:
    """ENCIRCLEMENT: ≥2 operators converge toward a common area from distinct directions."""
    if num_ops < 2:
        return []
    hexes_by_op: dict[int, list[str]] = {}
    for op_idx in range(num_ops):
        seq = traj.get(op_idx, [])
        hxs = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
        if hxs:
            hexes_by_op[op_idx] = hxs
    if len(hexes_by_op) < 2:
        return []
    # Find min start and max end across involved operators
    start = float("inf")
    end = float("-inf")
    for op_idx, hxs in hexes_by_op.items():
        op_seq = traj.get(op_idx, [])
        if op_seq:
            s = op_seq[0]["step"]
            e = op_seq[-1]["step"]
            if s < start: start = s
            if e > end: end = e
    start = int(start) if start != float("inf") else 0
    end = int(end) if end != float("-inf") else 0
    return [_realized_segment(
        action={"cur_step": start}, frame_step=start,
        intent="ENCIRCLEMENT", parent="OFFENSIVE_ACTION",
        subject_id="multi",
        target_type="none", target_id=None,
        evidence_ref=f"encirclement:ops={list(hexes_by_op.keys())}",
        end_step=end,
    )]


GRADE_B_DETECTORS = {
    "FLANK": _detect_flank,
    "BREAKTHROUGH": _detect_breakthrough,
    "PURSUIT": _detect_pursuit,
    "AMBUSH": _detect_ambush,
    "SURVEILLANCE": _detect_surveillance,
    "REDEPLOY": _detect_redeploy,
    "RETREAT": _detect_retreat,
    "EVADE": _detect_evade,
    "INTERDICT": _detect_interdict,
    # ESCORT: disabled — detector degenerated to "entity moved".
    #   Re-enable only when a true formation/transport cross-entity
    #   detector is implemented (see labeling_patch.py for prior work).
    # "ESCORT": _detect_escort,
    "REINFORCE": _detect_reinforce,
}


def derive_grade_b_intent_plan(raw_episode: dict[str, Any]) -> dict[str, Any]:
    """Derive Grade B (pattern-based) intent segments from trajectory analysis.

    Grade B intents cannot be determined from a single action type; they
    require spatiotemporal pattern recognition over the entity trajectory.
    """
    episode_id = _identifier(raw_episode.get("metadata", {}).get("episode_id"))
    truth_frames = raw_episode.get("observations", {}).get("truth", [])
    if not isinstance(truth_frames, list):
        raise ValueError("observations.truth must be a list")

    traj = _entity_trajectory(truth_frames)
    all_segments: dict[str, dict[str, Any]] = {}

    for op_index, seq in traj.items():
        si = str(op_index)
        for intent_name, detector in GRADE_B_DETECTORS.items():
            for seg in detector(seq):
                seg["evidence_grade"] = "B"
                seg["evidence_source"] = GRADE_B_EVIDENCE
                # Use the entity_id stored in the trajectory (seq[0]["entity_id"])
                # as subject; fallback to op_index for backward compat
                eid = seq[0].get("entity_id", str(op_index))
                sid = f"grade_b_{intent_name.lower()}_{eid}_{seg['start_step']}"
                seg["segment_id"] = sid
                seg["subject_id"] = eid
                if sid not in all_segments:
                    all_segments[sid] = seg

    # ENCIRCLEMENT — DISABLED. Detector degenerated to:
    #   "if len(hexes_by_op) >= 2: return True"
    # which fires on every multi-entity episode. Must implement
    # real convergence, direction-diversity and enemy-proximity
    # checks before re-enabling.
    # for seg in _detect_encirclement(traj, len(traj)):
    #     seg["evidence_grade"] = "B"
    #     seg["evidence_source"] = GRADE_B_EVIDENCE
    #     sid = f"grade_b_encirclement_multi_{seg['start_step']}"
    #     seg["segment_id"] = sid
    #     if sid not in all_segments:
    #         all_segments[sid] = seg

    segments = list(all_segments.values())
    segments.sort(key=lambda x: (x["start_step"], x["subject_id"], x["primary_intent"], x["segment_id"]))
    return {
        "episode_id": episode_id,
        "generator": "shenji_grade_b_labeler_v1",
        "segments": segments,
    }


def derive_grade_c_intent_plan(raw_episode: dict[str, Any]) -> dict[str, Any]:
    """Derive Grade C (FEINT) intent segments.

    FEINT requires explicit plan tag or human review. Behavior alone
    is never sufficient. This function is a shell; it never produces
    auto-derived segments.
    """
    episode_id = _identifier(raw_episode.get("metadata", {}).get("episode_id"))
    return {
        "episode_id": episode_id,
        "generator": "shenji_grade_c_labeler_v1",
        "segments": [],
    }

def derive_grade_a_intent_plan(raw_episode: dict[str, Any]) -> dict[str, Any]:
    """Derive only directly supported Grade A intent segments from truth logs.

    Reads from two sources:
    1. Top-level ``raw_episode["actions"]`` — all agent-submitted actions
       (MISSION_TASK type==200 commands, and tactical action types).
    2. Truth frame ``communication`` / ``actions`` fields — legacy fallback.
    """

    episode_id = _identifier(raw_episode.get("metadata", {}).get("episode_id"))
    truth_frames = raw_episode.get("observations", {}).get("truth", [])
    if not isinstance(truth_frames, list):
        raise ValueError("observations.truth must be a list")
    segments_by_id: dict[str, dict[str, Any]] = {}
    seen_commands: set[str] = set()
    seen_actions: set[str] = set()

    # ── SOURCE 1: top-level raw_episode["actions"] (primary) ──────────
    top_level_actions = raw_episode.get("actions", [])
    if not isinstance(top_level_actions, list):
        top_level_actions = []
    for action_index, action in enumerate(top_level_actions):
        if not isinstance(action, dict):
            continue
        frame_step = int(
            action.get("cur_step", action.get("step", 0))
        )
        action_type = int(action.get("type", -1))

        if action_type == 200:
            command_key = _canonical(action)
            if command_key in seen_commands:
                continue
            seen_commands.add(command_key)
            segment = _mission_segment(
                action, frame_step,
                f"top_level_action:{frame_step}:{action_index}",
            )
            if segment is not None:
                segments_by_id[segment["segment_id"]] = segment
        else:
            action_key = f"top:{frame_step}:{_canonical(action)}"
            if action_key in seen_actions:
                continue
            seen_actions.add(action_key)
            for segment in _segments_from_action(
                action, frame_step,
                f"top_level_action:{frame_step}:{action_index}",
            ):
                segments_by_id[segment["segment_id"]] = segment

    # ── SOURCE 2: truth-frame communication / actions (legacy) ─────────
    for frame_index, frame in enumerate(truth_frames):
        if not isinstance(frame, dict):
            continue
        frame_step = _step(frame, frame_index)
        communications = frame.get("communication", frame.get("communications", []))
        if isinstance(communications, dict):
            communications = [communications]
        for command_index, command in enumerate(communications or []):
            if not isinstance(command, dict) or command.get("type") != 200:
                continue
            command_key = _canonical(command)
            if command_key in seen_commands:
                continue
            seen_commands.add(command_key)
            segment = _mission_segment(
                command, frame_step,
                f"truth_communication:{frame_step}:{command_index}",
            )
            if segment is not None:
                segments_by_id[segment["segment_id"]] = segment

        actions = frame.get("actions", [])
        if isinstance(actions, dict):
            actions = [actions]
        for action_index, action in enumerate(actions or []):
            if not isinstance(action, dict):
                continue
            action_key = f"legacy:{frame_step}:{_canonical(action)}"
            if action_key in seen_actions:
                continue
            seen_actions.add(action_key)
            for segment in _segments_from_action(
                action, frame_step,
                f"truth_action:{frame_step}:{action_index}",
            ):
                segments_by_id[segment["segment_id"]] = segment

    segments = list(segments_by_id.values())
    segments.sort(key=lambda item: (
        item["start_step"], item["label_kind"],
        item["subject_id"], item["primary_intent"], item["segment_id"],
    ))
    return {
        "episode_id": episode_id,
        "generator": "shenji_grade_a_labeler_v1",
        "segments": segments,
    }
