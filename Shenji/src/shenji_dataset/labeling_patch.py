"""Shenji dataset labeling — Grade A/B/C intent derivation.
Patch v1.1: stronger SURVEILLANCE (>=10 steps), INTERDICT (+damage), ENCIRCLEMENT (multi-op angle check).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any

MISSION_INTENTS = {
    0: ("OFFENSIVE_ACTION", "ASSAULT"),
    1: ("AREA_CONTROL", "HOLD"),
    2: ("RECON_INFORMATION", "RECON"),
}

GRADE_B_EVIDENCE = "deterministic_rule"
GRADE_C_EVIDENCE = "plan_tag_or_human_review"


def _identifier(value: Any) -> str | None:
    if value is None: return None
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _step(frame, fallback):
    t = frame.get("time", {})
    return int(t.get("cur_step", frame.get("cur_step", fallback)))


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _segment_id(label_kind, intent, subject_id, start_step, evidence):
    identity = {"label_kind": label_kind, "intent": intent, "subject_id": subject_id,
                "start_step": start_step, "evidence": evidence}
    digest = hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()[:12]
    return f"{label_kind}_{intent.lower()}_{digest}"


def _mission_segment(command, frame_step, evidence_ref):
    info = command.get("info", {})
    if not isinstance(info, dict): return None
    try: mission_type = int(info.get("mission_type"))
    except (TypeError, ValueError): return None
    mapping = MISSION_INTENTS.get(mission_type)
    subject_id = _identifier(info.get("company_id"))
    if mapping is None or subject_id is None: return None
    start_step = int(info.get("start_time", frame_step))
    end_step = max(start_step, int(info.get("end_time", start_step)))
    parent, intent = mapping
    target_id = _identifier(info.get("target_pos"))
    return {"segment_id": _segment_id("commanded", intent, subject_id, start_step, command),
            "label_kind": "commanded", "subject_scope": "company", "subject_id": subject_id,
            "start_step": start_step, "end_step": end_step, "parent_intent": parent,
            "primary_intent": intent, "secondary_intents": [],
            "target_type": "objective_point" if target_id else "none", "target_id": target_id,
            "status": "planned", "outcome": "unknown", "evidence_source": "mission_command",
            "evidence_grade": "A", "evidence_refs": [evidence_ref], "confidence": 1.0}


def _action_was_accepted(action):
    if action.get("success") is False: return False
    for k in ("error", "error_message", "error_msg"):
        if action.get(k) not in (None, "", 0, False): return False
    if action.get("error_code") not in (None, "", 0, "0"): return False
    return True


def _realized_segment(*, action, frame_step, intent, parent, subject_id, target_type, target_id, evidence_ref):
    subject = _identifier(subject_id)
    if subject is None: return None
    step = int(action.get("cur_step", action.get("step", frame_step)))
    target = _identifier(target_id)
    return {"segment_id": _segment_id("realized", intent, subject, step, action), "label_kind": "realized",
            "subject_scope": "entity", "subject_id": subject, "start_step": step, "end_step": step,
            "parent_intent": parent, "primary_intent": intent, "secondary_intents": [],
            "target_type": target_type if target else "none", "target_id": target,
            "status": "completed", "outcome": "achieved", "evidence_source": "deterministic_rule",
            "evidence_grade": "A", "evidence_refs": [evidence_ref], "confidence": 1.0}


def _segments_from_action(action, frame_step, evidence_ref):
    if not _action_was_accepted(action): return []
    try: at = int(action.get("type"))
    except (TypeError, ValueError): return []
    segs = []
    if at == 5:
        segs.append(_realized_segment(action=action, frame_step=frame_step, intent="SEIZE",
            parent="AREA_CONTROL", subject_id=action.get("obj_id"),
            target_type="objective_point", target_id=action.get("target_pos"), evidence_ref=evidence_ref))
    elif at == 9:
        segs.append(_realized_segment(action=action, frame_step=frame_step, intent="TARGET_GUIDANCE",
            parent="RECON_INFORMATION", subject_id=action.get("obj_id"),
            target_type="enemy_entity", target_id=action.get("target_obj_id"), evidence_ref=evidence_ref))
        segs.append(_realized_segment(action=action, frame_step=frame_step, intent="FIRE_SUPPORT",
            parent="SUPPORT_COORDINATION", subject_id=action.get("guided_obj_id"),
            target_type="enemy_entity", target_id=action.get("target_obj_id"), evidence_ref=evidence_ref))
    elif at == 8:
        segs.append(_realized_segment(action=action, frame_step=frame_step, intent="FIRE_SUPPORT",
            parent="SUPPORT_COORDINATION", subject_id=action.get("obj_id"),
            target_type="terrain_region", target_id=action.get("jm_pos"), evidence_ref=evidence_ref))
    elif at in {15, 315}:
        segs.append(_realized_segment(action=action, frame_step=frame_step, intent="ASSEMBLE",
            parent="MANEUVER_POSTURE", subject_id=action.get("obj_id"),
            target_type="friendly_entity", target_id=action.get("target_obj_id"), evidence_ref=evidence_ref))
    elif at in {14, 314}:
        segs.append(_realized_segment(action=action, frame_step=frame_step, intent="DISPERSE",
            parent="MANEUVER_POSTURE", subject_id=action.get("obj_id"),
            target_type="none", target_id=None, evidence_ref=evidence_ref))
    elif at == 18:
        segs.append(_realized_segment(action=action, frame_step=frame_step, intent="FORTIFY",
            parent="MANEUVER_POSTURE", subject_id=action.get("obj_id"),
            target_type="terrain_region", target_id=action.get("target_obj_id"), evidence_ref=evidence_ref))
    elif at == 6 and action.get("target_state") == 4:
        segs.append(_realized_segment(action=action, frame_step=frame_step, intent="FORTIFY",
            parent="MANEUVER_POSTURE", subject_id=action.get("obj_id"),
            target_type="none", target_id=None, evidence_ref=evidence_ref))
    return [s for s in segs if s is not None]


def _hex_distance(a, b):
    if a is None or b is None: return float("inf")
    try: ra, ca = int(a)//100, int(a)%100; rb, cb = int(b)//100, int(b)%100
    except: return float("inf")
    dr, dc = abs(ra-rb), abs(ca-cb)
    return float(max(dr, dc, dr+dc) if dr>=0 and dc>=0 else dr+abs(dc))


def _bearing_change_degrees(hex_seq):
    if len(hex_seq) < 3: return None
    try: r0, c0 = int(hex_seq[0])//100, int(hex_seq[0])%100
    except: return None
    mc = 0.0
    for i in range(2, len(hex_seq)):
        if not hex_seq[i-1] or not hex_seq[i]: continue
        try:
            r1,c1=int(hex_seq[i-1])//100,int(hex_seq[i-1])%100; r2,c2=int(hex_seq[i])//100,int(hex_seq[i])%100
            b1=math.degrees(math.atan2(r1-r0,c1-c0))%360; b2=math.degrees(math.atan2(r2-r1,c2-c1))%360
            d=abs(b2-b1); mc=max(mc,min(d,360-d))
        except: continue
    return mc


def _entity_trajectory(truth_frames):
    traj = defaultdict(list)
    for si, frame in enumerate(truth_frames):
        if not isinstance(frame, dict): continue
        cs = _step(frame, si)
        for oi, op in enumerate(frame.get("operators", [])):
            traj[oi].append({"step": cs, "hex": _identifier(op.get("cur_hex")),
                             "hp": op.get("blood"), "ms": op.get("move_state"),
                             "speed": op.get("basic_speed", op.get("speed", 0))})
    return dict(traj)


def _detect_flank(seq):
    if len(seq) < 5: return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    bc = _bearing_change_degrees(hexes)
    if bc and bc > 45:
        return [_realized_segment(action={"cur_step": seq[0]["step"]}, frame_step=seq[0]["step"],
            intent="FLANK", parent="OFFENSIVE_ACTION", subject_id=str(seq[0].get("op_index", 0)),
            target_type="none", target_id=None, evidence_ref=f"flank:bc={bc:.0f}")]
    return []


def _detect_breakthrough(seq):
    if len(seq) < 5: return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    hps = [s["hp"] for s in seq if isinstance(s.get("hp"), (int, float))]
    if len(hexes) < 3 or len(hps) < 3: return []
    dist = _hex_distance(hexes[0], hexes[-1])
    lost = max(hps) - min(hps)
    if dist > 4 and lost > 0:
        return [_realized_segment(action={"cur_step": seq[0]["step"]}, frame_step=seq[0]["step"],
            intent="BREAKTHROUGH", parent="OFFENSIVE_ACTION", subject_id=str(seq[0].get("op_index", 0)),
            target_type="none", target_id=None, evidence_ref=f"breakthrough:d={dist:.0f}")]
    return []


def _detect_pursuit(seq):
    if len(seq) < 10: return []
    spds = [s["speed"] for s in seq[len(seq)//2:] if isinstance(s.get("speed"), (int, float))]
    if len(spds) < 3 or sum(spds)/len(spds) < 0.01: return []
    return [_realized_segment(action={"cur_step": seq[0]["step"]}, frame_step=seq[0]["step"],
        intent="PURSUIT", parent="OFFENSIVE_ACTION", subject_id=str(seq[0].get("op_index", 0)),
        target_type="none", target_id=None, evidence_ref=f"pursuit:spd={sum(spds)/len(spds):.2f}")]


def _detect_ambush(seq):
    if len(seq) < 5: return []
    concealed = sum(1 for s in seq if s.get("ms") in (4, 5))
    if concealed >= 3:
        return [_realized_segment(action={"cur_step": seq[0]["step"]}, frame_step=seq[0]["step"],
            intent="AMBUSH", parent="OFFENSIVE_ACTION", subject_id=str(seq[0].get("op_index", 0)),
            target_type="none", target_id=None, evidence_ref=f"ambush:concealed={concealed}")]
    return []


def _detect_surveillance(seq):
    if len(seq) < 10: return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    if not hexes: return []
    unique = set(hexes[-10:])
    if len(unique) <= 2:
        return [_realized_segment(action={"cur_step": seq[0]["step"]}, frame_step=seq[0]["step"],
            intent="SURVEILLANCE", parent="RECON_INFORMATION", subject_id=str(seq[0].get("op_index", 0)),
            target_type="none", target_id=None, evidence_ref="surveillance:10_plus_low_dispersion")]
    return []


def _detect_redeploy(seq):
    if len(seq) < 5: return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    if len(hexes) >= 2 and _hex_distance(hexes[0], hexes[-1]) > 5:
        return [_realized_segment(action={"cur_step": seq[0]["step"]}, frame_step=seq[0]["step"],
            intent="REDEPLOY", parent="MANEUVER_POSTURE", subject_id=str(seq[0].get("op_index", 0)),
            target_type="none", target_id=None,
            evidence_ref=f"redeploy:d={_hex_distance(hexes[0], hexes[-1]):.0f}")]
    return []


def _detect_retreat(seq):
    if len(seq) < 6: return []
    hps = [s["hp"] for s in seq if isinstance(s.get("hp"), (int, float))]
    if len(hps) < 3 or max(hps) - min(hps) <= 0: return []
    return [_realized_segment(action={"cur_step": seq[0]["step"]}, frame_step=seq[0]["step"],
        intent="RETREAT", parent="DISENGAGEMENT_DECEPTION", subject_id=str(seq[0].get("op_index", 0)),
        target_type="none", target_id=None, evidence_ref=f"retreat:hp_lost={max(hps)-min(hps)}")]


def _detect_evade(seq):
    if len(seq) < 5: return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    bc = _bearing_change_degrees(hexes)
    if bc and bc > 60:
        return [_realized_segment(action={"cur_step": seq[1]["step"]}, frame_step=seq[1]["step"],
            intent="EVADE", parent="DISENGAGEMENT_DECEPTION", subject_id=str(seq[0].get("op_index", 0)),
            target_type="none", target_id=None, evidence_ref=f"evade:bc={bc:.0f}")]
    return []


def _detect_interdict(seq):
    if len(seq) < 8: return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    hps = [s["hp"] for s in seq if isinstance(s.get("hp"), (int, float))]
    if not hexes or not hps: return []
    unique_hexes = set(hexes[-8:])
    if len(unique_hexes) <= 2 and max(hps) > min(hps):
        return [_realized_segment(action={"cur_step": seq[0]["step"]}, frame_step=seq[0]["step"],
            intent="INTERDICT", parent="AREA_CONTROL", subject_id=str(seq[0].get("op_index", 0)),
            target_type="none", target_id=None, evidence_ref="interdict:held_with_damage")]
    return []


def _detect_escort(seq):
    if len(seq) < 5: return []
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    if len(hexes) >= 3:
        return [_realized_segment(action={"cur_step": seq[0]["step"]}, frame_step=seq[0]["step"],
            intent="ESCORT", parent="SUPPORT_COORDINATION", subject_id=str(seq[0].get("op_index", 0)),
            target_type="none", target_id=None, evidence_ref=f"escort:steps={len(seq)}")]
    return []


def _detect_reinforce(seq):
    if len(seq) < 5: return []
    hps = [s["hp"] for s in seq if isinstance(s.get("hp"), (int, float))]
    hexes = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
    if len(hexes) >= 2 and len(hps) >= 3 and max(hps) - min(hps) == 0:
        return [_realized_segment(action={"cur_step": seq[0]["step"]}, frame_step=seq[0]["step"],
            intent="REINFORCE", parent="SUPPORT_COORDINATION", subject_id=str(seq[0].get("op_index", 0)),
            target_type="none", target_id=None,
            evidence_ref=f"reinforce:d={_hex_distance(hexes[0], hexes[-1]):.0f}")]
    return []


def _detect_encirclement(traj, num_ops):
    if num_ops < 2: return []
    hexes_by_op = {}
    for op_idx in range(num_ops):
        seq = traj.get(op_idx, [])
        hxs = [s["hex"] for s in seq if s["hex"] and s["hex"] != "null"]
        if hxs: hexes_by_op[op_idx] = hxs
    if len(hexes_by_op) < 2: return []
    return [_realized_segment(action={"cur_step": 0}, frame_step=0,
        intent="ENCIRCLEMENT", parent="OFFENSIVE_ACTION", subject_id="multi",
        target_type="none", target_id=None, evidence_ref=f"encirclement:ops={list(hexes_by_op.keys())}")]


GRADE_B_DETECTORS = {
    "FLANK": _detect_flank, "BREAKTHROUGH": _detect_breakthrough,
    "PURSUIT": _detect_pursuit, "AMBUSH": _detect_ambush,
    "SURVEILLANCE": _detect_surveillance, "REDEPLOY": _detect_redeploy,
    "RETREAT": _detect_retreat, "EVADE": _detect_evade,
    "INTERDICT": _detect_interdict, "ESCORT": _detect_escort,
    "REINFORCE": _detect_reinforce,
}


def derive_grade_b_intent_plan(raw_episode):
    episode_id = _identifier(raw_episode.get("metadata", {}).get("episode_id"))
    truth_frames = raw_episode.get("observations", {}).get("truth", [])
    if not isinstance(truth_frames, list): raise ValueError("observations.truth must be a list")
    traj = _entity_trajectory(truth_frames)
    all_segments = {}
    for op_index, seq in traj.items():
        si = str(op_index)
        for intent_name, detector in GRADE_B_DETECTORS.items():
            for seg in detector(seq):
                seg["evidence_grade"] = "B"; seg["evidence_source"] = GRADE_B_EVIDENCE
                sid = f"grade_b_{intent_name.lower()}_{si}_{seg['start_step']}"
                seg["segment_id"] = sid
                if sid not in all_segments: all_segments[sid] = seg
    # ENCIRCLEMENT
    for seg in _detect_encirclement(traj, len(traj)):
        seg["evidence_grade"] = "B"; seg["evidence_source"] = GRADE_B_EVIDENCE
        sid = f"grade_b_encirclement_multi_{seg['start_step']}"
        seg["segment_id"] = sid
        if sid not in all_segments: all_segments[sid] = seg
    segments = list(all_segments.values())
    segments.sort(key=lambda x: (x["start_step"], x["subject_id"], x["primary_intent"], x["segment_id"]))
    return {"episode_id": episode_id, "generator": "shenji_grade_b_labeler_v1", "segments": segments}


def derive_grade_c_intent_plan(raw_episode):
    return {"episode_id": _identifier(raw_episode.get("metadata", {}).get("episode_id")),
            "generator": "shenji_grade_c_labeler_v1", "segments": []}


def derive_grade_a_intent_plan(raw_episode):
    episode_id = _identifier(raw_episode.get("metadata", {}).get("episode_id"))
    truth_frames = raw_episode.get("observations", {}).get("truth", [])
    if not isinstance(truth_frames, list): raise ValueError("observations.truth must be a list")
    segments_by_id = {}; seen_commands = set(); seen_actions = set()
    top_level_actions = raw_episode.get("actions", [])
    if not isinstance(top_level_actions, list): top_level_actions = []
    for idx, action in enumerate(top_level_actions):
        if not isinstance(action, dict): continue
        frame_step = int(action.get("cur_step", action.get("step", 0)))
        action_type = int(action.get("type", -1))
        if action_type == 200:
            ck = _canonical(action)
            if ck in seen_commands: continue
            seen_commands.add(ck)
            s = _mission_segment(action, frame_step, f"top_level_action:{frame_step}:{idx}")
            if s: segments_by_id[s["segment_id"]] = s
        else:
            ak = f"top:{frame_step}:{_canonical(action)}"
            if ak in seen_actions: continue
            seen_actions.add(ak)
            for s in _segments_from_action(action, frame_step, f"top_level_action:{frame_step}:{idx}"):
                segments_by_id[s["segment_id"]] = s
    for fi, frame in enumerate(truth_frames):
        if not isinstance(frame, dict): continue
        frame_step = _step(frame, fi)
        comms = frame.get("communication", frame.get("communications", []))
        if isinstance(comms, dict): comms = [comms]
        for ci, cmd in enumerate(comms or []):
            if not isinstance(cmd, dict) or cmd.get("type") != 200: continue
            ck = _canonical(cmd)
            if ck in seen_commands: continue
            seen_commands.add(ck)
            s = _mission_segment(cmd, frame_step, f"truth_communication:{frame_step}:{ci}")
            if s: segments_by_id[s["segment_id"]] = s
        acts = frame.get("actions", [])
        if isinstance(acts, dict): acts = [acts]
        for ai, act in enumerate(acts or []):
            if not isinstance(act, dict): continue
            ak = f"legacy:{frame_step}:{_canonical(act)}"
            if ak in seen_actions: continue
            seen_actions.add(ak)
            for s in _segments_from_action(act, frame_step, f"truth_action:{frame_step}:{ai}"):
                segments_by_id[s["segment_id"]] = s
    segments = list(segments_by_id.values())
    segments.sort(key=lambda item: (item["start_step"], item["label_kind"],
                                     item["subject_id"], item["primary_intent"], item["segment_id"]))
    return {"episode_id": episode_id, "generator": "shenji_grade_a_labeler_v1", "segments": segments}
