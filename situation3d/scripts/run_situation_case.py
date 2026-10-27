#!/usr/bin/env python3
"""Run a TW2-style campaign through the geo engine and export a replay.

The case plugin only provides the initial scenario.  This script builds the
world, steps it, applies missions/ROE, and writes an immutable run artifact
(``trajectory.json`` / ``nodes.json`` / ``summary.json``) into the run store.

Usage:
    python3 scripts/run_situation_case.py --case taiwan_dense_transfer_v1
    python3 scripts/run_situation_case.py --case ... --ticks 120 --runs-dir /tmp/runs
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import date, datetime, timezone
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
for p in (SITUATION3D, os.path.join(SITUATION3D, "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine.scenario import Target  # noqa: E402
from engine.rolecards.legacy import card_from_legacy  # noqa: E402
from engine import symbology  # noqa: E402
from geo.geodesy import haversine_m  # noqa: E402
from sim.mission import MissionRunner  # noqa: E402
from sim import tasking  # noqa: E402
from sim.scenario import build_world  # noqa: E402
from sim.collision import obbs_overlap  # noqa: E402
from sim.motion import entity_profile  # noqa: E402
from run_store import RunStore  # noqa: E402
from resources import preflight, providers  # noqa: E402
from resources.scanners import scan_envpack, scan_repo  # noqa: E402
from resources.store import ResourceStore  # noqa: E402

CASE_MODULES = {"taiwan_dense_transfer_v1": "taiwan_dense"}
DEFAULT_ENVPACK = os.path.join(SITUATION3D, "envpacks", "taiwan_tw2_v1")


def load_plugin(case_id):
    module_name = CASE_MODULES.get(case_id)
    if not module_name:
        raise SystemExit("unknown engine case: %s" % case_id)
    path = os.path.join(SITUATION3D, "campaigns", module_name + ".py")
    spec = importlib.util.spec_from_file_location("case_" + module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _git_commit():
    try:
        out = subprocess.check_output(["git", "-C", SITUATION3D, "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return out.decode("ascii").strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _envpack_hash(path):
    manifest = os.path.join(path, "manifest.json")
    if not os.path.isfile(manifest):
        return None
    h = hashlib.sha256()
    with open(manifest, "rb") as fh:
        h.update(fh.read())
    # Include the layer payloads: re-baking geometry with an unchanged manifest
    # must change the provenance hash.
    try:
        with open(manifest, encoding="utf-8") as fh:
            doc = json.load(fh)
        layers = doc.get("layers") or {}
    except (OSError, ValueError):
        layers = {}
    for name in sorted(layers):
        rel = (layers[name] or {}).get("path")
        if not rel:
            continue
        full = os.path.join(path, rel)
        if not os.path.isfile(full):
            continue
        h.update(name.encode("utf-8"))
        with open(full, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def _clear_subcell_buildings(world):
    """Move stopped ground units out of small building footprints.

    The 30 m navigation field only samples cell centres; a footprint smaller
    than a cell can be missed. This pass only touches units that are already
    stopped and lands them on the nearest open point (road preferred), so the
    replay does not show a vehicle parked inside a house. Movement itself
    stays governed by the engine planning/collision.
    """
    fixed = 0
    for entity in world.sorted_entities():
        if entity.on_board or not entity.alive:
            continue
        if entity.domain not in ("land", "amphibious") or entity.velocity_ms > 0.15:
            continue
        if world.env.building_at(entity.pose.lon, entity.pose.lat) is None:
            continue
        lon0, lat0 = entity.pose.lon, entity.pose.lat
        kx = 111320.0 * math.cos(math.radians(lat0))
        best = None
        for radius in (15.0, 30.0, 50.0, 80.0):
            for k in range(12):
                ang = math.radians(30.0 * k)
                lon = lon0 + radius * math.cos(ang) / kx
                lat = lat0 + radius * math.sin(ang) / 110540.0
                if world.env.building_at(lon, lat) is not None:
                    continue
                depth = world.env.water_depth(lon, lat)
                if depth is not None and not getattr(entity.profile, "amphibious", False):
                    if depth > getattr(entity.profile, "wading_depth_m", 1.2):
                        continue
                road = False
                try:
                    road = bool(world.env.on_road(lon, lat, 25.0))
                except AttributeError:
                    road = True
                if getattr(entity.profile, "road_bound", False) and not road:
                    continue
                blocked = False
                try:
                    prof = entity_profile(entity)
                    for other in world.sorted_entities():
                        if other.uid == entity.uid or other.on_board:
                            continue
                        if other.domain not in ("land", "amphibious"):
                            continue
                        if obbs_overlap(prof, lon, lat, entity.pose.heading_deg,
                                        entity_profile(other), other.pose.lon,
                                        other.pose.lat, other.pose.heading_deg):
                            blocked = True
                            break
                except Exception:  # noqa: BLE001
                    blocked = False
                if blocked:
                    continue
                score = radius + (0.0 if road else 30.0)
                if best is None or score < best[0]:
                    best = (score, lon, lat)
            if best is not None:
                break
        if best is not None:
            entity.pose.lon, entity.pose.lat = best[1], best[2]
            entity.pose.alt_m = world.env.elevation(best[1], best[2])
            fixed += 1
    return fixed


def _phase_at(phases, tick):
    current = phases[0] if phases else {"id": "S0", "name": "", "first_tick": 0}
    for p in phases:
        if int(p.get("first_tick", 0)) <= tick:
            current = p
    return {"id": current.get("id"), "name": current.get("name"),
            "index": phases.index(current) if current in phases else 0}


def _stage_id(stages, tick):
    for st in stages:
        if int(st.get("start_tick", 0)) <= tick < int(st.get("end_tick", 10 ** 9)):
            return st.get("id")
    return stages[-1].get("id") if stages else "S0"


def _ui_domain(entity):
    if entity.domain in ("land", "amphibious"):
        return "land"
    if entity.domain == "surface":
        return "sea"
    return "air"


def _target_defs(world, meta):
    defs = []
    for cid in sorted(meta):
        m = meta[cid]
        entity = world.entities.get(int(m["uid"]))
        if entity is None:
            continue
        attrs = {"编组": m.get("group", ""), "任务": m.get("task", ""),
                 "合成对象": "是", "数据来源": "engine"}
        t = Target(
            target_id=cid, name=m["name"], platform_class=m["platform_class"],
            domain=_ui_domain(entity),
            affiliation="red" if entity.affiliation == 0 else "blue",
            waypoints=[], speed_kn=float(m.get("speed_kn") or 0.0),
            altitude_m=float(entity.pose.alt_m or 0.0),
            role=m.get("role", ""), model=m["model"], icon=m.get("icon", "unknown"),
            sensor_range_km=float(m.get("sensor_km") or 0.0),
            influence_km=float(m.get("influence_km") or 0.0),
            engage_km=float(m.get("engage_km") or 0.0),
            road_bound=(_ui_domain(entity) == "land"),
            attributes=attrs, role_card=entity.role_card,
        )
        entry = t.to_dataset_entry()
        entry["entity_kind"] = ("vehicle" if _ui_domain(entity) == "land" else
                                ("vessel" if _ui_domain(entity) == "sea" else "aircraft"))
        entry["representation"] = ("single_vehicle" if _ui_domain(entity) == "land" else
                                   ("single_vessel" if _ui_domain(entity) == "sea" else "single_aircraft"))
        entry["group_id"] = m.get("group", "")
        entry["task_id"] = (m.get("group", "") + "-T") if m.get("group") else ""
        entry["initial_status"] = "ready"
        entry["initial_mode"] = "embarked" if entity.on_board else (
            "sea" if _ui_domain(entity) == "sea" else ("air" if _ui_domain(entity) == "air" else "ground"))
        entry["collision_bbox"] = {
            "length_m": float(m.get("length_m") or 5.0),
            "width_m": float(m.get("width_m") or 2.5),
            "height_m": float(m.get("height_m") or 2.0),
            "source": "platform_profile",
        }
        # The browser panel expects the legacy snapshot shape
        # (class_id/summary/projection); the raw card is kept only for tests.
        try:
            entry["role_card"] = card_from_legacy({
                "target_id": cid, "name": m["name"],
                "platform_class": m["platform_class"], "domain": _ui_domain(entity),
                "affiliation": "red" if entity.affiliation == 0 else "blue",
                "role": m.get("role", ""), "model": m["model"],
                "icon": m.get("icon", "unknown"),
                "speed_kn": float(m.get("speed_kn") or 0.0),
                "altitude_m": float(entity.pose.alt_m or 0.0),
                "road_bound": _ui_domain(entity) == "land",
                "subclass_id": m.get("subclass_id"),
            }, include_effective=False)
        except Exception:  # noqa: BLE001 - keep the raw card if the registry rejects it
            pass
        if _ui_domain(entity) == "land":
            entry["lane"] = {"id": 0, "width_m": 3.75, "center_offset_m": 1.875,
                             "side": "right"}
        defs.append(entry)
    return defs


def _frame_row(entity, cid, time_s):
    on_board = bool(entity.on_board)
    domain = _ui_domain(entity)
    mode = ("embarked" if on_board else
            ("sea" if domain == "sea" else ("air" if domain == "air" else "ground")))
    if not entity.alive:
        status = "destroyed"
    elif entity.blood < entity.max_blood:
        status = "damaged"
    else:
        status = "ready"
    recent_fire = (entity.last_fired_s is not None and entity.last_fired_s >= 0.0
                   and time_s - entity.last_fired_s <= 120.0)
    if entity.route is not None or entity.moving:
        task_status = "moving"
    elif recent_fire:
        task_status = "engaged"
    else:
        task_status = "waiting"
    row = {
        "id": cid,
        "lat": round(float(entity.pose.lat), 6),
        "lon": round(float(entity.pose.lon), 6),
        "alt": round(float(entity.pose.alt_m or 0.0), 1),
        "speed_kn": round(float(entity.velocity_ms or 0.0) * 1.943844, 1),
        "heading": round(float(entity.pose.heading_deg or 0.0), 1),
        "travelled_m": 0.0,
        "laps": 0,
        "reached_end": False,
        "mode": mode,
        "status": status,
        "task_status": task_status,
        "strength": round(float(entity.blood) / max(1.0, float(entity.max_blood)), 3),
        "mobility_factor": round(float((entity.components or {}).get("mobility", 1.0)), 3),
    }
    return row


def _frame_row2(entity, cid, uid_to_cid, time_s):
    row = _frame_row(entity, cid, time_s)
    if entity.on_board and entity.carrier is not None:
        row["carrier_id"] = uid_to_cid.get(int(entity.carrier))
    return row


def _event_base(ev_id, tick, stage_id, kind, actors, location, desc,
                source="engine", cause=None, effects=None):
    return {
        "id": ev_id, "tick": int(tick), "stage_id": stage_id, "type": kind,
        "source": source, "producer": "sim.world", "actors": list(actors),
        "location": [round(float(location[0]), 6), round(float(location[1]), 6)] if location else [0.0, 0.0],
        "effects": effects or {"before": {}, "after": {}},
        "cause_event_ids": list(cause or []), "evidence_ids": [],
        "visibility": "global", "description": desc,
    }


def _convert_events(world, uid_to_cid, frames, tick_seconds, tick_count, stages):
    def pos(tick, cid):
        t = max(0, min(int(tick), len(frames) - 1))
        for row in frames[t]["targets"]:
            if row["id"] == cid:
                return (row["lon"], row["lat"])
        return None

    def uid(cid):
        return uid_to_cid.get(int(cid))

    events = []
    last_fire = {}
    last_impact = {}
    for ev in world.events:
        typ = str(ev.get("type") or "")
        tick = int(round(float(ev.get("tick_s") or (ev.get("tick") or 0) * tick_seconds) / tick_seconds))
        tick = max(0, min(tick_count, tick))
        stage = _stage_id(stages, tick)
        actors_uid = [int(a) for a in (ev.get("actors") or [])]
        actors = [uid(a) for a in actors_uid]
        actors = [a for a in actors if a]
        loc = ev.get("location_geo") or {}
        location = (loc.get("lon"), loc.get("lat")) if loc else None
        if location is None and actors:
            location = pos(tick, actors[0])
        eid = str(ev.get("id") or ("EV-%08d" % (len(events) + 1)))
        if typ == "direct_fire":
            shooter, target = (actors + [None, None])[:2]
            if not shooter or not target:
                continue
            fire_id = "F-" + eid
            tgt_pos = pos(tick, target)
            fire = _event_base(fire_id, tick, stage, "fire", [shooter, target],
                               location or tgt_pos, ev.get("description") or
                               ("%s 向 %s 开火" % (shooter, target)),
                               cause=[], effects={"before": {}, "after": {}})
            fire.update({"shooter_id": shooter, "target_id": target,
                         "target_location": list(tgt_pos) if tgt_pos else None,
                         "weapon": ev.get("weapon"), "result": ev.get("result"),
                         "damage": ev.get("damage"), "distance_m": ev.get("distance_m"),
                         "tof_s": ev.get("tof_s")})
            events.append(fire)
            last_fire[(shooter, target)] = fire_id
            impact_tick = min(tick_count, tick + int(float(ev.get("tof_s") or 0.0) // tick_seconds))
            impact_pos = pos(impact_tick, target) or tgt_pos
            detail = str(ev.get("result") or "")
            hit = detail not in ("", "miss", "none")
            imp = _event_base("I-" + eid, impact_tick, _stage_id(stages, impact_tick), "impact",
                              [shooter, target], impact_pos,
                              "命中" if hit else "未命中", cause=[fire_id])
            imp.update({"shooter_id": shooter, "target_id": target,
                        "result": "hit" if hit else "miss",
                        "result_detail": detail, "damage": ev.get("damage")})
            events.append(imp)
            if hit:
                last_impact[target] = imp["id"]
        elif typ == "indirect_fire":
            if not actors:
                continue
            shooter = actors[0]
            fire_id = "F-" + eid
            fire = _event_base(fire_id, tick, stage, "fire", actors, location,
                               ev.get("description") or ("%s 间瞄射击" % shooter))
            fire.update({"shooter_id": shooter, "weapon": ev.get("weapon"),
                         "result": ev.get("outcome") or "area",
                         "damage": ev.get("damage")})
            events.append(fire)
            for target in actors[1:]:
                tpos = pos(tick, target)
                imp = _event_base("I-%s-%s" % (eid, target), tick, stage, "impact",
                                  [shooter, target], tpos or location,
                                  "间瞄命中" if (ev.get("damage") or 0) > 0 else "间瞄落点",
                                  cause=[fire_id])
                imp.update({"shooter_id": shooter, "target_id": target,
                            "result": "hit" if (ev.get("damage") or 0) > 0 else "miss",
                            "damage": ev.get("damage")})
                events.append(imp)
                if (ev.get("damage") or 0) > 0:
                    last_impact[target] = imp["id"]
        elif typ == "sea_air_fire":
            shooter, target = (actors + [None, None])[:2]
            if not shooter or not target:
                continue
            fire_id = "F-" + eid
            fire = _event_base(fire_id, tick, stage, "fire", [shooter, target],
                               location or pos(tick, shooter) or pos(tick, target),
                               ev.get("description") or ("%s 发射导弹" % shooter))
            fire.update({"shooter_id": shooter, "target_id": target,
                         "weapon": ev.get("weapon"), "result": ev.get("result"),
                         "damage": ev.get("damage")})
            events.append(fire)
            last_fire[(shooter, target)] = fire_id
        elif typ == "missile_impact":
            shooter, target = (actors + [None, None])[:2]
            if not shooter or not target:
                continue
            cause = [last_fire[(shooter, target)]] if (shooter, target) in last_fire else []
            detail = str(ev.get("result") or "")
            label = ("被拦截" if detail == "intercepted" else
                     ("命中" if detail == "hit" else "未命中"))
            imp = _event_base("I-" + eid, tick, stage, "impact", [shooter, target],
                              location or pos(tick, target),
                              ev.get("description") or label, cause=cause)
            imp.update({"shooter_id": shooter, "target_id": target,
                        "result": "hit" if detail == "hit" else "miss",
                        "result_detail": detail, "damage": ev.get("damage"),
                        "intercepted": ev.get("intercepted")})
            events.append(imp)
            if detail == "hit":
                last_impact[target] = imp["id"]
        elif typ == "destroyed":
            target = actors[0] if actors else None
            if not target:
                continue
            destroy = _event_base("D-" + eid, tick, stage, "destroy", [target],
                                  location or pos(tick, target),
                                  ev.get("description") or ("%s 被摧毁" % target),
                                  cause=[last_impact[target]] if target in last_impact else [])
            destroy.update({"target_id": target})
            events.append(destroy)
        elif typ == "objective":
            desc = str(ev.get("description") or "")
            owner = -1
            obj_name = ""
            for cell, obj in world.objectives.items():
                if obj.get("name") and obj.get("name") in desc:
                    obj_name = obj.get("name")
                    owner = int(obj.get("owner", -1))
                    lon, lat = world.rule_grid.cell_center_lonlat(*cell)
                    location = (lon, lat)
                    break
            events.append(_event_base("O-" + eid, tick, stage, "objective_progress",
                                      [], location, desc, source="rule"))
        elif typ in ("mission_start", "mission_complete", "mission_failed"):
            events.append(_event_base("M-" + eid, tick, stage, "mission",
                                      actors, location, ev.get("description") or typ,
                                      source="mission"))
        elif typ == "disembark":
            events.append(_event_base("R-" + eid, tick, stage, "release", actors,
                                      location, ev.get("description") or "下船",
                                      source="rule"))
        elif typ in ("move_failed",):
            events.append(_event_base("W-" + eid, tick, stage, "warning", actors,
                                      location, ev.get("description") or typ,
                                      source="engine"))
    order = {"fire": 0, "impact": 1, "destroy": 2, "mission": 3,
             "release": 4, "objective_progress": 5, "warning": 6}
    events.sort(key=lambda e: (e["tick"], order.get(e["type"], 9), e["id"]))
    # The engine emits "destroyed" from inside the damage call, before the
    # direct-fire event is appended. With the impact sorted first, link the
    # destruction to the last hit on the same target.
    last_hit = {}
    for ev in events:
        if ev["type"] == "impact" and ev.get("result") == "hit" and ev.get("target_id"):
            last_hit[ev["target_id"]] = ev["id"]
        elif ev["type"] == "destroy":
            target = ev.get("target_id") or (ev.get("actors") or [None])[0]
            if target in last_hit and not ev.get("cause_event_ids"):
                ev["cause_event_ids"] = [last_hit[target]]
    return events


def _nodes(events, tick_count, phases, include_terminal=True):
    wanted = [("first_fire", "首次交火", "fire"),
              ("first_destroy", "首次毁伤", "destroy"),
              ("first_objective", "首次夺控", "objective_progress"),
              ("first_mission", "首次任务开始", "mission")]
    seen = {}
    nodes = []
    for ev in events:
        for key, title, typ in wanted:
            if key in seen or ev["type"] != typ:
                continue
            if typ == "mission" and ev.get("description", "").find("开始") < 0:
                continue
            seen[key] = True
            nodes.append({"tick": ev["tick"], "type": key, "title": title,
                          "actors": ev.get("actors") or [],
                          "location": ev.get("location"),
                          "description": ev.get("description") or ""})
    if include_terminal:
        nodes.append({"tick": tick_count, "type": "terminal", "title": "推演结束",
                      "actors": [], "location": None, "description": "达到想定时长"})
    nodes.sort(key=lambda n: n["tick"])
    return {"schema": "sim.nodes.v1", "nodes": nodes}


# ---------------------------------------------------------------------------
# 实时增量产物（live/）
# ---------------------------------------------------------------------------
#
# 测试端查看器不等推演结束：每一 tick 追加一帧到 live/frames.jsonl，并把该行
# 的字节偏移同步追加到 live/frames.idx；每 LIVE_EVERY_TICKS 个 tick 重写一次
# events.json / nodes.json / state.json。查看器用"帧数/事件数"作游标，
# serve.py 的 /live 接口按偏移直接 seek，不必每次重读 33 MB 的成品 trajectory。
#
# 写入顺序固定为 idx 先、行数据后，读取方只要忽略末尾不完整的行，就不会把
# 写了一半的 JSON 解析进场景。

LIVE_EVERY_TICKS = 10


class LiveRunWriter:
    def __init__(self, store, case_id, run_id):
        self.dir = os.path.join(store.run_dir(case_id, run_id), "live")
        os.makedirs(self.dir, exist_ok=True)
        self._frames = open(os.path.join(self.dir, "frames.jsonl"), "wb")
        self._index = open(os.path.join(self.dir, "frames.idx"), "wb")
        self._offset = 0
        self.frame_count = 0
        self.last_tick = 0

    def meta(self, doc):
        _atomic_write_json(os.path.join(self.dir, "meta.json"), doc)

    def append_frame(self, frame):
        data = (json.dumps(frame, ensure_ascii=False, separators=(",", ":"))
                + "\n").encode("utf-8")
        # idx 必须在行落地之前可见：读取方用 idx 定位，定位到尚未写好的行时
        # 会自然读到 EOF 并丢弃，不会错位。
        self._index.write(("%d\n" % self._offset).encode("ascii"))
        self._index.flush()
        self._frames.write(data)
        self._frames.flush()
        self._offset += len(data)
        self.frame_count += 1
        self.last_tick = int(frame.get("tick") or 0)

    def publish(self, tick, status, events=None, nodes=None, error=None):
        if events is not None:
            _atomic_write_json(os.path.join(self.dir, "events.json"),
                               {"tick": tick, "events": events})
        if nodes is not None:
            _atomic_write_json(os.path.join(self.dir, "nodes.json"),
                               {"schema": "sim.nodes.v1", "nodes": nodes})
        state = {
            "tick": int(tick),
            "status": status,
            "frame_count": self.frame_count,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if events is not None:
            state["event_count"] = len(events)
        if error:
            state["error"] = error
        _atomic_write_json(os.path.join(self.dir, "state.json"), state)

    def close(self):
        for fh in (self._frames, self._index):
            try:
                fh.close()
            except OSError:
                pass


def _atomic_write_json(path, obj):
    data = json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8")
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _load_decisions(path):
    if not path:
        return []
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return preflight.decisions_from(raw)


def _ensure_local_resources(store):
    """Register repo artifacts and the default envpack when absent.

    Local-only (no network) and idempotent.  It also refreshes entries whose
    content changed (roads/rolecards/platforms/weapons) so a stale index does
    not turn into a false ``integrity`` gap on every run.
    """
    try:
        entries, errors = scan_repo(SITUATION3D)
        try:
            current = {e.entry_id: e for e in store.entries()}
        except Exception:  # noqa: BLE001 - fall through to full registration
            current = {}
        for entry in entries:
            old = current.get(entry.entry_id)
            if old is not None and old.content_hash == entry.content_hash:
                continue
            try:
                store.register(entry, replace=True)
            except Exception as exc:  # noqa: BLE001 - keep the scan going
                print("[resource] register %s failed: %s"
                      % (entry.entry_id, exc), file=sys.stderr)
        if errors:
            print("[resource] scan warnings: %s" % errors, file=sys.stderr)
        if not any(e.kind == "envpack" for e in store.entries()) \
                and os.path.isdir(DEFAULT_ENVPACK):
            try:
                store.register(scan_envpack(DEFAULT_ENVPACK))
            except Exception as exc:  # noqa: BLE001
                print("[resource] default envpack not registered: %s" % exc,
                      file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - resolution will report the gap
        print("[resource] registry bootstrap failed: %s" % exc, file=sys.stderr)


def _allowed_resource_roots(store):
    raw = os.environ.get("SITUATION_RESOURCE_ROOTS", "")
    roots = [os.path.abspath(p) for p in raw.split(os.pathsep) if p.strip()]
    roots += [os.path.join(SITUATION3D, "envpacks"),
              os.path.join(str(store.root), "cache"), str(store.root)]
    return roots


def _recent_pbf(store, max_age_days=7.0):
    """A locally cached public .osm.pbf that is still reasonably fresh.

    The project covers all of China; never hard-code a Taiwan filename.  The
    env var wins, otherwise any ``cache/osm_pbf/*.osm.pbf`` (newest first) is
    eligible, so regional extracts (china-latest, province extracts, ...) work
    without code changes.
    """
    paths = []
    env_path = os.environ.get("SITUATION_OSM_PBF")
    if env_path:
        paths.append(env_path)
    pbf_dir = os.path.join(str(store.root), "cache", "osm_pbf")
    if os.path.isdir(pbf_dir):
        names = [n for n in os.listdir(pbf_dir) if n.endswith(".osm.pbf")]
        names.sort(key=lambda n: os.path.getmtime(os.path.join(pbf_dir, n)),
                   reverse=True)
        paths.extend(os.path.join(pbf_dir, n) for n in names)
    for path in paths:
        if not os.path.isfile(path):
            continue
        age_days = (time.time() - os.path.getmtime(path)) / 86400.0
        if age_days <= max_age_days:
            return path
    return None


def _check_task_points(missions, scenario, bbox):
    objectives = {}
    for item in scenario.get("objectives") or []:
        if item.get("name") is not None and item.get("lon") is not None:
            objectives[str(item["name"])] = (item["lon"], item["lat"])
    for mission in missions or []:
        lon, lat = mission.get("lon"), mission.get("lat")
        if lon is None:
            lon, lat = objectives.get(str(mission.get("objective")),
                                      (None, None))
        if lon is None:
            continue
        if not (bbox[0] <= float(lon) <= bbox[2]
                and bbox[1] <= float(lat) <= bbox[3]):
            raise SystemExit(
                "task request rejected: %s point %.6f,%.6f outside envpack bbox %s"
                % (mission.get("id"), float(lon), float(lat), list(bbox)))


def _wait_for_resources(store, case_id, run_id, summary, config):
    path = os.path.join(store.run_dir(case_id, run_id), "resources.json")
    _atomic_write_json(path, summary)
    run = store.get_run(case_id, run_id)
    merged = dict(run.get("config") or {})
    merged.update(config or {})
    store.update_run(case_id, run_id, status="waiting_resources",
                     resources=summary, resource_gaps=summary.get("gaps") or [],
                     config=merged, pid=None,
                     scenario_date=summary.get("scenario_date"))
    print(json.dumps({"status": "waiting_resources", "summary": summary},
                     ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="taiwan_dense_transfer_v1")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--run-id", default=None,
                    help="reserved run id (created by the test console)")
    ap.add_argument("--envpack", default=None,
                    help="explicit envpack override (skips registry selection)")
    ap.add_argument("--resource-root", default=None,
                    help="resource index root (default: situation3d/var/resources)")
    ap.add_argument("--resolutions", default=None,
                    help="JSON resource decisions (use_cached_stale/cancel/"
                         "fetch_online/provide_file)")
    ap.add_argument("--resource-mode", choices=("auto", "offline", "online"),
                    default="auto",
                    help="auto = refresh missing/stale envpack online; "
                         "offline = cache only; online = require refresh")
    ap.add_argument("--scenario-date", default=None,
                    help="ISO 8601 start time; default is the current UTC time")
    ap.add_argument("--tasks-file", default=None,
                    help="JSON task request (see sim/tasking.py)")
    ap.add_argument("--task-mode", choices=("default", "user", "mixed"),
                    default=None, help="override the request mode")
    ap.add_argument("--side", type=int, choices=(0, 1), default=0,
                    help="default side for tasks without an explicit side")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    scenario_date = args.scenario_date
    if not scenario_date:
        # Resolve "now" once so the scenario snapshot and the replay clock agree.
        scenario_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    runs_dir = args.runs_dir or os.path.join(SITUATION3D, "var", "sim_runs")
    store = RunStore(runs_dir)
    rstore = ResourceStore(args.resource_root)
    _ensure_local_resources(rstore)
    try:
        decisions = _load_decisions(args.resolutions)
    except ValueError as exc:
        raise SystemExit("invalid resolutions: %s" % exc)
    allowed_actions = ("use_cached_stale", "cancel", "fetch_online",
                       "provide_file")
    for decision in decisions:
        if decision.action not in allowed_actions:
            raise SystemExit("unsupported decision action: %s"
                             % decision.action)
        if decision.action == "provide_file":
            kind = decision.kind or (decision.entry_id.split(":", 1)[0]
                                     if ":" in decision.entry_id else "envpack")
            if kind != "envpack":
                raise SystemExit("provide_file 目前只支持 envpack")
            if not decision.path:
                raise SystemExit("provide_file 需要 path")
            try:
                providers.register_provided_envpack(
                    rstore, decision.path, decision.entry_id or None,
                    allowed_roots=_allowed_resource_roots(rstore))
            except providers.ProviderError as exc:
                raise SystemExit("provide_file failed: %s" % exc)
    force_fetch = any(d.action == "fetch_online" for d in decisions)
    if any(d.action == "cancel" for d in decisions):
        if args.run_id:
            store.update_run(args.case, args.run_id, status="cancelled")
        print(json.dumps({"status": "cancelled"}))
        return 0

    plugin = load_plugin(args.case)
    theatre = getattr(plugin, "THEATRE", None)
    if args.run_id:
        run_id = args.run_id
        store.get_run(args.case, run_id)  # must exist (reserved by the API)
        run = store.get_run(args.case, run_id)
        if run.get("status") == "waiting_resources" and args.resolutions:
            store.update_run(args.case, run_id, status="queued")
    else:
        run_id = store.create_run(args.case, seed=args.seed,
                                  config={"ticks": args.ticks,
                                          "seed": args.seed})

    early = preflight.check_case_data(rstore, theatre, scenario_date,
                                      decisions=decisions)
    if early is not None:
        _wait_for_resources(store, args.case, run_id,
                            preflight.plan_summary(early),
                            {"ticks": args.ticks, "seed": args.seed,
                             "scenario_date": scenario_date})
        return 0

    if args.envpack:
        envpack_override = os.path.abspath(args.envpack)
        if not os.path.isdir(envpack_override):
            raise SystemExit("envpack not found: %s" % envpack_override)
        try:
            entry = scan_envpack(envpack_override)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit("envpack invalid: %s" % exc)
        entry = type(entry)(**dict(entry.as_dict(), bbox=entry.bbox,
                                   provider="cli-override",
                                   notes=dict(entry.notes, manual_override=True)))
        rstore.register(entry, replace=True)

    scenario = plugin.scenario_dict(scenario_date=scenario_date)
    meta = plugin.target_meta()
    cm = plugin.case_meta(scenario_date=scenario_date)
    tick_seconds = float(cm.get("tick_seconds") or 30)
    tick_count = int(args.ticks or cm.get("tick_count") or 1440)
    scenario["duration_s"] = tick_seconds * tick_count
    if args.tasks_file:
        request = tasking.load_request(args.tasks_file, default_side=args.side)
    else:
        request = tasking.parse_request(None, default_side=args.side)
    if args.task_mode:
        request["mode"] = args.task_mode
    try:
        missions, tasking_summary = tasking.merge_missions(
            plugin.missions(), request, scenario)
    except tasking.TaskError as exc:
        raise SystemExit("task request rejected: %s" % exc)
    scenario["missions"] = missions
    scenario["tasking"] = tasking_summary
    # Reject coordinate typos before they are mistaken for a resource gap.
    anchor = scenario.get("geo_anchor") or {}
    candidates = []
    for entry in rstore.entries("envpack"):
        bbox = entry.bbox
        if bbox is None or anchor.get("lon") is None:
            continue
        if bbox[0] <= float(anchor["lon"]) <= bbox[2] \
                and bbox[1] <= float(anchor["lat"]) <= bbox[3]:
            candidates.append(entry)
    if candidates:
        candidates.sort(key=lambda e: e.as_of or "")
        _check_task_points(missions, scenario, candidates[-1].bbox)
    scenario_bytes = json.dumps(scenario, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    scenario_hash = _sha256_bytes(scenario_bytes)

    plan = preflight.make_plan(rstore, args.case, scenario_date, scenario,
                               missions, theatre=theatre,
                               engine=scenario.get("engine"),
                               decisions=decisions)
    summary = preflight.plan_summary(plan)
    envpack_candidate = preflight.entry_for(plan, "envpack")
    if envpack_candidate is not None and envpack_candidate.bbox is not None:
        _check_task_points(missions, scenario, envpack_candidate.bbox)
    config = {"tick_seconds": tick_seconds, "ticks": tick_count,
              "scenario_date": cm.get("start_time_utc"),
              "tasking": tasking_summary,
              "resource_mode": args.resource_mode,
              "resources": {"plan_hash": summary["plan_hash"],
                            "status": summary["status"]},
              "engine": scenario.get("engine")}
    fetch_error = None
    if plan.status == "blocked":
        gap_kinds = set(g.kind for g in plan.gaps
                        if g.criticality == "critical")
        envpack_reasons = set(
            (g.reason or "").split(":", 1)[0] for g in plan.gaps
            if g.kind == "envpack" and g.criticality == "critical")
        online_allowed = (args.resource_mode in ("auto", "online")
                          or force_fetch)
        try:
            target_date = date.fromisoformat(str(scenario_date)[:10])
        except ValueError:
            raise SystemExit("scenario_date must be ISO 8601: %r"
                             % (scenario_date,))
        if target_date < date.today():
            online_allowed = False
            if "envpack" in gap_kinds:
                fetch_error = ("想定日期早于今天，在线刷新无法生成该时点的数据；"
                               "请调整日期或提供历史快照")
        if "no_point_in_time" in envpack_reasons:
            online_allowed = False
            fetch_error = ("候选快照晚于想定日期；在线刷新无法解决，"
                           "请调整日期或提供历史快照")
        if "envpack" in gap_kinds and online_allowed:
            candidates = sorted(rstore.entries("envpack"),
                                key=lambda e: e.as_of or "", reverse=True)
            template = preflight.entry_for(plan, "envpack") or (
                candidates[0] if candidates else None)
            pbf = _recent_pbf(rstore)
            if template is None:
                fetch_error = "没有可用于在线重建的环境包模板"
            elif pbf is None and not providers.probe_online():
                fetch_error = ("网络探测失败（Overpass 不可达），已保留现有资源；"
                               "可先用 scripts/resource.py download-pbf 下载公开 PBF")
            else:
                try:
                    providers.refresh_envpack(
                        rstore, template,
                        os.path.join(str(rstore.root), "cache"),
                        osm_pbf=pbf,
                        mode=("online" if args.resource_mode == "online"
                              else "auto"))
                    plan = preflight.make_plan(
                        rstore, args.case, scenario_date, scenario, missions,
                        theatre=theatre, engine=scenario.get("engine"),
                        decisions=decisions)
                    summary = preflight.plan_summary(plan)
                    config["resources"] = {"plan_hash": summary["plan_hash"],
                                           "status": summary["status"]}
                except providers.ProviderError as exc:
                    fetch_error = str(exc)
    if plan.status == "blocked":
        if fetch_error:
            summary["fetch_error"] = fetch_error
        _wait_for_resources(store, args.case, run_id, summary, config)
        return 0
    envpack_entry = preflight.entry_for(plan, "envpack")
    if envpack_entry is None:
        raise SystemExit("resource plan has no envpack entry")
    envpack = envpack_entry.path
    config["envpack"] = os.path.basename(envpack)
    envpack_hash = _envpack_hash(envpack)
    engine_commit = _git_commit()
    store.update_run(args.case, run_id, status="running", config=config,
                     scenario_hash=scenario_hash,
                     engine_commit=engine_commit, envpack_hash=envpack_hash,
                     resources=summary, resource_gaps=[])
    t_start = time.time()
    live = LiveRunWriter(store, args.case, run_id)
    live.publish(0, "running")
    try:
        world = build_world(scenario, {}, seed=args.seed, dt_s=tick_seconds,
                            envpack=envpack)
        uid_to_cid = {}
        for cid, m in meta.items():
            entity = world.entities.get(int(m["uid"]))
            if entity is not None:
                entity._case_id = cid
                uid_to_cid[int(m["uid"])] = cid
        live_meta = dict(cm)
        live_meta.update({
            "schema": "situation3d.live.v1",
            "tick_count": tick_count,
            "terminal_tick": tick_count,
            "run_id": run_id,
            "targets": _target_defs(world, meta),
            "symbology": symbology.catalog(),
            "objectives": [
                {"id": obj["name"], "name": obj["name"], "lon": obj["lon"],
                 "lat": obj["lat"], "owner": -1,
                 "value": int(obj.get("value") or 0)}
                for obj in plugin.objectives()],
            "inference_model": None,
            "auto_plots": [],
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        live.meta(live_meta)
        runner = MissionRunner(world, missions, scenario.get("roe"),
                               interval_s=tick_seconds)
        # Occupation / objective progress are per-tick quantities: the live
        # viewer paints frames while the run is still going, so they must be
        # computed here rather than patched onto the finished array below.
        areas = {}
        for obj in plugin.objectives():
            areas[obj["name"]] = (float(obj["lon"]), float(obj["lat"]))
        aff = {cid: ("red" if m["affiliation"] == 0 else "blue") for cid, m in meta.items()}
        objective_defs = plugin.objectives()
        frames = []
        step_ms = []
        travelled = {cid: 0.0 for cid in meta}
        for tick in range(tick_count + 1):
            if tick > 0:
                t0 = time.perf_counter()
                world.step(tick_seconds)
                runner.update()
                _clear_subcell_buildings(world)
                step_ms.append((time.perf_counter() - t0) * 1000.0)
            rows = []
            for entity in world.sorted_entities():
                cid = getattr(entity, "_case_id", None)
                if cid is None:
                    continue
                rows.append(_frame_row2(entity, cid, uid_to_cid, world.time_s))
            rows.sort(key=lambda r: r["id"])
            prev_rows = {r["id"]: r for r in (frames[-1]["targets"] if frames else [])}
            for row in rows:
                prev = prev_rows.get(row["id"])
                if (prev and prev.get("mode") == row.get("mode")
                        and row.get("mode") != "embarked"):
                    travelled[row["id"]] = travelled.get(row["id"], 0.0) + haversine_m(
                        (prev["lon"], prev["lat"]), (row["lon"], row["lat"]))
                row["travelled_m"] = round(travelled.get(row["id"], 0.0), 1)
            occ = {}
            for name, (lon, lat) in areas.items():
                red = blue = 0
                for row in rows:
                    if row.get("mode") != "ground":
                        continue
                    d = math.hypot((row["lon"] - lon) * 111320.0 * math.cos(math.radians(lat)),
                                   (row["lat"] - lat) * 110540.0)
                    if d > 1500.0:
                        continue
                    if aff.get(row["id"]) == "red":
                        red += 1
                    else:
                        blue += 1
                occ[name] = ("red" if red > blue else ("blue" if blue > red else
                                                       ("neutral" if red or blue else "none")))
            phase = _phase_at(cm.get("phases") or [], tick)
            frame = {
                "tick": tick,
                "t_seconds": round(tick * tick_seconds, 0),
                "phase": phase,
                "unit_phase": {r["id"]: {"id": phase["id"], "name": phase["name"],
                                          "index": phase["index"]} for r in rows},
                "occupation": occ,
                "objectives": {
                    obj["name"]: (1.0 if world.objectives.get(
                        world.rule_grid.cell_of(float(obj["lon"]), float(obj["lat"])),
                        {}).get("owner") == 0 else 0.0)
                    for obj in objective_defs
                },
                "targets": rows,
            }
            frames.append(frame)
            live.append_frame(frame)
            if tick and tick % LIVE_EVERY_TICKS == 0:
                events_now = _convert_events(world, uid_to_cid, frames,
                                             tick_seconds, tick_count,
                                             cm.get("stages") or [])
                nodes_now = _nodes(events_now, tick_count,
                                   cm.get("phases") or [],
                                   include_terminal=False)["nodes"]
                live.publish(tick, "running", events=events_now, nodes=nodes_now)
                store.update_run(args.case, run_id,
                                 progress={"tick": tick, "ticks": tick_count,
                                           "events": len(world.events)})
                if not args.quiet and tick % 120 == 0:
                    print("[run] tick %d/%d events=%d elapsed=%.1fs"
                          % (tick, tick_count, len(world.events), time.time() - t_start),
                          flush=True)
        events = _convert_events(world, uid_to_cid, frames, tick_seconds,
                                 tick_count, cm.get("stages") or [])
        nodes = _nodes(events, tick_count, cm.get("phases") or [])
        counts = Counter(ev["type"] for ev in events)
        losses = {"red": 0, "blue": 0}
        for cid, m in meta.items():
            entity = world.entities.get(int(m["uid"]))
            if entity is not None and not entity.alive:
                losses["red" if m["affiliation"] == 0 else "blue"] += 1
        objectives = []
        for obj in plugin.objectives():
            cell = world.rule_grid.cell_of(float(obj["lon"]), float(obj["lat"]))
            state = world.objectives.get(cell) or {}
            objectives.append({"id": obj["name"], "name": obj["name"],
                               "lon": obj["lon"], "lat": obj["lat"],
                               "owner": int(state.get("owner", -1)),
                               "value": int(obj.get("value") or 0)})
        # Validation: finite positions, ground water/building intrusions,
        # per-frame displacement sanity (skip boarding transitions).
        finite = True
        water_hits = building_hits = 0
        ground_rows = 0
        amphibious = set()
        for cid, m in meta.items():
            ent = world.entities.get(int(m["uid"]))
            if ent is not None and getattr(ent.profile, "amphibious", False):
                amphibious.add(cid)
        for fr in frames:
            for row in fr["targets"]:
                if not all(math.isfinite(float(row[k])) for k in ("lat", "lon", "alt")):
                    finite = False
                if row.get("mode") == "ground":
                    ground_rows += 1
                    if (world.env.water_depth(row["lon"], row["lat"]) is not None
                            and row["id"] not in amphibious):
                        water_hits += 1
                    if world.env.building_at(row["lon"], row["lat"]) is not None:
                        building_hits += 1
        # Sub-cell OSM overlaps can still occur for a frame or two while a unit
        # is moving through a footprint smaller than the 30 m nav cell. The
        # tolerance is absolute and tiny (2 for a short run, +1 per 20k rows);
        # systematic violations still fail the run.
        transient_limit = max(2, ground_rows // 20000)
        validation = {
            "passed": bool(finite and water_hits <= transient_limit
                           and building_hits <= transient_limit),
            "finite_positions": finite,
            "ground_rows": ground_rows,
            "water_hits": water_hits,
            "building_hits": building_hits,
            "transient_limit": transient_limit,
            "note": "stopped units are cleared from sub-cell footprints; moving transients are tolerated up to the limit",
        }
        outcome = {
            "terminal_tick": tick_count,
            "engine": {"status": "sim.world",
                       "result": ("completed" if all(o["owner"] == 0 for o in objectives)
                                  else ("partial" if any(o["owner"] == 0 for o in objectives)
                                        else "limited")),
                       "objectives": objectives,
                       "red_losses": losses["red"], "blue_losses": losses["blue"]},
            "must_not_force_match": True,
        }
        doc = dict(cm)
        doc.update({
            "schema": "situation3d.trajectory.v2",
            "campaign_contract_version": 1,
            "tick_seconds": tick_seconds,
            "tick_count": tick_count,
            "terminal_tick": tick_count,
            "scenario_id": "taiwan_anping_engine_%d" % args.seed,
            "scenario_name": "台海方向多域态势演化（引擎推演）",
            "targets": _target_defs(world, meta),
            "frames": frames,
            "events": events,
            "nodes": nodes.get("nodes") or [],
            "objectives": objectives,
            "outcome": outcome,
            "provenance": {
                "generator": "sim.world", "runtime": "engine_run",
                "run_id": run_id, "engine_commit": engine_commit,
                "scenario_hash": scenario_hash, "envpack_hash": envpack_hash,
                "resource_plan_hash": summary["plan_hash"],
                "resources_status": summary["status"],
                "seed": args.seed, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                 time.gmtime()),
            },
            "inference_model": None,
            "symbology": symbology.catalog(),
            "auto_plots": [],
        })
        trajectory_bytes = json.dumps(doc, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")
        summary = {
            "case_id": args.case, "run_id": run_id, "seed": args.seed,
            "ticks": tick_count, "tick_seconds": tick_seconds,
            "events": dict(counts), "event_total": len(events),
            "losses": losses, "objectives": objectives,
            "elapsed_s": round(time.time() - t_start, 1),
            "step_ms_p50": round(statistics.median(step_ms), 1) if step_ms else 0,
            "step_ms_p95": round(sorted(step_ms)[int(len(step_ms) * 0.95)], 1) if step_ms else 0,
            "trajectory_bytes": len(trajectory_bytes),
            "validation": validation,
            "resources": {"status": summary["status"],
                          "plan_hash": summary["plan_hash"],
                          "warnings": summary["warnings"]},
            "engine_commit": engine_commit,
        }
        metrics = {"events": len(events), "fires": counts.get("fire", 0),
                   "destroyed": counts.get("destroy", 0),
                   "losses": losses, "elapsed_s": summary["elapsed_s"],
                   "step_ms_p95": summary["step_ms_p95"],
                   "trajectory_mb": round(len(trajectory_bytes) / 1048576.0, 1)}
        store.complete_run(
            args.case, run_id,
            {"scenario.json": scenario_bytes,
             "resources.json": json.dumps(summary, ensure_ascii=False,
                                          indent=1).encode("utf-8"),
             "trajectory.json": trajectory_bytes,
             "nodes.json": json.dumps(nodes, ensure_ascii=False, indent=1).encode("utf-8"),
             "summary.json": json.dumps(summary, ensure_ascii=False, indent=1).encode("utf-8")},
            metrics=metrics, validation=validation)
        live.publish(tick_count, "completed", events=events,
                     nodes=nodes["nodes"])
        live.close()
        print(json.dumps({"run_id": run_id, "metrics": metrics,
                          "validation": validation}, ensure_ascii=False, indent=1))
        return 0
    except Exception as exc:  # noqa: BLE001
        live.publish(live.last_tick, "failed",
                     error="%s: %s" % (type(exc).__name__, exc))
        live.close()
        store.fail_run(args.case, run_id, "%s: %s" % (type(exc).__name__, exc),
                       traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
