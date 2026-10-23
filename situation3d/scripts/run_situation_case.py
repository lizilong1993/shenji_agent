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
from sim.scenario import build_world  # noqa: E402
from run_store import RunStore  # noqa: E402

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


def _nodes(events, tick_count, phases):
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
    nodes.append({"tick": tick_count, "type": "terminal", "title": "推演结束",
                  "actors": [], "location": None, "description": "达到想定时长"})
    nodes.sort(key=lambda n: n["tick"])
    return {"schema": "sim.nodes.v1", "nodes": nodes}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="taiwan_dense_transfer_v1")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--run-id", default=None,
                    help="reserved run id (created by the test console)")
    ap.add_argument("--envpack", default=DEFAULT_ENVPACK)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    plugin = load_plugin(args.case)
    scenario = plugin.scenario_dict()
    meta = plugin.target_meta()
    cm = plugin.case_meta()
    missions = plugin.missions()
    tick_seconds = float(cm.get("tick_seconds") or 30)
    tick_count = int(args.ticks or cm.get("tick_count") or 1440)
    envpack = os.path.abspath(args.envpack)
    if not os.path.isdir(envpack):
        raise SystemExit("envpack not found: %s" % envpack)
    scenario["duration_s"] = tick_seconds * tick_count
    scenario_bytes = json.dumps(scenario, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    scenario_hash = _sha256_bytes(scenario_bytes)
    engine_commit = _git_commit()
    envpack_hash = _envpack_hash(envpack)

    runs_dir = args.runs_dir or os.path.join(SITUATION3D, "var", "sim_runs")
    store = RunStore(runs_dir)
    config = {"tick_seconds": tick_seconds, "ticks": tick_count,
              "envpack": os.path.basename(envpack),
              "engine": scenario.get("engine")}
    if args.run_id:
        run_id = args.run_id
        store.get_run(args.case, run_id)  # must exist (reserved by the API/CLI)
        store.update_run(args.case, run_id, status="running",
                         config=config, scenario_hash=scenario_hash,
                         engine_commit=engine_commit, envpack_hash=envpack_hash)
    else:
        run_id = store.create_run(
            args.case, seed=args.seed, config=config,
            scenario_hash=scenario_hash, engine_commit=engine_commit,
            envpack_hash=envpack_hash)
        store.update_run(args.case, run_id, status="running")
    t_start = time.time()
    try:
        world = build_world(scenario, {}, seed=args.seed, dt_s=tick_seconds,
                            envpack=envpack)
        uid_to_cid = {}
        for cid, m in meta.items():
            entity = world.entities.get(int(m["uid"]))
            if entity is not None:
                entity._case_id = cid
                uid_to_cid[int(m["uid"])] = cid
        runner = MissionRunner(world, missions, scenario.get("roe"),
                               interval_s=tick_seconds)
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
            phase = _phase_at(cm.get("phases") or [], tick)
            frames.append({
                "tick": tick,
                "t_seconds": round(tick * tick_seconds, 0),
                "phase": phase,
                "unit_phase": {r["id"]: {"id": phase["id"], "name": phase["name"],
                                          "index": phase["index"]} for r in rows},
                "occupation": {},
                "objectives": {},
                "targets": rows,
            })
            if tick and tick % 120 == 0:
                store.update_run(args.case, run_id,
                                 progress={"tick": tick, "ticks": tick_count,
                                           "events": len(world.events)})
                if not args.quiet:
                    print("[run] tick %d/%d events=%d elapsed=%.1fs"
                          % (tick, tick_count, len(world.events), time.time() - t_start),
                          flush=True)
        # Occupation and objective progress from the simulation state.
        areas = {}
        for obj in plugin.objectives():
            areas[obj["name"]] = (float(obj["lon"]), float(obj["lat"]))
        aff = {cid: ("red" if m["affiliation"] == 0 else "blue") for cid, m in meta.items()}
        for fr in frames:
            occ = {}
            for name, (lon, lat) in areas.items():
                red = blue = 0
                for row in fr["targets"]:
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
            fr["occupation"] = occ
            fr["objectives"] = {
                obj["name"]: (1.0 if world.objectives.get(
                    world.rule_grid.cell_of(float(obj["lon"]), float(obj["lat"])),
                    {}).get("owner") == 0 else 0.0)
                for obj in plugin.objectives()
            }
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
             "trajectory.json": trajectory_bytes,
             "nodes.json": json.dumps(nodes, ensure_ascii=False, indent=1).encode("utf-8"),
             "summary.json": json.dumps(summary, ensure_ascii=False, indent=1).encode("utf-8")},
            metrics=metrics, validation=validation)
        print(json.dumps({"run_id": run_id, "metrics": metrics,
                          "validation": validation}, ensure_ascii=False, indent=1))
        return 0
    except Exception as exc:  # noqa: BLE001
        store.fail_run(args.case, run_id, "%s: %s" % (type(exc).__name__, exc),
                       traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
