#!/usr/bin/env python3
"""Run a scenario on the continuous geo engine and export CZML/observation.

Usage:
  python3 scripts/run_geo_demo.py --scenario <scenario.json> --map <basic.json> \
      --ticks 600 --out /tmp/geo_demo
  python3 scripts/run_geo_demo.py --case taiwan_landing --ticks 600 --out /tmp/tw_geo
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from geo.geodesy import haversine_m  # noqa: E402
from geoio.czml import write_czml  # noqa: E402
from geoio.observation import observation  # noqa: E402
from sim.scenario import build_world  # noqa: E402


def _load_plugin(path: str):
    spec = importlib.util.spec_from_file_location("demo_case", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case_inputs(name: str):
    plugin = _load_plugin(os.path.join(SITUATION3D, "campaigns", name + ".py"))
    scenario = plugin.scenario_dict()
    basic = json.loads(open(plugin.map_path(), encoding="utf-8").read())
    spec = plugin.build_spec()
    return scenario, basic, getattr(spec, "objectives", []) or []


def _assign_orders(world, objectives) -> None:
    for entity in world.sorted_entities():
        if not entity.alive or entity.route is not None:
            continue
        candidates = [o for o in objectives if o["owner"] != entity.affiliation]
        if not candidates:
            candidates = list(objectives)
        if not candidates:
            continue
        ranked = []
        for obj in candidates:
            cell = world.rule_grid.cell_of(float(obj["lon"]), float(obj["lat"]))
            lon, lat = world.rule_grid.cell_center_lonlat(*cell)
            ranked.append((haversine_m((entity.pose.lon, entity.pose.lat), (lon, lat)), lon, lat))
        # Spread units around an objective so they do not end up at the same
        # point (which would be inside every weapon's minimum range).
        off_lon = ((entity.uid % 5) - 2) * 0.01
        off_lat = (((entity.uid // 5) % 5) - 2) * 0.01
        def preference(obj_name):
            name = entity.name
            landing = any(k in name for k in ("登陆", "两栖", "补给"))
            combat_ship = any(k in name for k in ("驱逐", "护卫", "巡逻"))
            if combat_ship and "STRAIT" in obj_name:
                return -1e9
            if landing and "BEACH" in obj_name:
                return -1e9
            if entity.domain == "land" and obj_name in ("OBJ-BEACH", "OBJ-TAINAN"):
                return -1e6
            return 0.0

        ranked = [(d + preference(o["name"]), lon + off_lon, lat + off_lat)
                  for (d, lon, lat), o in zip(ranked, candidates)]
        ranked.sort()
        for _, lon, lat in ranked:
            route = world.planner.plan(entity.profile, (entity.pose.lon, entity.pose.lat),
                                       (lon, lat))
            if route is not None:
                entity.goal = (lon, lat)
                entity.route = route
                entity.route_s = 0.0
                entity.moving = True
                break


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario")
    ap.add_argument("--map")
    ap.add_argument("--case")
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detection-model",
                    choices=("table", "spotting", "shadow"))
    ap.add_argument("--direct-fire-model",
                    choices=("crt", "physics", "shadow"))
    ap.add_argument("--tactics", action="store_true")
    ap.add_argument("--radio-share", action="store_true")
    ap.add_argument("--commander", action="store_true")
    args = ap.parse_args()

    if args.case:
        scenario, basic, raw_objs = _case_inputs(args.case)
    else:
        if not args.scenario or not args.map:
            print("need --scenario/--map or --case", file=sys.stderr)
            return 2
        scenario = json.loads(open(args.scenario, encoding="utf-8").read())
        basic = json.loads(open(args.map, encoding="utf-8").read())
        raw_objs = scenario.get("cities") or []

    if (args.detection_model or args.direct_fire_model
            or args.tactics or args.radio_share or args.commander):
        engine = scenario.get("engine")
        if not isinstance(engine, dict):
            engine = {}
            scenario["engine"] = engine
        if args.detection_model:
            engine["detection_model"] = args.detection_model
        if args.direct_fire_model:
            engine["direct_fire_model"] = args.direct_fire_model
        if args.tactics:
            engine["enable_tactics"] = True
        if args.radio_share:
            engine["enable_radio_share"] = True
        if args.commander:
            engine["enable_commander"] = True
    world = build_world(scenario, basic, seed=args.seed, dt_s=args.dt)
    objectives = []
    if raw_objs and "lon" in raw_objs[0]:
        for obj in raw_objs:
            lon = float(obj["lon"])
            lat = float(obj["lat"])
            flag = obj.get("flag", obj.get("owner", -1))
            owner = flag if flag in (0, 1) else -1
            cell = world.rule_grid.cell_of(lon, lat)
            world.add_objective(cell, value=int(obj.get("value") or 0),
                                name=str(obj.get("name") or ""), owner=owner)
            objectives.append({"lon": lon, "lat": lat, "owner": owner,
                               "value": int(obj.get("value") or 0),
                               "name": str(obj.get("name") or "")})
    else:
        grid = world.env.grid
        for obj in raw_objs:
            coord = obj.get("coord", obj.get("hex"))
            if coord is None:
                continue
            lon, lat = grid.to_lonlat(int(coord))
            flag = obj.get("flag", obj.get("owner", -1))
            owner = flag if flag in (0, 1) else -1
            cell = world.rule_grid.cell_of(lon, lat)
            world.add_objective(cell, value=int(obj.get("value") or 0),
                                name=str(obj.get("name") or ""), owner=owner)
            objectives.append({"lon": lon, "lat": lat, "owner": owner,
                               "value": int(obj.get("value") or 0),
                               "name": str(obj.get("name") or "")})

    if objectives:
        _assign_orders(world, objectives)
    else:
        # No objectives: send every unit toward the map centre.
        centre = world.rule_grid.cell_center_lonlat(world.rule_grid.rows // 2,
                                                    world.rule_grid.cols // 2)
        for entity in world.sorted_entities():
            world.order(entity.uid, {"type": "move_to", "lon": centre[0], "lat": centre[1]})

    step_ms = []
    fired = []
    t0 = time.time()
    for i in range(args.ticks):
        t = time.perf_counter()
        events = world.step()
        step_ms.append((time.perf_counter() - t) * 1000.0)
        fired.extend(events)
        if i % 120 == 59 and objectives:
            _assign_orders(world, objectives)
    elapsed = time.time() - t0

    os.makedirs(args.out, exist_ok=True)
    write_czml(world, os.path.join(args.out, "scene.czml"))
    with open(os.path.join(args.out, "observation.json"), "w", encoding="utf-8") as fh:
        json.dump(observation(world), fh, ensure_ascii=False, indent=1)
    summary = {
        "ticks": args.ticks,
        "detection_model": world.detection_model,
        "direct_fire_model": world.direct_fire_model,
        "detection_shadow": (None if world.detection_shadow is None else {
            k: v for k, v in world.detection_shadow.items() if k != "samples"}),
        "elapsed_s": round(elapsed, 2),
        "step_ms_p50": round(statistics.median(step_ms), 2),
        "step_ms_p95": round(sorted(step_ms)[int(len(step_ms) * 0.95)], 2),
        "entities": len(world.entities),
        "alive": sum(1 for e in world.entities.values() if e.alive),
        "events": len(world.events),
        "event_types": dict(__import__("collections").Counter(
            e.get("type") for e in world.events)),
        "direct_fire": sum(1 for e in world.events if e.get("type") == "direct_fire"),
        "sea_air_fire": sum(1 for e in world.events if e.get("type") == "sea_air_fire"),
        "destroyed": sum(1 for e in world.events if e.get("type") == "destroyed"),
        "objectives": len(world.objectives),
        "objective_owners": sorted({o["owner"] for o in world.objectives.values()}),
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
