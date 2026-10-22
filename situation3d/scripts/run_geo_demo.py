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


def _landing_berth(world, obj, entity):
    """Nearest water cell within 8 cells of an objective (landing berth)."""
    start = world.rule_grid.cell_of(float(obj["lon"]), float(obj["lat"]))
    frontier = [start]
    seen = {start}
    for radius in range(1, 9):
        next_frontier = []
        best = None
        for cell in frontier:
            for nb in world.rule_grid.neighbors(cell):
                if nb in seen:
                    continue
                seen.add(nb)
                next_frontier.append(nb)
                lon, lat = world.rule_grid.cell_center_lonlat(*nb)
                if world.env.water_depth(lon, lat) is None:
                    continue
                d = haversine_m((entity.pose.lon, entity.pose.lat), (lon, lat))
                if best is None or d < best[0]:
                    best = (d, lon, lat)
        if best is not None:
            return (best[1], best[2])
        frontier = next_frontier
        if not frontier:
            break
    return None


def _assign_orders(world, objectives) -> None:
    """Behaviour layer (P1.6): pick an objective cell/pace, not coordinates."""
    from sim.intent import Intent

    for entity in world.sorted_entities():
        if not entity.alive or entity.on_board:
            continue
        if entity.route is not None:
            # Clear routes that already arrived, so staged orders can advance.
            if (entity.goal is not None
                    and haversine_m((entity.pose.lon, entity.pose.lat),
                                    entity.goal) < 200.0):
                entity.route = None
                entity.goal = None
                entity.route_s = 0.0
                entity.moving = False
                entity.pace_scale = 1.0
            else:
                continue
        live = []
        for o in objectives:
            cell = world.rule_grid.cell_of(float(o["lon"]), float(o["lat"]))
            owner = world.objectives.get(cell, {}).get("owner", o["owner"])
            live.append(dict(o, owner=owner, cell=cell))
        candidates = [o for o in live if o["owner"] != entity.affiliation]
        if not candidates:
            candidates = list(live)
        if not candidates:
            continue
        landing_ship = any(k in entity.name for k in ("登陆", "两栖舰"))
        pace = ("dash" if entity.domain in ("land", "amphibious")
                and entity.unit_type != "infantry" else "normal")

        def preference(obj_name):
            name = entity.name
            landing = any(k in name for k in ("登陆", "两栖", "补给"))
            combat_ship = any(k in name for k in ("驱逐", "护卫", "巡逻"))
            if combat_ship and "STRAIT" in obj_name:
                return -1e9
            if landing and "BEACH" in obj_name:
                return -1e9
            if entity.domain in ("land", "amphibious") and obj_name in (
                    "OBJ-BEACH", "OBJ-TAINAN"):
                return -1e6
            return 0.0

        ranked = []
        for obj in candidates:
            cell = obj["cell"]
            lon, lat = world.rule_grid.cell_center_lonlat(*cell)
            berth = None
            if landing_ship and "BEACH" in str(obj.get("name")):
                berth = _landing_berth(world, obj, entity)
                if berth is not None:
                    lon, lat = berth
            score = (haversine_m((entity.pose.lon, entity.pose.lat),
                                 (lon, lat))
                     + preference(str(obj.get("name") or "")))
            ranked.append((score, obj, cell, lon, lat, berth))
        ranked.sort(key=lambda row: row[0])
        for _, obj, cell, lon, lat, berth in ranked:
            if berth is not None:
                intent = Intent(lon=lon, lat=lat, pace=pace,
                                objective=str(obj.get("name") or ""))
            else:
                intent = Intent(cell=cell, pace=pace,
                                objective=str(obj.get("name") or ""))
            if world.assign_intent(entity, intent):
                break


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario")
    ap.add_argument("--map")
    ap.add_argument("--case")
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--envpack", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detection-model",
                    choices=("table", "spotting", "shadow", "physical"))
    ap.add_argument("--direct-fire-model",
                    choices=("crt", "physics", "shadow"))
    ap.add_argument("--indirect-fire-model",
                    choices=("table", "physics"))
    ap.add_argument("--missile-model",
                    choices=("instant", "flight"))
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
            or args.indirect_fire_model or args.missile_model
            or args.tactics or args.radio_share or args.commander):
        engine = scenario.get("engine")
        if not isinstance(engine, dict):
            engine = {}
            scenario["engine"] = engine
        if args.detection_model:
            engine["detection_model"] = args.detection_model
        if args.direct_fire_model:
            engine["direct_fire_model"] = args.direct_fire_model
        if args.indirect_fire_model:
            engine["indirect_fire_model"] = args.indirect_fire_model
        if args.missile_model:
            engine["missile_model"] = args.missile_model
        if args.tactics:
            engine["enable_tactics"] = True
        if args.radio_share:
            engine["enable_radio_share"] = True
        if args.commander:
            engine["enable_commander"] = True
    world = build_world(scenario, basic, seed=args.seed, dt_s=args.dt,
                        envpack=args.envpack)
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
        if i % 30 == 29 and objectives:
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
        "indirect_fire_model": world.indirect_fire_model,
        "missile_model": world.missile_model,
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
