#!/usr/bin/env python3
"""Compare a JEV policy against the current heuristic controller.

Both runs use the same scenario, seed, blue controller and engine; only the
red decision policy changes.  The JEV model is optional: without ``--model``
the script runs the deterministic mock so the harness itself can be tested.

Usage:
    python3 scripts/evaluate_jev.py --model /tmp/jev_model/qwen3.5-0.8b-nli-v2s-long \
        --ticks 80 --decision-every 4 --out /tmp/jev_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
if SITUATION3D not in sys.path:
    sys.path.insert(0, SITUATION3D)

from geoio.actions import apply_action  # noqa: E402
from geoio.observation import observation  # noqa: E402
from policy.jev import (  # noqa: E402
    JevPolicy,
    LocalOpenJevClient,
    MockJevClient,
    build_candidates,
    summarize_entity,
)
from sim.scenario import build_world  # noqa: E402

GRID = {"center_lon": 119.9, "center_lat": 23.1, "cols": 30, "rows": 30,
        "width_m": 200.0}


def make_basic():
    cells = []
    for r in range(30):
        row = []
        for c in range(30):
            cond = 1 if 12 <= c <= 13 else 0
            row.append({"pos": r * 100 + c, "elev": 100,
                        "cond": cond, "roads": [0] * 6, "rivers": [0] * 6})
        cells.append(row)
    return {"map_id": 1, "ele_grade": 10, "grid": GRID, "map_data": cells}


def make_scenario():
    def op(oid, color, pos, blood=3, armor=4):
        return {"obj_id": oid, "color": color, "type": 2, "name": "重型坦克",
                "sub_type": 0, "blood": blood, "max_blood": blood,
                "armor": armor, "cur_hex": pos, "carry_weapon_ids": [36],
                "value": 10}
    return {"scenario_id": 1,
            "operators": [op(1, 0, 505), op(2, 0, 606),
                          op(3, 1, 2020), op(4, 1, 1919)],
            "cities": [{"coord": 1020, "value": 10, "flag": -1, "name": "OBJ-A"},
                       {"coord": 2010, "value": 10, "flag": -1, "name": "OBJ-B"}],
            "time": {"cur_step": 0, "tick": 10.0, "max_time": 999}}


def nearest_objective(world, entity):
    best = None
    best_d = None
    for cell, obj in world.objectives.items():
        if obj["owner"] == entity.affiliation:
            continue
        lon, lat = world.rule_grid.cell_center_lonlat(*cell)
        from geo.geodesy import haversine_m
        d = haversine_m((entity.pose.lon, entity.pose.lat), (lon, lat))
        if best_d is None or d < best_d:
            best_d = d
            best = (lon, lat)
    return best


def heuristic_controller(world, seat: int) -> int:
    """Current scheme: fire if possible, else advance to the nearest objective."""
    obs = observation(world, seat=seat)
    applied = 0
    for entity in world.sorted_entities(seat):
        if not entity.alive:
            continue
        own = next((e for e in obs["entities"] if e["id"] == entity.uid), None)
        actions = (own or {}).get("actions") or []
        fire = next((a for a in actions if a.get("type") == "fire"), None)
        if fire is not None:
            if apply_action(world, {"type": "fire", "entity": entity.uid,
                                    "target": fire["target"]}):
                applied += 1
            continue
        goal = nearest_objective(world, entity)
        if goal is not None and entity.route is None:
            if world.order(entity.uid, {"type": "move_to", "lon": goal[0], "lat": goal[1]}) is None:
                applied += 1
    return applied


def jev_controller(world, seat: int, policy: JevPolicy) -> Dict[str, float]:
    obs = observation(world, seat=seat)
    applied = 0
    latencies: List[float] = []
    for entity in world.sorted_entities(seat):
        if not entity.alive or entity.on_board:
            continue
        candidates = [c for c in build_candidates(obs, seat)
                      if c.action.get("entity") == entity.uid]
        if not candidates:
            continue
        action = policy.choose(obs, seat, candidates,
                               state=summarize_entity(obs, seat, entity.uid))
        if action is None:
            continue
        if apply_action(world, action):
            applied += 1
        client = policy.client
        if client is not None and getattr(client, "last_latency_s", None) is not None:
            latencies.append(client.last_latency_s)
    return {"applied": applied, "latencies": latencies}


def run_episode(policy_name: str, model_path: str, ticks: int,
                decision_every: int, seed: int, device: str = "cpu") -> dict:
    basic = make_basic()
    scenario = make_scenario()
    world = build_world(scenario, basic, seed=seed, dt_s=10.0)
    client = None
    if policy_name == "jev":
        client = (LocalOpenJevClient(model_path, device=device)
                  if model_path else MockJevClient())
    policy = JevPolicy(client=client) if policy_name == "jev" else None
    latencies: List[float] = []
    applied = 0
    decisions = 0
    for i in range(ticks):
        if i % decision_every == 0:
            decisions += 1
            if policy_name == "jev":
                res = jev_controller(world, 0, policy)
                applied += res["applied"]
                latencies.extend(res["latencies"])
            else:
                applied += heuristic_controller(world, 0)
            heuristic_controller(world, 1)
        world.step()
    red = [e for e in world.entities.values() if e.affiliation == 0]
    blue = [e for e in world.entities.values() if e.affiliation == 1]
    return {
        "policy": policy_name,
        "ticks": ticks,
        "decisions": decisions,
        "actions_applied": applied,
        "red_alive": sum(1 for e in red if e.alive),
        "blue_alive": sum(1 for e in blue if e.alive),
        "red_blood": sum(e.blood for e in red if e.alive),
        "blue_blood": sum(e.blood for e in blue if e.alive),
        "objectives": {str(k): v["owner"] for k, v in world.objectives.items()},
        "events": len(world.events),
        "direct_fire": sum(1 for e in world.events if e.get("type") == "direct_fire"),
        "destroyed": sum(1 for e in world.events if e.get("type") == "destroyed"),
        "decision_latency_ms": (round(1000.0 * sum(latencies) / len(latencies), 1)
                                if latencies else None),
        "decision_latency_max_ms": round(1000.0 * max(latencies), 1) if latencies else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None,
                    help="local openjev checkpoint; omit to use MockJevClient")
    ap.add_argument("--ticks", type=int, default=80)
    ap.add_argument("--decision-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    results = {
        "heuristic": run_episode("heuristic", args.model, args.ticks,
                                 args.decision_every, args.seed, args.device),
        "jev": run_episode("jev", args.model, args.ticks,
                           args.decision_every, args.seed, args.device),
    }
    payload = {"seed": args.seed, "model": args.model or "mock", "results": results}
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
