#!/usr/bin/env python3
"""P0 black-box probe for the closed 庙算 land engine (train_env 4.1.0).

Goal: empirically confirm data semantics that the public dictionary only
guesses at:
  * TarType (0/1/2/3) -> which WpData row a shot uses;
  * ele_rect_to_car/peo index semantics (row = elevation grade? col = what?);
  * dice (random1/random2) distributions and whether seeds are controllable;
  * time.tick meaning (frames per step vs seconds per frame).

Run with the py310env.sh environment (see /remote-home/lizilong/py310env.sh),
from the land_wargame_sdk root so relative Data paths resolve.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy

from train_env import TrainEnv

SDK_ROOT = Path("/remote-home/lizilong/shenji_agent/land_wargame_sdk")
MAP = SDK_ROOT / "Data/Data/maps/map_19"
TEMPLATE_SCENARIO = SDK_ROOT / "Data/Data/scenarios/1231.json"

RED, BLUE = 0, 1
SEAT_RED, SEAT_BLUE = 1, 11


def _load_json(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def load_map_bits():
    basic = _load_json(MAP / "basic.json")
    cost = pickle.loads((MAP / "cost.pickle").read_bytes())
    npz = numpy.load(str(MAP / "19see.npz"))
    key = "data" if "data" in npz.files else npz.files[0]
    see = npz[key]
    return basic, cost, see


def cube(row, col):
    x = col - (row - (row & 1)) // 2
    z = row
    return x, -x - z, z


def hex_distance(pos_a, pos_b):
    ar, ac = divmod(int(pos_a), 100)
    br, bc = divmod(int(pos_b), 100)
    ax, ay, az = cube(ar, ac)
    bx, by, bz = cube(br, bc)
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))


def cell_index(basic):
    """pos -> (row, col, elev, cond)."""
    out = {}
    for r, row in enumerate(basic["map_data"]):
        for c, cell in enumerate(row):
            out[int(cell["pos"])] = (r, c, int(cell["elev"]), int(cell["cond"]))
    return out


def find_pair(cells, d_min, d_max, elev_grade_delta=0, cond=0):
    """Find two open cells with given hex distance and elevation difference."""
    by_pos = sorted(cells)
    for a in by_pos:
        ra, ca, ea, conda = cells[a]
        if conda != cond:
            continue
        for b in by_pos:
            rb, cb, eb, condb = cells[b]
            if condb != cond or a == b:
                continue
            d = hex_distance(a, b)
            if not (d_min <= d <= d_max):
                continue
            if elev_grade_delta is not None:
                df = (ea - eb) / 20.0  # ele_grade of map_19 == 20 m
                if abs(df - elev_grade_delta) > 0.001:
                    continue
            return a, b, d, ea, eb
    raise SystemExit("no cell pair found for d=%s..%s dh=%s" % (d_min, d_max, elev_grade_delta))


def make_op(template, oid, color, hexpos, weapons, name="重型坦克", type_=2,
            armor=4, blood=3, sub_type=0):
    op = copy.deepcopy(template)
    op.update({
        "obj_id": oid, "color": color, "type": type_, "name": name,
        "sub_type": sub_type, "armor": armor, "blood": blood,
        "max_blood": blood, "cur_hex": hexpos,
        "carry_weapon_ids": list(weapons),
        "remain_bullet_nums": {str(w): 99 for w in weapons},
        "C2": 999, "C3": 999,
        "observe_distance": {"1": 60, "2": 60, "3": 60},
        "move_state": 0, "cur_pos": 0.0, "speed": 0.0,
        "move_to_stop_remain_time": 0.0, "can_to_move": 0, "stop": 1,
        "move_path": [], "keep": 0, "keep_remain_time": 0,
        "on_board": 0, "car": 0, "launcher": None,
        "passenger_ids": [], "launch_ids": [], "lose_control": 0,
        "see_enemy_bop_ids": [], "weapon_cool_time": 0.0,
        "weapon_unfold_time": 0.0, "weapon_unfold_state": 0,
        "target_state": 0, "change_state_remain_time": 0.0,
    })
    return op


def build_env(ops, scenario_id=990001, tick=1.0, max_time=600):
    env = TrainEnv()
    scenario = {
        "scenario_id": scenario_id,
        "operators": ops,
        "time": {"cur_step": 0, "tick": tick, "max_time": max_time},
        "cities": [],
    }
    basic, cost, see = load_map_bits()
    players = [
        {"seat": SEAT_RED, "faction": RED, "role": 1, "user_name": "probe", "user_id": 0},
        {"seat": SEAT_BLUE, "faction": BLUE, "role": 1, "user_name": "probe", "user_id": 0},
    ]
    state = env.setup({
        "scenario_data": scenario,
        "basic_data": basic,
        "cost_data": cost,
        "see_data": see,
        "player_info": players,
    })
    return env, state


def shoot_rounds(env, state, red_id, blue_id, weapon_id, rounds, tick=1.0,
                 step_limit=2000):
    """Fire up to ``rounds`` shots, respecting any cooldown, collecting judge_info."""
    shots = []
    step = 0
    while len(shots) < rounds and step < step_limit:
        action = [{"actor": SEAT_RED, "obj_id": red_id, "type": 2,
                   "target_obj_id": blue_id, "weapon_id": weapon_id}]
        state, done = env.step(action)
        step += 1
        for ji in state[1].get("judge_info", []):
            if ji.get("wp_id") == weapon_id:
                ji["_step"] = step
                shots.append(ji)
        if done:
            break
    return state, shots


def exp_smoke(args):
    tpl = _load_json(TEMPLATE_SCENARIO)["operators"][0]
    cells = cell_index(_load_json(MAP / "basic.json"))
    a, b, d, ea, eb = find_pair(cells, 1, 1, 0)
    print("pair", a, b, "dist", d, "elev", ea, eb)
    red = make_op(tpl, 1001, RED, a, [36])
    blue = make_op(tpl, 2001, BLUE, b, [36])
    env, state = build_env([red, blue], max_time=200)
    state, shots = shoot_rounds(env, state, 1001, 2001, 36, args.rounds)
    print("steps", state[1]["time"], "shots", len(shots))
    for s in shots[:5]:
        print(json.dumps(s, ensure_ascii=False))
    env.reset()


def exp_tartype(args):
    """Fire weapon 36 at targets of different unit type/blood and compare att_level."""
    tpl = _load_json(TEMPLATE_SCENARIO)["operators"][0]
    cells = cell_index(_load_json(MAP / "basic.json"))
    a, b, d, ea, eb = find_pair(cells, 1, 1, 0)
    rows = []
    for tgt_type, tgt_name, blood in [
        (2, "重型坦克", 1), (2, "重型坦克", 2), (2, "重型坦克", 3), (2, "重型坦克", 4),
        (1, "步兵小队", 1), (1, "步兵小队", 2), (1, "步兵小队", 3), (1, "步兵小队", 4),
    ]:
        red = make_op(tpl, 1001, RED, a, [36])
        blue = make_op(tpl, 2001, BLUE, b, [36], name=tgt_name, type_=tgt_type, blood=blood)
        env, state = build_env([red, blue], max_time=120)
        state, shots = shoot_rounds(env, state, 1001, 2001, 36, 1)
        if shots:
            s = shots[0]
            rows.append((tgt_type, tgt_name, blood, s.get("distance"), s.get("ele_diff"),
                         s.get("att_level"), s.get("random1"), s.get("random2"),
                         s.get("damage")))
        else:
            rows.append((tgt_type, tgt_name, blood, "NO SHOT", "", "", "", "", ""))
        env.reset()
    print("tgt_type tgt_name blood dist ele_diff att_level random1 random2 damage")
    for r in rows:
        print(" ".join(str(x) for x in r))


def exp_elevation(args):
    """Vary elevation grade difference at fixed hex distances; read ele_diff/att_level."""
    tpl = _load_json(TEMPLATE_SCENARIO)["operators"][0]
    cells = cell_index(_load_json(MAP / "basic.json"))
    print("elev_grade", _load_json(MAP / "basic.json")["ele_grade"])
    print("dh_grade dist elev_a elev_b ele_diff att_level")
    seen = set()
    for dh in range(-4, 5):
        for dmin, dmax in [(1, 1), (2, 2), (3, 3), (5, 5), (8, 8), (12, 12)]:
            try:
                a, b, d, ea, eb = find_pair(cells, dmin, dmax, dh)
            except SystemExit:
                continue
            key = (dh, d)
            if key in seen:
                continue
            seen.add(key)
            red = make_op(tpl, 1001, RED, a, [36])
            blue = make_op(tpl, 2001, BLUE, b, [36])
            env, state = build_env([red, blue], max_time=120)
            _, shots = shoot_rounds(env, state, 1001, 2001, 36, 1)
            if shots:
                s = shots[0]
                print(dh, d, ea, eb, s.get("ele_diff"), s.get("att_level"))
            else:
                print(dh, d, ea, eb, "NO SHOT", "")
            env.reset()


def exp_dice(args):
    tpl = _load_json(TEMPLATE_SCENARIO)["operators"][0]
    cells = cell_index(_load_json(MAP / "basic.json"))
    a, b, d, ea, eb = find_pair(cells, 1, 1, 0)
    red = make_op(tpl, 1001, RED, a, [36])
    blue = make_op(tpl, 2001, BLUE, b, [36])
    env, state = build_env([red, blue], max_time=20000)
    _, shots = shoot_rounds(env, state, 1001, 2001, 36, args.rounds, step_limit=40000)
    r1 = Counter(int(s["random1"]) for s in shots if "random1" in s)
    r2 = Counter(int(s["random2"]) for s in shots if "random2" in s)
    print("shots", len(shots))
    print("random1", sorted(r1.items()))
    print("random2", sorted(r2.items()))
    if len(shots) > 10:
        print("random1 max consecutive identical",
              max(len(list(g)) for _, g in __import__("itertools").groupby(
                  [s["random1"] for s in shots])))
    env.reset()


def exp_tick(args):
    tpl = _load_json(TEMPLATE_SCENARIO)["operators"][0]
    cells = cell_index(_load_json(MAP / "basic.json"))
    a, b, d, ea, eb = find_pair(cells, 1, 1, 0)
    for tick in (1.0, 2.0, 5.0):
        red = make_op(tpl, 1001, RED, a, [36])
        blue = make_op(tpl, 2001, BLUE, b, [36])
        env, state = build_env([red, blue], tick=tick, max_time=100)
        # shoot once on the first step; report when judge_info appears and cur_step values
        action = [{"actor": SEAT_RED, "obj_id": 1001, "type": 2,
                   "target_obj_id": 2001, "weapon_id": 36}]
        for i in range(1, 6):
            state, done = env.step(action)
            time = state[1]["time"]
            ji = state[1].get("judge_info", [])
            print("tick", tick, "call", i, "time", time, "judge", [(j["cur_step"], j["distance"]) for j in ji])
        env.reset()


EXPERIMENTS = {
    "smoke": exp_smoke,
    "tartype": exp_tartype,
    "elevation": exp_elevation,
    "dice": exp_dice,
    "tick": exp_tick,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=sorted(EXPERIMENTS))
    ap.add_argument("--rounds", type=int, default=20)
    args = ap.parse_args()
    EXPERIMENTS[args.experiment](args)


if __name__ == "__main__":
    main()
