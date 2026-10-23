#!/usr/bin/env python3
"""Probe time.tick semantics: steps per 75 s direct-fire cooldown."""
import sys

sys.path.insert(0, "/remote-home/lizilong/shenji_agent_geo/situation3d/tools/geo_p0")
import probe_env as P

def run(tick, steps=260):
    tpl = P._load_json(P.TEMPLATE_SCENARIO)["operators"][0]
    cells = P.cell_index(P._load_json(P.MAP / "basic.json"))
    a, b, d, ea, eb = P.find_pair(cells, 1, 1, 0)
    red = P.make_op(tpl, 1001, P.RED, a, [36])
    blue = P.make_op(tpl, 2001, P.BLUE, b, [36], blood=5)
    env, state = P.build_env([red, blue], tick=tick, max_time=int(steps * tick) + 10)
    state, _ = P.end_deployment(env, state)
    fire_steps = []
    cur_steps = []
    for i in range(steps):
        state, done = env.step([{"actor": 1, "obj_id": 1001, "type": 2,
                                 "target_obj_id": 2001, "weapon_id": 36}])
        cur_steps.append(state[-1]["time"]["cur_step"])
        if (state[-1].get("judge_info") or []):
            fire_steps.append(i)
        if done:
            break
    print("tick", tick, "fire_steps", fire_steps[:6],
          "gaps", [fire_steps[j + 1] - fire_steps[j] for j in range(min(4, len(fire_steps) - 1))],
          "cur_step_first_last", cur_steps[0], cur_steps[-1], "stage", state[-1]["time"]["stage"])
    env.reset()

if __name__ == "__main__":
    for t in (1.0, 2.0, 5.0):
        run(t)
