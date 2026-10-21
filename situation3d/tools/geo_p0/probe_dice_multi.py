#!/usr/bin/env python3
"""Sample judge dice by one shot per environment instance."""
import sys
from collections import Counter

sys.path.insert(0, "/remote-home/lizilong/shenji_agent_geo/situation3d/tools/geo_p0")
import probe_env as P

def main(n=100):
    tpl = P._load_json(P.TEMPLATE_SCENARIO)["operators"][0]
    cells = P.cell_index(P._load_json(P.MAP / "basic.json"))
    a, b, d, ea, eb = P.find_pair(cells, 1, 1, 0)
    r1 = Counter(); r2 = Counter(); dmg = Counter(); combos = Counter()
    got = 0
    for i in range(n):
        red = P.make_op(tpl, 1001, P.RED, a, [36])
        blue = P.make_op(tpl, 2001, P.BLUE, b, [36], blood=5)
        env, state = P.build_env([red, blue], max_time=80)
        state, _ = P.end_deployment(env, state)
        state, shots = P.shoot_rounds(env, state, 1001, 2001, 36, 1)
        if shots:
            s = shots[0]
            got += 1
            r1[s["random1"]] += 1; r2[s["random2"]] += 1
            dmg[s["damage"]] += 1
            combos[(s["random1"], s["ori_damage"], s["random2"], s["random2_rect"], s["rect_damage"], s["damage"])] += 1
        env.reset()
    print("episodes", n, "shots", got)
    print("random1", sorted(r1.items()))
    print("random2", sorted(r2.items()))
    print("damage", sorted(dmg.items()))
    print("combos", sorted(combos.items()))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
