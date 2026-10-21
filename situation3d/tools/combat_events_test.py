# -*- coding: utf-8 -*-
"""combat event schema sanity checks (run on generated case JSONs)."""
import json, sys, os
fails = []
def check(cond, msg):
    if not cond: fails.append(msg)
for path in sys.argv[1:]:
    d = json.load(open(path, encoding="utf-8"))
    evs = d.get("events") or []
    ids = {e["id"] for e in evs}
    fires = [e for e in evs if e.get("type") == "fire"]
    impacts = [e for e in evs if e.get("type") == "impact"]
    destroys = [e for e in evs if e.get("type") == "destroy"]
    print(os.path.basename(path), "events", len(evs), "fire", len(fires), "impact", len(impacts), "destroy", len(destroys))
    if "dense_transfer" not in d.get("case_id", ""):
        check(len(fires) > 0, "no fire events")
        check(len(impacts) == len(fires), "impact/fire count mismatch")
        check(len(destroys) >= 0, "destroy count")
    for e in fires:
        check(e.get("actors") and len(e["actors"]) == 2, e["id"] + " actors")
        check(e.get("location") and len(e["location"]) == 3, e["id"] + " location")
        check(abs(round(e["location"][0], 6) - e["location"][0]) < 1e-9, e["id"] + " lon precision")
    for e in impacts:
        check(e.get("cause_event_ids") and e["cause_event_ids"][0] in ids, e["id"] + " cause")
        check(e.get("result") in ("hit", "miss"), e["id"] + " result")
    for e in destroys:
        check(e.get("target_id"), e["id"] + " target")
    if "dense_transfer" in d.get("case_id", ""):
        check(not any(e.get("producer") == "combat.synth.v1" for e in evs), "dense case has combat events")
if fails:
    print("FAIL", len(fails)); print("\n".join(fails[:20])); sys.exit(1)
print("ALL COMBAT EVENT CHECKS PASSED")
