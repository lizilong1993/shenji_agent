# -*- coding: utf-8 -*-
"""combat event schema / consistency checks."""
import json, sys, os
fails = []
def check(cond, msg):
    if not cond: fails.append(msg)

for path in sys.argv[1:]:
    d = json.load(open(path, encoding="utf-8"))
    evs = d.get("events") or []
    cid = d.get("case_id", "")
    fires = [e for e in evs if e.get("type") == "fire"]
    impacts = [e for e in evs if e.get("type") == "impact"]
    destroys = [e for e in evs if e.get("type") == "destroy"]
    print(os.path.basename(path), "events", len(evs), "fire", len(fires), "impact", len(impacts), "destroy", len(destroys))
    if "dense_transfer" in cid:
        check(not any(e.get("producer") == "combat.synth.v1" for e in evs), "dense case must not have combat events")
        continue
    ids = {e["id"] for e in evs}
    fmap = {e["id"]: e for e in fires}
    check(len(fires) > 0, "no fire events")
    check(len(impacts) == len(fires), "impact/fire count mismatch")
    has_stage = bool(d.get("phases") or d.get("stages"))
    for e in fires:
        check(isinstance(e.get("actors"), list) and len(e["actors"]) == 2, e["id"] + " actors")
        loc = e.get("location") or []
        check(len(loc) == 3 and all(isinstance(x, (int, float)) for x in loc), e["id"] + " location")
        if loc: check(abs(loc[0] * 1e6 - round(loc[0] * 1e6)) < 1e-6, e["id"] + " lon precision")
        if has_stage: check(bool(e.get("stage_id")), e["id"] + " empty stage_id")
        # 武器与目标域的基本匹配
        a = next((t for t in d["targets"] if t["id"] == e["actors"][0]), {})
        b = next((t for t in d["targets"] if t["id"] == e["actors"][1]), {})
        sd, td, w = a.get("domain"), b.get("domain"), e.get("weapon", "")
        if sd == "sea" and td == "air": check("舰空" in w, e["id"] + " sea->air weapon")
        if sd == "air" and td == "air": check("空空" in w, e["id"] + " air->air weapon")
        if sd == "land" and "防空" in str(a.get("platform_class")) and td == "air": check("防空" in w, e["id"] + " sam weapon")
    for e in impacts:
        cause = (e.get("cause_event_ids") or [None])[0]
        check(cause in ids, e["id"] + " cause missing")
        f = fmap.get(cause)
        check(bool(f), e["id"] + " cause not fire")
        if f:
            check(f["actors"] == e["actors"], e["id"] + " actor mismatch")
            check(e["tick"] >= f["tick"], e["id"] + " impact before fire")
        check(e.get("result") in ("hit", "miss"), e["id"] + " result")
        if e.get("result") == "hit": check(e.get("damage", 0) > 0, e["id"] + " hit damage")
    for e in destroys:
        check(bool(e.get("target_id")), e["id"] + " target")
    # 事件 id 唯一
    check(len(ids) == len(evs), "duplicate event ids")
    # 排序
    ticks = [e["tick"] for e in evs]
    check(ticks == sorted(ticks), "events not sorted by tick")
if fails:
    print("FAIL", len(fails)); print("\n".join(fails[:30])); sys.exit(1)
print("ALL COMBAT EVENT CHECKS PASSED")
