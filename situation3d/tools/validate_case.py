#!/usr/bin/env python3
"""Offline contract / density check for a campaign case."""
import argparse, json, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="taiwan_dense_transfer_v1")
    ap.add_argument("--dir", default=None)
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = a.dir or os.path.join(root, "web", "data", "situations")
    p = os.path.join(d, a.case + ".json")
    x = json.load(open(p))
    errs = []
    def check(ok, msg):
        if not ok: errs.append(msg)
    t = x.get("targets") or []
    f = x.get("frames") or []
    inf = x.get("inference_frames") or []
    check(len(t) == 80, "targets != 80: %d" % len(t))
    check(len(f) == 1441, "frames != 1441: %d" % len(f))
    check(len(inf) == len(f), "inference frames length mismatch")
    check(all(fr.get("tick") == i for i, fr in enumerate(f)), "frame ticks not 0..1440")
    check(all(inf[i].get("tick") == i for i in range(len(inf))), "inference ticks not aligned")
    ids = [z.get("id") for z in t]
    check(len(ids) == len(set(ids)), "duplicate target ids")
    stages = x.get("stages") or []
    check(len(stages) == 6, "stages != 6")
    last = 0
    for s in stages:
        check(s["start_tick"] == last, "stage gap at %s" % s.get("id"))
        last = s["end_tick"]
    check(last == 1441, "stages do not cover 1441")
    ev = x.get("events") or []
    check(len(ev) >= 8, "events < 8")
    for e in ev:
        check(e.get("stage_id") in {s["id"] for s in stages}, "event stage missing: %s" % e.get("id"))
        for actor in e.get("actors") or []:
            check(actor in set(ids) or actor in {o["id"] for o in x.get("objectives") or []},
                  "event actor missing: %s" % actor)
    obj = x.get("objectives") or []
    check(len(obj) == 6, "objectives != 6")
    out = x.get("outcome") or {}
    check(out.get("script_expected") and out.get("engine", {}).get("status"),
          "outcome missing")
    # mode/carrier checks at t0 and after release
    def modes(k):
        fr = f[k]
        return {s["id"]: s for s in fr["targets"]}
    m0 = modes(0); m900 = modes(900)
    emb = [k for k, v in m0.items() if v.get("mode") == "embarked"]
    check(len(emb) == 16, "embarked at t0 != 16: %d" % len(emb))
    ground = [k for k, v in m900.items() if v.get("mode") == "ground" and k.startswith(("TRV", "TBV"))]
    check(len(ground) == 60, "ground vehicles at 900 != 60: %d" % len(ground))
    # density: red ground in D at 720/900 (synthetic area count)
    def area_count(tick, area, aff="red"):
        bbox = {"D": (120.145, 22.960, 120.180, 22.995),
                "E": (120.165, 22.950, 120.200, 22.985),
                "F": (120.185, 22.945, 120.220, 22.980)}[area]
        w, s, e, n = bbox
        return sum(1 for r in f[tick]["targets"]
                   if r["id"].startswith(("TRV", "TBV")) and r["mode"] == "ground"
                   and s <= r["lat"] <= n and w <= r["lon"] <= e)
    d720 = area_count(720, "D")
    check(d720 >= 12, "D ground at 720 < 12: %d" % d720)
    print(json.dumps({"file": p, "targets": len(t), "frames": len(f),
                      "events": len(ev), "objectives": len(obj),
                      "embarked_t0": len(emb), "ground_900": len(ground),
                      "D_ground_720": d720,
                      "outcome": out.get("engine", {}).get("result"),
                      "errors": errs}, ensure_ascii=False, indent=1))
    return 0 if not errs else 1

if __name__ == "__main__":
    sys.exit(main())
