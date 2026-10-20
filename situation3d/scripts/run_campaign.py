#!/usr/bin/env python3
"""Build a situation3d ``campaign.v3`` dataset from a wargame campaign.

This is the bridge between the joint wargame core and the 3D situation
display.  It writes:

  <out>/<case_id>.json          campaign.v3 (stages/events/frames/outcome)
  <out>/campaign_maps/map_<id>/ campaign hex map(s) copied for reference

Usage:
    python3 situation3d/scripts/run_campaign.py --case taiwan_landing \
        --out situation3d/web/data/situations
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
REPO = os.path.dirname(SITUATION3D)
for p in (SITUATION3D, os.path.join(SITUATION3D, "campaigns")):
    if p not in sys.path:
        sys.path.insert(0, p)

from wargame_core import scenario  # noqa: E402


def _load_plugin(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_json(path: str, payload) -> int:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    return os.path.getsize(path)


def _geo_from_hex(state, loc):
    """Event location is [row, col]; resolve to [lon, lat] when in range."""
    try:
        r, c = int(loc[0]), int(loc[1])
    except (TypeError, ValueError):
        return None
    if not (0 <= r < state.map.rows and 0 <= c < state.map.cols):
        return None
    try:
        lon, lat = state.map.pos_to_lonlat(r * 100 + c)
    except KeyError:
        return None
    return [round(lon, 6), round(lat, 6)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True,
                    help="campaign plugin name under situation3d/campaigns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-ticks-per-stage", type=int, default=None)
    args = ap.parse_args(argv)

    plugins = os.path.join(SITUATION3D, "campaigns")
    path = os.path.join(plugins, args.case + ".py")
    if not os.path.isfile(path):
        print("ERROR: campaign plugin not found: %s" % path, file=sys.stderr)
        return 2
    mod = _load_plugin(path, "campaign_" + args.case)
    spec = mod.build_spec()
    if args.seed is not None:
        spec.seed = args.seed

    map_path = mod.map_path()
    state = scenario.load_scenario_dict(mod.scenario_dict(), map_path,
                                        seed=spec.seed,
                                        tick_seconds=spec.tick_seconds)
    # Campaign spec objectives override/add to the scenario cities.
    from wargame_core.state import Objective
    if spec.objectives:
        state.objectives = [Objective(coord=int(o["coord"]),
                                      value=int(o.get("value") or 0),
                                      flag=int(o.get("flag", -1)),
                                      name=str(o.get("name") or ""))
                            for o in spec.objectives]
    runner = mod.runner_class()(spec, state)
    # Snapshot the starting order of battle before the run mutates it.
    import copy
    initial = copy.deepcopy(state)
    result = runner.run(max_ticks_per_stage=args.max_ticks_per_stage)

    # Compatibility fields so the existing player can open the case without
    # knowing about campaign.v3.  Hotspots are derived from the objectives;
    # symbology comes from the base data.
    base = {}
    base_path = os.path.join(SITUATION3D, "web", "data", "base.json")
    if os.path.isfile(base_path):
        try:
            base = json.load(open(base_path, encoding="utf-8"))
        except (OSError, ValueError):
            base = {}
    hotspots = []
    for o in spec.objectives:
        try:
            p = state.map.pos_to_lonlat(int(o["coord"]))
        except KeyError:
            continue
        hotspots.append({
            "name": o.get("name") or str(o["coord"]),
            "bbox": [round(p[0] - 0.02, 4), round(p[1] - 0.02, 4),
                     round(p[0] + 0.02, 4), round(p[1] + 0.02, 4)],
            "center": [round(p[1], 4), round(p[0], 4)],
            "span_deg": [0.04, 0.04],
            "note": "虚构演示目标点",
            "kind": "objective",
        })

    # L1/L2/L3 inference for the right-hand panels and the hotspot layer.
    # Campaign frames carry the wargame-core state, not the legacy static
    # definitions, so the levels are recomputed here.
    from campaign_inference import infer_campaign
    # Target definitions describe the starting order of battle; deriving them
    # from the final state would label a beached unit as "land" even though it
    # spent most of the run afloat (and vice versa).
    targets = mod.target_defs(initial)
    inference = infer_campaign(result["frames"], targets, hotspots)
    result["frames"], _ = result["frames"], None
    # Normalise the region list to the objective order used by the panel.
    inference["regions_order"] = [h["name"] for h in hotspots]

    payload = {
        "schema": "situation3d.campaign.v3",
        "case_id": spec.id,
        "case_name": spec.name,
        "case_description": ("虚构战役演示：%d 个阶段、%d 个目标、%d 个事件。"
                             "单位、参数与结局均为合成数据。" % (
                                 len(result["stages"]), len(state.units),
                                 len(result["events"]))),
        "region": mod.region(),
        "theatre": spec.theatre,
        "provenance": "fictional_demo",
        "provenance_note": ("由 situation3d.wargame_core 生成的虚构战役演示；"
                            "单位、参数、事件与结局均为合成数据，"
                            "不代表现实部署或战场预测。"),
        "tick_seconds": spec.tick_seconds,
        "tick_count": len(result["frames"]),
        "start_time_utc": mod.start_time_utc(),
        "camera": mod.camera(),
        "campaign": result["campaign"],
        "stages": result["stages"],
        "events": result["events"],
        "frames": result["frames"],
        "outcome": result["outcome"],
        "final": result["final"],
        "targets": targets,
        "hotspots": hotspots,
        "plots": mod.plots() if hasattr(mod, "plots") else [],
        "camera": mod.camera(),
        # The front end matches frames[].targets[].id against targets[].id with
        # ===; object keys are always strings, so both sides use strings.
        "target_id_type": "str",
        "urban_aoi": base.get("urban_aoi") or [],
        "symbology": base.get("symbology"),
        "inference_frames": inference["frames"],
        "inference_model": inference["model"],
        "inference_summary": inference["summary"],
        "supply_points": [
            {"pos": sp.get("pos"), "color": sp.get("color"), "kind": sp.get("kind")}
            for sp in (spec.supply_points or [])
        ],
    }
    # Resolve event locations (hex row/col) to lon/lat for the map UI.
    for e in payload["events"]:
        loc = e.get("location")
        if isinstance(loc, list) and len(loc) == 2:
            geo = _geo_from_hex(state, loc)
            if geo:
                e["geo"] = geo

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, spec.id + ".json")
    size = _write_json(out_path, payload)

    # Keep a copy of the map under web/data/campaign_maps for reproducibility.
    # Never write it inside data/situations: that directory is the case index.
    web_data = os.path.join(SITUATION3D, "web", "data")
    dst_dir = os.path.join(web_data, "campaign_maps", "map_%d" % mod.map_id())
    if os.path.abspath(dst_dir) != os.path.abspath(os.path.dirname(map_path)):
        os.makedirs(dst_dir, exist_ok=True)
        for name in os.listdir(os.path.dirname(map_path)):
            src = os.path.join(os.path.dirname(map_path), name)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(dst_dir, name))

    print("case      : %s (%s)" % (spec.id, spec.name))
    print("stages    : %d" % len(result["stages"]))
    print("frames    : %d  (tick=%ss)" % (len(result["frames"]), spec.tick_seconds))
    print("events    : %d" % len(result["events"]))
    print("outcome   : engine=%s script=%s match=%s" % (
        result["outcome"]["engine"]["winner"],
        (result["outcome"].get("script") or {}).get("winner"),
        result["outcome"].get("match")))
    print("written   : %s (%.1f KB)" % (out_path, size / 1024.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
