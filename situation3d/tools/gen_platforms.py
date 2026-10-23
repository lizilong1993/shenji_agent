"""One-off generator for situation3d/platforms/*.json (P0.3).

Values mirror mobility.profile.PROFILES exactly so that loading the JSON
reconstructs the built-in behaviour; extra physical metadata is demo data.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D = os.path.dirname(HERE)
sys.path.insert(0, SITUATION3D)

from mobility.profile import PROFILES  # noqa: E402

EXTRA = {
    "infantry": {
        "name_zh": "步兵小队",
        "mass_t": 0.15,
        "power_hp": 0.0,
        "armor_faces": {"front": 0, "side": 0, "rear": 0, "top": 0},
        "signature": {"rcs_m2": 0.05, "ir": 0.3, "acoustic": 0.6},
        "sensors": [{"kind": "optics", "range_m": 2000}],
        "weapons": [{"weapon_id": 29, "name": "步兵武器", "ammo": 30}],
    },
    "tracked": {
        "name_zh": "履带装甲车辆",
        "mass_t": 40.0,
        "power_hp": 800.0,
        "armor_faces": {"front": 500, "side": 80, "rear": 40, "top": 20},
        "signature": {"rcs_m2": 15.0, "ir": 1.0, "acoustic": 1.0},
        "sensors": [{"kind": "optics", "range_m": 4000}],
        "weapons": [{"weapon_id": 36, "name": "直瞄火炮", "ammo": 10}],
    },
    "wheeled": {
        "name_zh": "轮式装甲车辆",
        "mass_t": 12.0,
        "power_hp": 300.0,
        "armor_faces": {"front": 60, "side": 30, "rear": 20, "top": 10},
        "signature": {"rcs_m2": 8.0, "ir": 0.7, "acoustic": 0.8},
        "sensors": [{"kind": "optics", "range_m": 4000}],
        "weapons": [{"weapon_id": 56, "name": "机关炮", "ammo": 10}],
    },
    "artillery": {
        "name_zh": "自行火炮",
        "mass_t": 15.0,
        "power_hp": 300.0,
        "armor_faces": {"front": 40, "side": 20, "rear": 10, "top": 5},
        "signature": {"rcs_m2": 12.0, "ir": 0.8, "acoustic": 0.9},
        "sensors": [{"kind": "optics", "range_m": 3000}],
        "weapons": [{"weapon_id": 72, "name": "间瞄火炮", "ammo": 10}],
    },
    "amphibious": {
        "name_zh": "两栖战车",
        "mass_t": 25.0,
        "power_hp": 500.0,
        "armor_faces": {"front": 80, "side": 40, "rear": 20, "top": 10},
        "signature": {"rcs_m2": 10.0, "ir": 0.8, "acoustic": 0.9},
        "sensors": [{"kind": "optics", "range_m": 3500}],
        "weapons": [{"weapon_id": 54, "name": "车载机关炮", "ammo": 10}],
    },
    "ship": {
        "name_zh": "水面舰艇",
        "mass_t": 4000.0,
        "power_hp": 50000.0,
        "armor_faces": {"front": 30, "side": 20, "rear": 15, "top": 0},
        "signature": {"rcs_m2": 5000.0, "ir": 1.0, "acoustic": 1.0},
        "sensors": [{"kind": "radar", "range_m": 200000}],
        "weapons": [{"kind": "sea_air", "name": "舰载武器", "ammo": 8}],
    },
    "submarine": {
        "name_zh": "潜艇",
        "mass_t": 2000.0,
        "power_hp": 20000.0,
        "armor_faces": {"front": 20, "side": 15, "rear": 10, "top": 0},
        "signature": {"rcs_m2": 1.0, "ir": 0.2, "acoustic": 0.4},
        "sensors": [{"kind": "sonar", "range_m": 80000}],
        "weapons": [{"kind": "torpedo", "name": "鱼雷", "ammo": 6}],
    },
    "helicopter": {
        "name_zh": "武装直升机",
        "mass_t": 5.0,
        "power_hp": 2000.0,
        "armor_faces": {"front": 20, "side": 10, "rear": 10, "top": 0},
        "signature": {"rcs_m2": 5.0, "ir": 1.0, "acoustic": 1.0},
        "sensors": [{"kind": "optics", "range_m": 8000}],
        "weapons": [{"weapon_id": 54, "name": "空面武器", "ammo": 8}],
    },
    "uav": {
        "name_zh": "无人机",
        "mass_t": 0.5,
        "power_hp": 100.0,
        "armor_faces": {"front": 0, "side": 0, "rear": 0, "top": 0},
        "signature": {"rcs_m2": 0.5, "ir": 0.4, "acoustic": 0.2},
        "sensors": [{"kind": "eo_ir", "range_m": 10000}],
        "weapons": [{"kind": "loitering", "name": "巡飞弹", "ammo": 4}],
    },
    "aircraft": {
        "name_zh": "作战飞机",
        "mass_t": 20.0,
        "power_hp": 20000.0,
        "armor_faces": {"front": 10, "side": 5, "rear": 5, "top": 0},
        "signature": {"rcs_m2": 5.0, "ir": 1.0, "acoustic": 0.9},
        "sensors": [{"kind": "radar", "range_m": 120000}],
        "weapons": [{"kind": "air_air", "name": "空空/空面武器", "ammo": 6}],
    },
}

def main(out_dir=None):
    out = out_dir or os.path.join(SITUATION3D, "platforms")
    os.makedirs(out, exist_ok=True)
    for key, p in sorted(PROFILES.items()):
        extra = EXTRA[key]
        doc = {
            "schema": "msplatform.v1",
            "key": key,
            "identity": {
                "name_zh": extra["name_zh"],
                "domain": p.domain,
                "size_class": p.size_class,
            },
            "dimensions": {
                "length_m": p.length_m,
                "width_m": p.width_m,
                "height_m": p.vehicle_height_m,
            },
            "mass": {"combat_t": extra["mass_t"]},
            "power": {"power_hp": extra["power_hp"]},
            "motion": {
                "max_speed_ms": p.max_speed_ms,
                "accel_ms2": p.accel_ms2,
                "decel_ms2": p.decel_ms2,
                "turn_rate_dps": p.turn_rate_dps,
                "min_turn_radius_m": p.min_turn_radius_m,
                "pivot_turn": p.pivot_turn,
                "terrain_factors": p.terrain_factors,
                "speed_curve": [[0.0, 1.0],
                                [round(p.max_speed_ms, 1), 1.0]],
            },
            "terrain": {
                "max_gradient_deg": p.max_gradient_deg,
                "max_side_slope_deg": p.max_side_slope_deg,
                "wading_depth_m": p.wading_depth_m,
                "swim_depth_m": p.swim_depth_m,
                "amphibious": p.amphibious,
                "road_bound": p.road_bound,
            },
            "sea": {
                "draft_m": p.draft_m,
                "ukc_m": p.ukc_m,
                "max_operating_depth_m": p.max_operating_depth_m,
            },
            "air": {
                "min_clearance_m": p.min_clearance_m,
                "service_ceiling_m": p.service_ceiling_m,
                "agl_m": p.agl_m,
                "climb_rate_ms": p.climb_rate_ms,
                "descent_rate_ms": p.descent_rate_ms,
            },
            "road": {
                "weight_t": p.weight_t,
                "height_m": p.vehicle_height_m,
                "beam_m": p.beam_m,
            },
            "armor": {"class": p.armor_class,
                      "faces_mm_rha": extra["armor_faces"]},
            "signature": extra["signature"],
            "sensors": extra["sensors"],
            "weapons": extra["weapons"],
            "provenance": {
                "source": "demo-estimate",
                "confidence": "low",
                "note": "演示估计值，非真实装备数据；P0.3 v1 仅用于重建现有行为。",
            },
        }
        path = os.path.join(out, key + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print("wrote", path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="output directory (default: situation3d/platforms)")
    main(parser.parse_args().out)
