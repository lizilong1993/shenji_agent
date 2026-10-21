"""Action parsing (geo and legacy) into World orders (P5)."""

from __future__ import annotations

from typing import Optional

from compat.legacy import LegacyHexGrid


def apply_action(world, action: dict) -> bool:
    kind = action.get("type")
    uid = action.get("entity", action.get("obj_id"))
    if uid is None:
        return False
    if kind in ("move_to", "move"):
        target = action.get("target") or {}
        if "lon" in target and "lat" in target:
            world.order(int(uid), {"type": "move_to",
                                   "lon": float(target["lon"]),
                                   "lat": float(target["lat"])})
            return True
    elif kind == "stop":
        world.order(int(uid), {"type": "stop"})
        return True
    elif kind == "fire":
        world.order(int(uid), {"type": "fire", "target": int(action["target"])})
        return True
    return False


def legacy_move_path_to_orders(world, uid: int, move_path, grid: LegacyHexGrid) -> bool:
    """Convert an old ``move_path`` (hex ids) into a geo move order."""
    if not move_path:
        return False
    lon, lat = grid.to_lonlat(int(move_path[-1]))
    world.order(int(uid), {"type": "move_to", "lon": lon, "lat": lat})
    return True


def legacy_hex_to_lonlat(hex_pos: int, grid: LegacyHexGrid):
    return grid.to_lonlat(int(hex_pos))
