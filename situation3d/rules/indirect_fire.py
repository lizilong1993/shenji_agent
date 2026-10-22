"""Indirect fire: scatter tables, impact geometry and area adjudication (P4).

The authoritative sheets ``无校射火力散布裁决`` (planned fire, no observer
adjustment) and ``格内校射火力散布裁决`` (adjusted fire) map ``(distance band,
2d6)`` to either a hit on the aim cell or a scatter of N rule cells.  Each
scattered round walks N random neighbouring cells from the aim cell; units
of the opposing side in the impact cell are adjudicated with the artillery
TarType=0 attack table plus the shared CRT/correction helper.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from rules.combat import resolve_crt_shot
from rules.crt import load_crt
from rules.weapons import TAR_TYPE_INDIRECT

CELL_HIT = "hit"
CELL_SCATTER = "scatter"

# Column order of both sheets: 0-20 / 21-40 / 41-60 / 61+ rule cells.
_DISTANCE_BINS = ((0, 20), (21, 40), (41, 60), (61, 100000))
_SCATTER_RE = re.compile(r"散布(\d+)格")


def _parse_cell(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "命中" in text:
        return (CELL_HIT, 0)
    match = _SCATTER_RE.search(text)
    if match:
        return (CELL_SCATTER, int(match.group(1)))
    if "散布" in text:
        return (CELL_SCATTER, 1)
    return None


def _parse_sheet(sheet) -> List[Tuple[int, int, Dict[int, Tuple[str, int]]]]:
    parsed = []
    for idx, (lo, hi) in enumerate(_DISTANCE_BINS, start=1):
        cells = {}
        for row in sheet[2:]:
            try:
                roll = int(float(row[0]))
            except (TypeError, ValueError):
                continue
            if roll < 2 or roll > 12:
                continue
            value = _parse_cell(row[idx] if idx < len(row) else None)
            if value is not None:
                cells[roll] = value
        if cells:
            parsed.append((lo, hi, cells))
    return parsed


class ScatterTables(object):
    def __init__(self, doc: Optional[dict] = None):
        sheets = (doc or load_crt())["sheets"]
        self.unobserved = _parse_sheet(sheets["无校射火力散布裁决"])
        self.adjusted = _parse_sheet(sheets["格内校射火力散布裁决"])

    def lookup(self, observed: bool, distance_cells: int,
               roll: int) -> Tuple[str, int]:
        table = self.adjusted if observed else self.unobserved
        for lo, hi, cells in table:
            if lo <= distance_cells <= hi:
                return cells.get(roll, (CELL_HIT, 0))
        # Distance beyond the last band: treat as the farthest column.
        lo, hi, cells = max(table, key=lambda item: item[1])
        return cells.get(roll, (CELL_HIT, 0))


_TABLES = None


def tables() -> ScatterTables:
    global _TABLES
    if _TABLES is None:
        _TABLES = ScatterTables()
    return _TABLES


def select_indirect_weapon(attacker, distance_cells: int, ctx):
    """Return ``(weapon, table)`` for the first ready indirect weapon in range."""
    best = None
    for weapon in getattr(attacker, "weapons", []) or []:
        if getattr(weapon, "kind", None) != "indirect":
            continue
        if attacker.ammo_count(weapon.key) <= 0:
            continue
        if not ctx.weapon_ready(attacker.uid, weapon.key):
            continue
        table = (weapon.tables or {}).get((TAR_TYPE_INDIRECT, 0))
        if table is None:
            continue
        if table.level_at(distance_cells) is None:
            continue
        if best is None or table.att_max > best[1].att_max:
            best = (weapon, table)
    return best


def _impact_cell(ctx, aim_cell: Tuple[int, int], scatter: int) -> Tuple[int, int]:
    cell = aim_cell
    for _ in range(max(0, int(scatter))):
        neighbors = ctx.rule_neighbor_cells(cell)
        if not neighbors:
            break
        cell = neighbors[ctx.rng.randrange(len(neighbors))]
    return cell


def resolve_indirect_fire(attacker, aim_cell: Tuple[int, int], ctx,
                          observed: bool = False) -> Optional[dict]:
    """Fire one indirect round at ``aim_cell`` (rule-grid cell)."""
    if not attacker.alive:
        return None
    attacker_cell = ctx.rule_cell(attacker)
    distance = ctx.rule_distance_cells(attacker_cell, aim_cell)
    picked = select_indirect_weapon(attacker, distance, ctx)
    if picked is None:
        return None
    weapon, table = picked
    level = table.level_at(distance)
    roll = ctx.rng.randint(1, 6) + ctx.rng.randint(1, 6)
    outcome, scatter_cells = tables().lookup(observed, distance, roll)
    impact = aim_cell if outcome == CELL_HIT else _impact_cell(ctx, aim_cell, scatter_cells)
    impact_distance = ctx.rule_distance_cells(attacker_cell, impact)
    impact_level = table.level_at(impact_distance)
    if impact_level is None:
        impact_level = level

    ctx.consume_ammo(attacker.uid, weapon.key, 1)
    ctx.mark_fired(attacker.uid, weapon.key, getattr(weapon, "cooldown_s", 75.0))

    results = []
    for target in ctx.units_in_cell(impact):
        if not target.alive or target.affiliation == attacker.affiliation:
            continue
        if ctx.target_kind(target) not in ("infantry", "vehicle"):
            continue
        shot = resolve_crt_shot(attacker, target, ctx, impact_level, weapon,
                                distance=impact_distance, consume=False)
        shot["type"] = "indirect_shot"
        results.append(shot)
    lon, lat = ctx.rule_cell_center(impact)
    return {
        "type": "indirect_fire",
        "actors": [attacker.uid] + [r["actors"][1] for r in results],
        "weapon": weapon.key,
        "aim_cell": [int(aim_cell[0]), int(aim_cell[1])],
        "impact_cell": [int(impact[0]), int(impact[1])],
        "distance": distance,
        "impact_distance": impact_distance,
        "observed": bool(observed),
        "roll": roll,
        "outcome": outcome,
        "scatter_cells": int(scatter_cells),
        "att_level": int(impact_level),
        "targets": results,
        "damage": sum(int(r.get("damage", 0)) for r in results),
        "location_geo": {"lon": lon, "lat": lat},
    }
