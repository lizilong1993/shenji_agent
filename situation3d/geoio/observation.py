"""Seat-filtered observation JSON (P5).

Own units are always reported; enemy units only when a friendly observer
currently holds them (``detected`` flag plus the last-seen set).  Legacy
``hex``/``row``/``col`` fields are produced only when a legacy grid is passed.
"""

from __future__ import annotations

from typing import Optional

from rules import combat, engagement
from sim.context import WorldRuleContext


def _visible_to(world, seat: int, target_uid: int) -> bool:
    for uid, seen in world.seen.items():
        observer = world.entities.get(uid)
        if observer is not None and observer.affiliation == seat and target_uid in seen:
            return True
    return False


def _entity_dict(entity, legacy_grid=None) -> dict:
    d = {
        "id": entity.uid,
        "name": entity.name,
        "affiliation": entity.affiliation,
        "domain": entity.domain,
        "unit_type": entity.unit_type,
        "unit_kind": entity.unit_kind,
        "pos": {"lon": round(entity.pose.lon, 7),
                "lat": round(entity.pose.lat, 7),
                "alt": round(entity.pose.alt_m, 2)},
        "heading_deg": round(entity.pose.heading_deg, 1),
        "speed_ms": round(entity.velocity_ms, 2),
        "blood": entity.blood,
        "max_blood": entity.max_blood,
        "armor": entity.armor,
        "suppressed": entity.suppressed,
        "fuel": round(entity.fuel, 3),
        "alive": entity.alive,
        "value": entity.value,
        "stacked": entity.stacked,
        "concealed": entity.concealed,
    }
    if legacy_grid is not None:
        pos = legacy_grid.from_lonlat(entity.pose.lon, entity.pose.lat)
        row, col = divmod(pos, 100)
        d["hex"] = pos
        d["row"] = row
        d["col"] = col
    return d


def _valid_actions(world, ctx: WorldRuleContext, entity) -> list:
    actions = [{"type": "stop"}]
    if not entity.alive:
        return actions
    for target in world.sorted_entities():
        if target.uid == entity.uid or not target.alive:
            continue
        if target.affiliation == entity.affiliation:
            continue
        if target.uid not in world.seen.get(entity.uid, set()):
            continue
        if entity.domain == "land" and target.domain == "land":
            picked = combat.select_weapon(ctx.view(entity), ctx.view(target), ctx)
            if picked is not None:
                actions.append({"type": "fire", "target": target.uid,
                                "weapon": picked[0].key})
        else:
            kinds = engagement.allowed_kinds(entity.domain, target.domain)
            d_km = ctx.distance_m(ctx.view(entity), ctx.view(target)) / 1000.0
            for w in entity.sea_air_weapons or []:
                if (w.kind in kinds and w.min_range_km <= d_km <= w.max_range_km
                        and entity.ammo.get(w.key, 0) > 0):
                    actions.append({"type": "fire", "target": target.uid,
                                    "weapon": w.key})
                    break
    return actions


def observation(world, seat: Optional[int] = None, legacy_grid=None) -> dict:
    ctx = WorldRuleContext(world)
    ctx.refresh()
    entities = []
    for entity in world.sorted_entities():
        visible = seat is None or entity.affiliation == seat or _visible_to(world, seat, entity.uid)
        if not visible or not entity.alive:
            continue
        d = _entity_dict(entity, legacy_grid)
        d["detected"] = True
        if seat is not None and entity.affiliation == seat:
            d["actions"] = _valid_actions(world, ctx, entity)
        entities.append(d)
    objectives = []
    for cell, obj in sorted(world.objectives.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        lon, lat = world.rule_grid.cell_center_lonlat(*cell)
        objectives.append({"cell": list(cell), "pos": {"lon": lon, "lat": lat},
                           "owner": obj["owner"], "value": obj["value"],
                           "name": obj["name"]})
    return {
        "schema": "msgeo.observation.v1",
        "time_s": round(world.time_s, 3),
        "entities": entities,
        "objectives": objectives,
    }
