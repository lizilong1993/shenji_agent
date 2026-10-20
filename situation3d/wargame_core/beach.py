"""Ship-to-shore transfer for the joint wargame core.

The transfer is deliberately handled at the campaign-stage boundary rather than
inside the turn loop: a ship anchored in the transfer area spawns landing craft
or amphibious vehicles, which then move to the beach and become land units.
This mirrors the real event and keeps the turn loop simple.

Synthetic demonstration model.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import hexmap
from .state import Unit


# A beach hex is a land hex adjacent to water in the map's campaign profile.
def is_beach(hmap, pos: int) -> bool:
    if hmap.cond(pos) == 4:      # water
        return False
    for nb in hmap.neighbors(pos):
        if nb >= 0 and hmap.cond(nb) == 4:
            return True
    return False


def find_transfer_area(hmap, beach_pos: int, max_rings: int = 6) -> Optional[int]:
    """Nearest water hex within ``max_rings`` of a beach hex."""
    from collections import deque
    q = deque([(beach_pos, 0)])
    seen = {beach_pos}
    while q:
        cur, d = q.popleft()
        if d > 0 and hmap.cond(cur) == 4:
            return cur
        if d >= max_rings:
            continue
        for nb in hmap.neighbors(cur):
            if nb >= 0 and nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    return None


def spawn_landing_craft(state, ship: Unit, name: str,
                        unit_type: str = "amphibious",
                        blood: int = 2) -> Unit:
    """Create a landing craft/amphibious vehicle next to ``ship``.

    The craft starts in the ship's hex (stacking is checked by the engine on
    movement), has a small weapon fit and is flagged ``beaching=True`` until it
    reaches land.
    """
    from . import weapons
    profile = weapons.weapons_for("步兵", 1, 2)
    uid = state.allocate_uid()
    craft = Unit(
        uid=uid, name=name, color=ship.color, op_type=2, sub_type=4,
        pos=ship.pos, blood=blood, max_blood=blood, armor=1,
        speed_kmh=12.0, value=5, unit_type=unit_type, domain="sea",
        weapons=profile, ammo={w.key: w.ammo for w in profile},
        altitude_band="水面", size_class="small", readiness=0.9,
    )
    craft.raw["beaching"] = True
    craft.raw["parent_ship"] = ship.uid
    state.units[uid] = craft
    state.events.append({
        "id": "EV-%06d" % state.next_event_id(),
        "tick": state.tick,
        "type": "beaching",
        "source": "engine",
        "level": "info",
        "actors": [ship.uid, craft.uid],
        "location": list(hexmap.decode(ship.pos)),
        "description": "%s 释放 %s" % (ship.name, name),
        "raw": {"parent_ship": ship.uid, "craft": craft.uid},
    })
    return craft


def beached(state, craft: Unit) -> bool:
    """True once the craft has reached a land (beach) hex."""
    return craft.domain == "sea" and state.map.cond(craft.pos) != 4


def convert_to_land(state, craft: Unit, name: Optional[str] = None) -> Unit:
    """Turn an amphibious craft into a land unit in place."""
    craft.domain = "land"
    craft.unit_type = "vehicle"
    craft.raw["beaching"] = False
    craft.altitude_band = "ground"
    if name:
        craft.name = name
    return craft
