"""State model for the clean-room land wargame core."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import hexmap, weapons
from .los import LineOfSight


@dataclass
class Unit:
    uid: int
    name: str
    color: int                  # 0 red, 1 blue (public contract)
    op_type: int
    sub_type: int
    pos: int
    blood: int
    max_blood: int
    armor: int = 0
    speed_kmh: float = 30.0
    value: int = 0
    unit_type: str = "vehicle"  # infantry | vehicle | artillery | aa | helicopter | uav | fort
    domain: str = "land"        # land | air | sea | sub
    weapons: List[weapons.WeaponProfile] = field(default_factory=list)
    ammo: Dict[str, int] = field(default_factory=dict)
    observe_distance: Dict[str, int] = field(default_factory=dict)
    alive: bool = True
    suppressed: bool = False
    suppressed_ticks: int = 0
    concealed: bool = False
    moving: bool = False
    marching: bool = False
    stacked: bool = False
    path: List[int] = field(default_factory=list)
    path_index: int = 0
    # Fractional progress towards the next hex, carried across ticks.
    move_progress: float = 0.0
    move_state: int = 0
    raw: dict = field(default_factory=dict)

    # -- joint-domain extensions -------------------------------------------
    altitude_band: str = "ground"   # ground | 水面 | 超低空 | 低空 | 中空 | 高空
    depth: str = "surface"          # surface | periscope | deep (submarines)
    heading: float = 0.0
    size_class: str = "medium"      # small | medium | large
    readiness: float = 0.85
    fuel: float = 1.0
    emitting: bool = True
    sensors: List[object] = field(default_factory=list)
    sea_air_weapons: List[object] = field(default_factory=list)
    strength: float = 100.0

    def to_dict(self) -> dict:
        return {
            "id": self.uid,
            "name": self.name,
            "color": self.color,
            "type": self.op_type,
            "sub_type": self.sub_type,
            "unit_type": self.unit_type,
            "domain": self.domain,
            "pos": self.pos,
            "row": hexmap.decode(self.pos)[0],
            "col": hexmap.decode(self.pos)[1],
            "blood": self.blood,
            "max_blood": self.max_blood,
            "armor": self.armor,
            "alive": self.alive,
            "suppressed": self.suppressed and self.alive,
            "concealed": self.concealed,
            "moving": self.moving,
            "marching": self.marching,
            "domain": self.domain,
            "altitude_band": self.altitude_band,
            "depth": self.depth,
            "readiness": round(self.readiness, 2),
            "fuel": round(self.fuel, 2),
            "strength": round(self.strength, 1),
            "ammo": dict(self.ammo),
        }


@dataclass
class Objective:
    coord: int
    value: int
    flag: int = -1              # -1 neutral, 0 red, 1 blue
    name: str = ""

    def to_dict(self) -> dict:
        return {
            "coord": self.coord,
            "row": hexmap.decode(self.coord)[0],
            "col": hexmap.decode(self.coord)[1],
            "value": self.value,
            "owner": "red" if self.flag == 0 else "blue" if self.flag == 1 else "neutral",
            "name": self.name,
        }


# A captured objective must be held this many ticks before ownership flips.
CONTROL_HOLD_TICKS = 5


class GameState(object):
    def __init__(self, hmap: hexmap.HexMap, units: Dict[int, Unit],
                 objectives: List[Objective], seed: int = 20261019,
                 tick_seconds: float = 30.0, max_time: int = 1800,
                 scenario_id: Optional[int] = None):
        self.map = hmap
        self.units = units
        self.objectives = objectives
        self.seed = seed
        self.rng = random.Random(seed)
        self.tick = 0
        self.tick_seconds = tick_seconds
        self.max_time = max_time
        self.scenario_id = scenario_id
        self.events: List[dict] = []
        self._event_seq = 0
        self.los = LineOfSight(hmap)
        self.winner: Optional[str] = None
        self.reason: str = ""
        self.jam: float = 0.0
        self.stage_id: str = ""
        self.control: Dict[str, dict] = {}
        self.supply_points: List[dict] = []
        # (uid, goal) -> path or None.  Re-planning A* every tick is the
        # dominant cost once ships and aircraft are in the loop.
        self.path_cache: Dict[tuple, Optional[List[int]]] = {}

    # -- weapon cooldown -------------------------------------------------
    def weapon_ready_at(self, uid: int, key: str) -> int:
        return int((getattr(self, "cooldowns", {}) or {}).get((uid, key), 0))

    def mark_weapon_fired(self, uid: int, key: str, ticks: int) -> None:
        if not hasattr(self, "cooldowns"):
            self.cooldowns = {}
        self.cooldowns[(uid, key)] = self.tick + max(0, int(ticks))

    def allocate_uid(self) -> int:
        # Keep new IDs clear of public scenario IDs.
        self._uid_seq = max(getattr(self, "_uid_seq", 900000),
                            max(self.units.keys() or [900000])) + 1
        return self._uid_seq

    def next_event_id(self) -> int:
        self._event_seq += 1
        return self._event_seq

    def living(self, color: Optional[int] = None) -> List[Unit]:
        out = [u for u in self.units.values() if u.alive]
        if color is None:
            return out
        return [u for u in out if u.color == color]

    def unit_at(self, pos: int, color: Optional[int] = None) -> List[Unit]:
        out = [u for u in self.living() if u.pos == pos]
        if color is not None:
            out = [u for u in out if u.color == color]
        return out

    def control_objectives(self) -> List[dict]:
        """Apply the public capture rule: occupy the hex, no enemy adjacent."""
        changes = []
        for obj in self.objectives:
            reds = self.unit_at(obj.coord, 0)
            blues = self.unit_at(obj.coord, 1)
            if not reds and not blues:
                continue
            enemy_adjacent = False
            for u in self.living():
                if hexmap.distance(u.pos, obj.coord) == 1:
                    if reds and u.color == 1:
                        enemy_adjacent = True
                        break
                    if blues and u.color == 0:
                        enemy_adjacent = True
                        break
            if enemy_adjacent:
                continue
            new_flag = 0 if reds and not blues else 1 if blues and not reds else obj.flag
            # Hysteresis: a flip has to be held for CONTROL_HOLD_TICKS before
            # it counts.  Without this, units drifting across the objective
            # hex flip ownership every tick (measured: 356 events in 12 h).
            if new_flag != obj.flag:
                streaks = getattr(self, "_control_streak", None)
                if streaks is None:
                    streaks = self._control_streak = {}
                key = obj.coord
                cand = streaks.get(key, (obj.flag, 0))
                if cand[0] != new_flag:
                    cand = (new_flag, 1)
                else:
                    cand = (new_flag, cand[1] + 1)
                streaks[key] = cand
                if cand[1] < CONTROL_HOLD_TICKS:
                    continue
            else:
                streaks = getattr(self, "_control_streak", None)
                if streaks is not None:
                    streaks.pop(obj.coord, None)
            if new_flag != obj.flag:
                old = obj.flag
                obj.flag = new_flag
                changes.append({
                    "objective": obj.to_dict(),
                    "from": "red" if old == 0 else "blue" if old == 1 else "neutral",
                    "to": "red" if new_flag == 0 else "blue" if new_flag == 1 else "neutral",
                })
        return changes

    def score(self) -> Dict[str, int]:
        red = sum(o.value for o in self.objectives if o.flag == 0)
        blue = sum(o.value for o in self.objectives if o.flag == 1)
        red += sum(u.value for u in self.living(0))
        blue += sum(u.value for u in self.living(1))
        return {"red": red, "blue": blue}

    def evaluate(self) -> Optional[str]:
        reds = self.living(0)
        blues = self.living(1)
        if not reds and not blues:
            self.winner, self.reason = "draw", "双方均无存活单位"
            return self.winner
        if not reds:
            self.winner, self.reason = "blue", "红方无存活单位"
            return self.winner
        if not blues:
            self.winner, self.reason = "red", "蓝方无存活单位"
            return self.winner
        if self.tick >= self.max_time:
            s = self.score()
            if s["red"] > s["blue"]:
                self.winner, self.reason = "red", "时间到，红方分值领先"
            elif s["blue"] > s["red"]:
                self.winner, self.reason = "blue", "时间到，蓝方分值领先"
            else:
                self.winner, self.reason = "draw", "时间到，双方分值相同"
            return self.winner
        return None

    def snapshot(self) -> dict:
        return {
            "tick": self.tick,
            "units": [u.to_dict() for u in self.units.values()],
            "objectives": [o.to_dict() for o in self.objectives],
            "score": self.score(),
            "winner": self.winner,
            "reason": self.reason,
        }
