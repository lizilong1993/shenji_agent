"""RuleContext implementation over the continuous world (P4).

Rules only see this module's ``RuleEntityView`` projections; no ``geo`` or
``env`` types leak into ``rules/``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from env.service import LandCover
from geo.geodesy import haversine_m
from rules import detection
from rules.context import Cover, RuleContext, RuleEntityView
from rules.elevation import elevation_modifier
from sim.entity import Entity, EntityRuleView


_ELEV_DOC = None


def _elevation_doc():
    global _ELEV_DOC
    if _ELEV_DOC is None:
        from rules.elevation import load_elevation
        _ELEV_DOC = load_elevation()
    return _ELEV_DOC


def _cover(env, e: Entity) -> Cover:
    cover = env.landcover(e.pose.lon, e.pose.lat)
    if cover == LandCover.URBAN:
        return Cover.URBAN
    if cover == LandCover.FOREST:
        return Cover.FOREST
    return Cover.NONE


class WorldRuleContext(object):
    def __init__(self, world, rng: Optional[random.Random] = None):
        self.world = world
        self.env = world.env
        self.rule_grid = world.rule_grid
        self.now_s = world.time_s
        self.rng = rng or world.rng
        self.jam = 0.0
        self.direct_fire_model = getattr(world, "direct_fire_model", "crt")
        self.detection_model = getattr(world, "detection_model", "table")
        self._elev_doc = _elevation_doc()
        self._views: Dict[int, EntityRuleView] = {}
        self._grades: Dict[int, int] = {}

    # -- views -------------------------------------------------------------
    def view(self, entity: Entity) -> EntityRuleView:
        cached = self._views.get(entity.uid)
        if cached is None:
            cached = entity.to_rule_view(stacked=self.world.stacked(entity))
            self._views[entity.uid] = cached
        return cached

    def refresh(self):
        self.now_s = self.world.time_s
        self._views.clear()
        self._grades.clear()

    # -- spatial -----------------------------------------------------------
    def distance_m(self, a, b) -> float:
        ea, eb = self._entity(a), self._entity(b)
        return haversine_m((ea.pose.lon, ea.pose.lat), (eb.pose.lon, eb.pose.lat))

    def bearing_deg(self, a, b) -> float:
        from geo.geodesy import initial_bearing_deg
        ea, eb = self._entity(a), self._entity(b)
        return initial_bearing_deg((ea.pose.lon, ea.pose.lat),
                                   (eb.pose.lon, eb.pose.lat))

    def rule_distance(self, a, b) -> int:
        return self.world.rule_distance(self._entity(a), self._entity(b))

    def rule_cell(self, e) -> Tuple[int, int]:
        return self.world.rule_cell(self._entity(e))

    def elevation_delta_m(self, a, b) -> float:
        ea, eb = self._entity(a), self._entity(b)
        return self.env.elevation(ea.pose.lon, ea.pose.lat) - self.env.elevation(eb.pose.lon, eb.pose.lat)

    def elevation_grade(self, e) -> int:
        uid = getattr(e, "uid", e)
        if uid not in self._grades:
            ent = self._entity(e)
            self._grades[uid] = self.env.elevation_grade(ent.pose.lon, ent.pose.lat)
        return self._grades[uid]

    def ele_grade(self) -> float:
        return self.env.ele_grade

    def line_of_sight(self, a, b) -> bool:
        ea, eb = self._entity(a), self._entity(b)
        if ea.domain in ("land", "amphibious") and eb.domain in ("land", "amphibious"):
            ca = self.env.grid.cell_of(ea.pose.lon, ea.pose.lat)
            cb = self.env.grid.cell_of(eb.pose.lon, eb.pose.lat)
            return self.env.cell_line_of_sight(ca, cb)
        ha = ea.pose.alt_m + 2.0
        hb = eb.pose.alt_m + 1.5
        return self.env.line_of_sight((ea.pose.lon, ea.pose.lat), ha,
                                      (eb.pose.lon, eb.pose.lat), hb)

    def cover_at(self, e) -> Cover:
        return _cover(self.env, self._entity(e))

    def target_kind(self, e) -> str:
        ent = self._entity(e)
        if ent.unit_type == "infantry":
            return "infantry"
        if ent.domain == "air":
            return "air"
        return "vehicle"

    def stack_count(self, e) -> int:
        ent = self._entity(e)
        return max(1, int(ent.blood))

    def elevation_modifier(self, attacker, target) -> int:
        ga = self.elevation_grade(attacker)
        gb = self.elevation_grade(target)
        return elevation_modifier(self._elev_doc, ga >= gb, abs(ga - gb),
                                  self.rule_distance(attacker, target),
                                  self.target_kind(target))

    # -- state -------------------------------------------------------------
    def can_detect(self, observer, target) -> bool:
        if self.detection_model == "spotting":
            from sim import spotting
            obs = self._entity(observer)
            tgt = self._entity(target)
            if (obs.domain in ("land", "amphibious")
                    and tgt.domain in ("land", "amphibious")):
                return spotting.can_spot(
                    obs, tgt, self.env, self.now_s,
                    float(getattr(self.rule_grid, "cell_m", 200.0)),
                    line_of_sight=lambda a, b: self.line_of_sight(a, b))
        return detection.can_observe(self.view(self._entity(observer)),
                                     self.view(self._entity(target)), self)

    def sensor_detect(self, observer, target) -> bool:
        from sim import sensors
        obs = self._entity(observer)
        tgt = self._entity(target)
        if not getattr(obs, "sensors", None):
            return False
        return any(sensors.detects(s, obs, tgt, self.env) for s in obs.sensors)

    def entities(self, side: Optional[int] = None) -> List[RuleEntityView]:
        return [self.view(e) for e in self.world.entities.values()
                if e.alive and (side is None or e.affiliation == side)]

    def units_in_cell(self, cell) -> List[RuleEntityView]:
        return [self.view(e) for e in self.world.entities_in_cell(cell)]

    def side_detects(self, target) -> bool:
        return bool(self.world.detected_by(target.affiliation, target.uid))

    def apply_damage(self, uid: int, damage: int) -> None:
        self.world.apply_damage(uid, damage)

    def apply_suppression(self, uid: int, seconds: float) -> None:
        self.world.apply_suppression(uid, seconds)

    def side_jam(self, affiliation: int) -> float:
        return 0.0

    def weapon_ready(self, uid: int, key: str) -> bool:
        return self.world.weapon_ready(uid, key)

    def weapon_state(self, uid: int, key: str) -> str:
        return self.world.weapon_state(uid, key)

    def mark_fired(self, uid: int, key: str, cooldown_s: float) -> None:
        self.world.mark_fired(uid, key, cooldown_s)

    def consume_ammo(self, uid: int, key: str, n: int = 1) -> bool:
        return self.world.consume_ammo(uid, key, n)

    def _entity(self, view) -> Entity:
        uid = getattr(view, "uid", None)
        return self.world.entities[int(uid)]
