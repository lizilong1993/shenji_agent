"""Minimal world container for the P2 skeleton.

Owns entities, the 200 m rule grid and the environment service.  Motion,
collision, sensors and adjudication are P3/P4; this class only provides the
bookkeeping that already has a stable contract (rule-cell occupancy, entity
registration, time base).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from geo.rule_grid import RuleGrid
from sim.entity import Entity


class World(object):
    def __init__(self, env, rule_grid: RuleGrid, dt_s: float = 1.0):
        self.env = env
        self.rule_grid = rule_grid
        self.dt_s = float(dt_s)
        self.time_s = 0.0
        self.entities: Dict[int, Entity] = {}
        self._events: List[dict] = []

    # -- entities -----------------------------------------------------------
    def add_entity(self, entity: Entity) -> Entity:
        if entity.uid in self.entities:
            raise ValueError("duplicate entity uid %s" % entity.uid)
        self.entities[entity.uid] = entity
        return entity

    def rule_cell(self, entity: Entity) -> Tuple[int, int]:
        return self.rule_grid.cell_of(entity.pose.lon, entity.pose.lat)

    def entities_in_cell(self, cell: Tuple[int, int],
                         side: Optional[int] = None) -> List[Entity]:
        out = []
        for e in self._sorted_entities():
            if not e.alive:
                continue
            if side is not None and e.affiliation != side:
                continue
            if self.rule_cell(e) == cell:
                out.append(e)
        return out

    def stacked(self, entity: Entity) -> bool:
        """Combat ``stacked`` flag: another ground unit shares the rule cell.

        Baseline ``scenario._mark_stacking`` counts every ground unit in the
        cell, regardless of side; the +2 attack modifier uses that flag.  The
        "at most 4 friendly ground units" movement limit is a different
        concept -- see :meth:`friendly_ground_count`.
        """
        if entity.domain != "land":
            return False
        cell = self.rule_cell(entity)
        for other in self.entities_in_cell(cell):
            if other.uid != entity.uid and other.domain == "land":
                return True
        return False

    def friendly_ground_count(self, entity: Entity) -> int:
        """Friendly ground units in the same rule cell (stacking limit ≤4)."""
        if entity.domain != "land":
            return 0
        return len(self.entities_in_cell(self.rule_cell(entity),
                                         side=entity.affiliation))

    def rule_view(self, entity: Entity):
        """Rule-layer view with the stacking flag computed for the current tick."""
        return entity.to_rule_view(stacked=self.stacked(entity))

    def rule_distance(self, a: Entity, b: Entity) -> int:
        return self.rule_grid.distance(self.rule_cell(a), self.rule_cell(b))

    def _sorted_entities(self) -> Iterable[Entity]:
        for uid in sorted(self.entities):
            yield self.entities[uid]

    # -- time ---------------------------------------------------------------
    def step(self) -> List[dict]:
        """Advance one fixed step.  Motion/adjudication arrive in P3/P4."""
        self.time_s += self.dt_s
        events = list(self._events)
        self._events = []
        return events

    def emit(self, event: dict) -> None:
        self._events.append(event)
