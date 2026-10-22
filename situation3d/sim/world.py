"""World container and fixed-step loop (P3/P4).

Order of operations per step (plan §4.10):
    timers -> orders/planning -> motion -> rule-cell stacking -> sensors ->
    engagement -> objectives -> logistics -> time advance.

Rules are reached only through ``WorldRuleContext``; ``rules/`` never imports
``sim``/``env``/``geo``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from geo.geodesy import haversine_m
from geo.rule_grid import RuleGrid, cube_distance
from mobility.passability import passable
from mobility.planner import GridPlanner
from mobility.profile import PlatformProfile, profile_for
from rules import combat, engagement
from sim.context import WorldRuleContext
from sim.entity import Entity, Pose
from sim.motion import integrate
from sim import spotting as spotting_model
from sim.tactics import TacticalController


def _is_ground(entity) -> bool:
    return getattr(entity, "domain", "land") in ("land", "amphibious")


class World(object):
    def __init__(self, env, rule_grid: RuleGrid, dt_s: float = 1.0,
                 seed: int = 20261021, plan_grid=None,
                 detection_model: str = "table",
                 direct_fire_model: str = "crt",
                 enable_radio_share: bool = False,
                 enable_tactics: bool = False,
                 commander=None,
                 commander_interval_s: float = 2.0):
        self.env = env
        self.rule_grid = rule_grid
        self.dt_s = float(dt_s)
        self.time_s = 0.0
        self.entities: Dict[int, Entity] = {}
        self.rng = random.Random(seed)
        # Plan on the source map grid (200 m official, 1.5 km campaign) so A*
        # stays in the 10^4-10^5 node range; rule distances still use RuleGrid.
        self.plan_grid = plan_grid or rule_grid
        self.planner = GridPlanner(env, self.plan_grid)
        self.orders: Dict[int, dict] = {}
        self.cooldowns: Dict[Tuple[int, str], float] = {}
        self.weapon_states: Dict[Tuple[int, str], str] = {}
        self.seen: Dict[int, set] = {}
        self.objectives: Dict[Tuple[int, int], dict] = {}
        self.supply_points: List[Tuple[float, float]] = []
        self.minefields: List[dict] = []
        self.obstacles: List[dict] = []
        self._mine_hits: set = set()
        self.events: List[dict] = []
        self._event_id = 0
        self.intel: Dict[int, dict] = {}
        self._cell_cache: Dict[int, Tuple[int, int]] = {}
        self._detect_cache: Dict[Tuple[int, int], tuple] = {}
        self._los_cell_cache: Dict[tuple, bool] = {}
        self._scan_slice = 0
        self._stack_cache: Dict[int, bool] = {}
        self._friendly_cache: Dict[int, int] = {}
        self._count_cache: Dict[Tuple[Tuple[int, int], int], int] = {}
        self.winner: Optional[str] = None
        self.detection_model = str(detection_model)
        self.direct_fire_model = str(direct_fire_model)
        self.enable_radio_share = bool(enable_radio_share)
        self.detection_shadow = None
        self.tactics = TacticalController() if enable_tactics else None
        self.commander = None
        self.commander_interval_s = float(commander_interval_s)
        self._commander_next_s = 0.0
        self._order_source: Dict[int, str] = {}
        self._commander_control: Dict[int, float] = {}
        self._commander_owned: set = set()
        if commander is not None:
            self.set_commander(commander, commander_interval_s)

    # -- commander ----------------------------------------------------------
    def set_commander(self, model, interval_s: Optional[float] = None) -> None:
        from sim.commander import CommanderRegistry
        self.commander = CommanderRegistry(model, interval_s or self.commander_interval_s)
        self.commander_interval_s = self.commander.interval_s
        self._commander_next_s = self.time_s

    # -- entities -----------------------------------------------------------
    def add_entity(self, entity: Entity) -> Entity:
        if entity.uid in self.entities:
            raise ValueError("duplicate entity uid %s" % entity.uid)
        if entity.profile is None:
            entity.profile = profile_for(entity.unit_type, entity.domain)
        self.entities[entity.uid] = entity
        self._refresh_cell_cache()
        return entity

    def sorted_entities(self, side: Optional[int] = None) -> List[Entity]:
        out = []
        for uid in sorted(self.entities):
            e = self.entities[uid]
            if side is not None and e.affiliation != side:
                continue
            out.append(e)
        return out

    def rule_cell(self, entity: Entity) -> Tuple[int, int]:
        cell = self._cell_cache.get(entity.uid)
        if cell is None:
            cell = self.rule_grid.cell_of(entity.pose.lon, entity.pose.lat)
            self._cell_cache[entity.uid] = cell
        return cell

    def _refresh_cell_cache(self) -> None:
        self._cell_cache = {}
        counts: Dict[Tuple[Tuple[int, int], int], int] = {}
        any_counts: Dict[Tuple[int, int], int] = {}
        for e in self.sorted_entities():
            if not e.alive or e.on_board:
                continue
            cell = self.rule_grid.cell_of(e.pose.lon, e.pose.lat)
            self._cell_cache[e.uid] = cell
            if _is_ground(e):
                counts[(cell, e.affiliation)] = counts.get((cell, e.affiliation), 0) + 1
                any_counts[cell] = any_counts.get(cell, 0) + 1
        self._stack_cache = {}
        self._friendly_cache = {}
        self._count_cache = counts
        for e in self.sorted_entities():
            cell = self._cell_cache.get(e.uid)
            if cell is None:
                continue
            self._friendly_cache[e.uid] = counts.get((cell, e.affiliation), 0)
            self._stack_cache[e.uid] = _is_ground(e) and any_counts.get(cell, 0) > 1

    def entities_in_cell(self, cell: Tuple[int, int],
                         side: Optional[int] = None) -> List[Entity]:
        return [e for e in self.sorted_entities(side)
                if e.alive and not e.on_board and self.rule_cell(e) == cell]

    def stacked(self, entity: Entity) -> bool:
        if not _is_ground(entity):
            return False
        return self._stack_cache.get(entity.uid, False)

    def friendly_ground_count(self, entity: Entity) -> int:
        return self._friendly_cache.get(entity.uid, 0)

    def rule_distance(self, a: Entity, b: Entity) -> int:
        return self.rule_grid.distance(self.rule_cell(a), self.rule_cell(b))

    def rule_view(self, entity: Entity):
        return entity.to_rule_view(stacked=self.stacked(entity))

    # -- orders -------------------------------------------------------------
    def order(self, uid: int, action: dict, source: str = "manual") -> None:
        uid = int(uid)
        self.orders[uid] = dict(action)
        self._order_source[uid] = str(source)
        if source == "manual":
            self._commander_owned.discard(uid)
            self._commander_control.pop(uid, None)

    def _apply_orders(self) -> None:
        for uid in sorted(self.orders):
            entity = self.entities.get(uid)
            if entity is None or not entity.alive or entity.on_board:
                self.orders.pop(uid, None)
                self._order_source.pop(uid, None)
                continue
            action = self.orders[uid]
            kind = action.get("type")
            if kind == "move_to":
                goal = (float(action["lon"]), float(action["lat"]))
                if (entity.route is not None and entity.goal is not None
                        and haversine_m(entity.goal, goal) < 1.0):
                    self.orders.pop(uid, None)
                    self._order_source.pop(uid, None)
                    continue
                entity.goal = goal
                route = self.planner.plan(
                    entity.profile, (entity.pose.lon, entity.pose.lat),
                    entity.goal,
                    blocked=lambda lon, lat, ent=entity:
                    self._obstacle_blocks(ent, lon, lat))
                if route is not None:
                    entity.route = route
                    entity.route_s = 0.0
                    entity.moving = True
                else:
                    entity.goal = None
                    self.emit("move_failed", [uid], "no route to objective")
                self.orders.pop(uid, None)
                self._order_source.pop(uid, None)
            elif kind == "stop":
                entity.route = None
                entity.goal = None
                entity.velocity_ms = 0.0
                entity.moving = False
                self.orders.pop(uid, None)
                self._order_source.pop(uid, None)
            elif kind == "fire":
                target = self.entities.get(int(action.get("target", -1)))
                if target is None or not target.alive:
                    self.orders.pop(uid, None)
                    self._order_source.pop(uid, None)
            elif kind == "fire_indirect":
                lon = action.get("lon")
                lat = action.get("lat")
                if lon is not None and lat is not None:
                    entity.aim_cell = self.rule_grid.cell_of(float(lon), float(lat))
                self.orders.pop(uid, None)
                self._order_source.pop(uid, None)

    # -- timers -------------------------------------------------------------
    def _tick_timers(self, dt: float) -> None:
        for key in list(self.cooldowns):
            self.cooldowns[key] -= dt
            if self.cooldowns[key] <= 0:
                del self.cooldowns[key]
        for e in self.entities.values():
            if e.suppressed:
                e.suppressed_s -= dt
                if e.suppressed_s <= 0:
                    e.suppressed = False

    def weapon_ready(self, uid: int, key: str) -> bool:
        return self.cooldowns.get((int(uid), key), 0.0) <= 0.0

    def weapon_state(self, uid: int, key: str) -> str:
        if not self.weapon_ready(uid, key):
            return "reloading"
        return self.weapon_states.get((int(uid), key), "deployed")

    def mark_fired(self, uid: int, key: str, cooldown_s: float) -> None:
        self.cooldowns[(int(uid), key)] = float(cooldown_s)
        entity = self.entities.get(int(uid))
        if entity is not None:
            entity.last_fired_s = self.time_s

    def consume_ammo(self, uid: int, key: str, n: int = 1) -> bool:
        e = self.entities[int(uid)]
        have = int(e.ammo.get(key, 0))
        if have < n:
            return False
        e.ammo[key] = have - n
        return True

    # -- damage -------------------------------------------------------------
    def apply_damage(self, uid: int, damage: int) -> None:
        e = self.entities[int(uid)]
        if not e.alive:
            return
        e.blood = max(0, e.blood - int(damage))
        if e.blood <= 0:
            self._on_destroyed(e)

    def apply_suppression(self, uid: int, seconds: float) -> None:
        e = self.entities[int(uid)]
        if e.alive:
            e.suppressed = True
            e.suppressed_s = max(e.suppressed_s, float(seconds))

    # -- sensors ------------------------------------------------------------
    def _compute_table_seen(self, ctx: WorldRuleContext, full: bool = False) -> Dict[int, set]:
        for dead_uid in [uid for uid, e in self.entities.items() if not e.alive]:
            self.seen.pop(dead_uid, None)
        entities = [e for e in self.sorted_entities() if e.alive and not e.on_board]
        cells = self._cell_cache
        kind_of = {e.uid: ctx.view(e).unit_kind for e in entities}
        det_meta = {}
        for e in entities:
            view = ctx.view(e)
            det_meta[e.uid] = (int(ctx.cover_at(view)),
                               ctx.elevation_grade(view))
        table = __import__("rules.detection", fromlist=["observation_doc"]).observation_doc()["table"]
        # For large forces, scan observers round-robin (full picture every
        # ``slice_count`` seconds) so the per-tick sensor cost stays bounded.
        slice_count = 1 if full or len(entities) <= 50 else 5
        observers = entities
        if slice_count > 1:
            observers = [e for i, e in enumerate(entities)
                         if i % slice_count == self._scan_slice]
            self._scan_slice = (self._scan_slice + 1) % slice_count
        if slice_count == 1:
            self._scan_slice = 0
        result: Dict[int, set] = dict(self.seen)
        for uid, entity in self.entities.items():
            if not entity.alive or entity.on_board:
                result.pop(uid, None)
        for observer in observers:
            seen = set()
            result[observer.uid] = seen
            o_kind = kind_of[observer.uid]
            max_cells = int(table.get(o_kind, {}).get("vehicle", 0))
            if observer.sensors:
                km = max(float(getattr(s, "range_km", 0.0)) for s in observer.sensors)
                max_cells = max(max_cells, int(km * 1000.0 / self.rule_grid.cell_m) + 1)
            if max_cells <= 0:
                continue
            for target in entities:
                if target.uid == observer.uid or target.affiliation == observer.affiliation:
                    continue
                # Cubic-cell prefilter before terrain/cover work.
                co, ct = cells[observer.uid], cells[target.uid]
                distance = cube_distance(co, ct)
                if distance > max_cells:
                    continue
                land_pair = (observer.domain in ("land", "amphibious")
                             and target.domain in ("land", "amphibious"))
                if land_pair:
                    # Land observation depends on the cell pair, the target
                    # concealment AND sub-cell data: cover polygons and the
                    # continuous elevation grade.  Key on all of them so a
                    # unit crossing a forest edge inside one cell is not
                    # served a stale result.
                    key = (observer.uid, target.uid, co, ct,
                           bool(target.concealed),
                           det_meta[target.uid][0],
                           det_meta[observer.uid][1],
                           det_meta[target.uid][1])
                    visible = self._detect_cache.get(key)
                    if visible is None:
                        visible = ctx.can_detect(observer, target,
                                                 distance=distance)
                        if len(self._detect_cache) < 500000:
                            self._detect_cache[key] = visible
                else:
                    # Sea/air detection uses continuous positions and
                    # pointwise LOS; never reuse across ticks.
                    visible = ctx.can_detect(observer, target,
                                             distance=distance)
                if visible:
                    seen.add(target.uid)
        return result

    def _compute_spotting_seen(self, ctx: WorldRuleContext) -> Dict[int, set]:
        return spotting_model.compute_seen(self, ctx)

    def _update_sensors(self, ctx: WorldRuleContext) -> None:
        model = self.detection_model
        self.detection_shadow_error = None
        if model == "spotting":
            entities = [e for e in self.entities.values() if e.alive and not e.on_board]
            needs_table = any(e.domain not in ("land", "amphibious") for e in entities)
            table_seen = self._compute_table_seen(ctx) if needs_table else {}
            try:
                spotting_seen = self._compute_spotting_seen(ctx)
                seen = spotting_model.merge_seen(table_seen, spotting_seen, self)
            except Exception as exc:
                self.detection_shadow_error = str(exc)
                seen = table_seen
        elif model == "shadow":
            table_seen = self._compute_table_seen(ctx, full=True)
            try:
                spotting_seen = self._compute_spotting_seen(ctx)
            except Exception as exc:
                self.detection_shadow_error = str(exc)
                spotting_seen = table_seen
            self.detection_shadow = spotting_model.shadow_diff(table_seen, spotting_seen)
            seen = table_seen
        else:
            seen = self._compute_table_seen(ctx)
        if self.enable_radio_share or model == "spotting":
            seen = spotting_model.radio_share(self, seen)
        self.seen = seen

    def detected_by(self, side: int, target_uid: int) -> bool:
        for uid, seen in self.seen.items():
            if target_uid in seen and self.entities[uid].affiliation == side:
                return True
        return False

    # -- advisory intel -----------------------------------------------------
    def set_intel(self, side: int, data: dict) -> None:
        self.intel[int(side)] = dict(data)

    def intel_for(self, side: int) -> dict:
        return dict(self.intel.get(int(side), {}))

    # -- commander ----------------------------------------------------------
    def _commander_state(self, side: int) -> dict:
        from geo.geodesy import haversine_m, initial_bearing_deg
        objectives = []
        for cell, obj in sorted(self.objectives.items()):
            lon, lat = self.rule_grid.cell_center_lonlat(*cell)
            objectives.append({"lon": lon, "lat": lat,
                               "owner": obj.get("owner", -1)})
        units = []
        for entity in self.sorted_entities(side):
            if not entity.alive or entity.on_board:
                continue
            nearest = min(objectives, key=lambda o: haversine_m(
                (entity.pose.lon, entity.pose.lat), (o["lon"], o["lat"]))) if objectives else None
            units.append({
                "uid": entity.uid, "affiliation": entity.affiliation,
                "name": entity.name, "domain": entity.domain,
                "blood": entity.blood,
                "max_blood": entity.max_blood,
                "lon": entity.pose.lon, "lat": entity.pose.lat,
                "in_cover": self.env.landcover(entity.pose.lon, entity.pose.lat)
                            in ("forest", "urban"),
                "objective": nearest,
            })
        contacts = set()
        for uid, seen in self.seen.items():
            observer = self.entities.get(uid)
            if (observer is not None and observer.affiliation == side
                    and not observer.on_board):
                contacts.update(seen)
        enemies = []
        for uid in sorted(contacts):
            target = self.entities.get(uid)
            if target is None or not target.alive or target.affiliation == side:
                continue
            nearest_unit = min(units, key=lambda u: haversine_m(
                (target.pose.lon, target.pose.lat), (u["lon"], u["lat"]))) if units else None
            distance = haversine_m(
                (target.pose.lon, target.pose.lat),
                ((nearest_unit["lon"], nearest_unit["lat"]) if nearest_unit
                 else (target.pose.lon, target.pose.lat)))
            bearing = initial_bearing_deg(
                ((nearest_unit["lon"], nearest_unit["lat"]) if nearest_unit
                 else (target.pose.lon, target.pose.lat)),
                (target.pose.lon, target.pose.lat))
            enemies.append({"uid": uid, "domain": target.domain,
                            "distance_m": round(distance, 1),
                            "bearing": round(bearing, 1)})
        return {"side": side, "time_s": round(self.time_s, 3),
                "units": units, "enemies": enemies,
                "objectives": objectives,
                "intel": self.intel_for(side)}

    def _apply_commander_orders(self, ctx: WorldRuleContext) -> None:
        if self.commander is None or self.time_s < self._commander_next_s:
            return
        now = self.time_s
        for uid in list(self.commander.orders):
            if self.commander.active(uid, now) is None:
                if uid in self._commander_owned:
                    self.orders.pop(uid, None)
                    self._order_source.pop(uid, None)
                    entity = self.entities.get(uid)
                    if entity is not None:
                        entity.route = None
                        entity.goal = None
                        entity.velocity_ms = 0.0
                        entity.moving = False
                    self._commander_owned.discard(uid)
                    self._commander_control.pop(uid, None)
        self._commander_next_s = now + self.commander.interval_s
        for side in (0, 1):
            state = self._commander_state(side)
            for order in self.commander.decide(state, now):
                entity = self.entities.get(int(order.uid))
                if entity is None or not entity.alive or entity.on_board:
                    continue
                if entity.affiliation != side:
                    continue
                if self._order_source.get(entity.uid) == "manual":
                    continue
                self._commander_owned.add(entity.uid)
                self._commander_control[entity.uid] = now + max(
                    1.0, float(order.expires_s))
                if order.posture == "hold":
                    self.order(entity.uid, {"type": "stop"}, source="commander")
                elif order.lon is not None and order.lat is not None:
                    self.order(entity.uid, {"type": "move_to",
                                            "lon": float(order.lon),
                                            "lat": float(order.lat)},
                               source="commander")
                elif order.target_uid is not None:
                    target = self.entities.get(int(order.target_uid))
                    if target is not None and target.alive:
                        if ((entity.domain in ("land", "amphibious"))
                                != (target.domain in ("land", "amphibious"))):
                            continue
                        self.order(entity.uid, {"type": "move_to",
                                                "lon": target.pose.lon,
                                                "lat": target.pose.lat},
                                   source="commander")

    # -- engagement ---------------------------------------------------------
    def _engage(self, ctx: WorldRuleContext) -> None:
        ctx.refresh()
        for attacker in self.sorted_entities():
            if not attacker.alive or attacker.on_board:
                continue
            if not any(self.weapon_ready(attacker.uid, key) for key in attacker.ammo):
                continue
            order = self.orders.get(attacker.uid) or {}
            target = None
            if order.get("type") == "fire":
                candidate = self.entities.get(int(order.get("target", -1)))
                if candidate is not None and candidate.alive:
                    target = candidate
            if target is not None and not ctx.can_detect(ctx.view(attacker), ctx.view(target)):
                target = None
            if target is None:
                target = self._best_target(attacker, ctx)
            if target is None:
                aim = getattr(attacker, "aim_cell", None)
                if aim is not None:
                    from rules import indirect_fire as _indirect
                    event = _indirect.resolve_indirect_fire(
                        ctx.view(attacker), tuple(aim), ctx, observed=False)
                    attacker.aim_cell = None
                    if event is not None:
                        event["tick_s"] = round(self.time_s, 1)
                        self._append(event)
                continue
            if _is_ground(attacker) and _is_ground(target):
                if attacker.marching:
                    continue
                if attacker.velocity_ms > 0.1 and not any(
                        getattr(w, "shoot_on_move", False) for w in attacker.weapons):
                    continue
                direct = combat.select_weapon(ctx.view(attacker), ctx.view(target), ctx)
                if direct is None and self._has_indirect(attacker):
                    from rules import indirect_fire as _indirect
                    event = _indirect.resolve_indirect_fire(
                        ctx.view(attacker), ctx.rule_cell(ctx.view(target)), ctx,
                        observed=ctx.can_detect(ctx.view(attacker), ctx.view(target)))
                else:
                    event = combat.resolve_direct_fire(
                        ctx.view(attacker), ctx.view(target), ctx)
            else:
                event = engagement.resolve_sea_air(ctx.view(attacker), ctx.view(target), ctx)
            if event is not None:
                event["tick_s"] = round(self.time_s, 1)
                if event.get("type") == "indirect_fire":
                    event.setdefault("target_blood", None)
                else:
                    event["target_blood"] = target.blood
                event["location_geo"] = event.get("location_geo") or {
                    "lon": target.pose.lon, "lat": target.pose.lat}
                self._append(event)

    def _has_indirect(self, attacker: Entity) -> bool:
        return any(getattr(w, "kind", None) == "indirect"
                   for w in attacker.weapons)

    def _indirect_can_reach(self, attacker: Entity, distance: int) -> bool:
        for w in attacker.weapons:
            if getattr(w, "kind", None) != "indirect":
                continue
            if attacker.ammo.get(w.key, 0) <= 0:
                continue
            if not self.weapon_ready(attacker.uid, w.key):
                continue
            table = (w.tables or {}).get((0, 0))
            if table is not None and table.level_at(int(distance)) is not None:
                return True
        return False

    def _best_target(self, attacker: Entity, ctx: WorldRuleContext) -> Optional[Entity]:
        best = None
        best_key = None
        candidates = []
        for target_uid in sorted(self.seen.get(attacker.uid, ())):
            target = self.entities.get(target_uid)
            if target is None or not target.alive or target.on_board:
                continue
            if target.affiliation == attacker.affiliation:
                continue
            if _is_ground(attacker) and _is_ground(target):
                if self.direct_fire_model == "physics":
                    from rules import physics_direct_fire
                    distance_m = ctx.distance_m(ctx.view(attacker), ctx.view(target))
                    max_range_m = 0.0
                    for w in attacker.weapons:
                        profile = physics_direct_fire.profile_for_weapon(w)
                        if profile is not None:
                            max_range_m = max(max_range_m, physics_direct_fire._num(
                                profile.get("max_range_m"), 3000.0))
                    if max_range_m and distance_m > max_range_m:
                        continue
                    candidates.append((distance_m, -target.value, target))
                else:
                    dist = self.rule_grid.distance(self.rule_cell(attacker), self.rule_cell(target))
                    max_range = 0
                    for w in attacker.weapons:
                        for table in (w.tables or {}).values():
                            max_range = max(max_range, table.att_max)
                    if max_range and dist > max_range:
                        continue
                    candidates.append((dist, -target.value, target))
            else:
                candidates.append((0, -target.value, target))
        candidates.sort(key=lambda c: (c[0], c[1]))
        for _, _, target in candidates[:5]:
            if _is_ground(attacker) and _is_ground(target):
                if self.direct_fire_model == "physics":
                    from rules import physics_direct_fire
                    if not physics_direct_fire.can_engage(
                            ctx.view(attacker), ctx.view(target), ctx):
                        continue
                    event_key = (ctx.distance_m(ctx.view(attacker), ctx.view(target)),
                                 -target.value)
                else:
                    dist = ctx.rule_distance(ctx.view(attacker), ctx.view(target))
                    if (combat.select_weapon(ctx.view(attacker), ctx.view(target), ctx) is None
                            and not self._indirect_can_reach(attacker, dist)):
                        continue
                    event_key = (ctx.rule_distance(ctx.view(attacker), ctx.view(target)),
                                 -target.value)
            else:
                kinds = engagement.allowed_kinds(attacker.domain, target.domain)
                if not kinds:
                    continue
                d_km = ctx.distance_m(ctx.view(attacker), ctx.view(target)) / 1000.0
                in_range = any(w.kind in kinds and w.min_range_km <= d_km <= w.max_range_km
                               and attacker.ammo.get(w.key, 0) > 0
                               for w in (attacker.sea_air_weapons or []))
                if not in_range:
                    continue
                event_key = (d_km, -target.value)
            if best_key is None or event_key < best_key:
                best_key = event_key
                best = target
        return best

    def _on_destroyed(self, entity: Entity) -> None:
        if getattr(entity, "_destroyed", False):
            return
        entity._destroyed = True
        entity.alive = False
        entity.route = None
        entity.velocity_ms = 0.0
        self.emit("destroyed", [entity.uid], "%s 被摧毁" % entity.name)
        for pid in list(entity.passengers):
            passenger = self.entities.get(pid)
            if passenger is not None and passenger.alive:
                passenger.on_board = False
                passenger.carrier = None
                self.apply_damage(pid, passenger.blood + 1)
        entity.passengers = []

    @staticmethod
    def _point_in_ring(lon: float, lat: float, ring: List[List[float]]) -> bool:
        inside = False
        j = len(ring) - 1
        for i in range(len(ring)):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if (yi > lat) != (yj > lat) and lon < xi + (lat - yi) * (xj - xi) / (yj - yi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _inside_regions(lon: float, lat: float, regions) -> List[dict]:
        return [r for r in regions or []
                if r.get("ring") and World._point_in_ring(lon, lat, r["ring"])]

    def _obstacle_blocks(self, entity: Entity, lon: float, lat: float) -> bool:
        """Roadblocks stop all ground movement; fortifications stop vehicles."""
        if not _is_ground(entity) or not self.obstacles:
            return False
        for item in self._inside_regions(lon, lat, self.obstacles):
            kind = item.get("kind") or "roadblock"
            if kind == "fortification":
                if entity.unit_type != "infantry":
                    return True
            else:
                return True
        return False

    def _resolve_minefields(self) -> None:
        if not self.minefields:
            return
        from rules.minefield import tables as mine_tables
        for entity in self.sorted_entities():
            if not entity.alive or entity.on_board or not _is_ground(entity):
                continue
            kind = "infantry" if entity.unit_type == "infantry" else "vehicle"
            for field in self.minefields:
                ring = field.get("ring") or []
                if len(ring) < 3:
                    continue
                owner = field.get("owner", -1)
                if owner is not None and int(owner) == entity.affiliation:
                    continue
                key = (entity.uid, field.get("name") or "")
                if key in self._mine_hits:
                    continue
                if not self._point_in_ring(entity.pose.lon, entity.pose.lat, ring):
                    continue
                self._mine_hits.add(key)
                result = mine_tables().resolve(
                    kind, entity.armor, self.rng,
                    field_type=field.get("kind") or "临时雷场")
                event = {
                    "type": "mine_strike",
                    "actors": [entity.uid],
                    "field": field.get("name"),
                    "field_type": result["field_type"],
                    "roll": result["roll"],
                    "base_damage": result["base"],
                    "armor": result["armor"],
                    "rect_damage": result["rect"],
                    "damage": result["damage"],
                    "tick_s": round(self.time_s, 1),
                    "location_geo": {"lon": entity.pose.lon, "lat": entity.pose.lat},
                }
                if result["damage"] > 0:
                    self.apply_damage(entity.uid, result["damage"])
                self._append(event)

    def _can_enter_cell(self, entity: Entity, lon: float, lat: float) -> bool:
        if not _is_ground(entity):
            return True
        if self._obstacle_blocks(entity, lon, lat):
            return False
        cell = self.rule_grid.cell_of(lon, lat)
        if cell == self._cell_cache.get(entity.uid):
            return True
        return self._count_cache.get((cell, entity.affiliation), 0) < 4

    # -- embark / disembark -------------------------------------------------
    def _sync_passengers(self) -> None:
        for carrier in self.sorted_entities():
            if not carrier.alive or not carrier.passengers:
                continue
            depth = self.env.water_depth(carrier.pose.lon, carrier.pose.lat)
            can_land = depth is None or depth <= 1.2
            for pid in list(carrier.passengers):
                passenger = self.entities.get(pid)
                if passenger is None or not passenger.alive:
                    carrier.passengers.remove(pid)
                    continue
                if passenger.on_board:
                    passenger.pose = Pose(carrier.pose.lon, carrier.pose.lat,
                                          carrier.pose.alt_m)
                landing_point = None
                if not can_land and carrier.velocity_ms < 0.5:
                    landing_point = self._nearest_land(carrier, passenger)
                if (can_land or landing_point is not None) and carrier.velocity_ms < 0.5:
                    passenger.on_board = False
                    passenger.carrier = None
                    carrier.passengers.remove(pid)
                    if landing_point is not None:
                        lon, lat = landing_point
                    else:
                        # disembark 150 m inland (east) if passable
                        lon = carrier.pose.lon + 0.0015
                        lat = carrier.pose.lat
                    passenger.pose = Pose(lon, lat, self.env.elevation(lon, lat))
                    self.emit("disembark", [pid, carrier.uid],
                              "%s 从 %s 下船" % (passenger.name, carrier.name))

    def _nearest_land(self, carrier: Entity, passenger: Entity):
        """Nearest passable land point within 800 m (demo beach geometry)."""
        import math
        from geo.geodesy import haversine_m
        for radius in (100.0, 200.0, 300.0, 500.0, 800.0):
            for k in range(16):
                bearing = math.radians(k * 22.5)
                east = radius * math.sin(bearing)
                north = radius * math.cos(bearing)
                lat = carrier.pose.lat + north / 110540.0
                lon = carrier.pose.lon + east / (111320.0 * math.cos(math.radians(lat)))
                if self.env.water_depth(lon, lat) is not None:
                    continue
                if passable(passenger.profile, lon, lat, self.env,
                            self.env.elevation(lon, lat)):
                    return lon, lat
        return None

    # -- stacking -----------------------------------------------------------
    def _enforce_stacking(self) -> None:
        grouped: Dict[Tuple[Tuple[int, int], int], List[Entity]] = {}
        for e in self.sorted_entities():
            if not e.alive or e.domain != "land":
                continue
            grouped.setdefault((self.rule_cell(e), e.affiliation), []).append(e)
        for (cell, _), members in sorted(grouped.items(), key=lambda kv: str(kv[0])):
            for extra in members[4:]:
                extra.route = None
                extra.goal = None
                extra.velocity_ms = 0.0
                extra.moving = False
                self.emit("stack_blocked", [extra.uid],
                          "%s 因同格堆叠上限停在 %s" % (extra.name, cell))

    # -- objectives / logistics --------------------------------------------
    def add_objective(self, cell: Tuple[int, int], value: int = 0,
                      name: str = "", owner: int = -1) -> None:
        self.objectives[cell] = {"owner": int(owner), "value": value, "name": name}

    def _update_objectives(self) -> None:
        for cell, obj in self.objectives.items():
            owners = {e.affiliation for e in self.entities_in_cell(cell)
                      if e.alive and _is_ground(e)}
            if len(owners) != 1:
                continue
            owner = next(iter(owners))
            neighbours = set(self.rule_grid.neighbors(cell))
            enemy_near = any(e.alive and _is_ground(e) and e.affiliation != owner
                             and (self.rule_cell(e) == cell or self.rule_cell(e) in neighbours)
                             for e in self.entities.values())
            if not enemy_near and obj["owner"] != owner:
                obj["owner"] = owner
                self.emit("objective", [], "%s 被%s控制" % (obj["name"], "红方" if owner == 0 else "蓝方"))

    def _logistics(self, dt: float) -> None:
        for e in self.entities.values():
            if not e.alive:
                continue
            for p in self.supply_points:
                if haversine_m((e.pose.lon, e.pose.lat), p) <= 1000.0:
                    e.fuel = min(1.0, e.fuel + 0.02 * dt)
                    break

    # -- events -------------------------------------------------------------
    def _append(self, event: dict) -> None:
        self._event_id += 1
        event.setdefault("id", "EV-%06d" % self._event_id)
        event.setdefault("tick", int(self.time_s / self.dt_s))
        event["location_geo"] = event.get("location_geo") or None
        self.events.append(event)

    def emit(self, kind: str, actors: List[int], description: str) -> dict:
        event = {"type": kind, "actors": actors, "description": description,
                 "tick_s": round(self.time_s, 1)}
        self._append(event)
        return event

    # -- step ---------------------------------------------------------------
    def step(self, dt: Optional[float] = None) -> List[dict]:
        dt = self.dt_s if dt is None else float(dt)
        before = len(self.events)
        ctx = WorldRuleContext(self)
        self._tick_timers(dt)
        self._apply_orders()
        for e in self.sorted_entities():
            if e.alive and not e.on_board:
                reason = integrate(e, dt, self.env,
                                   cell_check=lambda lon, lat, ent=e: self._can_enter_cell(ent, lon, lat))
                if reason == "blocked" and e.goal is not None:
                    route = self.planner.plan(
                        e.profile, (e.pose.lon, e.pose.lat), e.goal,
                        blocked=lambda lon, lat, ent=e:
                        self._obstacle_blocks(ent, lon, lat))
                    if route is not None:
                        e.route = route
                        e.route_s = 0.0
                    else:
                        e.goal = None
        self._sync_passengers()
        self._refresh_cell_cache()
        self._resolve_minefields()
        self._enforce_stacking()
        self._refresh_cell_cache()
        ctx.refresh()
        self._update_sensors(ctx)
        if self.tactics is not None:
            self.tactics.update(self, ctx)
        if self.commander is not None:
            self._apply_commander_orders(ctx)
        self._engage(ctx)
        self._update_objectives()
        self._logistics(dt)
        self.time_s += dt
        return self.events[before:]
