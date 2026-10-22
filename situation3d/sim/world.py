"""World container and fixed-step loop (P3/P4).

Order of operations per step (plan §4.10):
    timers -> orders/planning -> motion -> rule-cell stacking -> sensors ->
    engagement -> objectives -> logistics -> time advance.

Rules are reached only through ``WorldRuleContext``; ``rules/`` never imports
``sim``/``env``/``geo``.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

from geo.geodesy import haversine_m, initial_bearing_deg
from geo.rule_grid import RuleGrid, cube_distance
from mobility.passability import passable
from mobility.planner import GridPlanner
from mobility.profile import PlatformProfile, profile_for
from rules import combat, engagement
from sim.context import WorldRuleContext
from sim.entity import Entity, Pose
from sim.motion import entity_profile, integrate
from sim.collision import CollisionIndex, obbs_overlap
from sim import spotting as spotting_model
from sim.tactics import TacticalController


def _is_ground(entity) -> bool:
    return getattr(entity, "domain", "land") in ("land", "amphibious")


class World(object):
    def __init__(self, env, rule_grid: RuleGrid, dt_s: float = 1.0,
                 seed: int = 20261021, plan_grid=None,
                 detection_model: str = "table",
                 direct_fire_model: str = "physics",
                 indirect_fire_model: str = "physics",
                 enable_radio_share: bool = False,
                 enable_tactics: bool = False,
                 commander=None,
                 commander_interval_s: float = 2.0,
                 nav=None):
        self.env = env
        self.rule_grid = rule_grid
        self.dt_s = float(dt_s)
        self.time_s = 0.0
        self.seed = int(seed)
        self.entities: Dict[int, Entity] = {}
        self.rng = random.Random(seed)
        # Plan on the source map grid (200 m official, 1.5 km campaign) so A*
        # stays in the 10^4-10^5 node range; rule distances still use RuleGrid.
        self.plan_grid = plan_grid or rule_grid
        self.planner = GridPlanner(env, self.plan_grid)
        # Fine navigation field (P0.1): single authority inside the pack.
        self.nav = nav
        # Traffic collision index (P0.2): physical footprints replace the
        # four-units-per-cell stop rule.
        self.traffic = CollisionIndex()
        self._crowd_warned: Dict[Tuple[int, int], float] = {}
        self._yield_streak: Dict[int, int] = {}
        # P1.6 behaviour intents (objective + pace/risk).
        self.intents: Dict[int, object] = {}
        # Physical perception state (P1.4).
        self.tracks: Dict[int, dict] = {}
        self._phys_last_seen: Dict[Tuple[int, int], float] = {}
        self.obscurants: List[dict] = []
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
        self.indirect_fire_model = str(indirect_fire_model)
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

    # -- routing ------------------------------------------------------------
    def assign_intent(self, entity: Entity, intent) -> bool:
        """Resolve an intent (objective cell / pace) into a planned route."""
        try:
            goal = intent.point(self.rule_grid)
        except (ValueError, TypeError, AttributeError, IndexError):
            return False
        if intent.cell is not None:
            # Deterministic small formation offset so several units sent to
            # one objective do not share the exact same point (~20 m).
            goal = (goal[0] + ((entity.uid % 5) - 2) * 0.0002,
                    goal[1] + (((entity.uid // 5) % 5) - 2) * 0.0002)
        route = self.plan_route(
            entity, goal,
            blocked=lambda lon, lat, ent=entity:
            self._obstacle_blocks(ent, lon, lat))
        if route is None:
            self.intents.pop(entity.uid, None)
            entity.pace_scale = 1.0
            return False
        entity.goal = (float(goal[0]), float(goal[1]))
        entity.route = route
        entity.route_s = 0.0
        entity.moving = True
        entity.pace_scale = float(intent.speed_scale())
        self.intents[entity.uid] = intent
        return True

    def plan_route(self, entity: Entity, goal, blocked=None):
        """Plan ``entity -> goal`` with the finest available authority.

        Ground/amphibious units inside an envpack use the navigation field
        (fine A* + coarse-route refinement); everything else stays on the
        base-map grid planner.
        """
        prof = entity_profile(entity)
        start = (entity.pose.lon, entity.pose.lat)
        if self.nav is not None and entity.domain in ("land", "amphibious"):
            # The nav field is authoritative for ground movement: a failed
            # fine plan must surface as "no route", never as a coarse route
            # that the pack geometry would block (the old G1 loop).
            return self.nav.plan_hybrid(self.planner, prof, start,
                                        (float(goal[0]), float(goal[1])),
                                        blocked=blocked)
        return self.planner.plan(prof, start, goal, blocked=blocked)

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
                route = self.plan_route(
                    entity, entity.goal,
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
                entity.pace_scale = 1.0
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

    def apply_component_damage(self, uid: int, component: str,
                               fraction: float) -> float:
        """Degrade a subsystem (mobility/firepower/sensors); returns health."""
        e = self.entities[int(uid)]
        if not e.alive:
            return 0.0
        key = str(component)
        if key not in e.components:
            return 1.0
        e.components[key] = max(0.0, float(e.components[key])
                                - max(0.0, float(fraction)))
        return float(e.components[key])

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
            health = 1.0
            if observer.sensors:
                try:
                    health = float((observer.components or {}).get(
                        "sensors", 1.0))
                except (TypeError, ValueError):
                    health = 1.0
                health = 0.0 if health <= 0.05 else max(0.2, health)
                km = max(float(getattr(s, "range_km", 0.0))
                         for s in observer.sensors) * health
                max_cells = max(max_cells,
                                int(km * 1000.0 / self.rule_grid.cell_m) + 1)
                max_cells = int(max_cells * health)
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
        if model == "physical":
            from sim import perception
            seen = perception.compute_seen(self)
        elif model == "spotting":
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

    def tracks_for(self, side: int) -> List[dict]:
        """Fused target tracks for one side (physical model)."""
        out = []
        for track in self.tracks.get(int(side), {}).values():
            item = dict(track)
            item["sensors"] = sorted(track.get("sensors") or ())
            out.append(item)
        out.sort(key=lambda t: t["uid"])
        return out

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
                    event = self._resolve_indirect(
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
                if self.direct_fire_model == "physics":
                    from rules import physics_direct_fire as _phys
                    direct_ok = _phys.can_engage(
                        ctx.view(attacker), ctx.view(target), ctx)
                else:
                    direct_ok = combat.select_weapon(
                        ctx.view(attacker), ctx.view(target), ctx) is not None
                if not direct_ok and self._has_indirect(attacker):
                    event = self._resolve_indirect(
                        ctx.view(attacker), ctx.rule_cell(ctx.view(target)), ctx,
                        observed=self.detected_by(attacker.affiliation,
                                                  target.uid))
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

    # -- indirect fire (P1.5: physics default, table compatibility) ---------
    def _resolve_indirect(self, attacker_view, aim_cell, ctx,
                          observed: bool = False):
        if self.indirect_fire_model == "physics":
            event = self._resolve_indirect_physics(
                attacker_view, aim_cell, ctx, observed)
            if event is not None:
                return event
            # Profiled guns that are out of range / not ready must not fall
            # back to the scatter table (that would bypass physical limits).
            from rules import physics_indirect
            if any(physics_indirect.has_profile(w)
                   for w in getattr(attacker_view, "weapons", []) or []):
                return None
        from rules import indirect_fire
        event = indirect_fire.resolve_indirect_fire(
            attacker_view, aim_cell, ctx, observed=observed)
        if event is not None:
            event.setdefault("model", "table")
        return event

    def _resolve_indirect_physics(self, attacker_view, aim_cell, ctx,
                                  observed: bool):
        from rules import physics_indirect
        attacker = self.entities.get(int(attacker_view.uid))
        if attacker is None or not attacker.alive:
            return None
        aim_lon, aim_lat = ctx.rule_cell_center(tuple(aim_cell))
        distance = haversine_m((attacker.pose.lon, attacker.pose.lat),
                               (aim_lon, aim_lat))
        picked = physics_indirect.select_weapon(
            attacker_view, distance,
            ready=lambda w: self.weapon_ready(attacker.uid, w.key))
        if picked is None:
            return None
        weapon, profile = picked
        impact = physics_indirect.solve_impact(
            profile, distance, observed, ctx.rng)
        if impact is None:
            return None
        bearing = initial_bearing_deg(
            (attacker.pose.lon, attacker.pose.lat), (aim_lon, aim_lat))
        br = math.radians(bearing)
        east = (impact["along_m"] * math.sin(br)
                + impact["across_m"] * math.cos(br))
        north = (impact["along_m"] * math.cos(br)
                 - impact["across_m"] * math.sin(br))
        kx = 111320.0 * math.cos(math.radians(aim_lat))
        lon = aim_lon + east / kx
        lat = aim_lat + north / 110540.0
        radius = float(profile.get("blast_radius_m", 50.0) or 50.0)
        targets = []
        total = 0
        for target in self.sorted_entities():
            if (not target.alive or target.on_board
                    or target.affiliation == attacker.affiliation):
                continue
            if not _is_ground(target):
                continue
            d = haversine_m((lon, lat), (target.pose.lon, target.pose.lat))
            if d > radius:
                continue
            kind = "infantry" if target.unit_type == "infantry" else "vehicle"
            damage = physics_indirect.blast_damage(
                profile, d, kind, target.armor)
            entry = {"uid": target.uid, "distance_m": round(d, 1),
                     "damage": damage, "hit": True}
            if damage > 0:
                total += damage
                self.apply_damage(target.uid, damage)
                entry["target_blood"] = target.blood
                if target.alive:
                    if kind == "infantry":
                        self.apply_suppression(target.uid, 45.0)
                    else:
                        comp = ctx.rng.choice(("mobility", "firepower",
                                               "sensors"))
                        frac = physics_indirect.component_fraction(profile,
                                                                   damage)
                        entry["component"] = comp
                        entry["component_health"] = round(
                            self.apply_component_damage(target.uid, comp, frac),
                            3)
            targets.append(entry)
        ctx.consume_ammo(attacker.uid, weapon.key, 1)
        ctx.mark_fired(attacker.uid, weapon.key,
                       getattr(weapon, "cooldown_s", 75.0))
        impact_cell = self.rule_grid.cell_of(lon, lat)
        scatter = ctx.rule_distance_cells(tuple(aim_cell), impact_cell)
        return {
            "type": "indirect_fire",
            "model": "physics",
            "actors": [attacker.uid] + [t["uid"] for t in targets
                                        if t["damage"] > 0],
            "weapon": weapon.key,
            "aim_cell": [int(aim_cell[0]), int(aim_cell[1])],
            "impact_cell": [int(impact_cell[0]), int(impact_cell[1])],
            "outcome": "hit" if scatter == 0 else "scatter",
            "scatter_cells": int(scatter),
            "att_level": None,
            "distance_m": round(distance, 1),
            "tof_s": round(impact["tof_s"], 2),
            "cep_m": round(impact["cep_m"], 1),
            "sigma_m": round(impact["sigma_m"], 1),
            "observed": bool(observed),
            "impact_geo": {"lon": round(lon, 7), "lat": round(lat, 7)},
            "blast_radius_m": radius,
            "targets": targets,
            "damage": total,
            "location_geo": {"lon": lon, "lat": lat},
        }

    def _indirect_reachable_physics(self, attacker,
                                    distance_m: float) -> bool:
        if self.indirect_fire_model != "physics":
            return False
        from rules import physics_indirect
        return physics_indirect.select_weapon(
            attacker, distance_m,
            ready=lambda w: self.weapon_ready(attacker.uid, w.key)) is not None

    def _best_target(self, attacker: Entity, ctx: WorldRuleContext) -> Optional[Entity]:
        best = None
        best_key = None
        candidates = []
        contacts = set(self.seen.get(attacker.uid, ()))
        if self._has_indirect(attacker):
            # Indirect fire can use the side's fused picture, not only the
            # gunner's own sensors (forward observers, battery FDC).
            for uid, seen in self.seen.items():
                observer = self.entities.get(uid)
                if (observer is not None
                        and observer.affiliation == attacker.affiliation):
                    contacts |= seen
        for target_uid in sorted(contacts):
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
                        rule_dist = ctx.rule_distance(ctx.view(attacker),
                                                      ctx.view(target))
                        if not (self._indirect_reachable_physics(
                                    attacker, distance_m)
                                or (self._has_indirect(attacker)
                                    and self._indirect_can_reach(
                                        attacker, rule_dist))):
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
                        distance_m = ctx.distance_m(ctx.view(attacker),
                                                    ctx.view(target))
                        reach = self._indirect_reachable_physics(
                            attacker, distance_m)
                        if not reach:
                            dist_cells = ctx.rule_distance(
                                ctx.view(attacker), ctx.view(target))
                            reach = (self._has_indirect(attacker)
                                     and self._indirect_can_reach(
                                         attacker, dist_cells))
                        if not reach:
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
        # P0.2: cell density is no longer a passability rule; separation is
        # handled by OBB traffic checks.  Obstacles still hard-block.
        return not self._obstacle_blocks(entity, lon, lat)

    # -- traffic collisions (P0.2) -----------------------------------------
    def _rebuild_traffic(self) -> None:
        self.traffic.rebuild(e for e in self.sorted_entities()
                             if e.alive and not e.on_board and _is_ground(e))

    def _traffic_clear(self, entity: Entity, lon: float, lat: float,
                       heading: float) -> bool:
        prof = entity_profile(entity)
        for other in self.traffic.near(lon, lat):
            if other.uid == entity.uid or not other.alive or other.on_board:
                continue
            if not _is_ground(other):
                continue
            oprof = entity_profile(other)
            if not obbs_overlap(prof, lon, lat, heading,
                                oprof, other.pose.lon, other.pose.lat,
                                other.pose.heading_deg):
                continue
            if obbs_overlap(prof, entity.pose.lon, entity.pose.lat,
                            entity.pose.heading_deg, oprof,
                            other.pose.lon, other.pose.lat,
                            other.pose.heading_deg):
                # Pairs already overlapping may separate, but only when the
                # candidate pose actually increases their separation.
                d_new = haversine_m((lon, lat),
                                    (other.pose.lon, other.pose.lat))
                d_cur = haversine_m((entity.pose.lon, entity.pose.lat),
                                    (other.pose.lon, other.pose.lat))
                if d_new > d_cur + 0.05:
                    continue
            return False
        return True

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
                    if landing_point is not None:
                        lon, lat = landing_point
                    else:
                        # disembark 150 m inland (east) if passable
                        lon = carrier.pose.lon + 0.0015
                        lat = carrier.pose.lat
                    seat = self._find_seat(passenger, lon, lat)
                    if seat is None:
                        # No passable, collision-free seat yet: stay aboard
                        # and retry next tick.
                        continue
                    lon, lat = seat
                    passenger.on_board = False
                    passenger.carrier = None
                    carrier.passengers.remove(pid)
                    passenger.pose = Pose(lon, lat, self.env.elevation(lon, lat))
                    passenger.route = None
                    passenger.goal = None
                    passenger.route_s = 0.0
                    passenger.velocity_ms = 0.0
                    passenger.moving = False
                    if _is_ground(passenger):
                        self.traffic.insert(passenger)
                    self.emit("disembark", [pid, carrier.uid],
                              "%s 从 %s 下船" % (passenger.name, carrier.name))

    def _nearest_land(self, carrier: Entity, passenger: Entity):
        """Nearest passable land point within 2.2 km (demo beach geometry)."""
        import math
        from geo.geodesy import haversine_m
        for radius in (100.0, 200.0, 300.0, 500.0, 800.0,
                       1200.0, 1600.0, 2200.0):
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

    def _find_seat(self, passenger: Entity, lon: float, lat: float):
        """Nearest passable, collision-free seat within ~40 m (P0.2).

        Returns ``None`` when no seat is free yet (the unit stays aboard).
        """
        import math as _math
        prof = entity_profile(passenger)
        kx = 111320.0 * _math.cos(_math.radians(lat))
        candidates = [(lon, lat)]
        for radius in (12.0, 24.0, 36.0):
            for k in range(8):
                angle = _math.radians(45.0 * k)
                candidates.append((lon + radius * _math.cos(angle) / kx,
                                   lat + radius * _math.sin(angle) / 110540.0))
        for clon, clat in candidates:
            if self.nav is not None and self.nav.covers(clon, clat):
                if not self.nav.motion_point_ok(prof, clon, clat):
                    continue
            elif not passable(prof, clon, clat, self.env,
                              self.env.elevation(clon, clat)):
                continue
            clear = True
            for other in self.traffic.near(clon, clat):
                if (other.uid == passenger.uid or not other.alive
                        or other.on_board or not _is_ground(other)):
                    continue
                if obbs_overlap(prof, clon, clat, 0.0,
                                entity_profile(other), other.pose.lon,
                                other.pose.lat, other.pose.heading_deg,
                                margin=0.5):
                    clear = False
                    break
            if clear:
                return clon, clat
        return None

    # -- stacking -----------------------------------------------------------
    def _enforce_stacking(self) -> None:
        """Soft crowding warning only (P0.2).

        Physical OBB collisions (``_traffic_clear``) produce queueing; the
        old four-per-cell stop is gone.  Keep a thinned warning for
        rule-cell density so stacking still shows up in replays.
        """
        grouped: Dict[Tuple[int, int], List[Entity]] = {}
        for e in self.sorted_entities():
            if not e.alive or e.on_board or not _is_ground(e):
                continue
            grouped.setdefault(self.rule_cell(e), []).append(e)
        for cell in sorted(grouped):
            members = grouped[cell]
            if len(members) <= 4:
                continue
            if self.time_s - self._crowd_warned.get(cell, -1e9) < 300.0:
                continue
            self._crowd_warned[cell] = self.time_s
            self.emit("crowding", [e.uid for e in members],
                      "规则格 %s 内地面单位 %d 个（软告警）"
                      % (cell, len(members)))

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
        self._rebuild_traffic()
        for e in self.sorted_entities():
            if e.alive and not e.on_board:
                ground = _is_ground(e)
                reason = integrate(
                    e, dt, self.env,
                    cell_check=lambda lon, lat, ent=e: self._can_enter_cell(ent, lon, lat),
                    nav=self.nav,
                    traffic_check=(lambda lon, lat, hd, ent=e:
                                   self._traffic_clear(ent, lon, lat, hd))
                    if ground else None)
                if ground:
                    self.traffic.refresh(e)
                if reason == "yield":
                    streak = self._yield_streak.get(e.uid, 0) + 1
                    self._yield_streak[e.uid] = streak
                    if streak >= 30:
                        # Head-on / permanent blockage: give the goal up so
                        # the tactical layer can re-task instead of waiting
                        # forever.  P1 motion planning will add passing rules.
                        self._yield_streak[e.uid] = 0
                        e.route = None
                        e.goal = None
                        e.velocity_ms = 0.0
                        e.moving = False
                        e.pace_scale = 1.0
                        self.emit("traffic_stuck", [e.uid],
                                  "%s 连续让行超时，放弃当前目标" % e.name)
                elif reason == "blocked" and e.goal is not None:
                    self._yield_streak.pop(e.uid, None)
                    route = self.plan_route(
                        e, e.goal,
                        blocked=lambda lon, lat, ent=e:
                        self._obstacle_blocks(ent, lon, lat))
                    if route is not None:
                        e.route = route
                        e.route_s = 0.0
                    else:
                        e.goal = None
                else:
                    self._yield_streak.pop(e.uid, None)
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
