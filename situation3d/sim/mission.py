"""Declarative missions and contact rules for campaign scenarios (D1).

A mission is a *task*, not a track: units, an objective, a pace, and optional
predecessors.  The runner turns a mission into engine intents/orders; it never
moves entities itself and never writes positions.  Contact handling implements
the rule-of-engagement part that the movement layer deliberately lacks:
ground units stop when an enemy is detected inside the engagement radius and
resume the task after the area has been quiet for a while.

Mission schema::

    {"id": "M1", "side": 0, "units": [2001, 2002],
     "task": "seize",              # move_to | seize | hold
     "objective": "D区岸海衔接",    # optional name ...
     "lon": 120.1625, "lat": 22.9775,  # ... or explicit point
     "after": ["M0"], "start_s": 0,
     "pace": "normal", "risk": "normal", "arrive_m": 250}

The exporter calls ``MissionRunner.update()`` after each ``World.step``.  The
runner is deterministic: all iteration order is sorted by uid / mission id.
"""

from __future__ import annotations

import math

from geo.geodesy import haversine_m
from sim.intent import Intent
from sim.collision import obbs_overlap
from sim.motion import entity_profile


class MissionRunner(object):
    def __init__(self, world, missions, roe=None, interval_s=30.0):
        self.world = world
        self.interval_s = float(interval_s)
        self.roe = dict(roe or {})
        self.halt_on_contact = bool(self.roe.get("halt_on_contact", True))
        self.engage_m = float(self.roe.get("engage_m", 2500.0))
        self.resume_after_s = float(self.roe.get("resume_after_s", 60.0))
        self.missions = {}
        for raw in missions or []:
            mid = str(raw.get("id") or ("M%d" % (len(self.missions) + 1)))
            self.missions[mid] = {
                "id": mid,
                "side": int(raw.get("side", 0)),
                "units": [int(u) for u in (raw.get("units") or [])],
                "task": str(raw.get("task") or "move_to"),
                "objective": str(raw.get("objective") or ""),
                "lon": (float(raw["lon"]) if raw.get("lon") is not None else None),
                "lat": (float(raw["lat"]) if raw.get("lat") is not None else None),
                "after": [str(x) for x in (raw.get("after") or [])],
                "start_s": float(raw.get("start_s") or 0.0),
                "pace": str(raw.get("pace") or "normal"),
                "risk": str(raw.get("risk") or "normal"),
                "arrive_m": float(raw.get("arrive_m") or 250.0),
                "status": "pending",
                "cell": None,
            }
        self.engaged = {}
        self._next_s = 0.0

    # -- public -------------------------------------------------------------

    def update(self, force=False):
        now = float(self.world.time_s)
        if not force and now + 1e-9 < self._next_s:
            return
        self._next_s = now + max(1.0, self.interval_s)
        self._activate(now)
        self._contacts(now)
        self._maintain(now)
        self._complete(now)

    def status(self):
        return [dict(self.missions[mid]) for mid in sorted(self.missions)]

    # -- activation ---------------------------------------------------------

    def _activate(self, now):
        for mid in sorted(self.missions):
            mission = self.missions[mid]
            if mission["status"] != "pending":
                continue
            if mission["start_s"] > now + 1e-9:
                continue
            deps = mission["after"]
            if deps and not all(self.missions.get(d, {}).get("status") == "complete"
                                for d in deps):
                continue
            mission["cell"] = self._objective_cell(mission)
            if mission["cell"] is None and mission["lon"] is None:
                mission["status"] = "failed"
                self.world.emit("mission_failed", mission["units"],
                                "%s 缺少目标点" % mid)
                continue
            mission["status"] = "active"
            self.world.emit("mission_start", mission["units"],
                            "%s 开始：%s" % (mid, mission["objective"] or mission["task"]))
            for uid in mission["units"]:
                self._assign(mission, uid)

    def _objective_cell(self, mission):
        grid = self.world.rule_grid
        if mission["lon"] is not None and mission["lat"] is not None:
            return grid.cell_of(mission["lon"], mission["lat"])
        name = mission["objective"]
        if not name:
            return None
        best = None
        for cell, obj in sorted(self.world.objectives.items()):
            if str(obj.get("name") or "") != name:
                continue
            lon, lat = self.world.rule_grid.cell_center_lonlat(*cell)
            distance = haversine_m((lon, lat), (mission["lon"] or lon,
                                                mission["lat"] or lat))
            if best is None or distance < best[0]:
                best = (distance, cell)
        return best[1] if best else None

    def _assign(self, mission, uid):
        entity = self.world.entities.get(int(uid))
        if entity is None or not entity.alive or entity.on_board:
            return False
        if uid in self.engaged:
            return False
        if entity.route is not None or entity.goal is not None:
            return False
        if mission["lon"] is not None and mission["lat"] is not None:
            # Prefer the exact passable point over the 200 m rule-cell centre:
            # a cell centre can sit inside a building and trap the unit there.
            lon, lat = self._passable_point(mission, uid)
            intent = Intent(lon=lon, lat=lat,
                            pace=mission["pace"], risk=mission["risk"],
                            objective=mission["objective"])
        elif mission["cell"] is not None:
            intent = Intent(cell=mission["cell"], pace=mission["pace"],
                            risk=mission["risk"], objective=mission["objective"])
        else:
            return False
        return bool(self.world.assign_intent(entity, intent))

    def _passable_point(self, mission, uid):
        """Objective point plus a small formation offset that is actually free.

        Rule-cell centres and blind offsets can land inside a building; walking
        the candidate ring keeps the stop point on open ground without moving
        the objective cell used for capture.
        """
        entity = self.world.entities.get(int(uid))
        base_lon, base_lat = mission["lon"], mission["lat"]
        candidates = [(0.0, 0.0)]
        for radius in (0.00018, 0.00036):
            for k in range(8):
                ang = math.radians(45.0 * k)
                candidates.append((radius * math.cos(ang), radius * math.sin(ang)))
        for dx, dy in candidates:
            lon, lat = base_lon + dx, base_lat + dy
            if entity is None or self._point_free(entity, lon, lat):
                return lon, lat
        return base_lon, base_lat

    def _point_free(self, entity, lon, lat):
        try:
            if self.world.nav is not None and self.world.nav.covers(lon, lat):
                if not self.world.nav.motion_point_ok(entity_profile(entity), lon, lat):
                    return False
        except Exception:  # noqa: BLE001 - nav is an optimisation, not a gate
            pass
        if self.world.env.building_at(lon, lat) is not None:
            return False
        depth = self.world.env.water_depth(lon, lat)
        if depth is not None and entity.domain == "land":
            limit = getattr(entity.profile, "wading_depth_m", 1.2)
            if depth > limit:
                return False
        # Strict convoy case: stay on the road network.
        if (getattr(entity.profile, "road_bound", False)
                and not self.world.env.on_road(lon, lat, 22.0)):
            return False
        # Do not teleport a stopping unit into another vehicle.
        try:
            prof = entity_profile(entity)
            # Use the live entity poses, not the traffic index: stop orders can
            # move several units in one update before the index is refreshed.
            for other in self.world.sorted_entities():
                if other.uid == entity.uid or other.on_board:
                    continue
                if other.domain not in ("land", "amphibious"):
                    continue
                if obbs_overlap(prof, lon, lat, entity.pose.heading_deg,
                                entity_profile(other), other.pose.lon,
                                other.pose.lat, other.pose.heading_deg):
                    return False
        except Exception:  # noqa: BLE001 - traffic index is best-effort
            pass
        return True

    # -- contact rules ------------------------------------------------------

    def _contacts(self, now):
        if not self.halt_on_contact:
            return
        for mid in sorted(self.missions):
            mission = self.missions[mid]
            if mission["status"] != "active":
                continue
            for uid in sorted(mission["units"]):
                entity = self.world.entities.get(uid)
                if entity is None or not entity.alive or entity.on_board:
                    continue
                if entity.domain not in ("land", "amphibious"):
                    continue
                threat = self._enemy_near(uid)
                if threat:
                    if uid not in self.engaged:
                        self._stop_legal(entity)
                        self.world.order(uid, {"type": "stop"}, source="mission")
                    self.engaged[uid] = now
                elif uid in self.engaged and now - self.engaged[uid] >= self.resume_after_s:
                    del self.engaged[uid]
                    self._assign(mission, uid)

    def _stop_legal(self, entity):
        """Pull over to a free spot before halting.

        A 30 m navigation cell can hide a small building footprint, so a unit
        ordered to stop mid-route may end up standing inside a house.  Look for
        a nearby open point (road preferred) and move the pose there before the
        stop order; the move is bounded to 80 m and never crosses water.
        """
        if self._point_free(entity, entity.pose.lon, entity.pose.lat):
            return
        lon0, lat0 = entity.pose.lon, entity.pose.lat
        kx = 111320.0 * math.cos(math.radians(lat0))
        best = None
        for radius in (15.0, 30.0, 50.0, 80.0):
            for k in range(12):
                ang = math.radians(30.0 * k)
                lon = lon0 + radius * math.cos(ang) / kx
                lat = lat0 + radius * math.sin(ang) / 110540.0
                if not self._point_free(entity, lon, lat):
                    continue
                try:
                    on_road = self.world.env.on_road(lon, lat, 25.0)
                except AttributeError:
                    on_road = True
                score = radius + (0.0 if on_road else 30.0)
                if best is None or score < best[0]:
                    best = (score, lon, lat)
            if best is not None:
                break
        if best is not None:
            entity.pose.lon, entity.pose.lat = best[1], best[2]
            entity.pose.alt_m = self.world.env.elevation(best[1], best[2])

    def _enemy_near(self, uid):
        entity = self.world.entities.get(uid)
        if entity is None:
            return None
        contacts = self.world.seen.get(uid) or set()
        best = None
        for other_uid in sorted(contacts):
            other = self.world.entities.get(other_uid)
            if other is None or not other.alive or other.affiliation == entity.affiliation:
                continue
            distance = haversine_m((entity.pose.lon, entity.pose.lat),
                                   (other.pose.lon, other.pose.lat))
            if distance > self.engage_m:
                continue
            if best is None or distance < best[0]:
                best = (distance, other)
        return best[1] if best else None

    # -- maintenance / completion ------------------------------------------

    def _maintain(self, now):
        for mid in sorted(self.missions):
            mission = self.missions[mid]
            if mission["status"] != "active":
                continue
            for uid in sorted(mission["units"]):
                entity = self.world.entities.get(uid)
                if entity is None or not entity.alive or entity.on_board:
                    continue
                if self._at_objective(mission, entity):
                    continue
                self._assign(mission, uid)

    def _at_objective(self, mission, entity):
        # Prefer the explicit passable point and its configured radius; the
        # 200 m rule cell is only the fallback for name-only missions.
        if mission["lon"] is not None and mission["lat"] is not None:
            return haversine_m((entity.pose.lon, entity.pose.lat),
                               (mission["lon"], mission["lat"])) <= mission["arrive_m"]
        if mission["cell"] is not None:
            return self.world.rule_cell(entity) == mission["cell"]
        return False

    def _complete(self, now):
        for mid in sorted(self.missions):
            mission = self.missions[mid]
            if mission["status"] != "active":
                continue
            alive = [uid for uid in mission["units"]
                     if (self.world.entities.get(uid) is not None
                         and self.world.entities[uid].alive)]
            if not alive:
                mission["status"] = "failed"
                self.world.emit("mission_failed", mission["units"], "%s 全部损失" % mid)
                continue
            task = mission["task"]
            if task == "hold":
                continue
            if task == "seize":
                obj = self.world.objectives.get(mission["cell"]) if mission["cell"] else None
                if obj is not None and obj.get("owner") == mission["side"]:
                    mission["status"] = "complete"
                    self.world.emit("mission_complete", mission["units"],
                                    "%s 完成：%s" % (mid, mission["objective"]))
                continue
            if all(self._at_objective(mission, self.world.entities[uid])
                   for uid in alive):
                mission["status"] = "complete"
                self.world.emit("mission_complete", mission["units"],
                                "%s 完成：%s" % (mid, mission["objective"] or task))
