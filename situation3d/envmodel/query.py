"""Time-bound spatial query facade for ``msgeo.envpack.v3``.

The facade deliberately does not reimplement terrain/road/LOS geometry: it
wraps the existing ``EnvPackEnvironment`` and adds the P0 contract semantics
(envelopes, quality, time binding, batch queries, explicit obstacles).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

from .gates import evaluate_gates
from .manifest import V3Manifest
from .obstacles import BridgeSet, ObstacleSet
from .policy import gate_overrides, validate_policy

T = TypeVar("T")
Coord = Tuple[float, float]


@dataclass(frozen=True)
class Envelope(Generic[T]):
    value: Optional[T]
    quality: str
    source_id: Optional[str] = None
    confidence: str = "unknown"
    as_of: float = 0.0


class SpatialQueryV3:
    """Spatial query view bound to one simulation time."""

    def __init__(self, env: Any, manifest: V3Manifest, time_s: float = 0.0,
                 policy: Optional[Dict[str, Any]] = None):
        self.env = env
        self.manifest = manifest
        self.time_s = float(time_s)
        if policy:
            policy = validate_policy(policy, require_approved=True)
        self.policy = dict(policy or {})
        self._obstacles = ObstacleSet(env, manifest)
        self._bridges = BridgeSet(env, manifest)

    def at(self, time_s: float) -> "SpatialQueryV3":
        return SpatialQueryV3(self.env, self.manifest, time_s, self.policy)

    def capabilities(self) -> set:
        return {
            "elevation", "landcover", "water", "building", "road",
            "query_batch", "nav_graph", "cost_surface", "obstacles",
            "road_graph", "weather", "sea_state", "tile", "los",
            "cover", "passability",
        }

    # -- helpers -----------------------------------------------------------
    def _meta(self, name: str) -> Dict[str, Any]:
        return self.manifest.layer(name)

    def _quality(self, name: str, value: Any) -> str:
        meta = self._meta(name)
        if value is None:
            return "unknown" if meta else "not_applicable"
        explicit = meta.get("quality")
        if explicit:
            return str(explicit)
        confidence = str(meta.get("confidence") or "unknown")
        return "measured" if confidence in ("high", "medium") else "derived"

    def _wrap(self, name: str, value: Any, quality: Optional[str] = None) -> Envelope:
        meta = self._meta(name)
        return Envelope(
            value=value,
            quality=quality or self._quality(name, value),
            source_id=meta.get("source"),
            confidence=str(meta.get("confidence") or "unknown"),
            as_of=self.time_s,
        )

    # -- point queries -----------------------------------------------------
    def elevation(self, lon: float, lat: float) -> Envelope:
        return self._wrap("terrain", float(self.env.elevation(lon, lat)))

    def landcover(self, lon: float, lat: float) -> Envelope:
        return self._wrap("landcover", self.env.landcover(lon, lat))

    def _is_water(self, lon: float, lat: float) -> bool:
        """Water membership from the pack polygons, not the legacy fallback."""
        polygon_at = getattr(self.env, "_polygon_at", None)
        index = getattr(self.env, "_water_index", None)
        if polygon_at is not None and index is not None:
            return polygon_at(index, lon, lat) is not None
        return bool(self.env.is_water(lon, lat))

    def water(self, lon: float, lat: float) -> Envelope:
        depth_kind = str(self._meta("hydrography").get("depth_kind") or "unknown")
        is_water = self._is_water(lon, lat)
        depth = (
            None
            if depth_kind == "unknown" or not is_water
            else self.env.water_depth(lon, lat)
        )
        # The legacy loader falls back to a 20 m default when bathymetry has
        # no negative (water) value at this point. Treat that as unknown.
        bathy = getattr(self.env, "_bathymetry", None)
        if depth is not None and bathy is not None:
            sound = self.env._grid_value(bathy, lon, lat)
            if sound is None or float(sound) >= 0.0:
                depth = None
        if depth is None:
            quality = "unknown" if is_water else "not_applicable"
            return self._wrap(
                "hydrography",
                {"is_water": is_water, "depth_m": None},
                quality=quality,
            )
        quality = "measured" if depth_kind in ("surveyed", "charted") else "derived"
        return self._wrap(
            "hydrography",
            {"is_water": is_water, "depth_m": float(depth)},
            quality=quality,
        )

    def building(self, lon: float, lat: float) -> Envelope:
        item = self._obstacles.at(lon, lat)
        if item is None:
            return self._wrap("buildings", None, quality="not_applicable")
        height_kind = str(self._meta("buildings").get("height_kind") or "unknown")
        source = str(item.get("height_source") or "")
        quality = {
            "tag": "measured",
            "levels": "derived",
            "fallback": "derived",
            "default": "default",
        }.get(source)
        if quality is None:
            quality = {
                "measured": "measured",
                "derived": "derived",
                "default": "default",
            }.get(height_kind, "unknown")
        return self._wrap("buildings", item, quality=quality)

    def road(self, lon: float, lat: float) -> Envelope:
        edge = self.env.road_edge_at(lon, lat)
        return self._wrap("roads", edge, quality=None if edge else "not_applicable")

    def soil(self, lon: float, lat: float) -> Envelope:
        return self._wrap("soil", None, quality="unknown")

    def airfield(self, airfield_id: str) -> Envelope:
        meta = self._meta("airfields")
        return self._wrap("airfields", None, quality="unknown" if meta else "not_applicable")

    # -- batch -------------------------------------------------------------
    def query_batch(self, points: Sequence[Coord], kind: str) -> List[Envelope]:
        dispatch = {
            "elevation": self.elevation,
            "landcover": self.landcover,
            "water": self.water,
            "building": self.building,
            "road": self.road,
            "soil": self.soil,
            "weather": self.weather,
        }
        if kind not in dispatch:
            raise ValueError("unsupported batch kind %r" % kind)
        fn = dispatch[kind]
        return [fn(float(lon), float(lat)) for lon, lat in points]

    # -- derived / structural ---------------------------------------------
    def nav_graph(self, profile: Any, bbox: Any = None, lod: int = 0) -> Envelope:
        from mobility.nav_field import NavField
        nav = NavField(self.env, res_m=30.0)
        return self._wrap("terrain", nav, quality="derived")

    def cost_surface(self, profile: Any, bbox: Any = None) -> Envelope:
        from mobility.nav_field import NavField
        nav = NavField(self.env, res_m=30.0)
        mask, speeds = nav.profile_mask(profile)
        return self._wrap(
            "terrain",
            {"mask": mask, "speeds": speeds, "nav_field": nav},
            quality="derived",
        )

    def obstacles(self, bbox: Any = None) -> Envelope:
        return self._wrap("obstacles", self._obstacles, quality="derived")

    def road_graph(self, bbox: Any = None) -> Envelope:
        nodes, edges = self.env.road_graph()
        return self._wrap(
            "roads", {"nodes": nodes, "edges": edges}, quality="measured"
        )

    def weather(self, lon: float, lat: float) -> Envelope:
        weather = self.env.weather(lon, lat)
        return self._wrap(
            "weather",
            {
                "visibility_m": weather.visibility_m,
                "wind_ms": weather.wind_ms,
                "sea_state": weather.sea_state,
            },
            quality="derived",
        )

    def sea_state(self, time_s: Optional[float] = None) -> Envelope:
        weather = self.env.weather(0.0, 0.0)
        return self._wrap("sea_state", weather.sea_state, quality="derived")

    def tile(self, tile_id: str, lod: int = 0) -> Envelope:
        meta = self._meta("terrain")
        path = meta.get("path")
        if not path:
            return self._wrap("terrain", None, quality="unknown")
        return self._wrap(
            "terrain", {"tile_id": tile_id, "path": path}, quality="derived"
        )

    def los(self, a: Coord, ha: float, b: Coord, hb: float,
            medium: str = "eo") -> Envelope:
        visible = bool(self.env.line_of_sight(a, ha, b, hb))
        building_blocked = bool(self._obstacles.blocked(a, ha, b, hb))
        return self._wrap(
            "obstacles",
            {
                "visible": visible and not building_blocked,
                "path_visible": visible,
                "building_blocked": building_blocked,
                "first_blocker": "building" if building_blocked else None,
                "blocker_kind": "building" if building_blocked else None,
                "clearance_m": None,
                "medium": medium,
            },
            quality="derived",
        )

    def cover(self, lon: float, lat: float, heading_deg: float = 0.0) -> Envelope:
        return self._wrap("landcover", self.env.cover_height(lon, lat), quality="derived")

    def _gate_policy(self) -> Dict[str, str]:
        gates = self.manifest.data.get("gates") or {}
        policy = {}
        for key, value in gates.items():
            if isinstance(value, dict):
                policy[str(key)] = str(value.get("unknown_policy") or "deny")
            else:
                policy[str(key)] = str(value)
        policy.update(gate_overrides(self.policy))
        return policy

    def passability(self, profile: Any, lon: float, lat: float,
                    altitude_m: Optional[float] = None) -> Envelope:
        depth_kind = str(self._meta("hydrography").get("depth_kind") or "unknown")
        result = evaluate_gates(
            profile, lon, lat, self.env, depth_kind, self._gate_policy(),
            is_water=self._is_water(lon, lat),
            on_bridge=self._bridges.at(lon, lat) is not None,
            altitude_m=altitude_m,
        )
        quality = "unknown" if result.unknown else "derived"
        return self._wrap(
            "terrain",
            {
                "allowed": result.allowed,
                "reason_codes": list(result.reason_codes),
                "gate_ids": list(result.gate_ids),
                "unknown": result.unknown,
                "cost": 1.0 if result.allowed else 0.0,
            },
            quality=quality,
        )
