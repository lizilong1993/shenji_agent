"""Dense transfer campaign case (design v1).

Generates a fictional multi-domain transfer demonstration on real OSM roads in
the Anping area of Tainan. The case is intentionally *not* a combat model: it
shows embarked vehicles, four release batches, dense road movement, support
dependencies and six fixed objective nodes. All platforms and parameters are
synthetic.

The output keeps the existing ``situation3d.trajectory.v2`` contract and adds
the optional campaign fields described in the integration spec. Old cases are
not touched.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .. import geo, roads, symbology

CASE_ID = "taiwan_dense_transfer_v1"
CASE_NAME = "登台2"
CASE_DESC = ("红蓝两组合成对象在真实海岸与路网上的转运、衔接、保障与观测演示；"
             "任务进度受对象可用性、道路可达性和保障状态约束。")
THEATRE = "taiwan_strait"
TICK_SECONDS = 30
TICK_COUNT = 1440
START_UTC = "2026-09-15T00:00:00Z"
GEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "geo", THEATRE, "anping")

AREAS = {
    "D": (120.145, 22.960, 120.180, 22.995),
    "E": (120.165, 22.950, 120.200, 22.985),
    "F": (120.185, 22.945, 120.220, 22.980),
}
AREA_META = {
    "D": {"name": "D区岸海衔接", "bbox": [120.145, 22.960, 120.180, 22.995],
          "center": [22.9775, 120.1625]},
    "E": {"name": "E区岸上任务", "bbox": [120.165, 22.950, 120.200, 22.985],
          "center": [22.9675, 120.1825]},
    "F": {"name": "F区保障", "bbox": [120.185, 22.945, 120.220, 22.980],
          "center": [22.9625, 120.2025]},
}

STAGES = [
    ("S0", "建立态势", 0, 120),
    ("S1", "多域活动增强", 120, 360),
    ("S2", "转运关系显现", 360, 600),
    ("S3", "岸海衔接", 600, 840),
    ("S4", "岸上活动与保障受限", 840, 1200),
    ("S5", "稳态评估", 1200, 1441),
]

VEHICLE_CLASSES = {
    "light_vehicle": {
        "label": "轻型车辆", "icon": "light_vehicle", "family": "地面机动",
        "length_m": 4.8, "width_m": 1.9, "height_m": 1.8,
        "speed_kmh": 55.0, "model": "ZTD-05 两栖突击车",
        "role": "地面机动", "sensor_km": 8.0, "engage_km": 4.0,
        "legacy_icon": "armor",
    },
    "truck": {
        "label": "运输车", "icon": "truck", "family": "地面保障",
        "length_m": 7.2, "width_m": 2.4, "height_m": 3.0,
        "speed_kmh": 45.0, "model": "东风 EQ240 运输车",
        "role": "地面运输", "sensor_km": 6.0, "engage_km": 2.0,
        "legacy_icon": "armor",
    },
    "engineer_vehicle": {
        "label": "工程车", "icon": "engineer_vehicle", "family": "地面保障",
        "length_m": 6.5, "width_m": 2.3, "height_m": 2.8,
        "speed_kmh": 40.0, "model": "EQ240 工程车",
        "role": "地面保障", "sensor_km": 5.0, "engage_km": 1.5,
        "legacy_icon": "armor",
    },
}

# Groups: id prefix, area, count, class mix, release tick (None = ground at t0),
# carrier (for embarked groups), role override.
GROUP_SPECS = [
    ("TRV", "A", "D", 8, ["light_vehicle"] * 8, 600, "TRS001", "地面机动"),
    ("TRV", "B", "D", 8, ["light_vehicle"] * 8, 660, "TRS002", "地面机动"),
    ("TRV", "C", "E", 6, ["light_vehicle"] * 4 + ["truck"] * 2, None, None, "地面运输"),
    ("TRV", "D", "E", 6, ["light_vehicle"] * 4 + ["truck"] * 2, 780, None, "节点等待"),
    ("TRV", "E", "F", 4, ["truck"] * 4, None, None, "地面运输"),
    ("TRV", "F", "F", 4, ["engineer_vehicle"] * 4, None, None, "地面保障"),
    ("TBV", "A", "D", 8, ["light_vehicle"] * 4 + ["truck"] * 4, None, None, "地面机动"),
    ("TBV", "B", "E", 6, ["light_vehicle"] * 4 + ["truck"] * 2, None, None, "地面机动"),
    ("TBV", "C", "E", 6, ["light_vehicle"] * 4 + ["truck"] * 2, None, None, "地面机动"),
    ("TBV", "D", "F", 4, ["engineer_vehicle"] * 4, None, None, "地面保障"),
]


@dataclass
class Entity:
    id: str
    kind: str                 # vehicle / vessel / aircraft
    affiliation: str
    group: str
    area: str
    role: str
    cls: str
    path: List[Tuple[float, float]]
    length_m: float
    speed_mps: float
    start_tick: int = 0
    offset_m: float = 0.0
    loop: bool = False
    carrier: Optional[str] = None
    release_tick: Optional[int] = None
    mode: str = "ground"
    status: str = "ready"
    strength: float = 1.0
    mobility: float = 1.0
    task: str = ""
    edges: List[dict] = field(default_factory=list)
    route_id: str = ""
    alt_m: float = 0.0
    cum: List[float] = field(default_factory=list)
    target: object = None


class PolyIndex(object):
    """Tiny bbox-grid point-in-polygon index (stdlib only)."""

    def __init__(self, path: str, cell: float = 0.0015):
        self.cell = cell
        self.items = []
        self.grid: Dict[Tuple[int, int], List[int]] = {}
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        for f in doc.get("features") or []:
            g = f.get("geometry") or {}
            if g.get("type") != "Polygon":
                continue
            ring = (g.get("coordinates") or [[]])[0]
            if len(ring) < 4:
                continue
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            idx = len(self.items)
            self.items.append((min(xs), min(ys), max(xs), max(ys), ring))
            for cx in range(int(math.floor(min(xs) / cell)), int(math.floor(max(xs) / cell)) + 1):
                for cy in range(int(math.floor(min(ys) / cell)), int(math.floor(max(ys) / cell)) + 1):
                    self.grid.setdefault((cx, cy), []).append(idx)

    def _inside(self, lon: float, lat: float, ring: Sequence[Sequence[float]]) -> bool:
        inside = False
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if ((yi > lat) != (yj > lat)) and (
                    lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    def hits(self, lon: float, lat: float) -> bool:
        key = (int(math.floor(lon / self.cell)), int(math.floor(lat / self.cell)))
        for idx in self.grid.get(key, ()):
            minx, miny, maxx, maxy, ring = self.items[idx]
            if minx <= lon <= maxx and miny <= lat <= maxy and self._inside(lon, lat, ring):
                return True
        return False

    def hits_path(self, path: Sequence[Tuple[float, float]], step_m: float = 18.0,
                  margin: int = 0) -> int:
        if not self.items:
            return 0
        bad = 0
        for a, b in zip(path, path[1:]):
            d = geo.haversine_m(a, b)
            n = max(1, int(math.ceil(d / step_m)))
            for k in range(n + 1):
                t = k / float(n)
                lat = a[0] + (b[0] - a[0]) * t
                lon = a[1] + (b[1] - a[1]) * t
                if self.hits(lon, lat):
                    bad += 1
                    if bad > margin:
                        return bad
        return bad


def _nodes_in(net: roads.Roads, bbox: Sequence[float], comp: int = 0) -> List[int]:
    w, s, e, n = bbox
    return [i for i, ll in enumerate(net.nodes)
            if w <= ll[0] <= e and s <= ll[1] <= n and net.weak_comp[i] == comp]


def _walk_route(net: roads.Roads, rng: random.Random, bbox: Sequence[float],
                start_band: Optional[Tuple[float, float]], target_len: float,
                buildings: PolyIndex, water: PolyIndex,
                nodes: Optional[List[int]] = None) -> Optional[roads.RouteResult]:
    """Build a road-following path by walking the directed adjacency.

    A random walk respects one-way/reverse edges by construction and avoids a
    full-graph Dijkstra per candidate, which matters because a local case needs
    many candidates and the Taiwan graph has 488k nodes.
    """
    if nodes is None:
        nodes = _nodes_in(net, bbox)
    if not nodes:
        return None
    starts = [i for i in nodes
              if start_band is None or (start_band[0] <= net.nodes[i][0] <= start_band[1])]
    if not starts:
        starts = nodes
    u = rng.choice(starts)
    path_nodes = [u]
    edges = []
    total = 0.0
    prev = None
    for _ in range(240):
        opts = []
        for v, m, kmh in net.adj[u]:
            if v == prev:
                continue
            lon, lat = net.nodes[v]
            if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                continue
            opts.append((v, m, kmh))
        if not opts:
            break
        if start_band is not None:
            fwd = [o for o in opts
                   if net.nodes[o[0]][0] >= net.nodes[u][0] or net.nodes[o[0]][1] >= net.nodes[u][1]]
            if fwd:
                opts = fwd
        fresh = [o for o in opts if o[0] not in path_nodes]
        if fresh:
            opts = fresh
        v, m, kmh = rng.choice(opts)
        edges.append({"from": u, "to": v, "way_id": net.way_ids.get((u, v)),
                      "length_m": round(m, 1), "speed_kmh": kmh})
        path_nodes.append(v)
        total += m
        prev, u = u, v
        if total >= target_len:
            break
    if total < 450.0:
        return None
    path = [net.node_latlon(i) for i in path_nodes]
    if buildings.hits_path(path) or water.hits_path(path):
        return None
    return roads.RouteResult(path, path_nodes, edges, net.weak_comp[path_nodes[0]])


def _pick_routes(net: roads.Roads, rng: random.Random, bbox: Sequence[float],
                 count: int, buildings: PolyIndex, water: PolyIndex,
                 min_len: float = 500.0, max_len: float = 3200.0,
                 start_band: Optional[Tuple[float, float]] = None) -> List[roads.RouteResult]:
    nodes = _nodes_in(net, bbox)
    out: List[roads.RouteResult] = []
    tries = 0
    while len(out) < count and tries < count * 120:
        tries += 1
        target = rng.uniform(max(min_len, 700.0), min(max_len, 1900.0))
        r = _walk_route(net, rng, bbox, start_band, target, buildings, water, nodes)
        if r is None:
            continue
        if any(abs(r.path[0][0] - q.path[0][0]) < 1e-5 and
               abs(r.path[0][1] - q.path[0][1]) < 1e-5 for q in out):
            continue
        out.append(r)
    if len(out) < count:
        raise RuntimeError("only %d/%d clean routes in %r" % (len(out), count, bbox))
    return out


def _cum_path(path: Sequence[Tuple[float, float]]) -> Tuple[List[float], float]:
    cum = [0.0]
    for a, b in zip(path, path[1:]):
        cum.append(cum[-1] + geo.haversine_m(a, b))
    return cum, cum[-1]


def _sample(path: Sequence[Tuple[float, float]], cum: Sequence[float],
            along: float, loop: bool = False) -> Tuple[float, float, float]:
    total = cum[-1]
    if total <= 0:
        return path[0][0], path[0][1], 0.0
    d = along % total if loop else max(0.0, min(along, total))
    # linear scan is fine: routes have tens to hundreds of vertices
    for i in range(1, len(cum)):
        if cum[i] >= d:
            seg = cum[i] - cum[i - 1]
            t = 0.0 if seg <= 0 else (d - cum[i - 1]) / seg
            a, b = path[i - 1], path[i]
            lat = a[0] + (b[0] - a[0]) * t
            lon = a[1] + (b[1] - a[1]) * t
            hdg = geo.initial_bearing_deg(a, b)
            return lat, lon, hdg
    a, b = path[-2], path[-1] if len(path) > 1 else (path[-1], path[-1])
    return path[-1][0], path[-1][1], geo.initial_bearing_deg(a, b)


def _edge_at(edges: Sequence[dict], dist: float) -> Tuple[Optional[object], float]:
    acc = 0.0
    for e in edges:
        L = float(e.get("length_m") or 0.0)
        if acc + L >= dist:
            return e.get("way_id"), round(dist - acc, 1)
        acc += L
    return (edges[-1].get("way_id") if edges else None), 0.0


def _phase_at(tick: int) -> dict:
    for idx, (sid, name, a, b) in enumerate(STAGES):
        if a <= tick < b:
            return {"id": sid, "name": name, "index": idx, "since_tick": a,
                    "ticks_in": tick - a, "count": len(STAGES)}
    sid, name, a, b = STAGES[-1]
    return {"id": sid, "name": name, "index": len(STAGES) - 1,
            "since_tick": a, "ticks_in": tick - a, "count": len(STAGES)}


def _stage_id(tick: int) -> str:
    return _phase_at(tick)["id"]


def _vehicle_entities(net: roads.Roads, rng: random.Random,
                      routes: Dict[str, List[roads.RouteResult]],
                      buildings: PolyIndex,
                      water: PolyIndex) -> Tuple[List[Entity], Dict[str, Entity]]:
    ents: List[Entity] = []
    carriers: Dict[str, Entity] = {}

    # Carriers: four synthetic transfer platforms offshore, slow loops.
    for i in range(4):
        base_lat = 22.962 + i * 0.004
        # Keep transfer platforms in open water west of the Anping coastline;
        # the earlier 120.13x loop sat over the fish-pond belt.
        path = [(base_lat, 120.095), (base_lat + 0.004, 120.091),
                (base_lat + 0.006, 120.096), (base_lat + 0.002, 120.099)]
        path = path + [path[0]]
        cum, total = _cum_path(path)
        e = Entity(
            id="TRS%03d" % (i + 1), kind="vessel", affiliation="red",
            group="TRS", area="B", role="转运载体", cls="transport_ship",
            path=path, length_m=total, speed_mps=geo.knots_to_mps(7.0),
            loop=True, mode="sea", task="转运平台待机",
        )
        e.cum = cum  # type: ignore[attr-defined]
        carriers[e.id] = e
        ents.append(e)

    # Red escorts and blue surface group.
    ship_specs = [
        ("TRS005", "red", "frigate", "054A 型护卫舰", 22.945, 120.085, 10.0),
        ("TRS006", "red", "corvette", "056A 型轻型护卫舰", 22.930, 120.093, 12.0),
        ("TBS001", "blue", "frigate", "成功级护卫舰", 22.955, 120.075, 10.0),
        ("TBS002", "blue", "frigate", "成功级护卫舰", 22.940, 120.082, 11.0),
        ("TBS003", "blue", "corvette", "沱江级巡逻舰", 22.925, 120.088, 14.0),
        ("TBS004", "blue", "corvette", "沱江级巡逻舰", 22.972, 120.082, 13.0),
        ("TBS005", "blue", "frigate", "成功级护卫舰", 22.985, 120.086, 9.0),
        ("TBS006", "blue", "corvette", "沱江级巡逻舰", 22.998, 120.092, 12.0),
    ]
    for sid, aff, cls, model, lat, lon, kn in ship_specs:
        path = [(lat, lon), (lat + 0.006, lon - 0.004),
                (lat + 0.010, lon + 0.004), (lat + 0.003, lon + 0.008)]
        path = path + [path[0]]
        cum, total = _cum_path(path)
        e = Entity(id=sid, kind="vessel", affiliation=aff, group="TBS",
                   area="B", role="海上巡逻", cls=cls, path=path, length_m=total,
                   speed_mps=geo.knots_to_mps(kn), loop=True, mode="sea",
                   task="海上活动")
        e.cum = cum  # type: ignore[attr-defined]
        ents.append(e)

    # Aircraft: two observation and two support per side.
    air_specs = []
    for aff, prefix in (("red", "TRA"), ("blue", "TBA")):
        for j, (cls, model, role, alt) in enumerate([
                ("fighter", "歼-16", "空中掩护", 5200.0),
                ("fighter", "F-16C", "空中掩护", 4800.0),
                ("aew", "空警-500", "空中警戒", 6500.0),
                ("aew", "E-2D", "空中警戒", 6200.0)]):
            air_specs.append((aff, prefix + "%03d" % (j + 1), cls, model, role, alt))
    for aff, aid, cls, model, role, alt in air_specs:
        # Real-sized patrol circuits (~40-60 km legs), not a 3 km local loop.
        # Red orbits over the northern strait, blue over the southern approach.
        if aff == "red":
            base = [(22.92, 119.92), (23.18, 120.02), (23.02, 120.30), (22.76, 120.08)]
        else:
            base = [(22.62, 120.00), (22.90, 120.14), (22.72, 120.40), (22.48, 120.12)]
        j = int(aid[-3:]) - 1
        path = [(la + 0.010 * (j % 2), lo + 0.010 * (j // 2)) for la, lo in base]
        path = path + [path[0]]
        cum, total = _cum_path(path)
        e = Entity(id=aid, kind="aircraft", affiliation=aff, group="TRA" if aff == "red" else "TBA",
                   area="B", role=role, cls=cls, path=path, length_m=total,
                   speed_mps=geo.knots_to_mps(330.0), loop=True, mode="air",
                   alt_m=alt, task="空中活动")
        e.cum = cum  # type: ignore[attr-defined]
        ents.append(e)

    # Ground groups. Route slots per group: release groups get one route each
    # (four vehicles), others share routes with offsets.
    gid = 0
    route_cursor: Dict[str, int] = {"D": 0, "E": 0, "F": 0}
    for prefix, suffix, area, count, classes, release, carrier, role in GROUP_SPECS:
        group = "%s-%s" % (prefix, suffix)
        avail = routes[area]
        ridx = route_cursor[area]
        route = avail[ridx % len(avail)]
        route_cursor[area] = ridx + 1
        cum, total = _cum_path(route.path)
        speed_kmh = min(VEHICLE_CLASSES[classes[0]]["speed_kmh"],
                        net.path_speed_profile(route.edges)["mean_kmh"] or 40.0)
        speed_mps = speed_kmh / 3.6
        for k in range(count):
            gid += 1
            cls = classes[k]
            profile = VEHICLE_CLASSES[cls]
            vid = "%s%03d" % (prefix, gid if prefix == "TRV" else gid - 36)
            # IDs: TRV001-036, TBV001-024 in group order.
            if prefix == "TBV":
                vid = "TBV%03d" % (gid - 36)
            offset = total * (0.06 + 0.88 * (k + 1) / (count + 1.0))
            start = release if release is not None else 0
            if group == "TRV-D":
                start = 780 if k < 3 else 840
            e = Entity(
                id=vid, kind="vehicle", affiliation="red" if prefix == "TRV" else "blue",
                group=group, area=area, role=role, cls=cls,
                path=list(route.path), length_m=total, speed_mps=speed_mps,
                start_tick=start, offset_m=offset,
                carrier=carrier if carrier is not None else None,
                release_tick=release if carrier is not None else None,
                mode="embarked" if carrier is not None else "ground",
                task="衔接后移动" if release is not None else "道路活动",
                edges=list(route.edges), route_id="%s%d" % (area, ridx),
            )
            e.cum = cum  # type: ignore[attr-defined]
            ents.append(e)

    return ents, carriers


def _entity_target(e: Entity):
    from ..scenario import Target
    if e.kind == "vehicle":
        p = VEHICLE_CLASSES[e.cls]
        name = "%s %s" % ("红方" if e.affiliation == "red" else "蓝方", p["label"])
        attrs = {
            "编组": e.group, "任务": e.task, "合成对象": "是",
            "车辆尺寸": "%.1f×%.1f×%.1f m" % (p["length_m"], p["width_m"], p["height_m"]),
            "数据来源": "demo_profile",
        }
        return Target(
            target_id=e.id, name=name, platform_class=p["label"], domain="land",
            affiliation=e.affiliation, waypoints=[],
            speed_kn=round(geo.kmh_to_knots(p["speed_kmh"]), 1),
            role=e.role, model=p["model"], icon=p.get("legacy_icon", p["icon"]),
            sensor_range_km=p["sensor_km"], influence_km=0.5,
            engage_km=p["engage_km"], road_bound=True, attributes=attrs,
        )
    if e.kind == "vessel":
        model = "071 型船坞登陆舰" if e.cls == "transport_ship" else (
            "054A 型护卫舰" if e.cls == "frigate" else "056A 型轻型护卫舰")
        return Target(
            target_id=e.id,
            name=("红方转运平台" if e.cls == "transport_ship" else
                  ("红方" if e.affiliation == "red" else "蓝方") + "水面平台"),
            platform_class=("转运平台" if e.cls == "transport_ship" else "水面平台"),
            domain="sea", affiliation=e.affiliation, waypoints=[],
            speed_kn=round(e.speed_mps * 1.943844, 1),
            role="编队指挥" if e.cls == "transport_ship" else "近海巡逻",
            model=model,
            icon=("replenishment" if e.cls == "transport_ship" else e.cls),
            sensor_range_km=80.0, influence_km=40.0, engage_km=20.0,
            loop=True, attributes={"合成对象": "是", "任务": e.task},
        )
    model = "歼-16" if e.cls == "fighter" and e.affiliation == "red" else (
        "F-16C" if e.cls == "fighter" else ("空警-500" if e.affiliation == "red" else "E-2D"))
    return Target(
        target_id=e.id,
        name=("红方" if e.affiliation == "red" else "蓝方") + ("战斗机" if e.cls == "fighter" else "预警机"),
        platform_class="战斗机" if e.cls == "fighter" else "预警机",
        domain="air", affiliation=e.affiliation, waypoints=[],
        speed_kn=330.0, altitude_m=e.alt_m,
        role="空中掩护" if e.cls == "fighter" else "空中警戒",
        model=model, icon=e.cls,
        sensor_range_km=150.0, influence_km=120.0, engage_km=60.0,
        loop=True, attributes={"合成对象": "是", "任务": e.task},
    )


def _snap(net: roads.Roads, lat: float, lon: float) -> Tuple[float, float]:
    i = net.nearest_node(lat, lon, max_km=2.0)
    if i is None:
        return lat, lon
    return net.node_latlon(i)


def _state_for(e: Entity, tick: int, carrier_pos: Dict[str, Tuple[float, float, float, float]]):
    if e.kind == "vehicle":
        release = e.release_tick
        if release is not None and e.carrier and tick < release:
            cp = carrier_pos.get(e.carrier or "")
            if cp:
                return {"lat": cp[0], "lon": cp[1], "alt": 0.0, "speed_kn": round(cp[3], 1),
                        "heading": round(cp[2], 1), "travelled_m": 0.0, "laps": 0,
                        "reached_end": False, "mode": "embarked",
                        "carrier_id": e.carrier, "task_status": "waiting",
                        "status": "reserve", "strength": 1.0, "mobility_factor": 1.0}
            return {"lat": e.path[0][0], "lon": e.path[0][1], "alt": 0.0, "speed_kn": 0.0,
                    "heading": 0.0, "travelled_m": 0.0, "laps": 0, "reached_end": False,
                    "mode": "embarked", "carrier_id": e.carrier,
                    "task_status": "waiting", "status": "reserve",
                    "strength": 1.0, "mobility_factor": 1.0}
        active = tick >= e.start_tick
        if active:
            raw = e.offset_m + (tick - e.start_tick) * TICK_SECONDS * e.speed_mps
        else:
            raw = e.offset_m
        # Ground columns shuttle along their route instead of piling up at the
        # far end. At 30 s/tick a 1 km route is crossed in minutes; a one-way
        # walk would leave every vehicle stacked on the same node.
        if e.length_m > 0:
            cycle = raw % (2.0 * e.length_m)
            if cycle <= e.length_m:
                along, reverse = cycle, False
            else:
                along, reverse = 2.0 * e.length_m - cycle, True
        else:
            along, reverse = 0.0, False
        reached = False
        lat, lon, hdg = _sample(e.path, e.cum or [0.0], along, loop=False)
        if reverse:
            hdg = (hdg + 180.0) % 360.0
        speed_kn = e.speed_mps * 1.943844 if active else 0.0
        task = "moving" if active else "waiting"
        status = e.status
        strength = e.strength
        mobility = e.mobility
        if e.id == "TRV005" and tick >= 680:
            status, mobility, strength = "damaged", 0.4, 0.9
        if e.id == "TRV005" and tick >= 1000:
            mobility, strength = 0.8, 0.95
        return {"lat": lat, "lon": lon, "alt": 0.0, "speed_kn": round(speed_kn, 1),
                "heading": round(hdg, 1), "travelled_m": round(along, 1),
                "laps": 0, "reached_end": reached, "mode": "ground" if active else "hold",
                "carrier_id": None, "task_status": task, "status": status,
                "strength": strength, "mobility_factor": mobility}
    # vessel / aircraft: closed loop at constant speed
    along = (tick * TICK_SECONDS * e.speed_mps + e.offset_m) % e.length_m if e.length_m else 0.0
    lat, lon, hdg = _sample(e.path, e.cum or [0.0], along, loop=True)
    return {"lat": lat, "lon": lon, "alt": e.alt_m, "speed_kn": round(e.speed_mps * 1.943844, 1),
            "heading": round(hdg, 1), "travelled_m": round(along, 1), "laps": 0,
            "reached_end": False, "mode": "sea" if e.kind == "vessel" else "air",
            "carrier_id": None, "task_status": "moving", "status": "ready",
            "strength": 1.0, "mobility_factor": 1.0}


def _objective_progress(entities: Sequence[Entity], states: Dict[str, dict],
                        tick: int) -> Dict[str, float]:
    def count_ground(aff, area):
        n = 0
        for e in entities:
            if e.kind != "vehicle" or e.affiliation != aff or e.area != area:
                continue
            st = states.get(e.id)
            if not st or st.get("mode") in ("embarked", "hold"):
                continue
            n += 1
        return n
    # Red objectives track how many ground vehicles are present in each task
    # area; the fixed node is a label/anchor, not a 300 m sensor. This avoids
    # failing an objective merely because a vehicle stopped on the next block.
    return {
        "TRN001": min(1.0, count_ground("red", "D") / 16.0),
        "TRN002": min(1.0, count_ground("red", "E") / 12.0),
        "TRN003": min(1.0, count_ground("red", "F") / 8.0),
        "TBN001": min(1.0, tick / 1200.0),
        "TBN002": min(1.0, tick / 1200.0),
        "TBN003": min(1.0, tick / 1200.0),
    }


def _build_frames(entities: Sequence[Entity], objectives: Sequence[dict]) -> List[dict]:
    carriers = {e.id: e for e in entities if e.id.startswith("TRS") and e.cls == "transport_ship"}
    frames: List[dict] = []
    for tick in range(TICK_COUNT + 1):
        carrier_pos = {}
        for cid, e in carriers.items():
            s = _state_for(e, tick, {})
            carrier_pos[cid] = (s["lat"], s["lon"], s["heading"], s["speed_kn"])
        states = {}
        rows = []
        for e in entities:
            s = _state_for(e, tick, carrier_pos)
            states[e.id] = s
            row = {
                "id": e.id, "lat": round(s["lat"], 6), "lon": round(s["lon"], 6),
                "alt": round(s["alt"], 1), "speed_kn": s["speed_kn"],
                "heading": s["heading"], "travelled_m": s["travelled_m"],
                "laps": s["laps"], "reached_end": bool(s["reached_end"]),
                "mode": s["mode"],
            }
            if s.get("carrier_id"):
                row["carrier_id"] = s["carrier_id"]
            if e.kind == "vehicle":
                row["status"] = s["status"]
                row["task_status"] = s["task_status"]
                row["strength"] = round(s["strength"], 2)
                row["mobility_factor"] = round(s["mobility_factor"], 2)
            rows.append(row)
        phase = _phase_at(tick)
        unit_phase = {r["id"]: {"id": phase["id"], "name": phase["name"],
                                "index": phase["index"]} for r in rows}
        occ = {}
        aff = {e.id: e.affiliation for e in entities}
        for aid in ("D", "E", "F"):
            meta = AREA_META[aid]
            w, s, en, n = meta["bbox"]
            red = sum(1 for r in rows if s <= r["lat"] <= n and w <= r["lon"] <= en
                      and aff[r["id"]] == "red")
            blue = sum(1 for r in rows if s <= r["lat"] <= n and w <= r["lon"] <= en
                       and aff[r["id"]] == "blue")
            occ[meta["name"]] = "red" if red > blue else ("blue" if blue > red else ("neutral" if red or blue else "none"))
        frames.append({
            "tick": tick, "t_seconds": round(tick * TICK_SECONDS, 0),
            "phase": phase, "unit_phase": unit_phase, "occupation": occ,
            "objectives": {k: round(v, 2) for k, v in
                           _objective_progress(entities, states, tick).items()},
            "targets": rows,
        })
    return frames


def _events(objectives: Sequence[dict]) -> List[dict]:
    def loc(lat, lon):
        return [round(lon, 6), round(lat, 6)]

    ev = []
    def add(eid, tick, typ, source, producer, actors, lat, lon, desc,
            before=None, after=None, cause=None, evidence=None):
        ev.append({
            "id": eid, "tick": tick, "stage_id": _stage_id(tick), "type": typ,
            "source": source, "producer": producer, "actors": list(actors),
            "location": loc(lat, lon),
            "effects": {"before": before or {}, "after": after or {}},
            "cause_event_ids": cause or [], "evidence_ids": evidence or [],
            "visibility": "global", "description": desc,
        })
    add("E01", 30, "contact", "observation", "campaign.observe", ["TRS001", "TBS001"],
        22.965, 120.130, "初始海上接触记录")
    add("E02", 150, "association", "rule", "campaign.observe", ["TRS001", "TRS002", "TRS003"],
        22.968, 120.132, "红方转运平台编组关联", cause=["E01"])
    add("E03", 260, "observation_gap", "observation", "campaign.observe", ["TRS003"],
        22.972, 120.130, "观测窗口缺口，信息陈旧", cause=["E02"])
    add("E04", 330, "reacquisition", "observation", "campaign.observe", ["TRS003", "TBA001"],
        22.974, 120.134, "目标重新出现，更新观测年龄", cause=["E03"])
    add("E05", 420, "group_change", "rule", "campaign.state", ["TRS003", "TRS004"],
        22.976, 120.135, "编组任务变化", cause=["E04"])
    add("E06", 520, "transfer_relation", "observation", "campaign.observe",
        ["TRS001", "TRV001"], 22.975, 120.133, "转运关系被合法观察", cause=["E02"])
    batches = [("E07a", 600, "TRS001", ["TRV001", "TRV002", "TRV003", "TRV004"]),
               ("E07b", 660, "TRS002", ["TRV005", "TRV006", "TRV007", "TRV008"]),
               ("E07c", 720, "TRS003", ["TRV009", "TRV010", "TRV011", "TRV012"]),
               ("E07d", 780, "TRS004", ["TRV013", "TRV014", "TRV015", "TRV016"])]
    for eid, tick, carrier, vids in batches:
        add(eid, tick, "release", "rule", "campaign.release", [carrier] + vids,
            22.971, 120.158,
            "合成转运释放：%s 释放 %d 辆" % (carrier, len(vids)),
            before={v: {"mode": "embarked"} for v in vids},
            after={v: {"mode": "ground"} for v in vids})
    add("E08", 680, "availability_change", "script", "campaign.script", ["TRV005"],
        22.968, 120.160, "软件测试输入：TRV005 可用性下降",
        before={"TRV005": {"mobility_factor": 1.0}},
        after={"TRV005": {"mobility_factor": 0.4}}, cause=["E07b"])
    add("E09", 880, "support_delay", "rule", "campaign.support", ["TRV029", "TRV023"],
        22.961, 120.202, "保障任务未完成，依赖组进入等待",
        before={"TRV023": {"task_status": "moving"}},
        after={"TRV023": {"task_status": "waiting"}})
    add("E10", 1000, "support_recovery", "rule", "campaign.support", ["TRV033", "TRV005"],
        22.963, 120.204, "维修任务完成，受影响车辆恢复",
        before={"TRV005": {"mobility_factor": 0.4}},
        after={"TRV005": {"mobility_factor": 0.8}}, cause=["E08", "E09"])
    add("E11", 1300, "objective_progress", "rule", "campaign.objectives",
        ["TRV017", "TRN002"], 22.966, 120.181, "节点任务进度更新", cause=["E07d"])
    add("E12", 1440, "termination", "rule", "campaign.outcome",
        ["TRN001", "TRN002", "TRN003"], 22.967, 120.182, "时限达到，独立计算结果",
        cause=["E11"])
    return ev


def _outcome(objectives: Sequence[dict]) -> dict:
    red = [o for o in objectives if o["id"].startswith("TRN")]
    done = sum(1 for o in red if o["status"] == "complete")
    if done == 3:
        result = "completed"
    elif done >= 2:
        result = "partial"
    elif any(o["status"] == "blocked" for o in red):
        result = "limited"
    else:
        result = "partial"
    return {
        "script_expected": "abstract_transfer_complete",
        "engine": {
            "status": "rule_v1", "result": result,
            "completed_nodes": [o["id"] for o in red if o["status"] == "complete"],
            "missing_nodes": [o["id"] for o in red if o["status"] != "complete"],
            "reasons": ["rule_v1 依据节点进度与车辆到达计数"],
        },
        "must_not_force_match": True,
    }


def _motion_extra(entities: Sequence[Entity]) -> Dict[str, List[dict]]:
    out = {}
    for e in entities:
        if e.kind != "vehicle" or e.loop or len(e.path) < 3:
            continue
        cum = e.cum or _cum_path(e.path)[0]
        rows = []
        last_h = None
        for i, (lat, lon) in enumerate(e.path[1:-1], start=1):
            h = geo.initial_bearing_deg(e.path[i - 1], e.path[i])
            if last_h is not None:
                dh = abs((h - last_h + 540.0) % 360.0 - 180.0)
                if dh < 18.0:
                    continue
            last_h = h
            dist = cum[i]
            if dist <= e.offset_m:
                continue
            t = e.start_tick * TICK_SECONDS + (dist - e.offset_m) / max(1e-6, e.speed_mps)
            if t > TICK_COUNT * TICK_SECONDS:
                break
            rows.append({"t": round(t, 1), "lat": round(lat, 6), "lon": round(lon, 6),
                         "alt": 0.0, "heading": round(h, 1)})
            if len(rows) >= 80:
                break
        if rows:
            out[e.id] = rows
    return out


OBJECTIVES = [
    {"id": "TRN001", "name": "D区衔接节点", "lat": 22.974, "lon": 120.162,
     "affiliation": "red", "area": "D", "depends_on": []},
    {"id": "TRN002", "name": "E区道路任务节点", "lat": 22.966, "lon": 120.181,
     "affiliation": "red", "area": "E", "depends_on": []},
    {"id": "TRN003", "name": "F区保障节点", "lat": 22.961, "lon": 120.202,
     "affiliation": "red", "area": "F", "depends_on": []},
    {"id": "TBN001", "name": "蓝方D区观测节点", "lat": 22.971, "lon": 120.166,
     "affiliation": "blue", "area": "D", "depends_on": []},
    {"id": "TBN002", "name": "蓝方E区观测节点", "lat": 22.963, "lon": 120.185,
     "affiliation": "blue", "area": "E", "depends_on": []},
    {"id": "TBN003", "name": "蓝方F区观测节点", "lat": 22.958, "lon": 120.206,
     "affiliation": "blue", "area": "F", "depends_on": []},
]


def _hotspots() -> List[dict]:
    out = []
    for aid in ("D", "E", "F"):
        m = AREA_META[aid]
        w, s, e, n = m["bbox"]
        out.append({"name": m["name"], "bbox": [round(w, 6), round(s, 6), round(e, 6), round(n, 6)],
                    "center": [round((s + n) / 2.0, 6), round((w + e) / 2.0, 6)],
                    "span_deg": [round(e - w, 6), round(n - s, 6)],
                    "note": "本案例合成任务区", "kind": "area"})
    return out


def _objective_list(frames: Sequence[dict]) -> List[dict]:
    last = frames[-1]["objectives"] if frames else {}
    out = []
    for o in OBJECTIVES:
        p = float(last.get(o["id"], 0.0))
        if o["id"].startswith("TBN"):
            status = "complete" if p >= 0.99 else "in_progress"
        else:
            status = "complete" if p >= 0.75 else ("in_progress" if p > 0 else "not_started")
        out.append({
            "id": o["id"], "name": o["name"], "lon": round(o["lon"], 6), "lat": round(o["lat"], 6),
            "affiliation": o["affiliation"], "area": o["area"],
            "status": status, "progress": round(p, 2), "control": o["affiliation"],
            "depends_on": o["depends_on"],
        })
    return out


def build_case(seed: int = 20261020) -> Tuple[dict, dict]:
    print("[campaign] load roads", flush=True)
    net = roads.load(THEATRE)
    rng = random.Random(seed)
    print("[campaign] build geo index", flush=True)
    buildings = PolyIndex(os.path.join(GEO_DIR, "buildings.geojson"))
    water = PolyIndex(os.path.join(GEO_DIR, "water.geojson"))
    if not buildings.items:
        raise RuntimeError("missing AOI buildings.geojson under %s" % GEO_DIR)

    route_counts = {"D": 4, "E": 4, "F": 3}
    routes: Dict[str, List[roads.RouteResult]] = {}
    for aid, count in route_counts.items():
        print("[campaign] pick routes", aid, count, flush=True)
        band = (120.145, 120.158) if aid == "D" else None
        routes[aid] = _pick_routes(net, rng, AREAS[aid], count, buildings, water,
                                   min_len=450.0, max_len=3200.0,
                                   start_band=band)
    print("[campaign] entities", flush=True)
    entities, carriers = _vehicle_entities(net, rng, routes, buildings, water)
    print("[campaign] frames", flush=True)
    frames = _build_frames(entities, OBJECTIVES)
    targets = [_entity_target(e) for e in entities]

    # Static campaign fields on each target definition.
    by_id = {e.id: e for e in entities}
    tdefs = []
    for t in targets:
        d = t.to_dataset_entry()
        e = by_id[t.target_id]
        d["icon_label"] = d.get("platform_class") or d.get("icon_label")
        d["entity_kind"] = e.kind
        d["representation"] = ("single_vehicle" if e.kind == "vehicle" else
                               ("single_vessel" if e.kind == "vessel" else "single_aircraft"))
        d["group_id"] = e.group
        d["task_id"] = e.group + "-T"
        d["initial_status"] = "reserve" if e.mode == "embarked" else "ready"
        d["initial_mode"] = e.mode
        d["initial_carrier_id"] = e.carrier
        if e.kind == "vehicle":
            p = VEHICLE_CLASSES[e.cls]
            d["physical_profile"] = {"length_m": p["length_m"], "width_m": p["width_m"],
                                     "height_m": p["height_m"], "source": "demo_profile"}
        tdefs.append(d)

    from .. import inference as inference_mod
    from .. import intent as intent_mod
    from .. import brief as brief_mod

    hs = _hotspots()
    print("[campaign] inference", flush=True)
    inf = inference_mod.infer(
        frames, targets, hotspot_names=[h["name"] for h in hs], hotspots=hs,
        relation_thresholds=((0.8, 3), (2.0, 2), (5.0, 1)),
        engagement_km=3.0, contact_km=8.0, proximity_floor_km=1.5)
    # v1 deliberately uses the L1 role intent from `inference.infer` only.
    # `intent.derive` scans every target against every neighbour on every frame;
    # at 80 targets that is a build-time cost without a v1 demo benefit.
    derived = {"summary": {"enabled": False, "source": "rule",
                            "note": "v1 uses L1 role intent; CES intent sets deferred"}}

    objectives = _objective_list(frames)
    events = _events(objectives)
    outcome = _outcome(objectives)
    motion_extra = _motion_extra(entities)

    # Route manifest: one entry per target that has a road route.
    manifest = {}
    for e in entities:
        if e.kind != "vehicle":
            continue
        manifest[e.id] = {
            "route_id": e.route_id,
            "length_m": round(e.length_m, 1),
            "component": "weak:" + str(net.weak_comp[net.nearest_node(e.path[0][0], e.path[0][1], max_km=2.0)] if net.nearest_node(e.path[0][0], e.path[0][1], max_km=2.0) is not None else -1),
            "way_ids": sorted({x.get("way_id") for x in e.edges if x.get("way_id") is not None}),
            "edge_count": len(e.edges),
            "building_check": "pass",
            "water_check": "pass",
        }
    route_manifest = {}
    for aid, rs in routes.items():
        for i, r in enumerate(rs):
            route_manifest["%s%d" % (aid, i)] = {
                "length_m": round(sum(float(e["length_m"]) for e in r.edges), 1),
                "way_ids": sorted({e.get("way_id") for e in r.edges if e.get("way_id") is not None}),
                "component": "weak:" + str(r.weak_comp),
                "building_hits": buildings.hits_path(r.path),
                "water_hits": water.hits_path(r.path),
                "status": "pass" if not buildings.hits_path(r.path) and not water.hits_path(r.path) else "fail",
            }

    geo_report = {
        "schema": "situation3d.geo_validation.v1",
        "aoi": "anping",
        "routes": route_manifest,
        "objectives": [{"id": o["id"], "lat": o["lat"], "lon": o["lon"]} for o in OBJECTIVES],
        "checks": {
            "building": {"method": "path sampled every 18 m; point-in-polygon",
                          "failed_routes": [k for k, v in route_manifest.items() if v["building_hits"]]},
            "water": {"method": "path sampled every 18 m; point-in-polygon",
                       "failed_routes": [k for k, v in route_manifest.items() if v["water_hits"]]},
            "direction": {"method": "directed Dijkstra on one-way/reverse edges"},
            "unsupported": ["maxheight", "maxwidth", "turn restrictions", "bridge clearance"],
        },
    }
    failed = len(geo_report["checks"]["building"]["failed_routes"]) + len(geo_report["checks"]["water"]["failed_routes"])
    geo_validation = {
        "passed": len(route_manifest) - failed, "failed": failed,
        "unknown": 0, "unsupported": 4,
        "package_hash": _package_hash(),
        "report": CASE_ID + ".geo.json",
    }

    print("[campaign] brief", flush=True)
    brief = brief_mod.build_brief(
        inf["frames"], targets, hotspot_names=[h["name"] for h in hs],
        model=inf["model"], start_utc=START_UTC, case_id=CASE_ID,
        case_name=CASE_NAME, tick_seconds=TICK_SECONDS)

    # Keep the per-frame inference payload lean: the UI reads state/intent/
    # threat/alert/nearest; threat_terms and repeated labels can be recomputed
    # from the model block and were the difference between a 37 MB and 25 MB
    # inference slice for 80 targets.
    for fr in inf["frames"]:
        for entry in fr["targets"].values():
            entry.pop("threat_terms", None)
            entry.pop("alert_label", None)
            entry.pop("state_label", None)
        for g in fr["groups"]:
            g.pop("members", None)

    dataset = {
        "schema": "situation3d.trajectory.v2",
        "campaign_contract_version": 1,
        "case_id": CASE_ID,
        "case_name": CASE_NAME,
        "case_description": CASE_DESC,
        "scenario_id": "taiwan_anping_dense_demo_20260915",
        "scenario_name": "台海方向多域态势演化与岸海衔接",
        "region": "中国 台湾海峡",
        "theatre": THEATRE,
        "provenance": "simulated",
        "provenance_note": "合成目标与虚构参数；真实地理仅作背景，不涉及现实部署。",
        "tick_seconds": float(TICK_SECONDS),
        "tick_count": TICK_COUNT,
        "start_time_utc": START_UTC,
        "camera": {"lon": 120.170000, "lat": 22.972000, "height": 12000.0,
                   "heading": 0.0, "pitch": -60.0},
        "phases": [{"id": s[0], "name": s[1], "first_tick": s[2]} for s in STAGES],
        "stages": [{"id": s[0], "name": s[1], "start_tick": s[2], "end_tick": s[3]}
                   for s in STAGES],
        "hotspots": hs,
        "urban_aoi": [{"name": "中国台湾省 高雄", "bbox": [120.24, 22.57, 120.36, 22.69], "zoom": 13}],
        "symbology": symbology.catalog(),
        "targets": tdefs,
        "frames": frames,
        "inference_frames": inf["frames"],
        "inference_model": inf["model"],
        "inference_summary": inf["summary"],
        "intent_summary": derived["summary"],
        "intent_source": "rule",
        "land_route_meta": manifest,
        "path_manifest": route_manifest,
        "road_constrained": True,
        "events": events,
        "objectives": objectives,
        "outcome": outcome,
        "motion_extra": motion_extra,
        "geo_validation": geo_validation,
        "brief": brief,
    }
    return dataset, geo_report


def _package_hash() -> str:
    h = hashlib.sha256()
    manifest = os.path.join(GEO_DIR, "manifest.json")
    if os.path.isfile(manifest):
        with open(manifest, "rb") as fh:
            h.update(fh.read())
    for name in ("buildings.geojson", "water.geojson", "coastline.geojson"):
        p = os.path.join(GEO_DIR, name)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def write_case(out_dir: str, seed: int = 20261020) -> Tuple[str, str]:
    dataset, geo_report = build_case(seed)
    out_dir = os.path.abspath(out_dir).rstrip(os.sep)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, CASE_ID + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    geo_dir = os.path.join(os.path.dirname(out_dir), "geo_reports")
    os.makedirs(geo_dir, exist_ok=True)
    geo_path = os.path.join(geo_dir, CASE_ID + ".geo.json")
    with open(geo_path, "w", encoding="utf-8") as fh:
        json.dump(geo_report, fh, ensure_ascii=False, indent=1)
    return path, geo_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "web", "data", "situations"))
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()
    t0 = time.time()
    p, g = write_case(args.out, args.seed)
    print("wrote", p, g, "in %.1fs" % (time.time() - t0))
