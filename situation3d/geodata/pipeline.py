"""Unified envpack builder for the online and offline data paths.

Both modes call the same normalisation code.  Online mode fetches OSM/DEM and
writes the raw responses into the shared cache; offline mode replays that
cache.  ``auto`` prefers the cache and only fetches missing entries.  When a
source is unavailable the builder records a fallback to the 庙算
``basic.json`` layers instead of failing silently.

Output (``msgeo.envpack.v1``):

    manifest.json
    terrain/grid.json          regular elevation grid
    landcover/polygons.json    OSM landuse/natural polygons
    water/polygons.json        water polygons
    buildings/buildings.json   footprint + height
    roads/graph.json           nodes/edges with road attributes
    base_map.json              optional fallback basic.json
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

from geodata.cache import CacheCorrupt, CacheStore, OfflineCacheMiss
from geodata.dem import DemSource, OnlineDemError
from geodata.osm import OnlineSourceError, OsmSource
from geo.geodesy import haversine_m


def _bbox_from_basic(basic: dict, grid) -> list:
    corners = [grid.cell_center_lonlat(0, 0),
               grid.cell_center_lonlat(0, grid.cols - 1),
               grid.cell_center_lonlat(grid.rows - 1, 0),
               grid.cell_center_lonlat(grid.rows - 1, grid.cols - 1)]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return [min(lons), min(lats), max(lons), max(lats)]


def _fallback_grid(basic: dict, grid, bbox, resolution_m: float):
    from env.grid import MapGridEnvironment
    env = MapGridEnvironment(basic, grid)
    lon0, lat0, lon1, lat1 = bbox
    step_deg = float(resolution_m) / 111320.0
    cols = max(2, int(math.ceil((lon1 - lon0) / step_deg)) + 1)
    rows = max(2, int(math.ceil((lat1 - lat0) / step_deg)) + 1)
    values = []
    for r in range(rows):
        lat = lat0 + (lat1 - lat0) * r / float(rows - 1)
        row = []
        for c in range(cols):
            lon = lon0 + (lon1 - lon0) * c / float(cols - 1)
            row.append(round(env.elevation(lon, lat), 1))
        values.append(row)
    return {"lon0": lon0, "lat0": lat0, "lon1": lon1, "lat1": lat1,
            "cols": cols, "rows": rows, "values": values}


def _fallback_roads(basic: dict, grid):
    nodes = {}
    edges = []
    road_cells = []
    for r in range(grid.rows):
        for c in range(grid.cols):
            cell = basic["map_data"][r][c]
            if any(int(x or 0) > 0 for x in (cell.get("roads") or [])):
                road_cells.append((r, c))
    for (r, c) in road_cells:
        lon, lat = grid.cell_center_lonlat(r, c)
        nodes["%d_%d" % (r, c)] = [lon, lat]
    for (r, c) in road_cells:
        cell = basic["map_data"][r][c]
        for direction, nb in enumerate(grid.neighbors((r, c))):
            nr, nc = nb
            if nb <= (r, c):
                continue
            ncell = basic["map_data"][nr][nc]
            if not any(int(x or 0) > 0 for x in (ncell.get("roads") or [])):
                continue
            a = "%d_%d" % (r, c)
            b = "%d_%d" % (nr, nc)
            lon_a, lat_a = nodes[a]
            lon_b, lat_b = nodes[b]
            edges.append({"a": a, "b": b,
                          "length_m": round(haversine_m((lon_a, lat_a), (lon_b, lat_b)), 1),
                          "class": "legacy", "bridge": False, "tunnel": False})
    return {"nodes": nodes, "edges": edges}


def _road_graph(features: dict):
    nodes = {}
    edges = []
    for way in features.get("roads") or []:
        coords = way.get("coords") or []
        ids = []
        for lon, lat in coords:
            key = "%.7f_%.7f" % (lon, lat)
            nodes.setdefault(key, [lon, lat])
            ids.append(key)
        for i in range(1, len(ids)):
            a, b = ids[i - 1], ids[i]
            if a == b:
                continue
            lon_a, lat_a = nodes[a]
            lon_b, lat_b = nodes[b]
            edges.append({"a": a, "b": b,
                          "length_m": round(haversine_m((lon_a, lat_a), (lon_b, lat_b)), 1),
                          "class": way.get("class"),
                          "surface": way.get("surface"),
                          "bridge": bool(way.get("bridge")),
                          "tunnel": bool(way.get("tunnel")),
                          "oneway": bool(way.get("oneway")),
                          "maxheight": way.get("maxheight"),
                          "maxweight": way.get("maxweight"),
                          "width": way.get("width")})
    return {"nodes": nodes, "edges": edges}


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)


def build_envpack(out: str, basic: dict, grid, theatre: str = "demo",
                  mode: str = "auto", cache_dir: Optional[str] = None,
                  resolution_m: float = 200.0, dem_zoom: int = 10,
                  use_osm: bool = True, use_dem: bool = True,
                  osm_source=None, dem_source=None,
                  bbox=None) -> dict:
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    cache = CacheStore(cache_dir or os.path.join(out, "_cache"), mode=mode)
    bbox = bbox or _bbox_from_basic(basic, grid)
    os.makedirs(out, exist_ok=True)

    provenance = {"mode": mode, "osm": "none", "dem": "none", "fallback": False}
    features = {"roads": [], "water": [], "buildings": [], "landuse": []}
    if use_osm:
        source = osm_source or OsmSource(cache)
        try:
            features = source.features(bbox)
            provenance["osm"] = "online" if getattr(source, "fetched_network", False) else "offline-cache"
        except (OfflineCacheMiss, OnlineSourceError, CacheCorrupt) as exc:
            provenance["osm"] = "fallback-basic (%s)" % type(exc).__name__
            provenance["fallback"] = True

    terrain = None
    if use_dem:
        try:
            dem = dem_source or DemSource(cache, zoom=dem_zoom)
            step_deg = float(resolution_m) / 111320.0
            terrain = dem.sample_grid(bbox, step_deg)
            provenance["dem"] = "online" if getattr(dem, "fetched_network", False) else "offline-cache"
        except (OfflineCacheMiss, OnlineDemError, CacheCorrupt):
            terrain = None
    if terrain is None:
        terrain = _fallback_grid(basic, grid, bbox, resolution_m)
        provenance["dem"] = "fallback-basic"
        provenance["fallback"] = True

    roads = _road_graph(features)
    if not roads["edges"]:
        roads = _fallback_roads(basic, grid)
        provenance["roads"] = "fallback-basic"
        provenance["fallback"] = True
    else:
        provenance["roads"] = provenance["osm"]

    _write_json(os.path.join(out, "terrain", "grid.json"), terrain)
    _write_json(os.path.join(out, "landcover", "polygons.json"),
                features.get("landuse") or [])
    _write_json(os.path.join(out, "water", "polygons.json"),
                features.get("water") or [])
    _write_json(os.path.join(out, "buildings", "buildings.json"),
                features.get("buildings") or [])
    _write_json(os.path.join(out, "roads", "graph.json"), roads)
    with open(os.path.join(out, "base_map.json"), "w", encoding="utf-8") as fh:
        json.dump(basic, fh, ensure_ascii=False)

    manifest = {
        "schema": "msgeo.envpack.v1",
        "theatre": theatre,
        "mode": mode,
        "built": __import__("datetime").date.today().isoformat(),
        "bbox": [round(float(v), 7) for v in bbox],
        "origin": {"lon": grid.center_lon, "lat": grid.center_lat, "alt": 0.0},
        "grid": {"center_lon": grid.center_lon, "center_lat": grid.center_lat,
                 "cols": grid.cols, "rows": grid.rows, "width_m": grid.cell_m},
        "tile_size_m": 5000.0,
        "resolution_m": float(resolution_m),
        "ele_grade": float(basic.get("ele_grade") or 10.0),
        "layers": {
            "terrain": {"format": "grid-json", "path": "terrain/grid.json",
                        "source": provenance["dem"], "resolution_m": resolution_m},
            "landcover": {"format": "polygons-json", "path": "landcover/polygons.json",
                          "source": provenance["osm"]},
            "water": {"format": "polygons-json", "path": "water/polygons.json",
                      "source": provenance["osm"], "depth_m": 20.0},
            "buildings": {"format": "polygons-json", "path": "buildings/buildings.json",
                          "source": provenance["osm"]},
            "roads": {"format": "graph-json", "path": "roads/graph.json",
                      "source": provenance.get("roads", provenance["osm"])},
        },
        "fallback_basic_json": "base_map.json",
        "provenance": provenance,
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    return manifest
