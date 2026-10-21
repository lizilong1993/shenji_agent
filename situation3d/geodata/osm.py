"""OSM source: Overpass online fetch with offline cache replay.

Both paths return the same normalized feature dict:

    {"roads": [...], "water": [...], "buildings": [...], "landuse": [...]}

Each road is a polyline of ``[lon, lat]`` plus tags; water/buildings/landuse
are closed rings.  The raw Overpass JSON is cached, so an offline rebuild is
byte-identical to the online one.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from geodata.cache import CacheCorrupt, CacheStore, OfflineCacheMiss


class OnlineSourceError(RuntimeError):
    pass


DEFAULT_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

QUERY_TEMPLATE = (
    "[out:json][timeout:{timeout}];"
    "(way({s},{w},{n},{e})[highway];"
    "way({s},{w},{n},{e})[natural=water];"
    "way({s},{w},{n},{e})[waterway=riverbank];"
    "way({s},{w},{n},{e})[landuse=reservoir];"
    "way({s},{w},{n},{e})[building];"
    "way({s},{w},{n},{e})[landuse];"
    "way({s},{w},{n},{e})[natural];"
    ");out geom;"
)


def _bbox_key(bbox) -> str:
    return "%.6f,%.6f,%.6f,%.6f" % tuple(float(v) for v in bbox)


class OsmSource(object):
    def __init__(self, cache: CacheStore, endpoints: Optional[List[str]] = None,
                 timeout_s: int = 60):
        self.cache = cache
        self.endpoints = endpoints or list(DEFAULT_ENDPOINTS)
        self.timeout_s = int(timeout_s)
        self.fetched_network = False

    def _fetch(self, bbox) -> dict:
        s, w, n, e = bbox[1], bbox[0], bbox[3], bbox[2]
        query = QUERY_TEMPLATE.format(s=s, w=w, n=n, e=e, timeout=self.timeout_s)
        last = None
        for url in self.endpoints:
            try:
                req = urllib.request.Request(
                    url, data=urllib.parse.urlencode({"data": query}).encode(),
                    headers={"User-Agent": "situation3d-geo-engine/1.0"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("unexpected Overpass payload")
                return data
            except Exception as exc:  # try the next mirror
                last = exc
        raise OnlineSourceError("all Overpass endpoints failed: %s" % last)

    def raw(self, bbox) -> dict:
        key = _bbox_key(bbox)
        if self.cache.mode == "offline" or (self.cache.mode == "auto" and self.cache.has("osm", key)):
            return self.cache.read_json("osm", key)
        data = self._fetch(bbox)
        self.fetched_network = True
        self.cache.write_json("osm", key, data)
        return data

    def features(self, bbox) -> Dict[str, list]:
        return normalize(self.raw(bbox))


def _way_coords(el: dict) -> List[List[float]]:
    out = []
    for g in el.get("geometry") or []:
        if g.get("lat") is None or g.get("lon") is None:
            continue
        out.append([float(g["lon"]), float(g["lat"])])
    if len(out) < 2 and el.get("nodes") and el.get("_nodes"):
        out = [el["_nodes"][n] for n in el["nodes"] if n in el["_nodes"]]
    return out


def _closed(coords: List[List[float]]) -> bool:
    return len(coords) >= 4 and coords[0][0] == coords[-1][0] and coords[0][1] == coords[-1][1]


def _height(tags: dict) -> float:
    try:
        if tags.get("height"):
            return float(str(tags["height"]).replace("m", "").strip())
        if tags.get("building:levels"):
            return float(tags["building:levels"]) * 3.0
    except ValueError:
        pass
    return 6.0


LANDUSE_MAP = {
    "forest": "forest", "wood": "forest", "scrub": "forest",
    "residential": "urban", "commercial": "urban", "industrial": "urban",
    "retail": "urban", "construction": "urban",
    "meadow": "open", "grass": "open", "farmland": "soft", "orchard": "soft",
    "beach": "beach", "sand": "beach", "quarry": "rock",
}


def normalize(raw: dict) -> Dict[str, list]:
    features = {"roads": [], "water": [], "buildings": [], "landuse": []}
    for el in raw.get("elements") or []:
        tags = el.get("tags") or {}
        coords = _way_coords(el)
        if len(coords) < 2:
            continue
        if "highway" in tags:
            features["roads"].append({
                "coords": coords,
                "class": tags.get("highway"),
                "surface": tags.get("surface"),
                "bridge": tags.get("bridge") in ("yes", "viaduct"),
                "tunnel": tags.get("tunnel") in ("yes", "building_passage"),
                "oneway": tags.get("oneway") in ("yes", "true", "1"),
                "maxheight": tags.get("maxheight"),
                "maxweight": tags.get("maxweight"),
                "width": tags.get("width"),
                "name": tags.get("name"),
            })
        elif "building" in tags and _closed(coords):
            features["buildings"].append({
                "ring": coords, "height_m": _height(tags),
                "name": tags.get("name"), "levels": tags.get("building:levels"),
            })
        elif ("natural" in tags or "waterway" in tags or "landuse" in tags) and _closed(coords):
            if tags.get("natural") == "water" or tags.get("waterway") == "riverbank" \
                    or tags.get("landuse") == "reservoir":
                features["water"].append({"ring": coords, "name": tags.get("name")})
            elif "landuse" in tags:
                cls = LANDUSE_MAP.get(str(tags.get("landuse")), "open")
                features["landuse"].append({"ring": coords, "class": cls})
            elif tags.get("natural") in ("wood", "scrub"):
                features["landuse"].append({"ring": coords, "class": "forest"})
    return features
