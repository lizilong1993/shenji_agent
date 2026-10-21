"""OSM PBF source: offline import of a local ``.osm.pbf`` extract.

Requires the optional ``osmium`` (pyosmium) package; the rest of the engine
does not.  Ways inside the requested bbox are converted into the same
Overpass ``out geom`` element shape consumed by :func:`geodata.osm.normalize`,
so both sources share one normalisation path.  The converted raw JSON is
cached, making offline rebuilds byte-identical to the PBF import.
"""

from __future__ import annotations

from typing import Dict, List

from geodata.cache import CacheStore, OfflineCacheMiss  # noqa: F401 (re-exported)
from geodata.osm import normalize

TAG_KEYS = ("highway", "natural", "waterway", "landuse", "building")


class PbfSourceError(RuntimeError):
    pass


def read_bbox(path: str, bbox) -> Dict[str, list]:
    """Return an Overpass-shaped ``{"elements": [...]}`` for ways in ``bbox``."""
    try:
        import osmium
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PbfSourceError(
            "pyosmium is required for PBF import: pip install osmium") from exc

    west, south, east, north = [float(v) for v in bbox]
    elements: List[dict] = []

    class _Handler(osmium.SimpleHandler):
        def way(self, w):
            tags = {t.k: t.v for t in w.tags}
            if not any(k in tags for k in TAG_KEYS):
                return
            coords = []
            inside = False
            for nd in w.nodes:
                if not nd.location.valid():
                    continue
                lon, lat = nd.location.lon, nd.location.lat
                coords.append([lon, lat])
                inside = inside or (west <= lon <= east and south <= lat <= north)
            if inside and len(coords) >= 2:
                elements.append({
                    "type": "way", "id": w.id, "tags": tags,
                    "geometry": [{"lon": c[0], "lat": c[1]} for c in coords],
                })

    handler = _Handler()
    handler.apply_file(path, locations=True)
    return {"elements": elements}


class PbfOsmSource(object):
    """Offline-first OSM source backed by a local ``.osm.pbf`` extract."""

    provenance_label = "pbf"

    def __init__(self, cache: CacheStore, pbf_path: str):
        self.cache = cache
        self.pbf_path = pbf_path
        self.fetched_network = False

    @staticmethod
    def _key(bbox) -> str:
        return "%.6f,%.6f,%.6f,%.6f" % tuple(float(v) for v in bbox)

    def raw(self, bbox) -> dict:
        key = self._key(bbox)
        if self.cache.mode == "offline" or (
                self.cache.mode == "auto" and self.cache.has("osm_pbf", key)):
            return self.cache.read_json("osm_pbf", key)
        data = read_bbox(self.pbf_path, bbox)
        self.fetched_network = True
        self.cache.write_json("osm_pbf", key, data)
        return data

    def features(self, bbox) -> Dict[str, list]:
        return normalize(self.raw(bbox))
