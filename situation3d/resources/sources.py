"""Source candidates for parallel acquisition and automatic fallback.

A candidate describes *how* a resource can be obtained, not just where:

* ``xyz``    -- download Web-Mercator tiles from a URL template
* ``raster`` -- download a complete GeoTIFF/COG and tile it locally
* ``file``   -- a local file/directory already on disk

The same candidate model is used by envpack sources (overpass/pbf) in a later
step; tile acquisition is the first vertical slice because it exercises both
"many small downloads" and "one big download + local processing".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Public imagery XYZ endpoints.  Licences are retained so callers can refuse
# sources that are not acceptable for their use.
IMAGERY_SOURCES: Dict[str, Dict[str, Any]] = {
    "esri": {
        "method": "xyz",
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "ext": "jpg",
        "max_zoom": 19,
        "attribution": "Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        "license": "Esri terms of use -- free with attribution; review before operational use.",
        "quality": "measured",
        "priority": 10,
    },
    "eox-s2": {
        "method": "xyz",
        "url": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/"
               "default/g/{z}/{y}/{x}.jpg",
        "ext": "jpg",
        "max_zoom": 14,
        "attribution": "Sentinel-2 cloudless 2020 by EOX IT Services GmbH",
        "license": "CC BY-NC-SA 4.0 -- non-commercial only.",
        "quality": "measured",
        "priority": 20,
    },
    "gibs-modis": {
        "method": "xyz",
        "url": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
               "MODIS_Terra_CorrectedReflectance_TrueColor/default/2023-06-01/"
               "GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg",
        "ext": "jpg",
        "max_zoom": 9,
        "attribution": "NASA EOSDIS GIBS / MODIS Terra",
        "license": "Public domain (US Government work). 250 m, includes cloud cover.",
        "quality": "measured",
        "priority": 30,
    },
}

TERRAIN_SOURCES: Dict[str, Dict[str, Any]] = {
    "terrarium": {
        "method": "xyz",
        "url": "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
        "ext": "png",
        "max_zoom": 15,
        "attribution": "Elevation tiles by Mapzen, hosted on AWS Open Data",
        "license": "Public domain / CC0.",
        "quality": "measured",
        "priority": 10,
    },
}

DEFAULT_IMAGERY = ("esri", "eox-s2", "gibs-modis")


@dataclass(frozen=True)
class SourceCandidate:
    source_id: str
    method: str                    # xyz | raster | file
    params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 100
    quality: str = "measured"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "method": self.method,
            "params": dict(self.params),
            "priority": int(self.priority),
            "quality": self.quality,
        }


def _from_catalog(source_id: str, cfg: Dict[str, Any]) -> SourceCandidate:
    params = {k: v for k, v in cfg.items()
              if k not in ("priority", "quality")}
    return SourceCandidate(source_id=source_id, method=cfg["method"],
                           params=params,
                           priority=int(cfg.get("priority") or 100),
                           quality=str(cfg.get("quality") or "measured"))


def imagery_candidates(source_ids: Optional[Sequence[str]] = None) -> List[SourceCandidate]:
    names = list(source_ids) if source_ids else list(DEFAULT_IMAGERY)
    out = []
    for name in names:
        if name in IMAGERY_SOURCES:
            out.append(_from_catalog(name, IMAGERY_SOURCES[name]))
        elif name in TERRAIN_SOURCES:
            out.append(_from_catalog(name, TERRAIN_SOURCES[name]))
        else:
            raise KeyError("unknown imagery source %r" % (name,))
    return out


def raster_candidate(url: str, source_id: Optional[str] = None,
                     license_text: str = "unknown",
                     cog: bool = True) -> SourceCandidate:
    return SourceCandidate(
        source_id=source_id or "raster-url",
        method="raster",
        params={"url": url, "ext": "png", "license": license_text,
                "cog": bool(cog),
                "attribution": "user-provided raster"},
        priority=5,
        quality="measured")


def local_candidate(path: str, source_id: str = "local-cache",
                    method: str = "file") -> SourceCandidate:
    return SourceCandidate(source_id=source_id, method=method,
                           params={"path": path}, priority=0,
                           quality="measured")


def candidates_for_tiles(source_ids: Optional[Sequence[str]] = None,
                         raster_urls: Optional[Sequence[str]] = None,
                         local_dirs: Optional[Sequence[str]] = None
                         ) -> List[SourceCandidate]:
    out: List[SourceCandidate] = []
    for url in raster_urls or []:
        out.append(raster_candidate(url))
    out.extend(imagery_candidates(source_ids))
    for path in local_dirs or []:
        out.append(local_candidate(path))
    return out
