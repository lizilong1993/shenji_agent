"""Source registry for the P0 VGI connectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    host: str
    license: str
    attribution: str
    schedule_s: int
    parser_version: str
    requires_env: Optional[str] = None
    optional: bool = False


SOURCES = {
    "usgs": SourceSpec(
        source_id="usgs",
        host="earthquake.usgs.gov",
        license="U.S. public domain",
        attribution="Data courtesy of the U.S. Geological Survey",
        schedule_s=300,
        parser_version="vgi-connectors-1",
    ),
    "gdacs": SourceSpec(
        source_id="gdacs",
        host="www.gdacs.org",
        license="GDACS public data (European Commission/JRC)",
        attribution="GDACS - European Commission/JRC",
        schedule_s=900,
        parser_version="vgi-connectors-1",
    ),
    "firms": SourceSpec(
        source_id="firms",
        host="firms.modaps.eosdis.nasa.gov",
        license="NASA open data; attribution required",
        attribution="NASA FIRMS",
        schedule_s=1800,
        parser_version="vgi-connectors-1",
        requires_env="FIRMS_MAP_KEY",
    ),
    "adsb": SourceSpec(
        source_id="adsb",
        host="api.adsb.lol",
        license="ODbL 1.0",
        attribution="adsb.lol (ODbL)",
        schedule_s=30,
        parser_version="vgi-sources-1",
    ),
    "eonet": SourceSpec(
        source_id="eonet",
        host="eonet.gsfc.nasa.gov",
        license="NASA open data",
        attribution="NASA EONET",
        schedule_s=1800,
        parser_version="vgi-sources-1",
    ),
    "gdelt": SourceSpec(
        source_id="gdelt",
        host="api.gdeltproject.org",
        license="GDELT terms; citation required",
        attribution="The GDELT Project",
        schedule_s=300,
        parser_version="vgi-sources-1",
        optional=True,
    ),
}

SOURCE_IDS = tuple(SOURCES)


def get_source(source_id: str) -> SourceSpec:
    key = str(source_id or "").strip().lower()
    if key not in SOURCES:
        raise KeyError("unsupported source %r" % (source_id,))
    return SOURCES[key]
