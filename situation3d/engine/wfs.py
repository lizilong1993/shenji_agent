"""OGC WFS 2.0.0 over the offline vector datasets.

Why serve WFS at all
--------------------
The vector layers are stored as files, and the front end could just fetch the
GeoJSON directly. Publishing them as WFS exists for three reasons:

1. **One contract for all vector layers.** New datasets (see
   `tools/fetch_vector.py`) join by adding a manifest entry; the client keeps
   talking to one endpoint with one parameter set.
2. **Server-side filtering and paging.** A client that only wants the 6 naval
   bases, or the first 200 rows, does not have to download and parse the whole
   file. That matters as datasets grow past the current ~450 KB.
3. **Licence travels with the data.** ODbL and CC BY-NC-SA both require
   attribution, and `GetCapabilities` is where a WFS client is expected to find
   it. Putting the licence in a README instead would be easy to lose.

Implemented operations
----------------------
  GetCapabilities          what feature types exist, their bbox, their licence
  DescribeFeatureType      the property schema of one type
  GetFeature                GeoJSON payload, with bbox / count / paging / properties

Deliberately NOT implemented: transactions (WFS-T), and non-GeoJSON output
formats. This is a read-only publication of static open data; write support would
add attack surface for nothing.

The two axis-order traps, both handled explicitly
-------------------------------------------------
* **WFS 2.0 `bbox` for EPSG:4326 is axis order lat,lon** (it follows the CRS
  definition, not the historical GeoJSON convention). Clients written against
  WFS 1.1 or against GeoJSON get this backwards, and the failure is silent --
  you simply get features from the wrong hemisphere. `parse_bbox()` therefore
  accepts `bbox=minlat,minlon,maxlat,maxlon` and says so in the capabilities
  doc. A `bboxCrs` of CRS84 is honoured as lon,lat, because that is what CRS84
  means.
* **`count` vs `maxFeatures`**: WFS 2.0 renamed the paging parameter to `count`.
  WFS 1.1 clients still send `maxFeatures`; both are accepted so a stock client
  does the expected thing.

Standard library only. Reads `engine/data/vector/*.geojson` + `manifest.json`,
both produced by tools/fetch_vector.py.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
VECTOR_DIR = os.path.join(HERE, "data", "vector")
MANIFEST_PATH = os.path.join(VECTOR_DIR, "manifest.json")

WFS_VERSION = "2.0.0"
DEFAULT_CRS = "urn:ogc:def:crs:EPSG::4326"
CRS84 = "urn:ogc:def:crs:OGC:1.3:CRS84"
DEFAULT_COUNT = 1000
MAX_COUNT = 20000

_MANIFEST_CACHE: Dict[str, object] = {}


class WfsError(ValueError):
    """Bad request (maps to HTTP 400 with an OGC exception report)."""


def load_manifest(refresh: bool = False) -> dict:
    """Read manifest.json, cached on (mtime, size) like the TLE catalogue."""
    try:
        st = os.stat(MANIFEST_PATH)
        key = (st.st_mtime, st.st_size)
    except OSError:
        if refresh or "manifest" not in _MANIFEST_CACHE:
            raise WfsError("vector manifest not found at %s -- run "
                           "tools/fetch_vector.py" % MANIFEST_PATH)
        return _MANIFEST_CACHE["manifest"]        # type: ignore[return-value]

    cached = _MANIFEST_CACHE.get("key")
    if not refresh and cached == key and "manifest" in _MANIFEST_CACHE:
        return _MANIFEST_CACHE["manifest"]        # type: ignore[return-value]

    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    _MANIFEST_CACHE["manifest"] = manifest
    _MANIFEST_CACHE["key"] = key
    return manifest


def feature_types() -> Dict[str, dict]:
    return (load_manifest().get("types") or {})


def load_features(type_name: str) -> dict:
    """Load one feature collection. Raises WfsError for an unknown type."""
    types = feature_types()
    if type_name not in types:
        raise WfsError("unknown feature type %r (have: %s)"
                       % (type_name, ", ".join(sorted(types)) or "none"))
    path = os.path.join(VECTOR_DIR, types[type_name]["file"])
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------

def parse_bbox(raw: Optional[str], crs: Optional[str] = None
               ) -> Optional[Tuple[float, float, float, float]]:
    """Parse a WFS 2.0 bbox into (west, south, east, north) in lon/lat order.

    The wire format is **lat,lon** for EPSG:4326 (CRS-defined axis order), which
    is the opposite of GeoJSON. This function is the single place that knows that,
    and it returns lon/lat so the rest of the module never has to think about it.

    `crs` may be CRS84, where the order is lon,lat.
    """
    if not raw:
        return None
    parts = [p for p in str(raw).replace(" ", "").split(",") if p]
    if len(parts) == 5:                     # optional trailing CRS uri
        parts = parts[:4]
    if len(parts) != 4:
        raise WfsError("bbox must be minlat,minlon,maxlat,maxlon (WFS 2.0 EPSG:4326 "
                       "axis order); got %r" % raw)
    try:
        a, b, c, d = (float(p) for p in parts)
    except ValueError:
        raise WfsError("bbox values must be numeric; got %r" % raw)

    if crs and "CRS84" in str(crs).upper():
        west, south, east, north = a, b, c, d
    else:
        south, west, north, east = a, b, c, d

    if south > north:
        south, north = north, south
    if west > east:
        west, east = east, west
    return (west, south, east, north)


def _geom_coords(geom: dict):
    return (geom or {}).get("coordinates")


def _point_in_box(lon: float, lat: float, box) -> bool:
    w, s, e, n = box
    return w <= lon <= e and s <= lat <= n


def _iter_points(coords):
    """Yield (lon, lat) for any nesting depth of a GeoJSON coordinate array."""
    if not coords:
        return
    first = coords[0]
    if isinstance(first, (int, float)):
        if len(coords) >= 2:
            yield (coords[0], coords[1])
        return
    for c in coords:
        for p in _iter_points(c):
            yield p


def _geom_bbox(geom: dict):
    pts = list(_iter_points(_geom_coords(geom)))
    if not pts:
        return None
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return (min(lons), min(lats), max(lons), max(lats))


def _geom_intersects(geom: dict, box) -> bool:
    """Bbox-overlap test.

    Deliberately a bounding-box test, not an exact geometric intersection: these
    are display layers, and a cable whose *envelope* clips the box is worth
    drawing. Exact clipping would need a geometry library the container does not
    have, for no visible benefit.
    """
    gb = _geom_bbox(geom)
    if gb is None:
        return False
    gw, gs, ge, gn = gb
    w, s, e, n = box
    return not (ge < w or gw > e or gn < s or gs > n)


def parse_count(raw: Optional[str], legacy: Optional[str] = None) -> int:
    """WFS 2.0 `count`, falling back to the WFS 1.1 `maxFeatures`."""
    for value in (raw, legacy):
        if value in (None, ""):
            continue
        try:
            n = int(value)
        except ValueError:
            raise WfsError("count must be an integer; got %r" % value)
        if n < 0:
            raise WfsError("count must be >= 0")
        return min(n, MAX_COUNT)
    return DEFAULT_COUNT


def parse_start_index(raw: Optional[str]) -> int:
    if raw in (None, ""):
        return 0
    try:
        n = int(raw)
    except ValueError:
        raise WfsError("startIndex must be an integer; got %r" % raw)
    return max(0, n)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def get_capabilities(base_url: str = "", type_names: Optional[Sequence[str]] = None
                     ) -> str:
    """Capabilities document, one FeatureType per manifest entry.

    `base_url` is echoed into the operation URLs so a client can call back; it is
    the caller's job to pass the externally reachable address.
    """
    manifest = load_manifest()
    types = manifest.get("types") or {}
    wanted = list(type_names) if type_names else sorted(types)
    earth = manifest.get("bbox") or [-180, -90, 180, 90]

    parts: List[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        '<wfs:WFS_Capabilities version="%s" '
        'xmlns:wfs="http://www.opengis.net/wfs/2.0" '
        'xmlns:ows="http://www.opengis.net/ows/1.1" '
        'xmlns:fes="http://www.opengis.net/fes/2.0" '
        'xmlns:xlink="http://www.w3.org/1999/xlink">' % WFS_VERSION)
    parts.append(
        '<ows:ServiceIdentification>'
        '<ows:Title>situation3d vector data</ows:Title>'
        '<ows:Abstract>离线矢量数据（开放许可）以 WFS 2.0.0 发布。'
        'bbox 采用 EPSG:4326 的轴序 minlat,minlon,maxlat,maxlon。'
        '</ows:Abstract>'
        '<ows:ServiceType>WFS</ows:ServiceType>'
        '<ows:ServiceTypeVersion>%s</ows:ServiceTypeVersion>'
        '</ows:ServiceIdentification>' % WFS_VERSION)

    # Licence lives here on purpose: this is where a WFS client is expected to
    # look for attribution, and both datasets require it (ODbL / CC BY-NC-SA).
    access = ''.join(
        '<ows:Constraint name="Licence">'
        '<ows:Value>%s -- %s</ows:Value>'
        '</ows:Constraint>' % (escape(t.get("licence", "")), escape(t.get("licence_url", "")))
        for t in (types[n] for n in wanted if n in types)
    )
    parts.append('<ows:ServiceProvider><ows:ProviderName>situation3d</ows:ProviderName>'
                 '<ows:ServiceContact/></ows:ServiceProvider>')
    parts.append('<ows:OperationsMetadata>%s</ows:OperationsMetadata>' % access)

    parts.append('<wfs:FeatureTypeList>')
    for name in wanted:
        meta = types.get(name)
        if not meta:
            continue
        parts.append(
            '<wfs:FeatureType>'
            '<wfs:Name>%s</wfs:Name>'
            '<wfs:Title>%s</wfs:Title>'
            '<wfs:Abstract>%s</wfs:Abstract>'
            '<wfs:DefaultCRS>%s</wfs:DefaultCRS>'
            '<ows:WGS84BoundingBox>'
            '<ows:LowerCorner>%s %s</ows:LowerCorner>'
            '<ows:UpperCorner>%s %s</ows:UpperCorner>'
            '</ows:WGS84BoundingBox>'
            '<MetadataURL xlink:href="%s"/>'
            '</wfs:FeatureType>' % (
                escape(name), escape(meta.get("title", name)),
                escape(meta.get("description", "")), DEFAULT_CRS,
                earth[1], earth[0], earth[3], earth[2],
                escape(meta.get("licence_url", "")),
            ))
    parts.append('</wfs:FeatureTypeList>')
    parts.append('<fes:Filter_Capabilities/>')
    parts.append('</wfs:WFS_Capabilities>')
    return "".join(parts)


def _property_schema(features: Sequence[dict]) -> List[Tuple[str, str]]:
    """(name, xsd type) pairs, inferred by scanning features.

    Inferred rather than declared because the sources are third-party files whose
    own schemas are not shipped. Scanning is honest about what is actually in the
    file, and a field that is null everywhere simply does not appear.
    """
    order: List[str] = []
    types: Dict[str, str] = {}
    for f in features[:500]:
        for k, v in (f.get("properties") or {}).items():
            if k not in types:
                order.append(k)          # keep first-seen order: stable output
                types[k] = "unknown"
            t = types[k]
            if t == "unknown":
                if isinstance(v, bool):
                    types[k] = "boolean"
                elif isinstance(v, int):
                    types[k] = "integer"
                elif isinstance(v, float):
                    types[k] = "double"
                elif isinstance(v, str):
                    types[k] = "string"
            elif t == "integer" and isinstance(v, float):
                types[k] = "double"      # widen, never narrow
    return [(k, types[k]) for k in order]


def describe_feature_type(type_name: str) -> str:
    features = load_features(type_name).get("features") or []
    props = _property_schema(features)
    xsd = {"string": "xsd:string", "integer": "xsd:int", "double": "xsd:double",
           "boolean": "xsd:boolean", "unknown": "xsd:string"}
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
             'xmlns:gml="http://www.opengis.net/gml/3.2" '
             'targetNamespace="situation3d" elementFormDefault="qualified">',
             '<xsd:element name="%s" type="xsd:anyType" substitutionGroup="gml:AbstractFeature"/>'
             % escape(type_name),
             '<xsd:complexType name="%sType"><xsd:sequence>' % escape(type_name)]
    for name, t in props:
        parts.append('<xsd:element name="%s" type="%s" minOccurs="0" nillable="true"/>'
                     % (escape(name), xsd.get(t, "xsd:string")))
    parts.append('</xsd:sequence></xsd:complexType></xsd:schema>')
    return "".join(parts)


def get_feature(type_name: str, bbox_raw: Optional[str] = None,
                bbox_crs: Optional[str] = None, count_raw: Optional[str] = None,
                max_features: Optional[str] = None,
                start_index_raw: Optional[str] = None,
                property_names: Optional[str] = None,
                ) -> dict:
    """Filtered, paged, server-side-projected GeoJSON FeatureCollection.

    Always returns CRS84 (lon,lat) GeoJSON, whatever the request CRS was: GeoJSON
    is defined that way, and mixing the two orders inside one document is how
    geometry ends up mirrored.
    """
    collection = load_features(type_name)
    features = list(collection.get("features") or [])
    total = len(features)

    box = parse_bbox(bbox_raw, bbox_crs)
    if box:
        features = [f for f in features if _geom_intersects(f.get("geometry") or {}, box)]

    matched = len(features)

    count = parse_count(count_raw, max_features)
    start = parse_start_index(start_index_raw)
    page = features[start:start + count] if count else []

    keep = None
    if property_names:
        keep = {p.strip() for p in str(property_names).split(",") if p.strip()}

    out = []
    for f in page:
        props = f.get("properties") or {}
        if keep:
            props = {k: v for k, v in props.items() if k in keep}
        out.append({"type": "Feature", "geometry": f.get("geometry"), "properties": props})

    types = feature_types()
    meta = types.get(type_name, {})
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": CRS84}},
        "features": out,
        # Not a WFS requirement: carrying the counts makes paging debuggable
        # without re-running the query, and lets the UI say "showing X of Y".
        "numberMatched": matched,
        "numberReturned": len(out),
        "totalAvailable": total,
        "startIndex": start,
        "bboxQuery": list(box) if box else None,
        "attribution": meta.get("attribution"),
        "licence": meta.get("licence"),
    }


def exception_report(message: str, code: str = "InvalidParameterValue") -> str:
    """OGC exception report, as a WFS client expects on a bad request."""
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<ows:ExceptionReport version="2.0.0" '
            'xmlns:ows="http://www.opengis.net/ows/1.1">'
            '<ows:Exception exceptionCode="%s"><ows:ExceptionText>%s</ows:ExceptionText>'
            '</ows:Exception></ows:ExceptionReport>'
            % (escape(code), escape(message)))