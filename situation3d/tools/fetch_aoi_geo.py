#!/usr/bin/env python3
"""Fetch a small OSM geo package (buildings/water/coastline) for a campaign AOI.

Standard library only. Usage:
  python3 tools/fetch_aoi_geo.py --theatre taiwan_strait --aoi anping \
      --bbox 120.135,22.945,120.225,22.998
"""
import argparse, hashlib, json, os, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
GEO_ROOT = os.path.join(ROOT, "engine", "data", "geo")

OVERPASS = "https://overpass-api.de/api/interpreter"


def fetch(bbox, timeout=180):
    w, s, e, n = bbox
    q = ("[out:json][timeout:%d];(way(%.6f,%.6f,%.6f,%.6f)[building];"
         "way(%.6f,%.6f,%.6f,%.6f)[\"building:part\"];"
         "way(%.6f,%.6f,%.6f,%.6f)[natural=water];"
         "way(%.6f,%.6f,%.6f,%.6f)[waterway=riverbank];"
         "way(%.6f,%.6f,%.6f,%.6f)[natural=coastline];);out geom;"
         % ((timeout,) + (s, w, n, e) * 5))
    data = urllib.parse.urlencode({"data": q}).encode()
    req = urllib.request.Request(OVERPASS, data=data,
                                 headers={"User-Agent": "situation3d-demo/1.0"})
    with urllib.request.urlopen(req, timeout=timeout + 30) as r:
        return json.loads(r.read().decode("utf-8"))


def ring(geom, bbox):
    w, s, e, n = bbox
    out = []
    for p in geom:
        lat, lon = p["lat"], p["lon"]
        if not (w <= lon <= e and s <= lat <= n):
            return None
        out.append([round(lon, 7), round(lat, 7)])
    return out if len(out) >= 4 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theatre", default="taiwan_strait")
    ap.add_argument("--aoi", default="anping")
    ap.add_argument("--bbox", required=True, help="west,south,east,north")
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args()
    bbox = [float(x) for x in a.bbox.split(",")]
    doc = fetch(bbox, a.timeout)
    out_dir = os.path.join(GEO_ROOT, a.theatre, a.aoi)
    os.makedirs(out_dir, exist_ok=True)
    buildings, water, coast = [], [], []
    for el in doc.get("elements") or []:
        tags = el.get("tags") or {}
        geom = el.get("geometry") or []
        if not geom:
            continue
        if "building" in tags or "building:part" in tags:
            r = ring(geom, bbox)
            if r:
                buildings.append({"type": "Feature",
                                  "properties": {"id": el["id"], "building": tags.get("building") or tags.get("building:part")},
                                  "geometry": {"type": "Polygon", "coordinates": [r]}})
        elif tags.get("natural") == "water" or tags.get("waterway") == "riverbank":
            r = ring(geom, bbox)
            if r:
                water.append({"type": "Feature", "properties": {"id": el["id"]},
                              "geometry": {"type": "Polygon", "coordinates": [r]}})
        elif tags.get("natural") == "coastline":
            pts = [[round(p["lon"], 7), round(p["lat"], 7)] for p in geom
                   if bbox[0] <= p["lon"] <= bbox[2] and bbox[1] <= p["lat"] <= bbox[3]]
            if len(pts) >= 2:
                coast.append({"type": "Feature", "properties": {"id": el["id"]},
                              "geometry": {"type": "LineString", "coordinates": pts}})

    def write(name, feats):
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"type": "FeatureCollection", "features": feats}, fh,
                      ensure_ascii=False, separators=(",", ":"))
        return {"file": name, "features": len(feats),
                "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                "bytes": os.path.getsize(p)}

    manifest = {
        "schema": "situation3d.geo_package.v1", "aoi": a.aoi,
        "theatre": a.theatre, "bbox": bbox,
        "source": "OpenStreetMap via Overpass API", "license": "ODbL 1.0",
        "snapshot_osm_base": doc.get("osm3s", {}).get("timestamp_osm_base"),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [write("buildings.geojson", buildings),
                  write("water.geojson", water),
                  write("coastline.geojson", coast)],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
