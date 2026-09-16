#!/usr/bin/env bash
# Pre-download the high-zoom WSF3D tiles used for the 3D building blocks.
#
# The main asset fetch (tools/fetch_assets.py) covers the whole Taiwan Strait
# at z0-z10, which is the right resolution for a situation display but far too
# coarse to extrude: one z10 pixel is about 150 m. The 3D building blocks need
# the native 90 m WSF3D resolution, so the urban cores listed in
# engine/scenario.py (URBAN_AOI) are fetched separately at z13, where one
# pixel is about 19 m -- finer than the source, so no resampling blur is added.
#
# Run this on the remote container, from the repository root:
#   sh situation3d/tools/fetch_urban_wsf3d.sh
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FETCH="$ROOT/tools/fetch_assets.py"
ZOOM="${ZOOM:-13}"
PY="${PY:-python3}"
IMAGERY_MAX="${IMAGERY_MAX:-14}"

# name<TAB>west,south,east,north -- must match URBAN_AOI in engine/scenario.py
# The boxes are padded by one z12 tile (~0.09 deg) beyond URBAN_AOI in
# engine/scenario.py. Cesium enumerates the imagery tiles covering a surface
# tile and then intersects with the provider rectangle, so the request set can
# reach one tile past the edge -- unpadded, those edge tiles 404 on every load.
set -- \
  "中国台湾省 台北" "121.415,24.883,121.715,25.183" \
  "中国台湾省 高雄" "120.150,22.480,120.450,22.780" \
  "中国 厦门"       "117.940,24.330,118.240,24.630" \
  "中国 福州"       "119.150,25.930,119.450,26.230"

while [ "$#" -ge 2 ]; do
  name="$1"; bbox="$2"; shift 2

  echo "=== $name  $bbox  WSF3D z$ZOOM ==="
  "$PY" "$FETCH" wsf3d \
    --bbox "$bbox" \
    --min-zoom "$ZOOM" \
    --max-zoom "$ZOOM" \
    --wsf3d-layers buildingheight,buildingvolume \
    --workers 8 --no-manifest || echo "  (WSF3D failed for $name)"

  # The main imagery fetch stops at z11, which is fine at theatre scale but
  # turns to mush once the camera drops over a city. These urban boxes are the
  # only places the demo zooms into, so they get z12-z14 as well.
  echo "=== $name  imagery z12-$IMAGERY_MAX ==="
  "$PY" "$FETCH" imagery \
    --bbox "$bbox" \
    --min-zoom 12 \
    --max-zoom "$IMAGERY_MAX" \
    --source esri \
    --workers 8 --no-manifest || echo "  (imagery failed for $name)"
done

# Merge the urban coverage into the primary manifest. These fetches run with
# --no-manifest precisely so they cannot overwrite the theatre-wide bbox and
# zoom ranges the basemap layer depends on.
"$PY" - "$ROOT" "$ZOOM" "$IMAGERY_MAX" <<'PYEOF'
import json, os, sys
root, wsf_zoom, img_max = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
path = os.path.join(root, "web", "tiles", "manifest.json")
man = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
man.setdefault("bbox", [115.0, 18.0, 125.0, 29.0])
man["urban_wsf3d"] = {"min_zoom": wsf_zoom, "max_zoom": wsf_zoom}
man["urban_imagery"] = {"min_zoom": 12, "max_zoom": img_max}
json.dump(man, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("manifest updated:", path)
PYEOF

echo "=== done: urban WSF3D z$ZOOM + imagery z12-$IMAGERY_MAX ==="
