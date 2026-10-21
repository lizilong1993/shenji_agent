"""ESRI ASCII grid helpers shared by the DEM and bathymetry sources.

The engine avoids GDAL at runtime: sources extract a small window with
``gdal_translate`` once, cache the ASCII grid, and sample it with plain
Python bilinear interpolation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

HEADER_KEYS = ("ncols", "nrows", "xllcorner", "xllcenter", "yllcorner",
               "yllcenter", "cellsize", "nodata_value")


def parse_aaigrid(text: str) -> Dict:
    """Parse an ESRI ASCII grid into ``{cols, rows, xll, yll, cell, values}``."""
    header = {}
    lines = text.strip().splitlines()
    idx = 0
    while idx < len(lines):
        parts = lines[idx].split()
        if not parts:
            idx += 1
            continue
        key = parts[0].lower()
        if key not in HEADER_KEYS:
            break
        header[key] = float(parts[1])
        idx += 1
    for key in ("ncols", "nrows", "cellsize"):
        if key not in header:
            raise ValueError("AAIGrid missing header %r" % key)
    cols, rows = int(header["ncols"]), int(header["nrows"])
    cell = float(header["cellsize"])
    xll = header.get("xllcorner", header.get("xllcenter", 0.0) - cell / 2.0)
    yll = header.get("yllcorner", header.get("yllcenter", 0.0) - cell / 2.0)
    nodata = header.get("nodata_value", -9999.0)
    values = []
    for line in lines[idx:]:
        parts = line.split()
        if not parts:
            continue
        values.append([float(p) for p in parts])
    if len(values) != rows:
        raise ValueError("AAIGrid row count %d != header %d" % (len(values), rows))
    return {"cols": cols, "rows": rows, "xll": xll, "yll": yll,
            "cell": cell, "nodata": nodata, "values": values}


def bilinear(grid: Dict, lon: float, lat: float) -> Optional[float]:
    """Bilinear sample at (lon, lat); ``None`` outside the grid or on nodata."""
    cell = grid["cell"]
    fx = (lon - grid["xll"]) / cell - 0.5
    fy = (lat - grid["yll"]) / cell - 0.5
    row = (grid["rows"] - 1) - fy
    eps = 1e-9
    if (fx < -0.5 - eps or row < -0.5 - eps
            or fx > grid["cols"] - 0.5 + eps
            or row > grid["rows"] - 0.5 + eps):
        return None
    fx = min(max(fx, 0.0), grid["cols"] - 1.0)
    row = min(max(row, 0.0), grid["rows"] - 1.0)
    c0, r0 = int(math.floor(fx)), int(math.floor(row))
    c1 = min(grid["cols"] - 1, c0 + 1)
    r1 = min(grid["rows"] - 1, r0 + 1)
    tx, ty = fx - c0, row - r0
    v = grid["values"]
    corners = (v[r0][c0], v[r0][c1], v[r1][c0], v[r1][c1])
    if any(c == grid["nodata"] for c in corners):
        return None
    top = corners[0] * (1 - tx) + corners[1] * tx
    bottom = corners[2] * (1 - tx) + corners[3] * tx
    return top * (1 - ty) + bottom * ty


def sample_regular_grid(grid: Dict, bbox, step_deg: float) -> Dict:
    """Resample ``grid`` onto the engine's regular lon/lat grid shape."""
    lon0, lat0, lon1, lat1 = [float(v) for v in bbox]
    cols = max(2, int(math.ceil((lon1 - lon0) / step_deg)) + 1)
    rows = max(2, int(math.ceil((lat1 - lat0) / step_deg)) + 1)
    values = []
    for r in range(rows):
        lat = lat0 + (lat1 - lat0) * r / float(rows - 1)
        row = []
        for c in range(cols):
            lon = lon0 + (lon1 - lon0) * c / float(cols - 1)
            value = bilinear(grid, lon, lat)
            if value is None:
                # nearest-valid fallback: first ring of valid cells
                value = _nearest_valid(grid, lon, lat)
            row.append(None if value is None else round(value, 1))
        values.append(row)
    return {"lon0": lon0, "lat0": lat0, "lon1": lon1, "lat1": lat1,
            "cols": cols, "rows": rows, "values": values}


def _nearest_valid(grid: Dict, lon: float, lat: float) -> Optional[float]:
    cell = grid["cell"]
    c = int(round((lon - grid["xll"]) / cell - 0.5))
    r = grid["rows"] - 1 - int(round((lat - grid["yll"]) / cell - 0.5))
    r = min(max(r, 0), grid["rows"] - 1)
    c = min(max(c, 0), grid["cols"] - 1)
    for radius in range(1, 6):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid["rows"] and 0 <= cc < grid["cols"]:
                    value = grid["values"][rr][cc]
                    if value != grid["nodata"]:
                        return value
    return None
