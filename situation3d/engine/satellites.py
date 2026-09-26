"""Satellite overpass queries over the offline TLE catalogue.

What this answers
-----------------
"Which satellites pass over the theatre in the next N hours, when, and how high?"
That is a recurring question for a situational picture -- imaging revisits,
comms windows, and simply "who can see this area right now".

Reads `engine/data/tle/*.tle`, which `tools/fetch_tle.py` downloads once and
commits. **No network at runtime**, matching the rest of the project.

Caching
-------
Parsing ~900 TLEs and building `sgp4.Sat` objects allocates a lot of constants;
doing it per request would make the endpoint unusably slow. Satellites are
cached per file and invalidated on (mtime, size), so a refreshed catalogue is
picked up without a restart.

Honesty about staleness
-----------------------
Element sets decay: a 7-day-old TLE can be minutes off in pass timing. Rather
than hide that, the response carries the catalogue's `fetched_utc` and an
`age_days` per group, plus a `stale` flag past a threshold. The UI is expected to
show it -- an overpass time presented as exact when the elements are old is a
worse failure than showing the caveat.

Deep-space TLEs (period >= 225 min) are reported in `skipped` rather than being
propagated with the wrong model; see engine/sgp4.py.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from . import sgp4

HERE = os.path.dirname(os.path.abspath(__file__))
TLE_DIR = os.path.join(HERE, "data", "tle")

# Past this many days the elements are flagged stale in the response.
STALE_AFTER_DAYS = 3.0

# Groups that contain mostly deep-space objects; still listed, but expect skips.
_CACHE: Dict[str, dict] = {}


def catalogue_path(group: str) -> str:
    return os.path.join(TLE_DIR, "%s.tle" % group)


def available_groups() -> List[str]:
    if not os.path.isdir(TLE_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(TLE_DIR)
                  if f.endswith(".tle") and os.path.getsize(os.path.join(TLE_DIR, f)) > 0)


def _stat_key(path: str) -> Optional[Tuple[float, int]]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime, st.st_size)


def load_group(group: str, max_age_days: Optional[float] = None) -> dict:
    """Load and prepare one group. Returns {sats, skipped, meta, age_days}.

    `sgp4.Sat` construction is the expensive part, so the whole list is cached
    and keyed on the file's (mtime, size).
    """
    path = catalogue_path(group)
    key = _stat_key(path)
    if key is None:
        raise FileNotFoundError("TLE group %r not found at %s -- run "
                                "tools/fetch_tle.py" % (group, path))

    cached = _CACHE.get(group)
    if cached and cached.get("key") == key:
        return cached["value"]

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    tles = sgp4.parse_tles(text)

    sats: List[sgp4.Sat] = []
    skipped: List[dict] = []
    for tle in tles:
        try:
            sats.append(sgp4.Sat(tle))
        except sgp4.TleError as e:
            # Deep-space or decayed elements: record why, do not guess.
            skipped.append({
                "norad_id": tle.norad_id,
                "name": tle.name,
                "period_min": round(tle.period_min, 1),
                "reason": str(e),
            })

    age = (datetime.now(timezone.utc)
           - datetime.fromtimestamp(key[0], timezone.utc)).total_seconds() / 86400.0
    value = {
        "group": group,
        "sats": sats,
        "skipped": skipped,
        "count": len(sats),
        "age_days": round(age, 2),
        "stale": age > (STALE_AFTER_DAYS if max_age_days is None else max_age_days),
        "path": path,
    }
    _CACHE[group] = {"key": key, "value": value}
    return value


def invalidate(group: Optional[str] = None) -> None:
    """Drop the cache (all groups, or one). Useful after a re-fetch."""
    if group is None:
        _CACHE.clear()
    else:
        _CACHE.pop(group, None)


def find_passes(lat: float, lon: float, groups: Optional[Sequence[str]] = None,
                hours: float = 12.0, min_elevation_deg: float = 10.0,
                start: Optional[datetime] = None, step_s: float = 30.0,
                max_per_group: int = 24, alt_km: float = 0.0) -> dict:
    """Passes over one observer point, merged across groups and time-sorted.

    `lat`/`lon` in degrees. Returns a dict with `passes`, per-group metadata and
    the elements that could not be propagated.
    """
    if start is None:
        start = datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)

    wanted = list(groups) if groups else available_groups()
    if not wanted:
        raise FileNotFoundError(
            "no TLE groups found in %s -- run tools/fetch_tle.py" % TLE_DIR)

    all_passes: List[dict] = []
    meta: Dict[str, dict] = {}
    skipped: List[dict] = []
    warnings: List[str] = []

    for group in wanted:
        try:
            g = load_group(group)
        except FileNotFoundError as e:
            warnings.append(str(e))
            continue
        meta[group] = {
            "count": g["count"],
            "age_days": g["age_days"],
            "stale": g["stale"],
        }
        for s in g["skipped"]:
            skipped.append(dict(s, group=group))

        found = 0
        for sat in g["sats"]:
            if found >= max_per_group:
                break
            try:
                ps = sgp4.find_passes(sat, lat, lon, start, hours=hours,
                                      step_s=step_s,
                                      min_elevation_deg=min_elevation_deg,
                                      alt_km=alt_km)
            except sgp4.TleError:
                # Decayed mid-window; skip this object, keep the rest.
                continue
            for p in ps:
                p["group"] = group
                all_passes.append(p)
            found += len(ps)

    all_passes.sort(key=lambda p: p["rise_utc"])
    return {
        "observer": {"lat": lat, "lon": lon, "alt_km": alt_km},
        "window": {
            "start_utc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hours": hours,
            "end_utc": (start + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "min_elevation_deg": min_elevation_deg,
        },
        "passes": all_passes,
        "pass_count": len(all_passes),
        "groups": meta,
        "skipped": skipped[:64],
        "skipped_count": len(skipped),
        "warnings": warnings,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def positions_at(when: datetime, groups: Optional[Sequence[str]] = None,
                 limit: int = 400) -> List[dict]:
    """Sub-satellite points for every propagatable object at `when`.

    Used to draw the constellation in the 3-D view. Capped by `limit` because the
    GEO group alone is 567 objects and the display cannot use them all.
    """
    wanted = list(groups) if groups else available_groups()
    out: List[dict] = []
    for group in wanted:
        try:
            g = load_group(group)
        except FileNotFoundError:
            continue
        for sat in g["sats"]:
            if len(out) >= limit:
                return out
            try:
                lat, lon, alt = sat.geodetic(when)
            except sgp4.TleError:
                continue
            out.append({
                "norad_id": sat.tle.norad_id,
                "name": sat.tle.name or ("NORAD %d" % sat.tle.norad_id),
                "group": group,
                "lat": round(lat, 3),
                "lon": round(lon, 3),
                "alt_km": round(alt, 1),
            })
    return out


if __name__ == "__main__":       # pragma: no cover - manual inspection entry
    import json
    import sys
    lat = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
    lon = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    print(json.dumps(find_passes(lat, lon, hours=6.0), indent=2, ensure_ascii=False)[:4000])