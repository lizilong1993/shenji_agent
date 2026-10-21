"""Rule-based intent inference (current-scheme baseline).

This is the deterministic alternative to a JEV intent model: it uses only
kinematics, recent fire events and objective ownership.  It is intentionally
simple and explainable; the JEV comparison uses the same windows.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from geo.geodesy import haversine_m


def _entity(obs: dict, uid: int) -> Optional[dict]:
    for e in obs.get("entities", []):
        if int(e.get("id")) == int(uid):
            return e
    return None


def _xy(entity: dict) -> Tuple[float, float]:
    pos = entity.get("pos") or {}
    return float(pos.get("lon", 0.0)), float(pos.get("lat", 0.0))


def _fired(events: Iterable[dict], uid: int) -> bool:
    for event in events or []:
        if str(event.get("type")) not in ("direct_fire", "sea_air_fire"):
            continue
        if int(uid) in (event.get("actors") or []):
            return True
    return False


def _nearest_objective(observation: dict, lon: float, lat: float):
    best = None
    best_d = None
    for obj in observation.get("objectives", []):
        pos = obj.get("pos") or {}
        d = haversine_m((lon, lat), (float(pos.get("lon", 0.0)),
                                     float(pos.get("lat", 0.0))))
        if best_d is None or d < best_d:
            best_d = d
            best = (d, obj)
    return best


class RuleIntentClassifier(object):
    def __init__(self, labels: Optional[Sequence[str]] = None):
        self.labels = tuple(labels or ("assault", "defend", "recon", "retreat",
                                       "reinforce", "hold", "fire_support",
                                       "resupply", "feint"))

    def features(self, history: List[dict], events: List[dict],
                 target_uid: int, reference_uid: int) -> Dict[str, float]:
        target_snaps = [(_entity(obs, target_uid), obs.get("time_s", 0.0))
                        for obs in history]
        target_snaps = [(e, t) for e, t in target_snaps if e is not None]
        if not target_snaps:
            return {"speed": 0.0, "closing": 0.0, "fired": 0.0,
                    "objective_dist": 1e9, "objective_owned": 0.0}
        first, t0 = target_snaps[0]
        last, t1 = target_snaps[-1]
        ref = _entity(history[-1], reference_uid)
        if ref is None:
            return {"speed": float(last.get("speed_ms", 0.0)), "closing": 0.0,
                    "fired": 0.0, "objective_dist": 1e9, "objective_owned": 0.0}
        x0, y0 = _xy(first)
        x1, y1 = _xy(last)
        rx, ry = _xy(ref)
        d0 = haversine_m((x0, y0), (rx, ry))
        d1 = haversine_m((x1, y1), (rx, ry))
        dt = max(1e-6, float(t1) - float(t0))
        closing = (d0 - d1) / dt
        # Lateral speed relative to the bearing to the reference.
        m_lon = 111320.0 * math.cos(math.radians(ry))
        dx, dy = (x1 - x0) * m_lon, (y1 - y0) * 110540.0
        ux, uy = ((rx - x0) * m_lon, (ry - y0) * 110540.0)
        un = math.hypot(ux, uy) or 1.0
        lateral = abs(dx * (uy / un) - dy * (ux / un)) / dt
        nearest = _nearest_objective(history[-1], *_xy(last))
        obj_owned = 0.0
        obj_dist = 1e9
        if nearest is not None:
            obj_dist, obj = nearest
            if obj.get("owner") == last.get("affiliation"):
                obj_owned = 1.0
        return {"speed": float(last.get("speed_ms", 0.0)),
                "closing": closing,
                "lateral": lateral,
                "fired": 1.0 if _fired(events, target_uid) else 0.0,
                "objective_dist": obj_dist,
                "objective_owned": obj_owned}

    def classify(self, history: List[dict], events: List[dict],
                 target_uid: int, reference_uid: int) -> str:
        f = self.features(history, events, target_uid, reference_uid)
        label = "hold"
        if f["fired"] and f["closing"] >= 0:
            label = "assault"
        elif f["speed"] < 0.6 and f["objective_owned"] and f["objective_dist"] < 1500:
            label = "defend"
        elif f["closing"] < -0.5 and f.get("lateral", 0.0) < 0.8:
            label = "retreat"
        elif f["speed"] > 0.5 and (f.get("lateral", 0.0) >= 0.5
                                    or abs(f["closing"]) < 0.8):
            label = "recon"
        elif f["objective_owned"] and f["objective_dist"] < 2000:
            label = "reinforce"
        return label if label in self.labels else "hold"
