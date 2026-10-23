"""Physical indirect fire (P1.5): ballistic arc, dispersion and blast.

Replaces the scatter-table random walk when ``indirect_fire_model="physics"``:
the gun solution gives a time of flight for the current range, the impact
point is drawn with a Gaussian CEP (observed fire halves the CEP), and
damage comes from a radial fragmentation falloff rather than a CRT lookup.
Targets in the blast radius lose subsystem health (P1.5 components).

The table model (``rules.indirect_fire``) stays as the compatibility path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

DATA = Path(__file__).resolve().parent / "data" / "ballistics_indirect.json"
_DOC = None
G = 9.81
CEP_TO_SIGMA = 1.1774


def doc() -> dict:
    global _DOC
    if _DOC is None:
        _DOC = json.loads(DATA.read_text(encoding="utf-8"))
    return _DOC


def profile_for_weapon(weapon) -> Optional[dict]:
    key = str(getattr(weapon, "weapon_id", "") or "")
    return (doc().get("by_weapon_id") or {}).get(key)


def has_profile(weapon) -> bool:
    return profile_for_weapon(weapon) is not None


def _ammo_count(attacker, key) -> int:
    counter = getattr(attacker, "ammo_count", None)
    if callable(counter):
        try:
            return int(counter(key))
        except (TypeError, ValueError):
            return 0
    try:
        return int((getattr(attacker, "ammo", {}) or {}).get(key, 0))
    except (TypeError, ValueError):
        return 0


def select_weapon(attacker, distance_m: float, ready=None):
    best = None
    best_score = -1.0
    for weapon in getattr(attacker, "weapons", []) or []:
        if getattr(weapon, "kind", None) != "indirect":
            continue
        profile = profile_for_weapon(weapon)
        if profile is None:
            continue
        if _ammo_count(attacker, weapon.key) <= 0:
            continue
        if ready is not None and not ready(weapon):
            continue
        max_range = float(profile.get("max_range_m", 0.0) or 0.0)
        min_range = float(profile.get("min_range_m", 0.0) or 0.0)
        if distance_m < min_range or distance_m > max_range:
            continue
        score = float(profile.get("lethality_infantry", 1.0) or 1.0)
        if best is None or score > best_score:
            best_score = score
            best = (weapon, profile)
    return best


def time_of_flight(profile: dict, distance_m: float) -> Optional[float]:
    velocity = max(1.0, float(profile.get("muzzle_velocity_mps", 500.0)))
    ratio = G * max(0.0, distance_m) / (velocity * velocity)
    if ratio >= 1.0:
        return None  # beyond the ballistic envelope
    theta = 0.5 * math.asin(ratio)  # low arc
    return 2.0 * velocity * math.sin(theta) / G


def solve_impact(profile: dict, distance_m: float, observed: bool,
                 rng) -> Optional[dict]:
    max_range = float(profile.get("max_range_m", 0.0) or 0.0)
    min_range = float(profile.get("min_range_m", 0.0) or 0.0)
    if distance_m < min_range or (max_range > 0 and distance_m > max_range):
        return None
    tof = time_of_flight(profile, distance_m)
    if tof is None:
        return None
    cep = float((profile.get("cep_observed_m") if observed
                 else profile.get("cep_m")) or 60.0)
    max_range = max(1.0, float(profile.get("max_range_m", 10000.0)))
    scale = max(0.3, min(1.5, distance_m / (0.6 * max_range)))
    sigma = cep * scale / CEP_TO_SIGMA
    along = rng.gauss(0.0, sigma)
    across = rng.gauss(0.0, sigma)
    return {"along_m": along, "across_m": across, "tof_s": tof,
            "cep_m": cep, "sigma_m": sigma}


def blast_damage(profile: dict, distance_m: float, target_kind: str,
                 armor_class: int) -> int:
    radius = float(profile.get("blast_radius_m", 50.0) or 50.0)
    if distance_m > radius:
        return 0
    falloff = max(0.0, 1.0 - (distance_m / max(1.0, radius)) ** 1.5)
    if target_kind == "infantry":
        base = float(profile.get("lethality_infantry", 2.0) or 0.0)
        factor = 1.0
    else:
        base = float(profile.get("lethality_vehicle", 1.0) or 0.0)
        armor_factor = float(profile.get("armor_factor", 0.25) or 0.0)
        factor = max(0.2, 1.0 - armor_factor * max(0, int(armor_class)))
    raw = base * falloff * factor
    if raw <= 0.0:
        return 0
    return max(1, int(round(raw)))


def component_fraction(profile: dict, damage: int) -> float:
    """Fractional subsystem loss per damaging blast (vehicles)."""
    if damage <= 0:
        return 0.0
    return min(1.0, 0.20 * damage)
