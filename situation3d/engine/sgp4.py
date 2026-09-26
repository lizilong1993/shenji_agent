"""SGP4 (near-Earth) propagation + pass prediction, standard library only.

Why hand-written
----------------
The container has python3.8 and no pip, so `pip install sgp4` is unavailable. A
circular-orbit shortcut is not an acceptable substitute: on eccentric or inclined
orbits it misplaces the ground track by hundreds of km -- the difference between
"overhead in 4 minutes" and "overhead in 40".

Scope, and what it is not
-------------------------
Implements the near-Earth SGP4 of Spacetrack Report #3 (the algorithm
`python-sgp4` wraps): J2/J3/J4 secular gravity, atmospheric drag, and the
short-period periodics. Two deliberate exclusions:

* **No SDP4.** Periods >= 225 min need the deep-space model. Those TLEs raise
  `TleError` instead of being quietly propagated with the wrong model -- a GEO
  satellite run through near-Earth SGP4 returns confident nonsense, and confident
  nonsense is worse than a refusal.
* **TEME treated as the inertial frame** when rotating to Earth-fixed; polar
  motion and the equinox offset (~1") are ignored. Far below the error already
  present in a day-old TLE, so fine for pass planning, **not** for antenna
  pointing.

Verification
------------
`verify_reference()` reproduces the published VANGUARD 1 case (NORAD 00005, the
canonical SGP4 test object) and compares against the WGS-72 TEME position from
the reference implementation:

    tsince = 309.67110720001529 min  ->  (-3754.2514743216166,
                                           7876.346817439062,
                                           4719.220856478582) km

`python3 -m engine.sgp4` runs it. If that check does not pass, nothing built on
top of this module should be trusted.

Units: SGP4 works internally in earth radii and minutes; this module exposes
kilometres, degrees and UTC.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

# --- WGS-72 constants, as SGP4 is defined against them -----------------------
# Do NOT substitute WGS-84 values. TLE mean elements are fitted in the WGS-72
# system, and mixing the two introduces an error larger than the terms computed
# (the reference implementation's own WGS-72 vs WGS-84 outputs differ by ~8 m,
# but a wrong J2/CK2 differ by far more).
XKE = 0.0743669161                  # sqrt(GM) in earth-radii^1.5 / min
J2 = 1.082616e-3
J3 = -2.53881e-6
J4 = -1.65597e-6
CK2 = 0.5 * J2
CK4 = -0.375 * J4
QOMS2T = 1.88027916e-9
S_CONST = 1.01222928               # earth radii; (1 + 78/6378.135)
XKMPER = 6378.135                  # km per earth radius
MIN_PER_DAY = 1440.0
TWOPI = 2.0 * math.pi
DEG = 180.0 / math.pi
# Earth rotation rate, rad/min (sidereal). 2*pi / 1436.0681752 min.
OMEGA_EARTH = TWOPI / 1436.0681752

# WGS-72 ellipsoid, for geodetic latitude / altitude.
WGS72_A = 6378.135
WGS72_F = 1.0 / 298.26
WGS72_E2 = WGS72_F * (2.0 - WGS72_F)


class TleError(ValueError):
    """Malformed TLE, or one outside the model this module implements."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _implied_exp(s: str, field: str = "") -> float:
    """Parse an 'implied decimal point' field, e.g. ' 00000+0' / '-11606-4'.

    There is no decimal point in the format: the leading digit run is the
    mantissa with an implied '0.' prefix, the trailing signed integer is the
    exponent.
    """
    t = (s or "").strip()
    if not t:
        return 0.0                       # blank means zero, e.g. 'nddot'
    try:
        return float(t)                  # some producers emit a plain float
    except ValueError:
        pass
    sign = 1.0
    if t[0] in "+-":
        sign = -1.0 if t[0] == "-" else 1.0
        t = t[1:]
    split = -1
    for i in range(1, len(t)):
        if t[i] in "+-":
            split = i
    if split < 0:
        raise TleError("cannot parse numeric field %s=%r" % (field, s))
    try:
        return sign * float("0." + t[:split]) * (10.0 ** int(t[split:]))
    except ValueError:
        raise TleError("cannot parse numeric field %s=%r" % (field, s))


def _parse_epoch(raw: str) -> datetime:
    """TLE epoch YYDDD.DDDDDDDD -> UTC datetime."""
    s = (raw or "").strip()
    if len(s) < 5:
        raise TleError("bad epoch %r" % raw)
    try:
        yy = int(s[:2])
        doy = float(s[2:])
    except ValueError:
        raise TleError("bad epoch %r" % raw)
    year = 1900 + yy if yy >= 57 else 2000 + yy      # TLE two-digit year window
    if not (1.0 <= doy < 367.0):
        raise TleError("epoch day-of-year out of range: %r" % raw)
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1.0)


class Tle(object):
    """One two-line element set."""

    __slots__ = ("name", "norad_id", "epoch", "inclo", "nodeo", "ecco", "argpo",
                 "mo", "no_kozai", "bstar", "ndot", "intl_designator",
                 "classification", "line1", "line2")

    def __init__(self, name: str, line1: str, line2: str):
        self.name = (name or "").strip()
        self.line1 = (line1 or "").rstrip()
        self.line2 = (line2 or "").rstrip()
        if not self.line1.startswith("1 ") or not self.line2.startswith("2 "):
            raise TleError("not a TLE pair for %r" % (self.name or "?"))
        if len(self.line1) < 61 or len(self.line2) < 63:
            raise TleError("TLE lines too short for %r" % (self.name or "?"))
        try:
            self.norad_id = int(self.line1[2:7])
            self.inclo = math.radians(float(self.line2[8:16]))
            self.nodeo = math.radians(float(self.line2[17:25]))
            self.ecco = float("0." + self.line2[26:33].strip())
            self.argpo = math.radians(float(self.line2[34:42]))
            self.mo = math.radians(float(self.line2[43:51]))
            self.no_kozai = float(self.line2[52:63]) * TWOPI / MIN_PER_DAY
        except ValueError:
            raise TleError("bad numeric field in %r / %r" % (self.line1, self.line2))
        self.classification = self.line1[7:8].strip() or "U"
        self.intl_designator = self.line1[9:17].strip()
        self.epoch = _parse_epoch(self.line1[18:32])
        self.ndot = _implied_exp(self.line1[33:43], "ndot")
        self.bstar = _implied_exp(self.line1[53:61], "bstar")
        if not (0.0 <= self.ecco < 1.0):
            raise TleError("eccentricity out of range for %s" % (self.name or self.norad_id))
        if self.no_kozai <= 0.0:
            raise TleError("non-positive mean motion for %s" % (self.name or self.norad_id))

    @property
    def period_min(self) -> float:
        return TWOPI / self.no_kozai

    @property
    def is_deep_space(self) -> bool:
        return self.period_min >= 225.0

    @property
    def altitude_km(self) -> float:
        """Mean altitude above the WGS-72 equatorial radius, from semi-major axis."""
        a = (XKE / self.no_kozai) ** (2.0 / 3.0) * XKMPER
        return a - XKMPER

    def __repr__(self) -> str:
        return "<Tle %s #%d %s>" % (self.name or "?", self.norad_id,
                                    self.epoch.strftime("%Y-%m-%dT%H:%M:%SZ"))


def parse_tles(text: str) -> List[Tle]:
    """Parse 3-line (name + 2) and/or bare 2-line TLE records, skipping bad ones."""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    out: List[Tle] = []
    i = 0
    while i < len(lines):
        a = lines[i]
        b = lines[i + 1] if i + 1 < len(lines) else ""
        c = lines[i + 2] if i + 2 < len(lines) else ""
        if a.startswith("1 ") and b.startswith("2 "):
            try:
                out.append(Tle("", a, b))
            except TleError:
                pass
            i += 2
            continue
        if b.startswith("1 ") and c.startswith("2 "):
            try:
                out.append(Tle(a, b, c))
            except TleError:
                pass
            i += 3
            continue
        i += 1
    return out


# ---------------------------------------------------------------------------
# SGP4
# ---------------------------------------------------------------------------

class Sat(object):
    """A TLE prepared for propagation. Construct once, propagate many times."""

    def __init__(self, tle: Tle):
        if tle.is_deep_space:
            raise TleError(
                "%s has a %.0f min period (>= 225 min): needs SDP4, which this "
                "module does not implement" % (tle.name or tle.norad_id, tle.period_min))
        self.tle = tle
        eo, inclo = tle.ecco, tle.inclo
        self.cosio = math.cos(inclo)
        self.sinio = math.sin(inclo)
        theta2 = self.cosio * self.cosio
        x3thm1 = 3.0 * theta2 - 1.0
        self.x3thm1 = x3thm1
        self.x1mth2 = 1.0 - theta2
        self.x7thm1 = 7.0 * theta2 - 1.0
        self.betao2 = 1.0 - eo * eo
        self.betao = math.sqrt(self.betao2)
        self.ecco = eo

        # Remove the J2 secular term carried by the TLE's mean motion, to get the
        # un-Kozai'd element SGP4 is actually defined on.
        ao = (XKE / tle.no_kozai) ** (2.0 / 3.0)
        delo = 1.5 * CK2 * x3thm1 / (ao * ao * self.betao * self.betao2)
        self.no_unkozai = tle.no_kozai / (1.0 + delo)
        self.a = (XKE / self.no_unkozai) ** (2.0 / 3.0)          # earth radii

        # Drag terms.
        self.tsi = 1.0 / (self.a - S_CONST)
        self.eta = self.a * eo * self.tsi
        self.etasq = self.eta * self.eta
        self.eeta = eo * self.eta
        self.psisq = abs(1.0 - self.etasq)
        if self.psisq < 1.0e-12:
            raise TleError("orbit too close to the drag singularity for %s"
                           % (tle.name or tle.norad_id))
        common = (self.a * (1.0 + 1.5 * self.etasq + self.eeta * (4.0 + self.etasq))
                  + 0.375 * CK2 * self.tsi / self.psisq * x3thm1
                  * (8.0 + 3.0 * self.etasq * (8.0 + self.etasq)))
        self.c2 = QOMS2T * self.tsi ** 4 / self.psisq ** 3.5 * self.no_unkozai * common
        self.c1 = tle.bstar * self.c2
        self.c3 = 0.0
        self.c4 = 0.0
        self.c5 = 0.0
        if eo > 1.0e-4:
            self.c3 = self.c2 * self.tsi
            self.c4 = 2.0 * self.no_unkozai * self.tsi ** 4 * self.psisq ** -3.5 * common
            self.c5 = (2.0 * self.tsi ** 3.5 * self.psisq ** -3.5
                       * (1.0 + 2.75 * (self.etasq + self.eeta)
                          + self.eeta * self.etasq))

        # Secular rates (rad/min).
        pinvsq = 1.0 / (self.a * self.a * self.betao2 * self.betao2)
        temp1 = 1.5 * J2 * pinvsq * self.no_unkozai
        temp2 = 0.5 * temp1 * J2 * pinvsq
        temp3 = -0.46875 * J4 * pinvsq * pinvsq * self.no_unkozai
        self.mdot = (self.no_unkozai + 0.5 * temp1 * self.betao * x3thm1
                     + 0.0625 * temp2 * self.betao
                     * (13.0 - 78.0 * theta2 + 137.0 * theta2 * theta2))
        self.argpdot = (-0.5 * temp1 * (1.0 - 5.0 * theta2)
                        + 0.0625 * temp2 * (7.0 - 114.0 * theta2 + 395.0 * theta2 * theta2)
                        + temp3 * (3.0 - 36.0 * theta2 + 49.0 * theta2 * theta2))
        xhdot1 = -temp1 * self.cosio
        self.nodedot = (xhdot1
                        + (0.5 * temp2 * (4.0 - 19.0 * theta2)
                           + 2.0 * temp3 * (3.0 - 7.0 * theta2)) * self.cosio)
        self.xpidot = self.argpdot + self.nodedot
        self.omgcof = tle.bstar * self.c3 * math.cos(tle.argpo)
        # Long-period periodic coefficient for the mean anomaly. Zero for
        # near-circular orbits, where the 1/e factor would blow up.
        self.xmcof = (-2.0 * CK2 * x3thm1 / (eo * self.betao2)) if eo > 1.0e-4 else 0.0
        self.nodecf = tle.bstar * self.c3 * self.sinio * math.cos(tle.nodeo)
        self.t2cof = 1.5 * self.c1
        if abs(self.cosio + 1.0) > 1.5e-12:
            self.xlcof = (-0.25 * J3 / J2 * self.sinio
                          * (3.0 + 5.0 * self.cosio) / (1.0 + self.cosio))
        else:
            # Retrograde equatorial: the (1+cos i) factor vanishes. The standard
            # clamps the denominator rather than letting the term diverge.
            self.xlcof = -0.25 * J3 / J2 * self.sinio * (3.0 + 5.0 * self.cosio) / 1.5e-12
        self.aycof = -0.5 * J3 / J2 * self.sinio
        # Flags a satellite as decayed once perigee drops below the surface.
        self.perigee_km = (self.a * (1.0 - eo) - 1.0) * XKMPER

    # -- propagation --------------------------------------------------------

    def position_eci(self, dt_min: float, allow_decay: bool = False
                     ) -> Tuple[float, float, float]:
        """TEME position in km at dt_min minutes after the TLE epoch."""
        t = self.tle
        t2 = dt_min * dt_min
        # Secular elements advanced to the requested time.
        xmdf = t.mo + self.mdot * dt_min
        argpdf = t.argpo + self.argpdot * dt_min
        nodedf = t.nodeo + self.nodedot * dt_min
        # Drag: the node picks up a t^2 term. For the near-Earth, non-resonant
        # case the higher-order t^3..t^5 corrections vanish, so the mean
        # eccentricity and semi-major axis are taken at their epoch values.
        nodem = (nodedf + self.nodecf * t2) % TWOPI
        argpm = argpdf % TWOPI
        mm = xmdf
        if not allow_decay and (1.0 - self.c1 * dt_min) <= 0.0:
            raise TleError("satellite %s has decayed (drag term drives a <= 0 at "
                           "t=%.1f min)" % (t.name or t.norad_id, dt_min))

        # Fold the node and argument of perigee into the mean longitude, then
        # recover the mean anomaly modulo 2*pi. Doing it this way keeps the
        # Kepler iteration's starting value well-conditioned for long dt.
        xlm = mm + argpm + nodem
        nodem = nodem % TWOPI
        argpm = argpm % TWOPI
        xlm = xlm % TWOPI
        mm = (xlm - argpm - nodem) % TWOPI

        sinim = self.sinio
        cosim = self.cosio

        # Long-period periodics (eq. 3-46 .. 3-48).
        axnl = self.ecco * math.cos(argpm)
        temp = 1.0 / (self.a * (1.0 - self.ecco * self.ecco))
        aynl = self.ecco * math.sin(argpm) + temp * self.aycof
        xl = mm + argpm + nodem + temp * self.xlcof * axnl

        # Kepler's equation for the eccentric anomaly.
        u = (xl - nodem) % TWOPI
        eo1 = u
        for _ in range(10):
            sineo1 = math.sin(eo1)
            coseo1 = math.cos(eo1)
            tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl
            if abs(tem5) < 1.0e-12:
                break
            tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5
            if abs(tem5) >= 0.95:
                tem5 = 0.95 if tem5 > 0 else -0.95
            eo1 += tem5
            if abs(tem5) < 1.0e-12:
                break

        # Short-period periodics (eq. 3-50 .. 3-54).
        ecose = axnl * math.cos(eo1) + aynl * math.sin(eo1)
        esine = axnl * math.sin(eo1) - aynl * math.cos(eo1)
        el2 = axnl * axnl + aynl * aynl
        pl = self.a * (1.0 - el2)
        if pl < 0.0:
            raise TleError("semilatus rectum is negative for %s at t=%.1f min"
                           % (t.name or t.norad_id, dt_min))
        rl = self.a * (1.0 - ecose)
        rdotl = math.sqrt(self.a) * esine / rl * XKE
        rvdotl = math.sqrt(pl) / rl * XKE
        betal = math.sqrt(max(1.0e-12, 1.0 - el2))
        temp = esine / (1.0 + betal)
        sinu = self.a / rl * (math.sin(eo1) - aynl - axnl * temp)
        cosu = self.a / rl * (math.cos(eo1) - axnl + aynl * temp)
        su = math.atan2(sinu, cosu)
        sin2u = 2.0 * sinu * cosu
        cos2u = 1.0 - 2.0 * sinu * sinu
        temp = 1.0 / pl
        temp1 = CK2 * temp
        temp2 = temp1 * temp

        # Update for short-period periodics.
        mrt = rl * (1.0 - 1.5 * temp2 * betal * self.x3thm1) + 0.5 * temp1 * self.x1mth2 * cos2u
        if mrt < 1.0:
            raise TleError("satellite %s has decayed (mrt %.6f < 1.0 at t=%.1f min)"
                           % (t.name or t.norad_id, mrt, dt_min))
        su = su - 0.25 * temp2 * self.x7thm1 * sin2u
        xnode = nodem + 1.5 * temp2 * cosim * sin2u
        xinc = self.tle.inclo + 1.5 * temp2 * cosim * sinim * cos2u
        mvt = rdotl - self.no_unkozai * temp1 * self.x1mth2 * sin2u / XKE
        rvdot = rvdotl + self.no_unkozai * temp1 * (
            self.x1mth2 * cos2u + 1.5 * self.x3thm1) / XKE

        # Orientation vectors.
        sinsu = math.sin(su)
        cossu = math.cos(su)
        snod = math.sin(xnode)
        cnod = math.cos(xnode)
        sini = math.sin(xinc)
        cosi = math.cos(xinc)
        xmx = -snod * cosi
        xmy = cnod * cosi
        ux = xmx * sinsu + cnod * cossu
        uy = xmy * sinsu + snod * cossu
        uz = sini * sinsu
        vx = xmx * cossu - cnod * sinsu
        vy = xmy * cossu - snod * sinsu
        vz = sini * cossu

        r = mrt * XKMPER
        return ((r * ux), (r * uy), (r * uz))

    def position_ecef(self, when: datetime) -> Tuple[float, float, float]:
        """Earth-fixed position in km."""
        dt = minutes_since_epoch(self.tle, when)
        x, y, z = self.position_eci(dt)
        return eci_to_ecef(x, y, z, dt)

    def geodetic(self, when: datetime) -> Tuple[float, float, float]:
        """(lat_deg, lon_deg, alt_km) at `when`, WGS-72 geodetic."""
        x, y, z = self.position_ecef(when)
        return ecef_to_geodetic(x, y, z)


def minutes_since_epoch(tle: Tle, when: datetime) -> float:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (when - tle.epoch).total_seconds() / 60.0


def eci_to_ecef(x: float, y: float, z: float, dt_min: float
                ) -> Tuple[float, float, float]:
    """Rotate TEME into Earth-fixed coordinates.

    TEME is treated as inertial (see the module docstring); the rotation is the
    Greenwich sidereal angle propagated from the TLE epoch at the sidereal rate.
    """
    g = OMEGA_EARTH * dt_min
    cg, sg = math.cos(g), math.sin(g)
    return (x * cg + y * sg, -x * sg + y * cg, z)


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """ECEF km -> (lat_deg, lon_deg, alt_km) on the WGS-72 ellipsoid.

    Uses Bowring's closed-form approximation, then two Newton refinements -- it
    converges to sub-metre in this altitude range, and unlike a single-step
    spherical conversion it stays accurate near the poles.
    """
    a = WGS72_A
    e2 = WGS72_E2
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    if p < 1.0e-9:                       # on the spin axis
        b = a * math.sqrt(1.0 - e2)
        return (90.0 if z >= 0 else -90.0, 0.0, abs(z) - b)
    # Bowring's initial guess.
    theta = math.atan2(z * a, p * (a * math.sqrt(1.0 - e2)))
    lat = math.atan2(z + e2 * a / math.sqrt(1.0 - e2) * math.sin(theta) ** 3,
                     p - e2 * a * math.cos(theta) ** 3)
    for _ in range(3):
        sinlat = math.sin(lat)
        rn = a / math.sqrt(1.0 - e2 * sinlat * sinlat)
        alt = p / math.cos(lat) - rn
        lat = math.atan2(z, p * (1.0 - e2 * rn / (rn + alt)))
    sinlat = math.sin(lat)
    rn = a / math.sqrt(1.0 - e2 * sinlat * sinlat)
    alt = p / math.cos(lat) - rn
    return (math.degrees(lat), math.degrees(lon), alt)


# ---------------------------------------------------------------------------
# Ground geometry
# ---------------------------------------------------------------------------

def observer_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0
                  ) -> Tuple[float, float, float]:
    """Geodetic observer -> ECEF km, WGS-72."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sinlat, coslat = math.sin(lat), math.cos(lat)
    rn = WGS72_A / math.sqrt(1.0 - WGS72_E2 * sinlat * sinlat)
    return ((rn + alt_km) * coslat * math.cos(lon),
            (rn + alt_km) * coslat * math.sin(lon),
            (rn * (1.0 - WGS72_E2) + alt_km) * sinlat)


def elevation_deg_at(lat_deg: float, lon_deg: float, alt_km: float,
                     sat_ecef: Sequence[float]) -> float:
    """Elevation of the satellite above the observer's local horizon, degrees.

    "Up" is the **geodetic** (ellipsoid) normal, which is what a horizon means on
    a flattened Earth. Using the geocentric radius instead would bias the
    elevation by up to ~0.2 deg at mid latitudes -- enough to move a 10-degree
    mask crossing by tens of seconds, which is the accuracy this is for.
    """
    obs = observer_ecef(lat_deg, lon_deg, alt_km)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    up = (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))
    dx = sat_ecef[0] - obs[0]
    dy = sat_ecef[1] - obs[1]
    dz = sat_ecef[2] - obs[2]
    rng = math.sqrt(dx * dx + dy * dy + dz * dz)
    if rng < 1.0e-9:
        return 90.0
    s = (dx * up[0] + dy * up[1] + dz * up[2]) / rng
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def find_passes(sat: Sat, lat: float, lon: float, start: datetime,
                hours: float = 24.0, step_s: float = 30.0,
                min_elevation_deg: float = 10.0, alt_km: float = 0.0,
                max_passes: int = 32) -> List[dict]:
    """Passes above `min_elevation_deg` for an observer, over the next `hours`.

    Scans at `step_s` and refines each crossing by bisection, because a 30 s step
    can be several degrees of elevation wide near the horizon -- reporting the
    step time as the rise time would be off by tens of seconds.

    A "pass" is one contiguous interval above the threshold. Peak elevation and
    its time are found by a coarse scan plus parabolic refinement, which is ample
    given TLE accuracy.
    """
    obs = observer_ecef(lat, lon, alt_km)
    out: List[dict] = []
    t = start
    end = start + timedelta(hours=hours)
    dt = timedelta(seconds=step_s)

    def elev(when: datetime) -> float:
        return elevation_deg_at(lat, lon, alt_km, sat.position_ecef(when))

    prev_t = t
    prev_e = elev(t)
    in_pass = prev_e >= min_elevation_deg
    rise = t if in_pass else None
    track = [(t, prev_e)] if in_pass else []

    while t < end and len(out) < max_passes:
        t = t + dt
        if t > end:
            break
        try:
            e = elev(t)
        except TleError:
            break
        if e >= min_elevation_deg:
            if not in_pass:
                in_pass = True
                rise = _bisect_crossing(elev, prev_t, t, min_elevation_deg)
                track = [(rise, min_elevation_deg)]
            track.append((t, e))
        else:
            if in_pass:
                in_pass = False
                sett = _bisect_crossing(elev, t, prev_t, min_elevation_deg)
                track.append((sett, min_elevation_deg))
                out.append(_summarise_pass(sat, track, rise, sett, lat, lon, alt_km))
                track = []
                rise = None
        prev_t, prev_e = t, e
    return out


def _bisect_crossing(elev_fn, t_above: datetime, t_below: datetime,
                     threshold: float, iters: int = 24) -> datetime:
    """Bisect the threshold crossing between two samples on opposite sides."""
    lo, hi = t_above, t_below
    for _ in range(iters):
        mid = lo + (hi - lo) / 2
        try:
            e = elev_fn(mid)
        except TleError:
            break
        if (e >= threshold) == (elev_fn(lo) >= threshold):
            lo = mid
        else:
            hi = mid
        if abs((hi - lo).total_seconds()) <= 0.5:
            break
    return lo + (hi - lo) / 2


def _summarise_pass(sat: Sat, track, rise, sett, lat, lon, alt_km) -> dict:
    peak_t, peak_e = max(track, key=lambda p: p[1])
    # Parabolic refinement around the peak sample (~3 point vertex).
    try:
        idx = track.index((peak_t, peak_e))
    except ValueError:
        idx = -1
    if 0 < idx < len(track) - 1:
        y0, y1, y2 = track[idx - 1][1], track[idx][1], track[idx + 1][1]
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-9:
            shift = 0.5 * (y0 - y2) / denom          # in units of the step
            step_s = (track[idx + 1][0] - track[idx][0]).total_seconds()
            peak_t = peak_t + timedelta(seconds=shift * step_s)
            peak_e = y1 - 0.25 * (y0 - y2) * shift
    geo = sat.geodetic(peak_t)
    return {
        "rise_utc": rise.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "set_utc": sett.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "peak_utc": peak_t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_s": int(round((sett - rise).total_seconds())),
        "max_elevation_deg": round(peak_e, 1),
        "peak_lat": round(geo[0], 3),
        "peak_lon": round(geo[1], 3),
        "peak_alt_km": round(geo[2], 1),
        "satellite": sat.tle.name or ("NORAD %d" % sat.tle.norad_id),
        "norad_id": sat.tle.norad_id,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

# The canonical SGP4 test object: VANGUARD 1 (NORAD 00005). Expected output is
# the WGS-72 TEME position from the reference implementation at the given tsince.
REFERENCE_LINE1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
REFERENCE_LINE2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
REFERENCE_TSINCE_MIN = 309.67110720001529
REFERENCE_POSITION_KM = (-3754.2514743216166, 7876.346817439062, 4719.220856478582)
REFERENCE_TOLERANCE_KM = 0.05


def verify_reference(tolerance_km: float = REFERENCE_TOLERANCE_KM) -> dict:
    """Propagate the canonical case and compare with the reference position.

    Returns a report; raises AssertionError if the position is outside tolerance.
    Kept in the shipped module (not a test file) because this project has no test
    runner on the target container, and an unverified propagator is worse than
    none.
    """
    tle = Tle("VANGUARD 1", REFERENCE_LINE1, REFERENCE_LINE2)
    sat = Sat(tle)
    got = sat.position_eci(REFERENCE_TSINCE_MIN)
    diffs = [g - w for g, w in zip(got, REFERENCE_POSITION_KM)]
    err = math.sqrt(sum(d * d for d in diffs))
    report = {
        "tle": tle.name,
        "norad_id": tle.norad_id,
        "tsince_min": REFERENCE_TSINCE_MIN,
        "expected_km": REFERENCE_POSITION_KM,
        "got_km": tuple(round(v, 6) for v in got),
        "abs_error_km": round(err, 6),
        "tolerance_km": tolerance_km,
        "ok": err <= tolerance_km,
        "sanity": {
            "period_min": round(tle.period_min, 4),
            "altitude_km": round(tle.altitude_km, 2),
            "position_radius_km": round(math.sqrt(sum(v * v for v in got)), 2),
            "eccentricity": tle.ecco,
        },
    }
    if not report["ok"]:
        raise AssertionError(
            "SGP4 reference check FAILED: got %s, expected %s, |err|=%.6f km > %.6f km"
            % (report["got_km"], REFERENCE_POSITION_KM, err, tolerance_km))
    return report


if __name__ == "__main__":       # pragma: no cover - manual verification entry
    import json
    print(json.dumps(verify_reference(), indent=2, ensure_ascii=False, default=str))