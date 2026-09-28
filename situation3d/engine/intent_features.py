"""Feature extraction for the intent model: the rules become inputs, not verdicts.

Why this layer exists
---------------------
`intent.py` currently decides intent with hand-written rules. The rules encode
real domain knowledge, but as a *decision* they cannot be improved, cannot take a
new input, and — as measured on the demo data — produce an intent that changes
235 times for a single platform and asserts `HOLD` and `FORTIFY` simultaneously
in 1435 frames even though the ontology declares the two mutually exclusive.

So the rules are demoted here to **features**: each one contributes a continuous
margin (how strongly it fires, how far past its threshold), and a model decides.
This is the "internalise the wargame rules as model input" direction: the rules
stay explicit and auditable, but they no longer have the last word, and new
evidence (road context, and later the MiaoSuan action/judge streams) can be added
without rewriting the decision logic.

Feature groups
--------------
1. **kinematic**    — from our own tracks: speed, heading deltas, displacement
                      over several windows, heading dispersion, altitude.
2. **relative**     — geometry against the nearest opponent and the nearest friend:
                      range, bearing, closure rate, whether inside engagement or
                      sensor radius.
3. **rule margins** — one continuous margin per derivable rule, plus a 0/1 fire
                      flag and an approximation flag. The verdict is not here.
4. **context**      — theatre-level situation: force counts, group sizes, region
                      membership, distance to the hotspot centre.
5. **roads**        — `on_road`, distance to the network, road class, alignment of
                      heading with the route. A land platform that never left the
                      road cannot have manoeuvred off it.
6. **bridge**       — placeholders for the MiaoSuan model inputs (`mission_type`,
                      `action_type_*`, `judge_*`, `city_owner_flag`, `move_state`,
                      `fatigue`, `suppression`, `see_enemy`). These are **not
                      available yet** and are emitted as 0 with the name listed in
                      `MISSING_BRIDGE_FIELDS`, so a reviewer can see exactly what
                      is absent instead of reading a silent zero as a real zero.

Determinism
-----------
`feature_names()` is the single source of the vector layout, and extraction
iterates that list. Two runs on the same input must produce byte-identical
vectors — the self-test asserts it, because a feature whose value depends on dict
ordering would make "the same" trajectory produce different intents.

Only the standard library is used: the target container has python3.8 and no pip.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import geo

FEATURE_SCHEMA_VERSION = "ifv2"

# Windows, in frames. At 30 s per tick these are 1.5 min / 4 min / 15 min.
W_SHORT = 3
W_MID = 8
W_LONG = 30

# A fixed order is the schema. Never append in the middle: the trained weights
# are positional, and reordering silently reinterprets every weight.
_KINEMATIC = [
    "speed_kn",
    "speed_norm",            # speed / platform nominal speed
    "speed_ratio_class",     # speed / nominal speed of its platform class
    "alt_norm",
    "d_heading_1",           # |heading change| over 1 frame, deg
    "d_heading_3",
    "d_heading_8",
    "heading_std_8",         # dispersion: high means turning, not transiting
    "disp_3_km",
    "disp_8_km",
    "disp_30_km",
    "net_speed_3_kmh",       # net displacement rate: transit vs holding
    "net_speed_8_kmh",
    "net_speed_30_kmh",
    "path_ratio_8",          # path length / net displacement: 1 straight, high loiter
    "stationary_8",          # 1 if displacement over 8 frames < 2 km
    "stationary_30",
    "monotone_sep_3",        # 1 if range to nearest opponent strictly grows
    "monotone_close_3",      # 1 if it strictly shrinks
]

_RELATIVE = [
    "nearest_opp_km",
    "nearest_opp_km_norm",   # / max(engage, sensor) — scale-free proximity
    "in_engage_range",
    "in_sensor_range",
    "neg_in_engage_range",   # d/dw first difference of in_engage_range
    "bearing_to_opp_deg",
    "bearing_rate_3",        # deg change of that bearing over 3 frames
    "opp_heading_delta",     # ours minus theirs: 180 => head-on
    "closure_rate_kmh",      # negative => closing
    "nearest_friend_km",
    "friend_in_contact",     # nearest friend is inside its own sensor/engage range
    "closing_on_friend_3",   # 1 if range to nearest friend shrank over 3 frames
    "friend_lt_3hex",        # ontology's escort criterion
]

_CONTEXT = [
    "n_friendly",
    "n_opponent",
    "n_friendly_same_domain",
    "n_opponent_same_domain",
    "force_ratio",           # (friendly+1)/(opponent+1)
    "group_spread_km",       # own-side spread in the same domain
    "region_index_norm",
    "dist_hotspot_center_km",
    "global_level_norm",     # overall situation level / 4
]

_ROADS = [
    "on_road",               # 1 = within the constraint radius of the network
    "dist_to_road_km",
    "road_class_code",       # ordered: tertiary .. motorway
    "heading_alignment",     # cos(heading - route bearing): 1 aligned, -1 reversed
    "route_deviation_km",    # distance from the planned route polyline
    "is_land_domain",        # road features only mean something for land platforms
]

_BRIDGE = [
    "bridge_mission_type",
    "bridge_action_type_5",
    "bridge_action_type_9",
    "bridge_action_type_14",
    "bridge_action_type_15",
    "bridge_action_type_18",
    "bridge_judge_attack_count",
    "bridge_judge_damage",
    "bridge_city_owner_flag",
    "bridge_move_state",
    "bridge_fatigue",
    "bridge_suppression",
    "bridge_see_enemy",
]

# Names whose source data does not exist in this system yet. Emitted as 0 so the
# vector layout is stable, and listed here so nobody mistakes the zero for a
# measurement.
MISSING_BRIDGE_FIELDS = list(_BRIDGE)

_DERIVABLE = [
    "ASSAULT", "FLANK", "ENCIRCLEMENT", "PURSUIT", "HOLD", "RECON",
    "SURVEILLANCE", "REINFORCE", "ESCORT", "FIRE_SUPPORT", "REDEPLOY",
    "ASSEMBLE", "DISPERSE", "FORTIFY", "RETREAT", "EVADE",
]


def _rule_feature_names() -> List[str]:
    out = []
    for code in _DERIVABLE:
        c = code.lower()
        out.append("rule_%s_margin" % c)
        out.append("rule_%s_fire" % c)
        out.append("rule_%s_approx" % c)
    return out


_RULE = _rule_feature_names()

FEATURE_NAMES: List[str] = _KINEMATIC + _RELATIVE + _RULE + _CONTEXT + _ROADS + _BRIDGE

_ROAD_CLASS_CODE = {"tertiary": 1.0, "secondary": 2.0, "primary": 3.0,
                    "trunk": 4.0, "motorway": 5.0}


def feature_names() -> List[str]:
    """The vector layout. One source of truth for training and inference."""
    return list(FEATURE_NAMES)


def feature_dim() -> int:
    return len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _norm_speed(speed_kn: Optional[float], nominal: Optional[float]) -> float:
    if not speed_kn or not nominal or nominal <= 0:
        return 0.0
    return max(0.0, min(3.0, speed_kn / nominal))


def _displacement_km(pos: Sequence[Tuple[float, float]]) -> float:
    if len(pos) < 2:
        return 0.0
    return geo.path_length_m(pos) / 1000.0


def _heading_std(headings: Sequence[float]) -> float:
    """Circular standard deviation of headings, in degrees.

    A plain standard deviation is wrong here: 359 and 1 degrees are two degrees
    apart, not 358, and a platform crossing north would look like it was thrashing.
    """
    vals = [h for h in headings if h is not None]
    if len(vals) < 2:
        return 0.0
    s = sum(math.sin(math.radians(h)) for h in vals) / len(vals)
    c = sum(math.cos(math.radians(h)) for h in vals) / len(vals)
    r = math.sqrt(s * s + c * c)
    r = max(1e-9, min(1.0, r))
    return math.degrees(math.sqrt(-2.0 * math.log(r)))


def _bearing_delta(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d


class TrackView(object):
    """Read-only view over one target's per-frame samples.

    Built once per case; `extract_target_features` then answers questions about
    any frame without rescanning the raw frames.
    """

    __slots__ = ("tid", "domain", "aff", "nominal_speed", "engage_km", "sensor_km",
                 "alt_m", "lat", "lon", "spd", "hdg")

    def __init__(self, tid, domain, aff, nominal_speed, engage_km, sensor_km, alt_m):
        self.tid = tid
        self.domain = domain
        self.aff = aff
        self.nominal_speed = nominal_speed
        self.engage_km = float(engage_km or 0.0)
        self.sensor_km = float(sensor_km or 0.0)
        self.alt_m = float(alt_m or 0.0)
        self.lat: List[Optional[float]] = []
        self.lon: List[Optional[float]] = []
        self.spd: List[Optional[float]] = []
        self.hdg: List[Optional[float]] = []

    def at(self, i: int) -> Optional[Tuple[float, float]]:
        if 0 <= i < len(self.lat) and self.lat[i] is not None:
            return (self.lat[i], self.lon[i])  # type: ignore[index]
        return None

    def window(self, i: int, back: int) -> List[Tuple[float, float]]:
        out = []
        for j in range(max(0, i - back), i + 1):
            p = self.at(j)
            if p is not None:
                out.append(p)
        return out

    def heading(self, i: int, back: int = 0) -> Optional[float]:
        j = i - back
        if 0 <= j < len(self.hdg):
            return self.hdg[j]
        return None

    def speed(self, i: int) -> Optional[float]:
        if 0 <= i < len(self.spd):
            return self.spd[i]
        return None

    def km_to(self, i: int, other: "TrackView", j: int) -> Optional[float]:
        a = self.at(i)
        b = other.at(j)
        if a is None or b is None:
            return None
        return geo.haversine_m(a, b) / 1000.0


def build_tracks(frames: Sequence[dict], targets: Sequence[object]
                 ) -> Dict[str, TrackView]:
    """Index every target's samples by frame, once."""
    views: Dict[str, TrackView] = {}
    for t in targets:
        views[t.target_id] = TrackView(
            t.target_id, getattr(t, "domain", ""), getattr(t, "affiliation", ""),
            getattr(t, "speed_kn", 0.0), getattr(t, "engage_km", 0.0),
            getattr(t, "sensor_range_km", 0.0), getattr(t, "altitude_m", 0.0))
    for f in frames:
        for s in f.get("targets") or []:
            v = views.get(s["id"])
            if v is None:
                continue
            v.lat.append(s.get("lat"))
            v.lon.append(s.get("lon"))
            v.spd.append(s.get("speed_kn"))
            v.hdg.append(s.get("heading"))
    return views

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# ---------------------------------------------------------------------------
# rule margins: each rule reports *how far past* its criterion it is
# ---------------------------------------------------------------------------

def rule_margins(tracks: Dict[str, TrackView], order: Sequence[str], i: int,
                 tid: str) -> Dict[str, dict]:
    """Continuous margin per derivable rule, plus the evidence lines.

    A margin is normalised so 0.0 means "did not fire at all" and 1.0 means "well
    past the criterion". The ontology's thresholds are the reference points; where
    a threshold had to be substituted (because our data lacks a wargame field)
    the entry is flagged `approximated`, exactly as `intent.SUPPORTED_RULES` does.
    """
    me = tracks[tid]
    out: Dict[str, dict] = {}

    def put(code, margin, evidence, approx=False, sec=False):
        out[code] = {
            "margin": round(_clamp01(margin), 4),
            "evidence": [e for e in evidence if e],
            "approximated": bool(approx),
            "secondary_only": bool(sec),
        }

    here = me.at(i)
    if here is None:
        return out

    opp = _nearest(tracks, order, i, tid, want_same_aff=False)
    fri = _nearest(tracks, order, i, tid, want_same_aff=True)
    opp_km = opp[1] if opp else None
    fri_km = fri[1] if fri else None
    engaged = opp_km is not None and me.engage_km > 0 and opp_km <= me.engage_km
    in_sensor = opp_km is not None and me.sensor_km > 0 and opp_km <= me.sensor_km
    contact = engaged or in_sensor
    spd = me.speed(i)
    hdg = me.heading(i)

    w3 = me.window(i, W_SHORT)
    w8 = me.window(i, W_MID)
    w30 = me.window(i, W_LONG)
    disp3 = _displacement_km(w3)
    disp8 = _displacement_km(w8)
    disp30 = _displacement_km(w30)

    # closure over the short window, against the principal opponent
    closure = None
    sep_series = []
    if opp and w3:
        other = tracks[opp[0]]
        sep_series = [geo.haversine_m(p, other.at(i) or p) / 1000.0 for p in w3] \
            if other.at(i) else []
        if len(w3) >= 2 and other.at(i) is not None:
            closure = geo.haversine_m(w3[-1], other.at(i)) / 1000.0 - \
                      geo.haversine_m(w3[0], other.at(i)) / 1000.0

    # ---- RETREAT: sustained separation from the principal threat, in contact
    if opp and len(sep_series) >= 3 and contact:
        growing = all(sep_series[k + 1] > sep_series[k]
                      for k in range(len(sep_series) - 1))
        if growing:
            put("RETREAT", 0.4 + 0.6 * min(1.0, (sep_series[-1] - sep_series[0]) / 5.0),
                ["连续 %d 帧与 %s 拉开距离（%.1f→%.1f km）"
                 % (len(sep_series), opp[0], sep_series[0], sep_series[-1]),
                 "接触中：最近对方 %.1f km" % opp_km])

    # ---- EVADE: sharp turn then return to the previous bearing
    if hdg is not None:
        turn = 0.0
        for back in (1, 2):
            h0, h1 = me.heading(i, back + 1), me.heading(i, back)
            if h0 is not None and h1 is not None:
                turn = max(turn, _bearing_delta(h0, h1))
        base = me.heading(i, 3)
        if turn > 60.0 and base is not None and _bearing_delta(base, hdg) <= 20.0:
            put("EVADE", 0.5 + 0.5 * min(1.0, (turn - 60.0) / 60.0),
                ["1–2 帧内方位突变 %.0f° > 60°，且回到原航向 %.0f°" % (turn, base),
                 "处于对方 %.1f km 处" % opp_km if opp_km is not None else ""])

    # ---- ASSAULT: inside engagement range and still closing
    if engaged and closure is not None and closure < 0:
        depth = 1.0 - (opp_km / me.engage_km if me.engage_km else 1.0)
        put("ASSAULT", 0.5 + 0.5 * max(depth, min(1.0, -closure / 10.0)),
            ["进入 %s 交战半径（%.1f ≤ %.1f km）" % (opp[0], opp_km, me.engage_km),
             "近 %d 帧持续接近（Δ %.1f km）" % (W_SHORT, closure),
             "近似：本体另需 SHOOT 事件，本系统无该字段"], approx=True)

    # ---- FLANK: terminal bearing differs from the initial one by > 45 deg
    if opp and opp_km is not None and me.engage_km > 0 and opp_km <= 4.0 * max(me.engage_km, 20.0):
        other = tracks[opp[0]]
        far = me.at(i - 12)
        if far is not None and other.at(i) is not None:
            b_far = geo.initial_bearing_deg(far, other.at(i))
            b_now = geo.initial_bearing_deg(here, other.at(i))
            delta = _bearing_delta(b_far, b_now)
            if delta > 45.0 and closure is not None and closure < 0:
                put("FLANK", 0.4 + 0.6 * min(1.0, (delta - 45.0) / 90.0),
                    ["终端方位 %.0f° 相对初始方位 %.0f° 偏转 %.0f° > 45°"
                     % (b_now, b_far, delta),
                     "同时仍在接近（Δ %.1f km）" % closure])

    # ---- ENCIRCLEMENT: >=2 same-side platforms converging from > 90 deg apart
    if opp:
        other = tracks[opp[0]]
        tgt = other.at(i)
        if tgt is not None:
            bearings = []
            for oid in order:
                if oid == tid or tracks[oid].aff != me.aff:
                    continue
                p = tracks[oid].at(i)
                if p is None:
                    continue
                if geo.haversine_m(p, tgt) / 1000.0 > 3.0 * max(me.engage_km, 20.0):
                    continue
                w = tracks[oid].window(i, W_SHORT)
                if len(w) >= 2 and geo.haversine_m(w[-1], tgt) < geo.haversine_m(w[0], tgt):
                    bearings.append(geo.initial_bearing_deg(p, tgt))
            best_spread = 0.0
            for a in range(len(bearings)):
                for b in range(a + 1, len(bearings)):
                    best_spread = max(best_spread, _bearing_delta(bearings[a], bearings[b]))
            if len(bearings) >= 2 and best_spread > 90.0:
                put("ENCIRCLEMENT", 0.5 + 0.5 * min(1.0, (best_spread - 90.0) / 90.0),
                    ["%d 个同方目标自夹角 %.0f° > 90° 逼近 %s"
                     % (len(bearings), best_spread, opp[0])])

    # ---- PURSUIT: our target is closing on an opponent that is retreating
    if opp and closure is not None and closure < 0:
        other = tracks[opp[0]]
        ow = other.window(i, W_SHORT)
        tgt_fleeing = False
        if len(ow) >= 3:
            seps = [geo.haversine_m(p, here) for p in ow]
            tgt_fleeing = all(seps[k + 1] > seps[k] for k in range(len(seps) - 1))
        if tgt_fleeing:
            put("PURSUIT", 0.5 + 0.5 * min(1.0, -closure / 10.0),
                ["%s 正在拉开距离，本方近 %d 帧追击接近（Δ %.1f km）"
                 % (opp[0], W_SHORT, closure),
                 "近似：本体以'对方撤退 + 我方相关轨迹'判定"], approx=True)

    # ---- HOLD: essentially stationary while an opponent is near
    if spd is not None and spd < 5.0 and not engaged:
        stayed = 0
        for back in range(1, W_MID):
            p = me.at(i - back)
            if p is None or geo.haversine_m(p, here) / 1000.0 > 5.0:
                break
            stayed += 1
        if stayed >= 5 and opp_km is not None and opp_km <= 3.0 * max(me.engage_km, 20.0):
            put("HOLD", 0.4 + 0.4 * (stayed / float(W_MID)) + 0.2 * (1.0 - spd / 5.0),
                ["连续 %d 帧在 5 km 内基本静止（%.1f kn）" % (stayed, spd),
                 "对方 %s 在 %.1f km 外" % (opp[0], opp_km),
                 "近似：本体要求'己方/指定目标点'，以静止位置代替"], approx=True)

    # ---- FORTIFY: sustained slow with no displacement
    if spd is not None and spd < 3.0:
        still = 0
        for back in range(1, W_MID):
            p = me.at(i - back)
            if p is None or geo.haversine_m(p, here) / 1000.0 > 2.0:
                break
            still += 1
        if still >= 6:
            put("FORTIFY", 0.4 + 0.6 * min(1.0, still / float(W_MID)),
                ["连续 %d 帧位移 < 2 km 且航速 %.1f kn" % (still, spd),
                 "近似：本体用 move_state=4 隐蔽态，本系统无该字段"], approx=True)

    # ---- REDEPLOY: long move without contact (ontology: > 5 hex)
    if not contact and disp30 > 0:
        hex_km = 10.0
        if disp30 > 5.0 * hex_km:
            put("REDEPLOY", 0.4 + 0.6 * min(1.0, (disp30 - 5 * hex_km) / (15 * hex_km)),
                ["%d 帧无接触位移 %.1f km > 5 hex（%.0f km）"
                 % (W_LONG, disp30, 5 * hex_km)],
                approx=True)

    # ---- RECON: advancing without contact
    if not contact and spd and spd > 0 and disp3 > 2.0:
        put("RECON", 0.3 + 0.7 * min(1.0, disp3 / 15.0),
            ["无接触推进：近 %d 帧位移 %.1f km" % (W_SHORT, disp3),
             "与最近对方 %.1f km" % opp_km if opp_km is not None else "无对方目标"],
            approx=True)

    # ---- SURVEILLANCE: persistent sensor contact without engagement
    if in_sensor and not engaged and opp:
        held = 0
        for back in range(1, W_MID):
            p, q = me.at(i - back), tracks[opp[0]].at(i - back)
            if p is None or q is None or geo.haversine_m(p, q) / 1000.0 > me.sensor_km:
                break
            held += 1
        if held >= 5:
            put("SURVEILLANCE", 0.4 + 0.6 * min(1.0, held / float(W_MID)),
                ["对 %s 连续 %d 帧保持传感器接触（≤ %.0f km）"
                 % (opp[0], held, me.sensor_km),
                 "未进入交战半径（%.1f > %.1f km）" % (opp_km, me.engage_km)])

    # ---- REINFORCE: closing on a friend that is itself in contact
    if fri and fri_km is not None:
        ft = tracks[fri[0]]
        fp = ft.at(i)
        friend_contact = False
        if fp is not None:
            for oid in order:
                if tracks[oid].aff == me.aff:
                    continue
                q = tracks[oid].at(i)
                if q is not None and geo.haversine_m(fp, q) / 1000.0 <= max(
                        ft.engage_km, ft.sensor_km):
                    friend_contact = True
                    break
        closer = None
        if len(w3) >= 2 and fp is not None:
            old = me.at(i - W_SHORT)
            if old is not None:
                closer = geo.haversine_m(w3[-1], fp) - geo.haversine_m(w3[0], fp)
        if friend_contact and closer is not None and closer < 0:
            put("REINFORCE", 0.5 + 0.5 * min(1.0, -closer / 5.0),
                ["%s 处于接触中，本方近 %d 帧向其靠拢（Δ %.1f km）"
                 % (fri[0], W_SHORT, closer)], approx=True)

    # ---- ESCORT: bounded proximity to a friend (ontology: < 3 hex)
    if fri and fri_km is not None and fri_km < 3.0 * 10.0:
        held = 0
        for back in range(1, W_MID):
            p, q = me.at(i - back), tracks[fri[0]].at(i - back)
            if p is None or q is None or geo.haversine_m(p, q) / 1000.0 > 30.0:
                break
            held += 1
        if held >= 5:
            put("ESCORT", 0.4 + 0.6 * min(1.0, held / float(W_MID)),
                ["与 %s 连续 %d 帧保持 < 3 hex（30 km）邻近" % (fri[0], held)],
                approx=True)

    # ---- FIRE_SUPPORT: secondary only, requires a friend in contact
    if not engaged and in_sensor and opp:
        any_friend_contact = False
        for oid in order:
            if oid == tid or tracks[oid].aff != me.aff:
                continue
            p = tracks[oid].at(i)
            q = tracks[opp[0]].at(i)
            if p is None or q is None:
                continue
            if geo.haversine_m(p, q) / 1000.0 <= max(tracks[oid].engage_km,
                                                     tracks[oid].sensor_km):
                any_friend_contact = True
                break
        if any_friend_contact:
            put("FIRE_SUPPORT", 0.4 + 0.4 * min(1.0, me.sensor_km and opp_km / me.sensor_km or 0.0),
                ["同方目标在与 %s 对抗，本方保持火力距离（%.1f ≤ %.0f km）"
                 % (opp[0], opp_km, me.sensor_km),
                 "本体标 secondary only：只能作为次要意图"],
                approx=True, sec=True)

    # ---- ASSEMBLE / DISPERSE: same-side pairwise distance trend
    mates = [oid for oid in order if oid != tid and tracks[oid].aff == me.aff]
    if mates:
        spreads = []
        for back in range(0, W_MID):
            j = i - back
            if j < 0:
                break
            pts = [tracks[o].at(j) for o in [tid] + mates]
            pts = [p for p in pts if p is not None]
            if len(pts) < 2:
                continue
            lat = sum(p[0] for p in pts) / len(pts)
            lon = sum(p[1] for p in pts) / len(pts)
            spreads.append(max(geo.haversine_m(p, (lat, lon)) for p in pts) / 1000.0)
        if len(spreads) >= 4:
            # spreads[0] is the newest; compare newest against oldest in window
            delta = spreads[0] - spreads[-1]
            if delta < -0.5:      # converging
                put("ASSEMBLE", 0.4 + 0.6 * min(1.0, -delta / 10.0),
                    ["同方两两跨度近 %d 帧由 %.1f km 收敛到 %.1f km"
                     % (len(spreads), spreads[-1], spreads[0])])
            elif delta > 0.5:     # diverging
                put("DISPERSE", 0.4 + 0.6 * min(1.0, delta / 10.0),
                    ["同方两两跨度近 %d 帧由 %.1f km 发散到 %.1f km"
                     % (len(spreads), spreads[-1], spreads[0])])

    return out


def _nearest(tracks: Dict[str, TrackView], order: Sequence[str], i: int,
             tid: str, want_same_aff: bool):
    """(id, km) of the nearest platform on the requested side, or None."""
    me = tracks[tid]
    here = me.at(i)
    if here is None:
        return None
    best, best_km = None, None
    for oid in order:
        if oid == tid:
            continue
        other = tracks[oid]
        if want_same_aff and other.aff != me.aff:
            continue
        if (not want_same_aff) and other.aff == me.aff:
            continue
        p = other.at(i)
        if p is None:
            continue
        km = geo.haversine_m(here, p) / 1000.0
        if best_km is None or km < best_km:
            best, best_km = oid, km
    return (best, best_km) if best is not None else None


# ---------------------------------------------------------------------------
# context: force counts, group spread, region membership
# ---------------------------------------------------------------------------

def context_features(tracks: Dict[str, TrackView], order: Sequence[str], i: int,
                     tid: str, inference_frame: Optional[dict],
                     hotspots: Sequence[dict], route: Sequence[Tuple[float, float]],
                     road: Optional[object] = None) -> Dict[str, float]:
    me = tracks[tid]
    here = me.at(i)
    f: Dict[str, float] = {}

    n_f = n_o = n_fd = n_od = 0
    for oid in order:
        other = tracks[oid]
        p = other.at(i)
        if p is None:
            continue
        if other.aff == me.aff:
            n_f += 1
            if other.domain == me.domain:
                n_fd += 1
        else:
            n_o += 1
            if other.domain == me.domain:
                n_od += 1
    f["n_friendly"] = float(n_f)
    f["n_opponent"] = float(n_o)
    f["n_friendly_same_domain"] = float(n_fd)
    f["n_opponent_same_domain"] = float(n_od)
    f["force_ratio"] = (n_f + 1.0) / (n_o + 1.0)

    # own-side spread within the same domain
    mates = [tracks[o].at(i) for o in order
             if tracks[o].aff == me.aff and tracks[o].domain == me.domain]
    mates = [p for p in mates if p is not None]
    if len(mates) >= 2 and here is not None:
        lat = sum(p[0] for p in mates) / len(mates)
        lon = sum(p[1] for p in mates) / len(mates)
        f["group_spread_km"] = max(geo.haversine_m(p, (lat, lon)) for p in mates) / 1000.0
    else:
        f["group_spread_km"] = 0.0

    region_idx = -1.0
    dist_center = 0.0
    if here is not None and hotspots:
        best = None
        for k, h in enumerate(hotspots):
            bb = h.get("bbox")
            if not bb:
                continue
            inside = (bb[0] <= here[1] <= bb[2] and bb[1] <= here[0] <= bb[3])
            c = h.get("center") or [(bb[1] + bb[3]) / 2.0, (bb[0] + bb[2]) / 2.0]
            d = geo.haversine_m(here, (c[0], c[1])) / 1000.0
            score = (0 if inside else 1, d)
            if best is None or score < best[0]:
                best = (score, k, d)
        if best is not None:
            region_idx = float(best[1])
            dist_center = best[2]
    f["region_index_norm"] = region_idx / max(1.0, float(len(hotspots))) if region_idx >= 0 else -1.0
    f["dist_hotspot_center_km"] = dist_center

    lvl = 0.0
    if inference_frame:
        ov = inference_frame.get("overall") or {}
        lvl = float(ov.get("level") or 0) / 4.0
    f["global_level_norm"] = lvl

    # ---- roads ----
    f["is_land_domain"] = 1.0 if me.domain == "land" else 0.0
    d_road = None
    if here is not None and road is not None:
        try:
            d_road = road.distance_to_network_km(here[0], here[1])
        except Exception:  # noqa: BLE001 - road data is optional
            d_road = None
    f["dist_to_road_km"] = float(d_road) if d_road is not None else -1.0
    f["on_road"] = 1.0 if (d_road is not None and d_road <= 3.0) else 0.0
    f["road_class_code"] = 0.0
    f["heading_alignment"] = 0.0
    f["route_deviation_km"] = -1.0

    if here is not None and route and len(route) >= 2:
        # nearest vertex of the planned route, and the bearing to the next one
        best_d = None
        best_k = 0
        for k, p in enumerate(route):
            d = geo.haversine_m(here, p)
            if best_d is None or d < best_d:
                best_d, best_k = d, k
        f["route_deviation_km"] = best_d / 1000.0
        nxt = route[min(best_k + 1, len(route) - 1)]
        prv = route[max(best_k - 1, 0)]
        hdg = me.heading(i)
        if hdg is not None and (nxt != prv):
            brg = geo.initial_bearing_deg(prv, nxt)
            f["heading_alignment"] = math.cos(math.radians(brg - hdg))
        f["road_class_code"] = 3.0

    return f


# ---------------------------------------------------------------------------
# the feature vector
# ---------------------------------------------------------------------------

def extract_target_features(tracks: Dict[str, TrackView], order: Sequence[str],
                            i: int, tid: str,
                            inference_frame: Optional[dict] = None,
                            hotspots: Sequence[dict] = (),
                            route: Sequence[Tuple[float, float]] = (),
                            road: Optional[object] = None) -> Dict[str, float]:
    """One target's feature vector at frame `i`, keyed by `feature_names()`.

    Every key in `FEATURE_NAMES` is present in the result, so the vector can be
    materialised positionally without a key lookup that might silently miss.
    """
    me = tracks[tid]
    here = me.at(i)
    f: Dict[str, float] = dict.fromkeys(FEATURE_NAMES, 0.0)

    spd = me.speed(i)
    hdg = me.heading(i)
    f["speed_kn"] = float(spd or 0.0)
    f["speed_norm"] = _norm_speed(spd, me.nominal_speed)
    f["speed_ratio_class"] = _norm_speed(spd, me.nominal_speed)
    f["alt_norm"] = min(3.0, me.alt_m / 10000.0)

    for back, key in ((1, "d_heading_1"), (3, "d_heading_3"), (8, "d_heading_8")):
        h0 = me.heading(i, back)
        if h0 is not None and hdg is not None:
            f[key] = _bearing_delta(h0, hdg) / 180.0
    f["heading_std_8"] = _heading_std([me.heading(i, b) for b in range(W_MID)]) / 180.0

    w3, w8, w30 = me.window(i, W_SHORT), me.window(i, W_MID), me.window(i, W_LONG)
    d3, d8, d30 = _displacement_km(w3), _displacement_km(w8), _displacement_km(w30)
    f["disp_3_km"] = d3
    f["disp_8_km"] = d8
    f["disp_30_km"] = d30
    # tick length is 30 s; km per hour over the window
    f["net_speed_3_kmh"] = (d3 / W_SHORT) * 120.0
    f["net_speed_8_kmh"] = (d8 / W_MID) * 120.0
    f["net_speed_30_kmh"] = (d30 / W_LONG) * 120.0
    # Path length over net displacement: 1.0 means a straight transit, and higher
    # values mean the platform wandered (loiter, patrol orbit, re-attack run).
    # Both a stationary platform (no path) and one whose endpoints coincide have
    # no meaningful ratio, and dividing by zero here previously crashed training
    # outright -- the guard is explicit rather than relying on `max(1e-6, ...)`
    # after the division has already happened.
    f["path_ratio_8"] = 0.0
    if len(w8) >= 2:
        straight = geo.haversine_m(w8[0], w8[-1]) / 1000.0
        if straight > 1e-6:
            f["path_ratio_8"] = min(10.0, d8 / straight)
        elif d8 > 1e-6:
            # Returned to its own start: a closed loop, ratio is unbounded, so cap.
            f["path_ratio_8"] = 10.0
    f["stationary_8"] = 1.0 if d8 < 2.0 else 0.0
    f["stationary_30"] = 1.0 if d30 < 5.0 else 0.0

    # ---- relative ----
    opp = _nearest(tracks, order, i, tid, want_same_aff=False)
    fri = _nearest(tracks, order, i, tid, want_same_aff=True)
    if opp and opp[1] is not None and here is not None:
        opp_km = opp[1]
        other = tracks[opp[0]]
        scale = max(me.engage_km, me.sensor_km, 1.0)
        f["nearest_opp_km"] = opp_km
        f["nearest_opp_km_norm"] = opp_km / scale
        f["in_engage_range"] = 1.0 if (me.engage_km > 0 and opp_km <= me.engage_km) else 0.0
        f["in_sensor_range"] = 1.0 if (me.sensor_km > 0 and opp_km <= me.sensor_km) else 0.0
        op = other.at(i)
        if op is not None:
            f["bearing_to_opp_deg"] = geo.initial_bearing_deg(here, op) / 360.0
            old = me.at(i - W_SHORT)
            if old is not None:
                b0 = geo.initial_bearing_deg(old, op)
                b1 = geo.initial_bearing_deg(here, op)
                f["bearing_rate_3"] = ((b1 - b0 + 540.0) % 360.0 - 180.0) / 180.0
            oh = other.heading(i)
            if oh is not None and hdg is not None:
                f["opp_heading_delta"] = _bearing_delta(hdg, oh) / 180.0
        if len(w3) >= 2 and op is not None:
            f["closure_rate_kmh"] = ((geo.haversine_m(w3[-1], op)
                                      - geo.haversine_m(w3[0], op)) / 1000.0
                                     / W_SHORT) * 120.0
        # monotone separation series against the opponent
        if op is not None and len(w3) >= 3:
            seps = [geo.haversine_m(p, op) for p in w3]
            f["monotone_sep_3"] = 1.0 if all(seps[k + 1] > seps[k]
                                             for k in range(len(seps) - 1)) else 0.0
            f["monotone_close_3"] = 1.0 if all(seps[k + 1] < seps[k]
                                               for k in range(len(seps) - 1)) else 0.0
        # was it in range a frame ago? gives the entry/exit direction
        prev = other.at(i - 1)
        old = me.at(i - 1)
        if prev is not None and old is not None and me.engage_km > 0:
            was = geo.haversine_m(old, prev) / 1000.0 <= me.engage_km
            now = f["in_engage_range"] > 0.5
            f["neg_in_engage_range"] = 1.0 if (was and not now) else 0.0

    if fri and fri[1] is not None:
        f["nearest_friend_km"] = fri[1]
        f["friend_lt_3hex"] = 1.0 if fri[1] < 30.0 else 0.0
        ft = tracks[fri[0]]
        fp = ft.at(i)
        if fp is not None:
            for oid in order:
                if tracks[oid].aff == me.aff:
                    continue
                q = tracks[oid].at(i)
                if q is not None and geo.haversine_m(fp, q) / 1000.0 <= max(
                        ft.engage_km, ft.sensor_km):
                    f["friend_in_contact"] = 1.0
                    break
        if len(w3) >= 2 and fp is not None:
            f["closing_on_friend_3"] = 1.0 if (
                geo.haversine_m(w3[-1], fp) < geo.haversine_m(w3[0], fp)) else 0.0

    # ---- rule margins ----
    margins = rule_margins(tracks, order, i, tid)
    for code, info in margins.items():
        c = code.lower()
        f["rule_%s_margin" % c] = float(info["margin"])
        f["rule_%s_fire" % c] = 1.0 if info["margin"] > 0 else 0.0
        f["rule_%s_approx" % c] = 1.0 if info["approximated"] else 0.0

    # ---- context + roads ----
    f.update(context_features(tracks, order, i, tid, inference_frame,
                              hotspots, route, road))
    # bridge fields stay 0.0 (see MISSING_BRIDGE_FIELDS)
    return f


def vector(features: Dict[str, float]) -> List[float]:
    """Materialise a feature dict positionally, in `feature_names()` order."""
    return [float(features.get(n, 0.0)) for n in FEATURE_NAMES]
