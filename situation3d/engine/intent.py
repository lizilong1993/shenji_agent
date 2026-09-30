"""Derive the 22-intent tactical ontology from the tracks we actually have.

Why this module exists
----------------------
The ontology in `Shenji/ontology/intent_ontology_v2.json` defines 22 tactical
intents under 6 parent categories, plus 4 judgment-status layers. The inference
layer's own `ROLE_INTENT` table does **not** speak that vocabulary: it maps a
platform's role to a near-synonym of itself (`"编队指挥" -> "编队机动"`,
`"空中警戒" -> "空中预警指挥"`, `"近海巡逻" -> "近海巡逻"`). That is a *function*
restated, not an *intent* judged, which is exactly the semantic gap the progress
report raises ("跟 JS 中的种类（攻击、通信、撤退等）还有一些语义上的差异").

So this module judges intent from observable behaviour instead, following the
ontology's own `necessary_conditions` / `sufficient_evidence` fields.

Honesty rules (these matter more than coverage)
-----------------------------------------------
1. **A rule whose required field we do not have is not guessed.** The ontology's
   conditions frequently bind to wargame fields (`mission_type`, `action type
   8/9/14/15/18/315`, `move_state`, `guided_obj_id`). This system has positions,
   headings, speeds, sensor ranges and engagement radii. `SUPPORTED_RULES` below
   states, per intent, whether it is derivable and what is missing otherwise; the
   UI surfaces that instead of inventing an answer.
2. **FEINT is never derived.** The ontology itself sets `auto_label_grade: C` and
   `requires_plan_tag: true`, with `boundary_rules`: "Behaviour alone is never
   sufficient for FEINT." Guessing it would be the single most misleading thing
   this module could do.
3. **Secondary-only intents stay secondary.** `FIRE_SUPPORT` and
   `TARGET_GUIDANCE` are marked `primary_vs_secondary: secondary only`; they can
   co-occur but never *be* the answer.
4. **AMBIGUOUS is a first-class result**, matching the report's model finding
   that a set of candidate intents (sometimes shrinking only once decisive
   evidence arrives) is more honest than one low-confidence label.
5. **Rules are marked `approximated`** where we substitute a proxy (e.g. sustained
   low speed standing in for `move_state=4` concealment), so a reviewer can see
   which conclusions rest on a substitution.

Units and thresholds
--------------------
The ontology states distances in *hexes* ("< 3 hexes", "> 5 hexes") because it was
written for a hex wargame. `HEX_KM` converts to kilometres; it is a declared
assumption, not a measurement, and is called out in the evidence text of any rule
that uses it.

Only standard library is used: the target container has python3.8 and no pip.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import geo

# ---------------------------------------------------------------------------
# Ontology
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
SITUATION3D_ROOT = os.path.dirname(HERE)

# Relative location of the ontology inside the repository that holds both the
# agent checkout and `situation3d`.
#
# NOTE (term replacement): the ontology's parent-category name is spelled
# `机动TS` in the vendored copy at `engine/data/intent_ontology_v2.json` and is
# emitted to the front end as-is. The authority file this constant points at
# lives in the Shenji checkout, which is **not** modified here, so re-syncing the
# vendored copy will restore the old spelling `机动TS` and the UI will follow.
# That divergence is the accepted cost of translating the term at all; see
# NOTES_term_replacement.md.
_ONTOLOGY_REL = os.path.join("code", "shenji_agent", "Shenji", "ontology",
                             "intent_ontology_v2.json")


def _repo_ontology_candidates() -> List[str]:
    """Candidate ontology paths, walking up from here to find the repo root.

    Deriving the root with a fixed number of `dirname()` calls broke once
    already: the layouts differ -- the agent code sits *below* the directory
    holding `situation3d` in one checkout and beside it in another -- so a
    hard-coded depth silently pointed outside the tree and the module could not
    find its own authority file. Walking up and probing is layout-proof.
    """
    out = [os.path.join(SITUATION3D_ROOT, "engine", "data", "intent_ontology_v2.json")]
    d = SITUATION3D_ROOT
    for _ in range(5):
        out.append(os.path.join(d, _ONTOLOGY_REL))
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return out


ONTOLOGY_CANDIDATES = _repo_ontology_candidates()

# Status layers, in the order the ontology declares them.
STATUS_LAYERS = ("BACKGROUND", "UNKNOWN", "AMBIGUOUS", "OUT_OF_ONTOLOGY")

# Hex -> km. The ontology counts distance in hexes; this is the declared
# conversion used by every rule that mentions hexes.
HEX_KM = 10.0

# Two intents count as tied (=> AMBIGUOUS) when their weights differ by less
# than this fraction of the stronger one.
AMBIGUITY_MARGIN = 0.15

# Below this weight nothing is asserted at all.
MIN_WEIGHT = 0.25

_state: Dict[str, object] = {}


class IntentError(ValueError):
    """Ontology missing or malformed."""


def ontology_path() -> Optional[str]:
    for p in ONTOLOGY_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def load_ontology(path: Optional[str] = None, refresh: bool = False) -> dict:
    """Load and cache the ontology. Raises IntentError if it cannot be found."""
    key = os.path.abspath(path) if path else ontology_path()
    if key is None:
        raise IntentError(
            "intent ontology not found; looked in: %s" % ", ".join(ONTOLOGY_CANDIDATES))
    st = os.stat(key)
    cached = _state.get("ontology")
    if (not refresh and cached and _state.get("path") == key
            and _state.get("mtime") == st.st_mtime):
        return cached  # type: ignore[return-value]
    with open(key, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc.get("intents"), list):
        raise IntentError("ontology at %s has no 'intents' list" % key)
    _state.update({"ontology": doc, "path": key, "mtime": st.st_mtime})
    return doc


def _doc(refresh: bool = False) -> dict:
    return load_ontology(refresh=refresh)


def intents(refresh: bool = False) -> List[dict]:
    return list(_doc(refresh).get("intents") or [])


def parents(refresh: bool = False) -> Dict[str, dict]:
    return {p["code"]: p for p in (_doc(refresh).get("parent_categories") or [])}


def intent_codes(refresh: bool = False) -> List[str]:
    return [i["code"] for i in intents(refresh)]


def intent_by_code(code: str, refresh: bool = False) -> Optional[dict]:
    for i in intents(refresh):
        if i["code"] == code:
            return i
    return None


def confusable_pairs(refresh: bool = False) -> set:
    """Pairs the ontology's confusion matrix says are easy to *confuse*.

    These are **not** mutually exclusive. The matrix exists to explain how to tell
    two intents apart ("HOLD is area control over a designated objective; FORTIFY
    is a posture"), so a decoder that treated them as exclusive would drop
    legitimate co-occurring hypotheses. Use `conflict_pairs()` for exclusions.
    """
    out = set()
    for entry in (_doc(refresh).get("confusion_matrix") or []):
        pair = entry.get("pair") or entry.get("confusable_pair") or []
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            a, b = str(pair[0]), str(pair[1])
            # The matrix carries a wildcard row, `['FIRE_SUPPORT', 'any primary
            # intent']`, which states the secondary-only rule rather than a pair.
            # Keeping it as a pair would create a tuple that can never match
            # anything, i.e. silently dead code.
            if a.startswith("any ") or b.startswith("any "):
                continue
            out.add((a, b))
    return out


def conflict_pairs(refresh: bool = False) -> set:
    """Pairs of intents the ontology declares mutually exclusive.

    Taken from each intent's own `conflicts_with` list, which is an
    exclusivity declaration: ASSAULT cannot hold at the same time as RETREAT,
    EVADE, HOLD or RECON. This is the set a decoder should use to prune.

    Kept separate from `confusable_pairs()` on purpose. Folding the two together
    (as an earlier build did) yields ~71 exclusion pairs, and because the two
    relations are independent the closure then excludes almost every pair, so the
    prediction set collapses to a singleton and the ambiguity the system exists to
    express disappears. The asymmetry is easy to miss and expensive to get wrong.
    """
    out = set()
    for i in intents(refresh):
        code = i["code"]
        for other in (i.get("conflicts_with") or []):
            other = str(other)
            if other and other != code and not other.startswith("any "):
                out.add((code, other))
    return out


def secondary_only_codes(refresh: bool = False) -> List[str]:
    """Intents the ontology forbids as a standalone answer."""
    out = []
    for i in intents(refresh):
        if str(i.get("primary_vs_secondary", "")).startswith("secondary"):
            out.append(i["code"])
    return out
    return out


def ontology_summary(refresh: bool = False) -> dict:
    """Compact form for the model block sent to the front end."""
    doc = _doc(refresh)
    return {
        "version": doc.get("version"),
        "ontology_name": doc.get("ontology_name"),
        "status_layers": [s["code"] for s in (doc.get("status_layers") or [])],
        "parents": [
            {"code": p["code"], "name_zh": p.get("name_zh"),
             "name_en": p.get("name_en"), "children": p.get("children") or []}
            for p in (doc.get("parent_categories") or [])
        ],
        "intents": [
            {"code": i["code"], "name_zh": i.get("name_zh"),
             "name_en": i.get("name_en"), "parent_code": i.get("parent_code"),
             "primary_vs_secondary": i.get("primary_vs_secondary"),
             "auto_label_grade": i.get("auto_label_grade")}
            for i in intents(refresh)
        ],
    }


# ---------------------------------------------------------------------------
# What we can and cannot judge
# ---------------------------------------------------------------------------

# Per intent: can this system derive it from tracks alone, and if not, why not.
# 'fields' lists what the ontology's necessary_conditions actually need.
SUPPORTED_RULES: Dict[str, dict] = {
    "ASSAULT": {
        "derivable": True,
        "note": "抵近 + 进入对方交战半径；本体另需 SHOOT 事件，我们用'进入 engage_km'近似",
        "approximated": True,
    },
    "BREAKTHROUGH": {
        "derivable": False, "fields": ["defensive_line", "SHOOT"],
        "note": "需要'敌方防守线'这一声明字段，本系统无此语义对象",
    },
    "FLANK": {
        "derivable": True,
        "note": "本体给出可判定判据：终端方位相对初始方位偏转 > 45°",
        "approximated": False,
    },
    "ENCIRCLEMENT": {
        "derivable": True,
        "note": "≥2 个同方目标自夹角 > 90° 的方向逼近同一目标（编组级）",
        "approximated": False,
    },
    "PURSUIT": {
        "derivable": True,
        "note": "对方呈 RETREAT/EVADE 且我方与其保持相关轨迹并持续接近",
        "approximated": True,
    },
    "AMBUSH": {
        "derivable": False, "fields": ["move_state", "SHOOT"],
        "note": "需要隐蔽态与'敌进入射程后立即开火'的事件序列；低速可作近似但误报高，不启用",
    },
    "SEIZE": {
        "derivable": False, "fields": ["action type 5", "objective"],
        "note": "需要'夺控动作/归属变更'事件与目标点对象",
    },
    "HOLD": {
        "derivable": True,
        "note": "在己方/指定目标点附近保持位置且对方逼近",
        "approximated": True,
    },
    "INTERDICT": {
        "derivable": False, "fields": ["choke_point", "enemy_transit"],
        "note": "需要'隘口/航道'与'敌方通过'两个语义对象",
    },
    "RECON": {
        "derivable": True,
        "note": "向未观测区域推进且与对方保持距离（不进入交战半径）",
        "approximated": True,
    },
    "SURVEILLANCE": {
        "derivable": True,
        "note": "对同一目标持续保持传感器接触且不进入交战半径",
        "approximated": False,
    },
    "TARGET_GUIDANCE": {
        "derivable": False, "fields": ["action type 9", "guided_obj_id"],
        "note": "需要'目标引导'事件与制导对象 id，本系统无此字段",
    },
    "REINFORCE": {
        "derivable": True,
        "note": "向处于接触中的同方单位靠拢，抵达后进入交战半径",
        "approximated": True,
    },
    "ESCORT": {
        "derivable": True,
        "note": "与特定同方目标保持有界邻近（本体给 < 3 hex），持续多帧",
        "approximated": True,
    },
    "FIRE_SUPPORT": {
        "derivable": True, "secondary_only": True,
        "note": "本体标 secondary only；用'同方主攻目标 + 本方在其侧后方保持火力距离'近似",
        "approximated": True,
    },
    "REDEPLOY": {
        "derivable": True,
        "note": "本体给可判定判据：> 5 hex 的无接触位移",
        "approximated": True,
    },
    "ASSEMBLE": {
        "derivable": True,
        "note": "本体给可判定判据：同方两两距离持续收敛",
        "approximated": False,
    },
    "DISPERSE": {
        "derivable": True,
        "note": "本体给可判定判据：同方两两距离持续发散",
        "approximated": False,
    },
    "FORTIFY": {
        "derivable": True,
        "note": "本体需 move_state=4；用'长时间低速且无位移'近似",
        "approximated": True,
    },
    "RETREAT": {
        "derivable": True,
        "note": "本体给可判定判据：接触后持续 > 3 帧与威胁拉开距离",
        "approximated": False,
    },
    "EVADE": {
        "derivable": True,
        "note": "本体给可判定判据：1–2 帧内方位突变 > 60° 且 5 帧内回到原航向",
        "approximated": False,
    },
    "FEINT": {
        "derivable": False, "fields": ["plan_tag"],
        "note": "本体 auto_label_grade=C 且 requires_plan_tag=true，"
                "明示'仅凭行为永不足够'。本模块**永不**推导 FEINT。",
    },
}


def derivable_codes() -> List[str]:
    return sorted(c for c, v in SUPPORTED_RULES.items() if v.get("derivable"))


def underivable_report() -> List[dict]:
    """Per-intent statement of what is missing. Surfaced in the UI and in tests."""
    out = []
    for i in intents():
        code = i["code"]
        spec = SUPPORTED_RULES.get(code) or {"derivable": False, "note": "未定义规则"}
        if spec.get("derivable"):
            continue
        out.append({
            "code": code,
            "name_zh": i.get("name_zh"),
            "parent_code": i.get("parent_code"),
            "missing_fields": spec.get("fields") or [],
            "reason": spec.get("note") or "",
        })
    return out

# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------

def _bearing_delta(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, 0..180."""
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d


class _Track(object):
    """One target's motion history, so rules can look backwards."""

    __slots__ = ("tid", "aff", "domain", "engage_km", "sensor_km", "speed_kn",
                 "route", "positions", "headings", "speeds")

    def __init__(self, tid, aff, domain, engage_km, sensor_km, speed_kn):
        self.tid = tid
        self.aff = aff
        self.domain = domain
        self.engage_km = float(engage_km or 0.0)
        self.sensor_km = float(sensor_km or 0.0)
        self.speed_kn = float(speed_kn or 0.0)
        self.route = []          # waypoints, (lat, lon)
        self.positions = []      # per frame, (lat, lon) or None
        self.headings = []       # per frame, deg or None
        self.speeds = []         # per frame, knots or None

    def offset(self, frame_idx, back):
        """Position `back` frames before frame_idx, or None."""
        j = frame_idx - back
        if j < 0 or j >= len(self.positions):
            return None
        return self.positions[j]

    def window(self, frame_idx, back):
        """Positions over (frame_idx-back, frame_idx], oldest first."""
        lo = max(0, frame_idx - back)
        out = []
        for j in range(lo, min(frame_idx + 1, len(self.positions))):
            if self.positions[j] is not None:
                out.append(self.positions[j])
        return out

    def heading_at(self, frame_idx, back=0):
        j = frame_idx - back
        if j < 0 or j >= len(self.headings):
            return None
        return self.headings[j]

    def speed_at(self, frame_idx):
        if 0 <= frame_idx < len(self.speeds):
            return self.speeds[frame_idx]
        return None


def _km(a, b):
    return geo.haversine_m(a, b) / 1000.0


def _event(code, weight, evidence, role="primary", approximated=False):
    return {
        "code": code,
        "weight": round(float(weight), 3),
        "role": role,
        "evidence": list(evidence),
        "approximated": bool(approximated),
    }


def derive(frames: Sequence[dict],
           targets: Sequence[object],
           ontology: Optional[dict] = None) -> dict:
    """Judge intents for every target on every frame.

    `frames` are the raw frames from `scenario.build_frames` (each carrying
    per-target `lat`/`lon`/`speed_kn`/`heading`). `targets` are the scenario
    `Target` objects, which carry `affiliation` / `engage_km` / `sensor_range_km`
    / `waypoints`.

    Returns `{"targets": [{tid: entry}], "summary": {...}, "underivable": [...]}`.

    The traversal only ever looks at earlier frames, so the result is
    deterministic and safe to run online.
    """
    doc = ontology or _doc()
    known = {i["code"]: i for i in (doc.get("intents") or [])}
    sec_only = {c for c, i in known.items()
                if str(i.get("primary_vs_secondary", "")).startswith("secondary")}

    by_id = {}
    for t in targets:
        by_id[t.target_id] = t

    tracks: Dict[str, _Track] = {}
    for t in targets:
        tracks[t.target_id] = _Track(t.target_id, t.affiliation, t.domain,
                                     t.engage_km, t.sensor_range_km, t.speed_kn)
        # Both entry points exist depending on whether the caller passes Target
        # objects or their dict form.
        wp = getattr(t, "waypoints", None)
        if wp is None and isinstance(t, dict):
            wp = t.get("waypoints")
        tracks[t.target_id].route = list(wp or [])

    order = [t.target_id for t in targets]

    # ---- fill histories -------------------------------------------------
    for f in frames:
        present = {}
        for s in f.get("targets") or []:
            present[s["id"]] = s
        for tid in order:
            tk = tracks[tid]
            s = present.get(tid)
            if s is None:
                tk.positions.append(None)
                tk.headings.append(None)
                tk.speeds.append(None)
                continue
            # NOTE: frames carry latitude and longitude; the ontology speaks of
            # position, so normalise to (lat, lon) once here and never again.
            tk.positions.append((s["lat"], s["lon"]))
            tk.headings.append(s.get("heading"))
            tk.speeds.append(s.get("speed_kn"))

    per_frame: List[Dict[str, dict]] = []
    rule_hits: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}

    for i, f in enumerate(frames):
        cur = {}
        for tid in order:
            tk = tracks[tid]
            if tk.positions[i] is None:
                continue
            cur[tid] = tk.positions[i]

        entries: Dict[str, dict] = {}
        for tid in order:
            tk = tracks[tid]
            here = cur.get(tid)
            if here is None:
                continue

            # Nearest opponent, and the closest friend for escort/reinforce.
            nearest_opp = None
            nearest_opp_km = None
            for oid in order:
                other = tracks[oid]
                if other.aff != tk.aff:
                    p = cur.get(oid)
                    if p is None:
                        continue
                    d = _km(here, p)
                    if nearest_opp_km is None or d < nearest_opp_km:
                        nearest_opp_km, nearest_opp = d, oid

            nearest_friend = None
            nearest_friend_km = None
            for oid in order:
                if oid == tid:
                    continue
                other = tracks[oid]
                if other.aff != tk.aff:
                    continue
                p = cur.get(oid)
                if p is None:
                    continue
                d = _km(here, p)
                if nearest_friend_km is None or d < nearest_friend_km:
                    nearest_friend_km, nearest_friend = d, oid

            events: List[dict] = []
            missing: List[str] = []
            engaged = (nearest_opp_km is not None
                       and tk.engage_km > 0 and nearest_opp_km <= tk.engage_km)
            in_contact = (nearest_opp_km is not None and tk.engage_km > 0
                          and nearest_opp_km <= max(tk.engage_km, tk.sensor_km))
            speed = tk.speed_at(i)
            hdg = tk.heading_at(i)

            # ---- movement vector over the last 3 frames ----
            w = tk.window(i, 3)
            closing = None
            if len(w) >= 2 and nearest_opp_km is not None:
                p_opp = cur.get(nearest_opp)
                if p_opp is not None:
                    d0 = _km(w[0], p_opp)
                    d1 = _km(w[-1], p_opp)
                    closing = d1 - d0          # < 0 => getting closer

            # ---- RETREAT: sustained separation from the principal threat ----
            if nearest_opp_km is not None and len(w) >= 3:
                seps = [_km(p, cur[nearest_opp]) for p in w]
                growing = all(seps[k + 1] > seps[k] for k in range(len(seps) - 1))
                if in_contact and growing:
                    events.append(_event(
                        "RETREAT", 0.9,
                        ["与 %s 连续 %d 帧拉开距离（%.1f→%.1f km）"
                         % (nearest_opp, len(seps), seps[0], seps[-1]),
                         "接触中：最近对方 %.1f km ≤ 交战/传感半径 %.1f km"
                         % (nearest_opp_km, max(tk.engage_km, tk.sensor_km))]))

            # ---- EVADE: abrupt turn then return to the prior heading ----
            if hdg is not None and nearest_opp_km is not None and tk.engage_km > 0:
                turned = False
                for back in (1, 2):
                    h0 = tk.heading_at(i, back + 1)
                    h1 = tk.heading_at(i, back)
                    if h0 is None or h1 is None:
                        continue
                    if _bearing_delta(h0, h1) > 60.0:
                        turned = True
                        break
                if turned:
                    base = tk.heading_at(i, 3)
                    if base is not None and _bearing_delta(base, hdg) <= 20.0:
                        events.append(_event(
                            "EVADE", 0.8,
                            ["1–2 帧内方位突变 > 60°，且 %d 帧内回到原航向 %.0f°"
                             % (3, base),
                             "处于对方交战半径内（%.1f km）" % nearest_opp_km]))

            # ---- ASSAULT: sustained closure into engagement range ----
            if engaged and closing is not None and closing < 0:
                events.append(_event(
                    "ASSAULT", 0.85,
                    ["已进入 %s 交战半径（%.1f ≤ %.1f km）"
                     % (nearest_opp, nearest_opp_km, tk.engage_km),
                     "近 3 帧持续接近（Δ %.1f km）" % closing,
                     "近似：本体另需 SHOOT 事件，本系统无该字段"]))

            # ---- FLANK: terminal bearing differs from initial by > 45 deg ----
            if nearest_opp_km is not None and nearest_opp_km <= 4.0 * max(tk.engage_km, 20.0):
                p_opp = cur.get(nearest_opp)
                far = tk.offset(i, 12) or (tk.window(i, 12) or [None])[0]
                if p_opp is not None and far is not None:
                    b_far = geo.initial_bearing_deg(far, p_opp)
                    b_now = geo.initial_bearing_deg(here, p_opp)
                    delta = _bearing_delta(b_far, b_now)
                    if delta > 45.0 and closing is not None and closing < 0:
                        events.append(_event(
                            "FLANK", 0.7,
                            ["终端方位 %.0f° 相对初始方位 %.0f° 偏转 %.0f° > 45°"
                             % (b_now, b_far, delta),
                             "同时仍在接近（Δ %.1f km）" % closing]))

            # ---- SURVEILLANCE: persistent sensor contact, no engagement ----
            if (nearest_opp_km is not None and tk.sensor_km > 0
                    and nearest_opp_km <= tk.sensor_km
                    and not engaged):
                held = 0
                for back in range(1, 6):
                    p = tk.offset(i, back)
                    if p is None:
                        break
                    pp = tracks[nearest_opp].offset(i, back)
                    if pp is None:
                        break
                    if _km(p, pp) > tk.sensor_km:
                        break
                    held += 1
                if held >= 5:
                    events.append(_event(
                        "SURVEILLANCE", 0.75,
                        ["对 %s 连续 %d 帧保持传感器接触（≤ %.0f km）"
                         % (nearest_opp, held, tk.sensor_km),
                         "未进入交战半径（%.1f > %.1f km）"
                         % (nearest_opp_km, tk.engage_km)]))

            # ---- RECON: advancing without contact ----
            if not in_contact and speed and speed > 0:
                moved = 0.0
                if len(w) >= 2:
                    moved = _km(w[0], w[-1])
                if moved > max(2.0, (speed * 0.5144 * 3 * 30) / 1000.0 * 0.4):
                    events.append(_event(
                        "RECON", 0.5,
                        ["无接触推进：近 3 帧位移 %.1f km" % moved,
                         "与最近对方 %.1f km，超出传感半径 %.0f km"
                         % (nearest_opp_km, tk.sensor_km) if nearest_opp_km is not None
                         else "无对方目标"]))
            elif nearest_opp_km is None and speed and speed > 0:
                events.append(_event("RECON", 0.5, ["无可比对方目标，按推进判定"]))

            # ---- HOLD: staying put near a friendly objective while threatened ----
            if speed is not None and speed < 5.0 and not engaged:
                stayed = 0
                for back in range(1, 6):
                    p = tk.offset(i, back)
                    if p is None:
                        break
                    if _km(p, here) > 5.0:
                        break
                    stayed += 1
                if stayed >= 5 and nearest_opp_km is not None and nearest_opp_km <= 3.0 * max(tk.engage_km, 20.0):
                    events.append(_event(
                        "HOLD", 0.55,
                        ["连续 %d 帧在 5 km 内基本静止（%.1f kn）" % (stayed, speed),
                         "对方 %s 在 %.1f km 外逼近" % (nearest_opp, nearest_opp_km),
                         "近似：本体要求'己方/指定目标点'，本系统以静止位置代替"]))

            # ---- FORTIFY: sustained slow, no displacement ----
            if speed is not None and speed < 3.0:
                still = 0
                for back in range(1, 8):
                    p = tk.offset(i, back)
                    if p is None:
                        break
                    if _km(p, here) > 2.0:
                        break
                    still += 1
                if still >= 6:
                    events.append(_event(
                        "FORTIFY", 0.5,
                        ["连续 %d 帧位移 < 2 km 且航速 %.1f kn" % (still, speed),
                         "近似：本体用 move_state=4 隐蔽态，本系统无该字段"]))

            # ---- REDEPLOY: long non-contact move (ontology > 5 hex) ----
            if not in_contact:
                far = tk.offset(i, 30)
                if far is not None:
                    moved = _km(far, here)
                    if moved > 5.0 * HEX_KM:
                        events.append(_event(
                            "REDEPLOY", 0.5,
                            ["30 帧无接触位移 %.1f km > 5 hex（%d km，HEX_KM=%g）"
                             % (moved, int(5 * HEX_KM), HEX_KM)]))

            # ---- REINFORCE: closing on a friendly that is in contact ----
            if nearest_friend is not None and nearest_friend_km is not None:
                friend_tk = tracks[nearest_friend]
                fw = friend_tk.window(i, 3)
                friend_contact = False
                for oid in order:
                    other = tracks[oid]
                    if other.aff == tk.aff:
                        continue
                    pp = cur.get(oid)
                    if pp is None or not fw:
                        continue
                    if _km(fw[-1], pp) <= max(friend_tk.engage_km, friend_tk.sensor_km):
                        friend_contact = True
                        break
                closer = None
                if len(w) >= 2 and fw:
                    closer = _km(w[-1], fw[-1]) - _km(w[0], fw[0])
                if friend_contact and closer is not None and closer < 0:
                    events.append(_event(
                        "REINFORCE", 0.6,
                        ["%s 处于接触中，本方近 3 帧向其靠拢（Δ %.1f km）"
                         % (nearest_friend, closer)]))

            # ---- ESCORT: bounded proximity to a friend (ontology < 3 hex) ----
            if nearest_friend is not None and nearest_friend_km is not None:
                if nearest_friend_km < 3.0 * HEX_KM:
                    held = 0
                    for back in range(1, 6):
                        p = tk.offset(i, back)
                        pp = tracks[nearest_friend].offset(i, back)
                        if p is None or pp is None:
                            break
                        if _km(p, pp) > 3.0 * HEX_KM:
                            break
                        held += 1
                    if held >= 5:
                        events.append(_event(
                            "ESCORT", 0.55,
                            ["与 %s 连续 %d 帧保持 < 3 hex（%.0f km）邻近"
                             % (nearest_friend, held, 3.0 * HEX_KM)]))

            # ---- FIRE_SUPPORT: secondary-only, co-occurrence required ----
            engaged_friends = [oid for oid in order
                               if tracks[oid].aff == tk.aff and oid != tid]
            if engaged_friends and nearest_opp_km is not None and not engaged:
                if tk.sensor_km > 0 and nearest_opp_km <= tk.sensor_km:
                    events.append(_event(
                        "FIRE_SUPPORT", 0.45,
                        ["同方目标在与 %s 对抗，本方保持火力距离（%.1f ≤ %.0f km）"
                         % (nearest_opp, nearest_opp_km, tk.sensor_km),
                         "本体标 secondary only：只能作为次要意图"],
                        role="secondary", approximated=True))

            if not events:
                missing.append("action type / mission_type / move_state")

            # ---- drop intents this system cannot judge, keep the reason ----
            events = [e for e in events
                      if (SUPPORTED_RULES.get(e["code"]) or {}).get("derivable")]

            # ---- separate primary / secondary, sort by weight ----
            prim = sorted([e for e in events if e["code"] not in sec_only
                           and e["role"] != "secondary"],
                          key=lambda e: -e["weight"])
            sec = sorted([e for e in events if e["code"] in sec_only
                          or e["role"] == "secondary"],
                         key=lambda e: -e["weight"])

            # ---- status layer ----
            status = None
            if not prim and not sec:
                status = "BACKGROUND" if in_contact is False else "UNKNOWN"
                if missing:
                    status = "UNKNOWN"
            elif prim:
                strong = [e for e in prim if e["weight"] >= MIN_WEIGHT]
                if not strong:
                    status = "UNKNOWN"
                elif (len(strong) >= 2
                      and (strong[0]["weight"] - strong[1]["weight"])
                      <= AMBIGUITY_MARGIN * strong[0]["weight"]):
                    status = "AMBIGUOUS"

            for e in prim + sec:
                rule_hits[e["code"]] = rule_hits.get(e["code"], 0) + 1
            if status:
                status_counts[status] = status_counts.get(status, 0) + 1

            entries[tid] = {
                "intent_set": prim + sec,
                "intent_status": status,
                "primary_intent": (prim[0]["code"] if prim else None),
                "missing_evidence": sorted(set(missing)),
            }
        per_frame.append(entries)

    return {
        "targets": per_frame,
        "summary": {
            "rule_hits": rule_hits,
            "status_counts": status_counts,
            "operational_targets": len([t for t in targets if getattr(t, "affiliation", "")]),
            "derivable": derivable_codes(),
            "underivable": underivable_report(),
            "ontology_version": doc.get("version"),
        },
    }


def merge_into_frames(inference_frames: Sequence[dict], derived: dict) -> int:
    """Copy derived intents onto `inference.infer()`'s per-frame target entries.

    Kept as a separate step (rather than having `inference.py` call this module)
    so the inference layer stays free of any IO and of any dependency on the
    ontology file. Returns the number of targets annotated.

    Existing `intent` / `intent_label` fields are **left untouched**: the brief
    generator and any older client still read them.
    """
    per_frame = derived.get("targets") or []
    n = 0
    for i, frame in enumerate(inference_frames):
        if i >= len(per_frame):
            break
        entries = per_frame[i] or {}
        tgts = frame.get("targets") or {}
        for tid, entry in entries.items():
            if tid in tgts:
                tgts[tid].update(entry)
                n += 1
    return n
