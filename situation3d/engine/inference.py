"""Three-level situation inference for situation3d.

The platform is required to *infer* a situation rather than merely replay
positions, and to show the result at three levels of abstraction:

===== ========== ==========================================================
Level Chinese    Question answered
===== ========== ==========================================================
L1    目标级TS  What is this platform doing, and how threatened is it?
L2    编组级TS  How do the red and blue groupings relate to each other?
L3    区域级TS  What is the situation inside each hotspot region?
===== ========== ==========================================================

Every level is produced by an explicit, inspectable rule set -- no black box --
because the output is meant to be defended in a review. The scoring weights and
the code tables are exported in the ``model`` block of the result so the front
end can display the reasoning behind any number it shows.

Conventions
-----------
Distances are kilometres, bearings and headings degrees. Threat is 0-100 and
rises with the platform's own combat value, its proximity to the opposing side
and the number of opposing sensors that can see it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from . import geo

# --------------------------------------------------------------------------
# Code tables. These are exported verbatim in the `model` block so the front
# end never has to duplicate them.
# --------------------------------------------------------------------------

ALERT_CODES = {0: "正常", 1: "注意", 2: "警告", 3: "严重"}
RELATION_CODES = {0: "无接触", 1: "远距监视", 2: "近距对峙", 3: "高危接触"}
LEVEL_CODES = {1: "和缓", 2: "关注", 3: "紧张", 4: "高危"}
# Formation descriptors are domain specific: "疏散编队" reads correctly for
# ships, but two aircraft 200 km apart are flying a patrol line, not a formation.
POSTURE_CODES = {
    "sea": {0: "单舰", 1: "密集编队", 2: "疏散编队", 3: "分散部署"},
    "air": {0: "单机", 1: "密集编队", 2: "双机编队", 3: "巡逻航线"},
    "land": {0: "单台", 1: "集中配置", 2: "机动转移", 3: "分散配置", 4: "阵地待机"},
}
DOMINANCE_CODES = {0: "无", 1: "红方", 2: "蓝方", 3: "均势"}

# L1 state / intent, keyed by the platform's declared role. Keeping this a
# lookup table rather than a chain of conditionals makes the mapping auditable:
# every platform class has exactly one declared behaviour.
ROLE_STATE: Dict[str, int] = {
    "编队指挥": 0,
    "伴随护航": 1,
    "后勤保障": 0,
    "岸海打击": 6,
    "要地防空": 3,
    "区域防空": 3,
    "空中掩护": 4,
    "空中指挥": 5,
    "空中警戒": 5,
    "海上拦截": 1,
    "近海巡逻": 1,
    "战斗空中巡逻": 4,
    # Campaign v1 land roles. These keys are only used by the dense-transfer
    # case; old cases keep their existing lookup results.
    "地面机动": 2,
    "地面运输": 2,
    "地面保障": 3,
    "节点等待": 3,
}

STATE_LABELS = {
    0: "编队巡航",
    1: "海上巡逻",
    2: "机动转移",
    3: "阵地待机",
    4: "空中战斗巡逻",
    5: "空中警戒",
    6: "岸海封控",
    7: "抵近跟踪",
}

ROLE_INTENT: Dict[str, int] = {
    "编队指挥": 0,
    "伴随护航": 1,
    "后勤保障": 2,
    "岸海打击": 3,
    "要地防空": 4,
    "区域防空": 4,
    "空中掩护": 5,
    "空中指挥": 6,
    "空中警戒": 6,
    "海上拦截": 7,
    "近海巡逻": 8,
    "战斗空中巡逻": 10,
    "地面机动": 13,
    "地面运输": 13,
    "地面保障": 14,
    "节点等待": 14,
}

INTENT_LABELS = {
    0: "编队机动",
    1: "伴随护航",
    2: "后勤保障",
    3: "岸海打击待机",
    4: "区域防空警戒",
    5: "空中掩护",
    6: "空中预警指挥",
    7: "海上拦截",
    8: "近海巡逻",
    9: "要地防空",
    10: "战斗空中巡逻",
    11: "跟踪监视",
    12: "接近查证",
    13: "地面机动",
    14: "地面保障",
}

# Own combat value, 0-35. Feeds the L1 threat score.
PLATFORM_WEIGHT: Dict[str, float] = {
    "coastal_missile": 33.0,
    "fighter": 30.0,
    "destroyer": 28.0,
    "sam_battalion": 22.0,
    "frigate": 22.0,
    "corvette": 15.0,
    "aew": 12.0,
    "replenishment": 6.0,
    "armor": 24.0,
    "artillery": 20.0,
    "helicopter": 18.0,
    "uav": 8.0,
    "landing_ship": 10.0,
    "transport": 4.0,
    "outpost": 8.0,
    # 大类兜底同样给一个中位数权重，避免未知型号落到默认值后偏高/偏低。
    "generic_sea": 15.0,
    "generic_land": 15.0,
    "generic_air": 15.0,
    "unknown": 10.0,
    # Campaign v1 synthetic platforms. Support vehicles are deliberately low:
    # a truck column must not read as a combat formation.
    "light_vehicle": 10.0,
    "truck": 6.0,
    "engineer_vehicle": 5.0,
    "transport_ship": 12.0,
}

# Proximity / coverage contributions to the L1 threat score. The proximity
# term decays as exp(-PROXIMITY_DECAY * d / R), where R is the nearest opposing
# platform's influence radius. A decay of 2.0 means a target sitting on the edge
# of the opponent's reach contributes about 7 of the 50 available points, while
# one inside a quarter of that reach contributes about 30.
PROXIMITY_MAX = 50.0
PROXIMITY_DECAY = 2.0
COVERAGE_PER_SENSOR = 3.0
COVERAGE_MAX = 9.0

# L1 alert thresholds on the 0-100 threat score.
ALERT_THRESHOLDS = ((70.0, 3), (45.0, 2), (25.0, 1))

# L2 relation thresholds, kilometres, on the closest opposing pair.
RELATION_THRESHOLDS = ((60.0, 3), (150.0, 2), (300.0, 1))

# Platforms closer than this are treated as operating against each other.
ENGAGEMENT_KM = 60.0
# Platforms closer than this are considered to have made contact.
CONTACT_KM = 300.0


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _km(a: Sequence[float], b: Sequence[float]) -> float:
    """Great-circle distance in kilometres between (lat, lon) pairs."""
    return geo.haversine_m((a[0], a[1]), (b[0], b[1])) / 1000.0


def _centroid(points: List[Sequence[float]]) -> List[float]:
    if not points:
        return [0.0, 0.0]
    return [
        round(sum(p[0] for p in points) / len(points), 4),
        round(sum(p[1] for p in points) / len(points), 4),
    ]


def _spread_km(points: List[Sequence[float]]) -> float:
    """Maximum pairwise separation, a cheap proxy for formation tightness."""
    if len(points) < 2:
        return 0.0
    worst = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            worst = max(worst, _km(points[i], points[j]))
    return worst


# --------------------------------------------------------------------------
# L1 -- target level
# --------------------------------------------------------------------------


def _threat(target, position, opponents, proximity_floor_km: float = 15.0) -> tuple:
    """Threat score for one target, plus the opposing target driving it.

    Three additive terms, all visible in the returned breakdown:

    * the platform's own combat value (constant per class);
    * a proximity term driven by the *nearest opponent's weapon engagement
      radius* -- an unarmed platform projects no threat of its own no matter how
      large its radar picture is, which is what keeps a distant AEW orbit from
      pinning every score in the theatre at "warning";
    * a coverage term counting how many opposing *sensors* can already see it,
      measured against each opponent's sensor range rather than its weapons
      reach.
    """
    base = PLATFORM_WEIGHT.get(target.icon, 15.0)

    nearest = None
    nearest_km = float("inf")
    covered_by = 0
    coverage_raw = 0.0

    for opp in opponents:
        d = _km(position, opp["pos"])
        if d < nearest_km:
            nearest_km = d
            nearest = opp
        rng = opp["sensor_range_km"]
        if rng > 0.0 and d <= rng:
            covered_by += 1
            # Scored by depth: a target on the edge of a radar's range is
            # barely tracked, one directly overhead is tracked well. Without
            # this every wide-area sensor saturates the term and it stops
            # carrying information.
            coverage_raw += COVERAGE_PER_SENSOR * (1.0 - d / rng)

    if nearest is None:
        return 0.0, None, nearest_km, 0, {"base": base, "proximity": 0.0, "coverage": 0.0}

    reach = max(nearest["engage_km"], proximity_floor_km)
    proximity = PROXIMITY_MAX * pow(2.718281828, -PROXIMITY_DECAY * (nearest_km / reach))
    coverage = min(COVERAGE_MAX, coverage_raw)

    total = min(100.0, base + proximity + coverage)
    return (
        round(total, 1),
        nearest["id"],
        round(nearest_km, 1),
        covered_by,
        {"base": round(base, 1), "proximity": round(proximity, 1), "coverage": round(coverage, 1)},
    )


def _alert_for(threat: float) -> int:
    for threshold, code in ALERT_THRESHOLDS:
        if threat >= threshold:
            return code
    return 0


# --------------------------------------------------------------------------
# L2 -- grouping level
# --------------------------------------------------------------------------


def _posture(targets, spread: float) -> int:
    """Formation descriptor code for a grouping."""
    if len(targets) == 1:
        return 0
    if all(t.domain == "land" for t in targets):
        # Ground units are either dug in or moving; spacing says little.
        return 4 if all(t.speed_kn < 5.0 for t in targets) else 2
    if spread < 15.0:
        return 1
    if spread < 60.0:
        return 2
    return 3


def _posture_label(domain: str, code: int) -> str:
    return POSTURE_CODES.get(domain, POSTURE_CODES["sea"]).get(code, "未知")


def _relation(min_km: float,
              thresholds: Sequence[Tuple[float, int]] = RELATION_THRESHOLDS) -> int:
    for threshold, code in thresholds:
        if min_km < threshold:
            return code
    return 0


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------


def infer(frames: List[dict], targets, hotspot_names: Sequence[str] = (),
          hotspots: Optional[Sequence[dict]] = None,
          relation_thresholds: Optional[Sequence[Tuple[float, int]]] = None,
          engagement_km: Optional[float] = None,
          contact_km: Optional[float] = None,
          proximity_floor_km: float = 15.0) -> dict:
    """Run the three-level inference over a frame sequence.

    Returns ``{"model": ..., "summary": ..., "frames": [...]}`` where each entry
    of ``frames`` is aligned index-for-index with the input ``frames``.

    The optional parameters let a local-scale case use its own thresholds and
    hotspot list without changing the global constants. Omitting them reproduces
    the old theatre-scale behaviour exactly.
    """
    from . import scenario as scenario_mod

    target_by_id = {t.target_id: t for t in targets}
    order = [t.target_id for t in targets]
    domains = sorted({t.domain for t in targets})
    affiliations = sorted({t.affiliation for t in targets})

    if relation_thresholds is None:
        relation_thresholds = RELATION_THRESHOLDS
    if engagement_km is None:
        engagement_km = ENGAGEMENT_KM
    if contact_km is None:
        contact_km = CONTACT_KM
    if hotspots is None:
        hotspots = [h for h in scenario_mod.HOTSPOTS
                    if h["name"] in set(hotspot_names)]
    else:
        wanted = set(hotspot_names) if hotspot_names else None
        hotspots = [h for h in hotspots
                    if wanted is None or h["name"] in wanted]
    # Pre-compute the bounding box test inputs once.
    boxes = [(h["bbox"], h["name"]) for h in hotspots]

    out_frames: List[dict] = []
    peak_level = 1
    peak_tick = 0
    threat_sum = 0.0
    threat_n = 0
    level_hist = {code: 0 for code in LEVEL_CODES}

    for frame in frames:
        positions = {s["id"]: [s["lat"], s["lon"]] for s in frame["targets"]}
        alts = {s["id"]: s["alt"] for s in frame["targets"]}

        # ---- L1 ---------------------------------------------------------
        l1 = {}
        for tid in order:
            t = target_by_id[tid]
            pos = positions[tid]
            opponents = [
                {
                    "id": oid,
                    "pos": positions[oid],
                    "engage_km": target_by_id[oid].engage_km,
                    "sensor_range_km": target_by_id[oid].sensor_range_km,
                }
                for oid in order
                if target_by_id[oid].affiliation != t.affiliation
            ]
            threat, nearest_id, nearest_km, covered, breakdown = _threat(
                t, pos, opponents, proximity_floor_km=proximity_floor_km)
            alert = _alert_for(threat)

            state = ROLE_STATE.get(t.role, 0)
            intent = ROLE_INTENT.get(t.role, 0)
            # Dynamic override: anything close to the other side is tracking it,
            # which is what an operator would actually be told.
            if nearest_km <= contact_km and nearest_id:
                intent = 11
            if nearest_km <= engagement_km and t.domain in ("sea", "air"):
                state = 7
                intent = 12

            l1[tid] = {
                "state": state,
                "state_label": STATE_LABELS[state],
                "intent": intent,
                "intent_label": INTENT_LABELS[intent],
                "threat": threat,
                "alert": alert,
                "alert_label": ALERT_CODES[alert],
                "nearest_id": nearest_id,
                "nearest_km": nearest_km if nearest_id else None,
                "covered_by": covered,
                "threat_terms": breakdown,
            }

        # ---- L2 ---------------------------------------------------------
        l2 = []
        for aff in affiliations:
            for dom in domains:
                members = [tid for tid in order
                           if target_by_id[tid].affiliation == aff and target_by_id[tid].domain == dom]
                if not members:
                    continue
                pts = [positions[m] for m in members]
                centre = _centroid(pts)
                spread = round(_spread_km(pts), 1)

                opp_members = [tid for tid in order
                               if target_by_id[tid].affiliation != aff and target_by_id[tid].domain == dom]
                if opp_members:
                    min_km = round(min(_km(positions[a], positions[b])
                                       for a in members for b in opp_members), 1)
                else:
                    min_km = None

                l2.append({
                    "key": aff + ":" + dom,
                    "affiliation": aff,
                    "domain": dom,
                    "count": len(members),
                    "members": members,
                    "centroid": centre,
                    "spread_km": spread,
                    "posture": _posture([target_by_id[m] for m in members], spread),
                    "posture_label": _posture_label(
                        dom, _posture([target_by_id[m] for m in members], spread)),
                    "opposing_key": (("blue" if aff == "red" else "red") + ":" + dom) if opp_members else None,
                    "min_distance_km": min_km,
                    "relation": _relation(min_km, relation_thresholds) if min_km is not None else 0,
                    "relation_label": RELATION_CODES[_relation(min_km, relation_thresholds) if min_km is not None else 0],
                    "mean_threat": round(sum(l1[m]["threat"] for m in members) / len(members), 1),
                })

        # ---- L3 ---------------------------------------------------------
        # Emitted as a compact array aligned with `hotspots`, because the
        # region list itself never changes during a run.
        l3 = []
        frame_level = 1
        for bbox, name in boxes:
            w, s, e, n = bbox
            inside = [tid for tid in order
                      if s <= positions[tid][0] <= n and w <= positions[tid][1] <= e]
            n_red = sum(1 for tid in inside if target_by_id[tid].affiliation == "red")
            n_blue = sum(1 for tid in inside if target_by_id[tid].affiliation == "blue")

            if inside:
                max_threat = max(l1[tid]["threat"] for tid in inside)
            else:
                max_threat = 0.0
            level = _alert_to_level(max_threat)
            # Both sides present in one box is a shared area, which is worth
            # flagging on its own -- but only as "关注". Escalating to 紧张 or
            # 高危 is reserved for actual proximity, otherwise a single large
            # box covering the whole theatre pins the level for the entire run.
            if n_red > 0 and n_blue > 0:
                level = max(level, 2)

            if n_red > n_blue:
                dominance = 1
            elif n_blue > n_red:
                dominance = 2
            elif n_red and n_blue:
                dominance = 3
            else:
                dominance = 0

            l3.append([level, n_red, n_blue, dominance])
            frame_level = max(frame_level, level)

        overall_note = _overall_note(frame_level)
        level_hist[frame_level] += 1
        if frame_level > peak_level:
            peak_level = frame_level
            peak_tick = frame["tick"]

        threats = [l1[tid]["threat"] for tid in order]
        threat_sum += sum(threats)
        threat_n += len(threats)

        out_frames.append({
            "tick": frame["tick"],
            "targets": l1,
            "groups": l2,
            "regions": l3,
            "overall": {
                "level": frame_level,
                "level_label": LEVEL_CODES[frame_level],
                "mean_threat": round(sum(threats) / len(threats), 1),
                "max_threat": round(max(threats), 1),
                "engaged_pairs": sum(
                    1 for g in l2 if g["affiliation"] == "red" and g["relation"] >= 2
                ),
                "note": overall_note,
            },
        })

    return {
        "model": {
            "levels": [
                {"id": "L1", "name": "目标级TS",
                 "description": "单平台状态、意图与威胁度，由装备类别权重、与对方最近目标距离、被对方传感器覆盖数三项合成。"},
                {"id": "L2", "name": "编组级TS",
                 "description": "按红蓝双方与海陆空域聚合为编组，给出编队形态、编组质心、与对方同域编组的最小距离及对抗关系。"},
                {"id": "L3", "name": "区域级TS",
                 "description": "逐热点区域统计双方兵力存在与最高威胁，输出区域TS等级与主导方；双方共处一区即判为紧张及以上。"},
            ],
            "alert_codes": ALERT_CODES,
            "relation_codes": RELATION_CODES,
            "level_codes": LEVEL_CODES,
            "posture_codes": POSTURE_CODES,
            "dominance_codes": DOMINANCE_CODES,
            "state_labels": STATE_LABELS,
            "intent_labels": INTENT_LABELS,
            "weights": {
                "platform_weight": PLATFORM_WEIGHT,
                "proximity_max": PROXIMITY_MAX,
                "proximity_decay": PROXIMITY_DECAY,
                "coverage_per_sensor": COVERAGE_PER_SENSOR,
                "coverage_max": COVERAGE_MAX,
            },
            "thresholds": {
                "alert": [list(t) for t in ALERT_THRESHOLDS],
                "relation_km": [list(t) for t in relation_thresholds],
                "contact_km": contact_km,
                "engagement_km": engagement_km,
                "proximity_floor_km": proximity_floor_km,
            },
        },
        "summary": {
            "peak_level": peak_level,
            "peak_level_label": LEVEL_CODES[peak_level],
            "peak_tick": peak_tick,
            "mean_threat": round(threat_sum / max(1, threat_n), 1),
            "level_histogram": {LEVEL_CODES[k]: v for k, v in level_hist.items()},
            "target_count": len(order),
            "group_count": len(domains) * len(affiliations),
        },
        "frames": out_frames,
    }


def _alert_to_level(max_threat: float) -> int:
    """Map a region's worst threat score onto the 1-4 region level scale."""
    if max_threat >= 70.0:
        return 4
    if max_threat >= 45.0:
        return 3
    if max_threat >= 25.0:
        return 2
    return 1


def _overall_note(level: int) -> str:
    return {
        1: "双方兵力分散，未形成有效接触，整体TS和缓。",
        2: "存在远距监视关系，需持续跟踪对方动向。",
        3: "双方兵力在同一区域共处或已进入对方影响范围，TS紧张。",
        4: "存在高危接触，双方兵力距离已进入武器有效作用范围。",
    }[level]
