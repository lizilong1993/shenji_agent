"""Lightweight three-level inference for campaign.v3 datasets.

The legacy datasets carry `inference_frames` produced by ``engine/inference.py``
from static target definitions.  Campaign datasets are generated from the
wargame core instead, so this module recomputes the same three levels per frame
from the campaign frames and the objective list:

  L1  target : threat = platform weight + proximity + sensor coverage
  L2  group  : red/blue x land/sea/air centroid, spread, relation
  L3  region : objectives -> level, red/blue counts, dominance

It intentionally mirrors the public weights in ``engine/inference.py`` so the
right-hand panel and the hotspot layer keep the same semantics.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

ALERT_CODES = {0: "正常", 1: "注意", 2: "警告", 3: "严重"}
RELATION_CODES = {0: "无接触", 1: "远距监视", 2: "近距对峙", 3: "高危接触"}
LEVEL_CODES = {1: "和缓", 2: "关注", 3: "紧张", 4: "高危"}
DOMINANCE_CODES = {0: "无", 1: "红方", 2: "蓝方", 3: "均势"}

PLATFORM_WEIGHT = {
    "destroyer": 28.0, "frigate": 22.0, "replenishment": 6.0,
    "corvette": 15.0, "coastal_missile": 33.0, "sam_battalion": 22.0,
    "armor": 24.0, "artillery": 20.0, "fighter": 30.0, "aew": 12.0,
}
PROXIMITY_MAX = 50.0
PROXIMITY_DECAY = 2.0
COVERAGE_PER_SENSOR = 3.0
COVERAGE_MAX = 9.0
ALERT_THRESHOLDS = ((70.0, 3), (45.0, 2), (25.0, 1))
RELATION_THRESHOLDS = ((60.0, 3), (150.0, 2), (300.0, 1))

STATE_LABELS = {
    "sea": "编队巡航", "air": "空中战斗巡逻", "land": "地面机动",
}
INTENT_LABELS = {
    "sea": "海上行动", "air": "空中行动", "land": "地面行动",
}


def _km(a, b) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


def _level(threat: float) -> int:
    if threat >= 70.0:
        return 4
    if threat >= 45.0:
        return 3
    if threat >= 25.0:
        return 2
    return 1


def _alert(threat: float) -> int:
    for t, code in ALERT_THRESHOLDS:
        if threat >= t:
            return code
    return 0


def _relation(km: float) -> int:
    for t, code in RELATION_THRESHOLDS:
        if km < t:
            return code
    return 0


def _posture(domain: str, members: Sequence[dict], spread: float) -> int:
    if len(members) == 1:
        return 0
    if domain == "land":
        return 2
    if spread < 15.0:
        return 1
    if spread < 60.0:
        return 2
    return 3


POSTURE_LABELS = {
    "sea": {0: "单舰", 1: "密集编队", 2: "疏散编队", 3: "分散部署"},
    "air": {0: "单机", 1: "密集编队", 2: "双机编队", 3: "巡逻航线"},
    "land": {0: "单台", 1: "集中配置", 2: "机动转移", 3: "分散配置", 4: "阵地待机"},
}


def infer_campaign(frames: List[dict], targets: List[dict],
                   objectives: List[dict]) -> dict:
    """Return {'model', 'summary', 'frames'} shaped like engine.inference."""
    by_id = {str(t["id"]): t for t in targets}
    order = [str(t["id"]) for t in targets]
    domains = sorted({t["domain"] for t in targets})
    affiliations = sorted({t["affiliation"] for t in targets})
    # Objective boxes (0.04 deg) for the region level.
    regions = []
    for o in objectives:
        center = o.get("center")
        if center:
            lat, lon = center
        else:
            lat, lon = o.get("lat"), o.get("lon")
        if lat is None or lon is None:
            continue
        regions.append((o.get("name") or str(o.get("coord")), lat, lon))

    out_frames: List[dict] = []
    peak_level, peak_tick = 1, 0
    level_hist = {c: 0 for c in LEVEL_CODES}
    threat_sum, threat_n = 0.0, 0

    for frame in frames:
        units = [t for t in frame.get("targets") or [] if t.get("alive", True)]
        pos = {str(u["id"]): (u["lat"], u["lon"]) for u in units}

        l1 = {}
        for tid in order:
            if tid not in pos:
                continue
            t = by_id[tid]
            me = pos[tid]
            nearest, nkm = None, float("inf")
            covered = 0
            coverage_raw = 0.0
            for oid, p in pos.items():
                o = by_id.get(oid)
                if not o or o["affiliation"] == t["affiliation"]:
                    continue
                d = _km(me, p)
                if d < nkm:
                    nkm, nearest = d, oid
                sr = float(o.get("influence_km") or 0.0)
                if sr > 0 and d <= sr:
                    covered += 1
                    coverage_raw += COVERAGE_PER_SENSOR * (1.0 - d / sr)
            base = PLATFORM_WEIGHT.get(t.get("icon"), 15.0)
            if nearest is None:
                threat, nearest_km = 0.0, None
            else:
                reach = max(float(by_id[nearest].get("engage_km") or 0.0), 15.0)
                prox = PROXIMITY_MAX * math.exp(-PROXIMITY_DECAY * nkm / reach)
                threat = min(100.0, base + prox + min(COVERAGE_MAX, coverage_raw))
                nearest_km = round(nkm, 1)
            threat = round(threat, 1)
            l1[tid] = {
                "state": 0,
                "state_label": STATE_LABELS.get(t["domain"], "机动"),
                "intent": 0,
                "intent_label": INTENT_LABELS.get(t["domain"], "行动"),
                "threat": threat,
                "alert": _alert(threat),
                "alert_label": ALERT_CODES[_alert(threat)],
                "nearest_id": nearest,
                "nearest_km": nearest_km,
                "covered_by": covered,
                "threat_terms": {"base": base, "proximity": round(
                    threat - base - min(COVERAGE_MAX, coverage_raw), 1),
                    "coverage": round(min(COVERAGE_MAX, coverage_raw), 1)},
            }

        l2 = []
        for aff in affiliations:
            for dom in domains:
                members = [tid for tid in order
                           if tid in pos and by_id[tid]["affiliation"] == aff
                           and by_id[tid]["domain"] == dom]
                if not members:
                    continue
                pts = [pos[m] for m in members]
                centre = [round(sum(p[0] for p in pts) / len(pts), 4),
                          round(sum(p[1] for p in pts) / len(pts), 4)]
                spread = 0.0
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        spread = max(spread, _km(pts[i], pts[j]))
                spread = round(spread, 1)
                opp = [tid for tid in order if tid in pos
                       and by_id[tid]["affiliation"] != aff
                       and by_id[tid]["domain"] == dom]
                min_km = round(min(_km(pos[a], pos[b]) for a in members for b in opp), 1) if opp else None
                l2.append({
                    "key": aff + ":" + dom,
                    "affiliation": aff,
                    "domain": dom,
                    "count": len(members),
                    "members": members,
                    "centroid": centre,
                    "spread_km": spread,
                    "posture": _posture(dom, members, spread),
                    "posture_label": POSTURE_LABELS.get(dom, POSTURE_LABELS["sea"])[
                        _posture(dom, members, spread)],
                    "opposing_key": (("blue" if aff == "red" else "red") + ":" + dom) if opp else None,
                    "min_distance_km": min_km,
                    "relation": _relation(min_km) if min_km is not None else 0,
                    "relation_label": RELATION_CODES[_relation(min_km) if min_km is not None else 0],
                    "mean_threat": round(sum(l1[m]["threat"] for m in members) / len(members), 1),
                })

        l3 = []
        frame_level = 1
        for name, lat, lon in regions:
            inside = [tid for tid in order if tid in pos and
                      abs(pos[tid][0] - lat) <= 0.08 and abs(pos[tid][1] - lon) <= 0.08]
            n_red = sum(1 for tid in inside if by_id[tid]["affiliation"] == "red")
            n_blue = sum(1 for tid in inside if by_id[tid]["affiliation"] == "blue")
            max_threat = max((l1[tid]["threat"] for tid in inside), default=0.0)
            level = _level(max_threat)
            if n_red and n_blue:
                level = max(level, 2)
            dom = 1 if n_red > n_blue else 2 if n_blue > n_red else 3 if n_red else 0
            l3.append([level, n_red, n_blue, dom])
            frame_level = max(frame_level, level)

        offenders = [s for s in l2 if s.get("relation", 0) >= 2]
        overall = {
            "level": frame_level,
            "level_label": LEVEL_CODES[frame_level],
            "mean_threat": round(sum(v["threat"] for v in l1.values()) / max(1, len(l1)), 1),
            "max_threat": round(max((v["threat"] for v in l1.values()), default=0.0), 1),
            "engaged_pairs": sum(1 for s in offenders if s["affiliation"] == "red"),
            "note": {
                1: "双方兵力分散，未形成有效接触，整体TS和缓。",
                2: "存在远距监视关系，需持续跟踪对方动向。",
                3: "双方兵力在同一区域共处或已进入对方影响范围，TS紧张。",
                4: "存在高危接触，双方兵力距离已进入武器有效作用范围。",
            }[frame_level],
        }
        out_frames.append({
            "tick": frame["tick"],
            "targets": l1,
            "groups": l2,
            "regions": l3,
            "overall": overall,
        })
        level_hist[frame_level] = level_hist.get(frame_level, 0) + 1
        threat_sum += overall["mean_threat"]
        threat_n += 1
        if frame_level > peak_level:
            peak_level, peak_tick = frame_level, frame["tick"]

    return {
        "model": {
            "levels": [
                {"id": "L1", "name": "目标级TS",
                 "description": "单平台状态、意图与威胁度：装备权重 + 与最近对方目标的接近度 + 被对方传感器覆盖数。"},
                {"id": "L2", "name": "编组级TS",
                 "description": "按红蓝与海陆空聚合，给出编队形态、质心、跨度与对抗关系。"},
                {"id": "L3", "name": "区域级TS",
                 "description": "按战役目标点统计双方兵力存在与最高威胁，输出区域等级与主导方。"},
            ],
            "alert_codes": ALERT_CODES,
            "relation_codes": RELATION_CODES,
            "level_codes": LEVEL_CODES,
            "posture_codes": POSTURE_LABELS,
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
                "relation_km": [list(t) for t in RELATION_THRESHOLDS],
                "contact_km": 300.0,
                "engagement_km": 60.0,
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
