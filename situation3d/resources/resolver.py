"""Requirement -> ResourcePlan resolution (P2).

Selection rules, in order:

* the candidate must satisfy the requirement scope: matching ``name`` (for
  theatre-scoped data) and a bbox that contains the scenario anchor plus all
  mission points (entries without a bbox are global, e.g. role cards);
* data from after ``scenario_date + 1 day`` is never used (point-in-time);
* remaining candidates are ordered by ``as_of`` then ``fetched_at``;
* an entry that fails integrity verification is skipped;
* a selected entry older than the kind TTL is ``stale``: critical gaps block
  the run unless a ``use_cached_stale`` decision names that entry; advisory
  gaps are recorded as degradations and the run continues.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .catalog import kind_spec
from .model import (ResourceDecision, ResourceEntry, ResourceGap, ResourcePlan,
                    ResourceRequirement, slugify, utc_now)
from .store import ResourceStore

IMPACT = {
    "envpack": "缺少可用的环境包时，地形/路网/水系/建筑没有权威来源，引擎不能安全推演",
    "case_roads": "案例初始编成与道路路由依赖该路网，缺失时无法生成初始想定",
    "rolecards": "角色卡缺失时平台类型/装备/能力只能退回内置常量，结果不可比",
    "platforms": "平台剖面缺失时尺寸/运动/传感器退回代码常量，机动与交战会失真",
    "weapons": "武器/CRT/观察规则数据缺失时无法进行裁决",
    "intel_bundle": "情报包缺失时 VGI 线索为空，态势提示减少（不影响核心推演）",
    "weather": "天气时间线缺失时使用场景默认天气（影响感知精度提示）",
    "commander_service": "远程指挥不可用时回退确定性 RuleCommander（结果可复现）",
    "bathymetry": "水深缺失时按环境包默认值处理，两栖/涉水判断精度下降",
    "tiles3d": "三维显示数据集缺失，仅影响显示",
    "imagery": "底图/影像缺失，仅影响显示",
}


def _target_date(scenario_date: str) -> date:
    try:
        return date.fromisoformat(str(scenario_date or "")[:10])
    except ValueError:
        raise ValueError("scenario_date must be ISO 8601: %r" % (scenario_date,))


def _entry_date(as_of: str) -> Optional[date]:
    try:
        return date.fromisoformat(str(as_of or "")[:10])
    except ValueError:
        return None


def _scope_matches(entry: ResourceEntry, scope: Dict[str, Any]) -> bool:
    name = scope.get("name")
    if name:
        suffix = entry.entry_id.split(":", 1)[-1]
        theatre = str((entry.notes or {}).get("theatre") or "")
        if suffix != slugify(str(name)) and theatre != str(name):
            return False
    if entry.bbox is None:
        return True
    points: List[Tuple[float, float]] = []
    contains = scope.get("contains")
    if (isinstance(contains, (list, tuple)) and len(contains) == 2
            and contains[0] is not None and contains[1] is not None):
        points.append((float(contains[0]), float(contains[1])))
    for item in scope.get("points") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2 \
                and item[0] is not None and item[1] is not None:
            points.append((float(item[0]), float(item[1])))
    lon_min, lat_min, lon_max, lat_max = entry.bbox
    return all(lon_min <= lon <= lon_max and lat_min <= lat <= lat_max
               for lon, lat in points)


def _recommendation(req: ResourceRequirement, reason: str,
                    candidates: Sequence[ResourceEntry],
                    detail: str = "") -> Dict[str, Any]:
    if reason == "no_point_in_time":
        return {"action": "choose_date_or_older_snapshot",
                "detail": detail or "候选快照晚于想定日期",
                "command": "python3 scripts/resource.py list --kind %s" % req.kind}
    if reason == "stale":
        return {"action": "refresh_online_or_accept",
                "detail": detail or "数据时效超过该类资源上限",
                "choices": ["fetch_online", "use_cached_stale"],
                "command": "python3 scripts/resource.py scan"}
    if reason == "integrity":
        return {"action": "rescan_or_rebuild",
                "detail": detail or "登记内容与磁盘不一致",
                "command": "python3 scripts/resource.py verify"}
    if req.kind == "envpack":
        return {"action": "build_or_restore_envpack",
                "path": "situation3d/envpacks/",
                "command": ("python3 scripts/build_envpack.py --basic <basic.json> "
                            "--out envpacks/<name> --mode auto"),
                "detail": "也可以恢复已有环境包后执行 "
                          "python3 scripts/resource.py scan"}
    if req.kind == "case_roads":
        path = detail or "situation3d/engine/data/roads/<theatre>.json"
        return {"action": "restore_file", "path": path,
                "detail": "从仓库恢复该路网文件后执行 python3 scripts/resource.py scan"}
    return {"action": "scan_or_restore",
            "command": "python3 scripts/resource.py scan",
            "detail": "补齐该类资源后重新扫描登记"}


def _verify(store: ResourceStore, entry: ResourceEntry) -> Dict[str, Any]:
    try:
        return store.verify(entry.entry_id)
    except Exception as exc:  # noqa: BLE001 - a bad entry must not abort planning
        return {"ok": False, "status": "unreadable", "detail": str(exc)}


def resolve_requirements(store: ResourceStore,
                         requirements: Iterable[ResourceRequirement],
                         scenario_date: str,
                         decisions: Iterable[ResourceDecision] = (),
                         case_id: str = "case") -> ResourcePlan:
    target = _target_date(scenario_date)
    accepted_stale = set(d.entry_id for d in decisions
                         if d.action == "use_cached_stale" and d.entry_id)
    chosen: List[ResourceEntry] = []
    gaps: List[ResourceGap] = []
    for req in requirements:
        spec = kind_spec(req.kind)
        entries = store.entries(req.kind)
        matched = [e for e in entries if _scope_matches(e, req.scope)]
        future = [e for e in matched
                  if not (e.notes or {}).get("manual_override")
                  and _entry_date(e.as_of) is not None
                  and _entry_date(e.as_of) > target + timedelta(days=1)]
        usable = [e for e in matched if e not in future]
        usable.sort(key=lambda e: (e.as_of or "", e.fetched_at or "",
                                   e.entry_id), reverse=True)
        if req.multi:
            good = []
            bad = None
            for entry in usable:
                result = _verify(store, entry)
                if result["ok"]:
                    good.append(entry)
                else:
                    bad = (entry, result)
            if not good:
                if bad is not None:
                    gaps.append(ResourceGap(
                        kind=req.kind, criticality=req.criticality,
                        impact=IMPACT.get(req.kind) or "",
                        reason="integrity: %s" % bad[1].get("status"),
                        recommendation=_recommendation(req, "integrity", [],
                                                       bad[0].entry_id),
                        candidates=tuple(e.as_dict() for e in usable[:3])))
                else:
                    gaps.append(ResourceGap(
                        kind=req.kind, criticality=req.criticality,
                        impact=IMPACT.get(req.kind) or "",
                        reason="missing: %s" % (req.reason or ""),
                        recommendation=_recommendation(req, "missing", matched)))
                continue
            chosen.extend(good)
            if bad is not None:
                gaps.append(ResourceGap(
                    kind=req.kind, criticality=req.criticality,
                    impact=IMPACT.get(req.kind) or "",
                    reason="integrity: %s" % bad[1].get("status"),
                    recommendation=_recommendation(req, "integrity", [],
                                                   bad[0].entry_id),
                    candidates=(bad[0].as_dict(),)))
            continue
        selected = None
        integrity = None
        for entry in usable:
            result = _verify(store, entry)
            if result["ok"]:
                selected = entry
                break
            integrity = (entry, result)
        if selected is None:
            if integrity is not None:
                reason = "integrity"
                detail = "%s: %s" % (integrity[0].entry_id,
                                     integrity[1].get("status"))
            elif matched and future and len(future) == len(matched):
                reason = "no_point_in_time"
                latest = max(matched, key=lambda e: e.as_of or "")
                detail = "最新快照 as_of=%s，晚于想定日期 %s" % (
                    latest.as_of, target.isoformat())
            else:
                reason = "missing"
                name = req.scope.get("name")
                if req.kind == "case_roads" and name:
                    detail = "situation3d/engine/data/roads/%s.json" % name
                elif name:
                    detail = "应提供: %s" % name
                else:
                    detail = ""
            gaps.append(ResourceGap(
                kind=req.kind,
                criticality=req.criticality,
                impact=IMPACT.get(req.kind, req.reason or "运行所需资源缺失"),
                reason="%s%s" % (reason, (": " + req.reason) if req.reason else ""),
                recommendation=_recommendation(req, reason, matched, detail),
                candidates=tuple(e.as_dict() for e in matched[:3])))
            continue
        chosen.append(selected)
        ttl = spec.ttl_days
        stale = False
        detail = ""
        if (selected.notes or {}).get("manual_override"):
            continue
        if ttl is not None:
            as_of = _entry_date(selected.as_of)
            if as_of is None:
                stale = True
                detail = "登记缺少 as_of，无法证明时效"
            elif target - as_of > timedelta(days=float(ttl)):
                stale = True
                detail = "as_of=%s，距想定日期 %d 天，上限 %g 天" % (
                    selected.as_of, (target - as_of).days, float(ttl))
        if not stale:
            continue
        if selected.entry_id in accepted_stale:
            gaps.append(ResourceGap(
                kind=req.kind, criticality="advisory",
                impact="已接受过期数据（%s）" % detail,
                reason="stale_accepted: " + detail,
                recommendation={"action": "accepted",
                                "entry_id": selected.entry_id},
                candidates=(selected.as_dict(),)))
            continue
        gaps.append(ResourceGap(
            kind=req.kind, criticality=req.criticality,
            impact=IMPACT.get(req.kind, req.reason or "数据时效不足"),
            reason="stale: " + detail,
            recommendation=dict(_recommendation(req, "stale", [selected], detail),
                               entry_id=selected.entry_id,
                               as_of=selected.as_of),
            candidates=(selected.as_dict(),)))
    blocked = any(g.criticality == "critical" for g in gaps)
    status = "blocked" if blocked else ("degraded" if gaps else "ready")
    return ResourcePlan(plan_id="%s@%s" % (case_id, scenario_date),
                        scenario_date=scenario_date,
                        entries=tuple(chosen), gaps=tuple(gaps),
                        decisions=tuple(decisions), status=status)


def requirements_for(scenario: Dict[str, Any],
                     missions: Sequence[Dict[str, Any]],
                     theatre: Optional[str] = None,
                     engine: Optional[Dict[str, Any]] = None) -> List[ResourceRequirement]:
    engine = engine or {}
    objectives = {}
    for item in scenario.get("objectives") or []:
        if item.get("name") is not None and item.get("lon") is not None:
            objectives[str(item["name"])] = (item["lon"], item["lat"])
    points = []
    for mission in missions or []:
        if mission.get("lon") is not None and mission.get("lat") is not None:
            points.append((mission["lon"], mission["lat"]))
        elif mission.get("objective") in objectives:
            points.append(objectives[mission["objective"]])
    anchor = scenario.get("geo_anchor") or {}
    reqs = [ResourceRequirement(
        "envpack", "critical", "引擎空间权威",
        {"contains": [anchor.get("lon"), anchor.get("lat")],
         "points": points})]
    if theatre:
        reqs.append(ResourceRequirement(
            "case_roads", "critical", "案例初始路网",
            {"name": theatre}))
    if engine.get("use_role_cards"):
        reqs.append(ResourceRequirement(
            "rolecards", "critical", "场景启用角色卡"))
    reqs.append(ResourceRequirement("weapons", "critical", "裁决规则数据"))
    reqs.append(ResourceRequirement("platforms", "critical", "平台剖面数据",
                                    multi=True))
    if scenario.get("intel") or scenario.get("intel_bundle"):
        reqs.append(ResourceRequirement(
            "intel_bundle", "advisory", "场景引用情报包"))
    if scenario.get("weather") or scenario.get("weather_base"):
        reqs.append(ResourceRequirement(
            "weather", "advisory", "场景配置天气时间线"))
    if engine.get("enable_commander") and str(engine.get("commander") or "") \
            in ("remote", "http"):
        reqs.append(ResourceRequirement(
            "commander_service", "advisory", "远程指挥已启用"))
    return reqs
