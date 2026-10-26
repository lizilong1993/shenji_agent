"""Run preflight: requirements -> plan -> waiting/degraded/ready summary."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .model import ResourceDecision, ResourceGap, ResourcePlan, ResourceRequirement, utc_now
from .resolver import requirements_for, resolve_requirements
from .store import ResourceStore


def check_case_data(store: ResourceStore, theatre: Optional[str],
                    scenario_date: str,
                    decisions: Iterable[ResourceDecision] = ()) -> Optional[ResourcePlan]:
    """Early check for case data that the scenario builder cannot work without."""
    if not theatre:
        return None
    reqs = [ResourceRequirement("case_roads", "critical",
                                "案例初始编成依赖路网",
                                {"name": theatre})]
    plan = resolve_requirements(store, reqs, scenario_date,
                                decisions=decisions, case_id="case-data")
    return plan if plan.status == "blocked" else None


def make_plan(store: ResourceStore, case_id: str, scenario_date: str,
              scenario: Dict[str, Any], missions: Sequence[Dict[str, Any]],
              theatre: Optional[str] = None,
              engine: Optional[Dict[str, Any]] = None,
              decisions: Iterable[ResourceDecision] = ()) -> ResourcePlan:
    reqs = requirements_for(scenario, missions, theatre=theatre, engine=engine)
    return resolve_requirements(store, reqs, scenario_date,
                                decisions=decisions, case_id=case_id)


def format_gap(gap: ResourceGap) -> str:
    level = "必须处理" if gap.criticality == "critical" else "提示"
    rec = gap.recommendation or {}
    target = rec.get("path") or rec.get("command") or rec.get("action") or ""
    return "[%s] %s —— %s；建议：%s" % (level, gap.kind, gap.impact, target)


def plan_summary(plan: ResourcePlan, store: Optional[ResourceStore] = None) -> Dict[str, Any]:
    entries = []
    for entry in plan.entries:
        entries.append({
            "kind": entry.kind,
            "entry_id": entry.entry_id,
            "path": entry.path,
            "as_of": entry.as_of,
            "fetched_at": entry.fetched_at,
            "quality": entry.quality,
            "source": entry.source,
        })
    warnings = [format_gap(g) for g in plan.gaps
                if g.criticality != "critical"]
    blockers = [format_gap(g) for g in plan.gaps
                if g.criticality == "critical"]
    return {
        "status": plan.status,
        "plan_id": plan.plan_id,
        "plan_hash": plan.hash(),
        "scenario_date": plan.scenario_date,
        "entries": entries,
        "gaps": [g.as_dict() for g in plan.gaps],
        "blockers": blockers,
        "warnings": warnings,
        "generated_at": utc_now(),
    }


def entry_for(plan: ResourcePlan, kind: str):
    for entry in plan.entries:
        if entry.kind == kind:
            return entry
    return None


def decisions_from(raw: Any) -> List[ResourceDecision]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        items = raw.get("decisions") or []
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("decisions must be a list or an object with decisions")
    out = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each decision must be an object")
        decision = ResourceDecision(
            action=str(item.get("action") or ""),
            entry_id=str(item.get("entry_id") or ""),
            note=str(item.get("note") or ""),
            decided_by=str(item.get("decided_by") or ""),
            decided_at=str(item.get("decided_at") or ""),
            accept_stale=bool(item.get("accept_stale") or False),
            kind=str(item.get("kind") or ""),
            path=str(item.get("path") or ""),
        ).validate()
        out.append(decision)
    return out
