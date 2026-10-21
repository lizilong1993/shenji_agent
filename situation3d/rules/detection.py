"""Observation rules (public 观察规则 table).

Table (hexes): infantry/vehicle/helicopter see each other at 10/25/25;
UAV/cruise missile sees ground at 2 and cannot observe air/UAV.  Concealed
targets are halved, except vehicles when the observer is on higher ground;
targets in visible urban/forest cover are halved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rules.context import Cover, RuleContext, RuleEntityView

DEFAULT_OBSERVATION = Path(__file__).resolve().parent / "data" / "observation.json"
_DOC = None


def observation_doc() -> dict:
    global _DOC
    if _DOC is None:
        _DOC = json.loads(DEFAULT_OBSERVATION.read_text(encoding="utf-8"))
    return _DOC


def _kind(view: RuleEntityView) -> str:
    kind = str(getattr(view, "unit_kind", "vehicle"))
    if kind in ("infantry", "vehicle", "helicopter", "uav"):
        return kind
    if kind in ("recon", "aa", "artillery", "engineering"):
        return "vehicle"
    if kind in ("aircraft",):
        return "helicopter"
    return "vehicle"


def observe_range(observer: RuleEntityView, target: RuleEntityView,
                  ctx: RuleContext) -> int:
    table = observation_doc()["table"]
    base = int(table.get(_kind(observer), {}).get(_kind(target), 0))
    override = getattr(observer, "observe_distance", None) or {}
    if override:
        try:
            base = max([base] + [int(v) for v in override.values()])
        except (TypeError, ValueError):
            pass
    if base <= 0:
        return 0
    if target.concealed:
        # docs.md: the high-ground exception only applies to vehicle targets.
        if not (ctx.target_kind(target) == "vehicle"
                and ctx.elevation_grade(observer) > ctx.elevation_grade(target)):
            base = max(1, base // 2)
    if ctx.cover_at(target) != Cover.NONE:
        base = max(1, base // 2)
    return base


def can_observe(observer: RuleEntityView, target: RuleEntityView,
                ctx: RuleContext) -> bool:
    if observer.uid == target.uid or not target.alive:
        return False
    # Sea/air domains are covered by the radar/IR/sonar/ESM model, not the
    # land observation table.
    if (observer.domain not in ("land", "amphibious")
            or target.domain not in ("land", "amphibious")):
        return ctx.sensor_detect(observer, target)
    # Cheap rejection before terrain/cover work: the table base range is an
    # upper bound, so anything farther cannot be observed.
    table = observation_doc()["table"]
    max_range = int(table.get(_kind(observer), {}).get(_kind(target), 0))
    distance = ctx.rule_distance(observer, target)
    if max_range <= 0 or distance > max_range:
        return False
    rng = observe_range(observer, target, ctx)
    if rng <= 0 or distance > rng:
        return False
    return ctx.line_of_sight(observer, target)
