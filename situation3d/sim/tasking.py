"""User task requests: parse, validate, merge with the case doctrine.

P1 scope is *launch-time tasking* (no mid-run injection).  A request is a JSON
object::

    {"mode": "user" | "mixed" | "default",
     "side": 0,
     "tasks": [{"id": "U1", "side": 0, "units": [2001, 2002],
                "task": "move_to" | "seize" | "hold",
                "objective": "D区岸海衔接",          # or lon/lat
                "lon": 120.16, "lat": 22.97,
                "after": ["M0"], "start_s": 0,
                "pace": "normal", "risk": "normal", "arrive_m": 250}]}

``mode=user`` removes the case default missions for the requesting side and
keeps the other side untouched; ``mode=mixed`` keeps all defaults.  A bare
list of tasks is accepted as ``{"mode": "user", "tasks": [...]}``.

Validation is intentionally strict: unknown units, foreign affiliations,
duplicate unit assignment, unknown objectives, dangling ``after`` references
and cycles are rejected with a readable error instead of failing mid-run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from resources.model import canonical_json

SUPPORTED_TASKS = ("move_to", "seize", "hold")
PACES = ("dash", "normal", "cautious")
RISKS = ("low", "normal", "high")
MODES = ("default", "user", "mixed")
SIDES = (0, 1)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")


class TaskError(ValueError):
    """Invalid user task request."""


def parse_request(raw: Any, default_side: int = 0) -> Dict[str, Any]:
    if raw is None:
        return {"mode": "default", "side": int(default_side), "tasks": []}
    if isinstance(raw, list):
        data = {"tasks": raw}
    elif isinstance(raw, dict):
        data = raw
    else:
        raise TaskError("task request must be an object or a list")
    tasks = data.get("tasks")
    if tasks is None:
        tasks = []
    if not isinstance(tasks, list):
        raise TaskError("tasks must be a list")
    mode = str(data.get("mode") or ("user" if tasks else "default")).strip().lower()
    if mode not in MODES:
        raise TaskError("mode must be one of %s" % (", ".join(MODES)))
    try:
        side = int(data.get("side", default_side))
    except (TypeError, ValueError):
        raise TaskError("side must be 0 or 1")
    if side not in SIDES:
        raise TaskError("side must be 0 or 1")
    return {"mode": mode, "side": side, "tasks": list(tasks)}


def request_digest(request: Dict[str, Any]) -> str:
    payload = canonical_json({"mode": request.get("mode"),
                              "side": request.get("side"),
                              "tasks": request.get("tasks") or []})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unit_index(scenario: Dict[str, Any]) -> Dict[int, int]:
    index = {}
    for op in scenario.get("operators") or []:
        try:
            uid = int(op.get("obj_id"))
        except (TypeError, ValueError):
            continue
        try:
            color = int(op.get("color", op.get("affiliation", -1)))
        except (TypeError, ValueError):
            color = -1
        index[uid] = color
    return index


def _objective_names(scenario: Dict[str, Any]) -> List[str]:
    return [str(o.get("name")) for o in scenario.get("objectives") or []
            if o.get("name")]


def _check_number(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise TaskError("%s must be a finite number" % label)
    return value


def _within_bbox(lon: float, lat: float, bbox: Sequence[float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = [float(v) for v in bbox]
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def _check_cycles(tasks: List[Dict[str, Any]]) -> None:
    ids = set(task["id"] for task in tasks)
    graph = {task["id"]: [dep for dep in task["after"] if dep in ids]
             for task in tasks}
    state: Dict[str, int] = {}

    def visit(node: str, stack: List[str]) -> None:
        mark = state.get(node, 0)
        if mark == 1:
            raise TaskError("任务依赖成环: %s" % " -> ".join(stack + [node]))
        if mark == 2:
            return
        state[node] = 1
        for dep in graph.get(node, []):
            visit(dep, stack + [node])
        state[node] = 2

    for mid in sorted(graph):
        visit(mid, [])


def validate_tasks(tasks: Iterable[Any], scenario: Dict[str, Any],
                   default_side: int = 0,
                   known_ids: Iterable[str] = (),
                   bbox: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    units_index = _unit_index(scenario)
    objectives = set(_objective_names(scenario))
    try:
        duration = float(scenario.get("duration_s") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    seen_ids = set(str(x) for x in known_ids)
    used_units: Dict[int, str] = {}
    out: List[Dict[str, Any]] = []
    for index, raw in enumerate(tasks):
        if not isinstance(raw, dict):
            raise TaskError("task #%d must be an object" % (index + 1))
        mid = str(raw.get("id") or ("U%02d" % (index + 1))).strip()
        if not _ID_RE.match(mid):
            raise TaskError("task id %r is invalid (letters, digits, _.:- only)" % mid)
        if mid in seen_ids:
            raise TaskError("任务 id 重复: %s" % mid)
        try:
            side = int(raw.get("side", default_side))
        except (TypeError, ValueError):
            raise TaskError("%s: side must be 0 or 1" % mid)
        if side not in SIDES:
            raise TaskError("%s: side must be 0 or 1" % mid)
        raw_units = raw.get("units")
        if not isinstance(raw_units, list) or not raw_units:
            raise TaskError("%s: units must be a non-empty list" % mid)
        units: List[int] = []
        for value in raw_units:
            try:
                uid = int(value)
            except (TypeError, ValueError):
                raise TaskError("%s: unit ids must be integers" % mid)
            if uid not in units_index:
                raise TaskError("%s: unknown unit %s" % (mid, uid))
            if units_index[uid] != side:
                raise TaskError("%s: unit %s belongs to side %s, not %s"
                                % (mid, uid, units_index[uid], side))
            if uid in used_units:
                raise TaskError("单位 %s 已属于任务 %s；同一单位只允许一条用户任务"
                                % (uid, used_units[uid]))
            used_units[uid] = mid
            if uid not in units:
                units.append(uid)
        task = str(raw.get("task") or "move_to").strip().lower()
        if task not in SUPPORTED_TASKS:
            raise TaskError("%s: task must be one of %s"
                            % (mid, ", ".join(SUPPORTED_TASKS)))
        lon = raw.get("lon")
        lat = raw.get("lat")
        if (lon is None) != (lat is None):
            raise TaskError("%s: lon and lat must be given together" % mid)
        if lon is not None:
            try:
                lon = _check_number(float(lon), "%s lon" % mid)
                lat = _check_number(float(lat), "%s lat" % mid)
            except (TypeError, ValueError):
                raise TaskError("%s: lon/lat must be finite numbers" % mid)
            if bbox is not None and not _within_bbox(lon, lat, bbox):
                raise TaskError("%s: point %.6f,%.6f outside envpack bbox %s"
                                % (mid, lon, lat, list(bbox)))
        objective = str(raw.get("objective") or "").strip()
        if lon is None:
            if not objective:
                raise TaskError("%s: objective or lon/lat is required" % mid)
            if objectives and objective not in objectives:
                raise TaskError("%s: unknown objective %r (known: %s)"
                                % (mid, objective, ", ".join(sorted(objectives))))
        after = raw.get("after") or []
        if not isinstance(after, list):
            raise TaskError("%s: after must be a list" % mid)
        after = [str(item) for item in after]
        for dep in after:
            if dep not in seen_ids:
                raise TaskError("%s: unknown predecessor %r" % (mid, dep))
        try:
            start_s = float(raw.get("start_s") or 0.0)
        except (TypeError, ValueError):
            raise TaskError("%s: start_s must be a number" % mid)
        if start_s < 0:
            raise TaskError("%s: start_s must be >= 0" % mid)
        if duration > 0 and start_s > duration:
            raise TaskError("%s: start_s %.0f exceeds scenario duration %.0f"
                            % (mid, start_s, duration))
        pace = str(raw.get("pace") or "normal").strip().lower()
        if pace not in PACES:
            raise TaskError("%s: pace must be one of %s"
                            % (mid, ", ".join(PACES)))
        risk = str(raw.get("risk") or "normal").strip().lower()
        if risk not in RISKS:
            raise TaskError("%s: risk must be one of %s"
                            % (mid, ", ".join(RISKS)))
        arrive_raw = raw.get("arrive_m")
        try:
            arrive_m = 250.0 if arrive_raw is None else float(arrive_raw)
        except (TypeError, ValueError):
            raise TaskError("%s: arrive_m must be a number" % mid)
        if arrive_m <= 0 or not math.isfinite(arrive_m):
            raise TaskError("%s: arrive_m must be a positive number" % mid)
        out.append({
            "id": mid,
            "side": side,
            "units": units,
            "task": task,
            "objective": objective,
            "lon": lon,
            "lat": lat,
            "after": after,
            "start_s": start_s,
            "pace": pace,
            "risk": risk,
            "arrive_m": arrive_m,
        })
        seen_ids.add(mid)
    _check_cycles(out)
    return out


def _unit_override(defaults, tasks, side):
    """Remove user-tasked units from own-side defaults.

    Returns ``(kept, dropped_ids, trimmed_ids)``.  A default mission that
    loses all of its units is dropped; a partially overridden one keeps the
    remaining units.  Kept missions must not depend on a dropped mission.
    """
    tasked = set()
    for task in tasks:
        tasked.update(task["units"])
    kept = []
    dropped = []
    trimmed = []
    for item in defaults:
        if int(item.get("side", 0)) != side:
            kept.append(item)
            continue
        units = [int(u) for u in item.get("units") or []]
        overlap = [u for u in units if u in tasked]
        if not overlap:
            kept.append(item)
            continue
        if len(overlap) == len(units):
            dropped.append(str(item.get("id")))
            continue
        trimmed_item = dict(item)
        trimmed_item["units"] = [u for u in units if u not in tasked]
        trimmed.append(str(item.get("id")))
        kept.append(trimmed_item)
    dropped_set = set(dropped)
    for item in kept:
        for dep in item.get("after") or []:
            if str(dep) in dropped_set:
                raise TaskError(
                    "默认任务 %s 依赖被用户任务整体覆盖的 %s"
                    % (item.get("id"), dep))
    return kept, dropped, trimmed


def merge_missions(default_missions: Iterable[Dict[str, Any]],
                   request: Dict[str, Any],
                   scenario: Dict[str, Any],
                   bbox: Optional[Sequence[float]] = None
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return ``(missions, summary)`` for the runner and run manifest.

    ``mode=user``    user tasks override the same units in own-side defaults;
    ``mode=mixed``   defaults stay untouched and unit overlap is rejected;
    ``mode=default`` no user tasks are accepted.
    """
    mode = str(request.get("mode") or "default")
    side = int(request.get("side", 0))
    raw_tasks = list(request.get("tasks") or [])
    defaults = [dict(item) for item in (default_missions or [])]
    if mode == "default":
        if raw_tasks:
            raise TaskError("mode=default does not accept user tasks")
        return defaults, {"mode": mode, "side": side, "user_tasks": 0}
    all_default_ids = set(str(item.get("id")) for item in defaults)
    tasks = validate_tasks(raw_tasks, scenario, default_side=side,
                           known_ids=all_default_ids, bbox=bbox)
    dropped: List[str] = []
    trimmed: List[str] = []
    if mode == "user":
        kept, dropped, trimmed = _unit_override(defaults, tasks, side)
    else:
        kept = defaults
        default_units = {}
        for item in kept:
            if int(item.get("side", 0)) != side:
                continue
            for uid in item.get("units") or []:
                default_units[int(uid)] = str(item.get("id"))
        for task in tasks:
            for uid in task["units"]:
                if uid in default_units:
                    raise TaskError(
                        "单位 %s 已属于默认任务 %s；请使用 mode=user 或调整单位"
                        % (uid, default_units[uid]))
    kept_ids = set(str(item.get("id")) for item in kept)
    for task in tasks:
        for dep in task["after"]:
            if dep in all_default_ids and dep not in kept_ids:
                raise TaskError("%s: predecessor %s 已被用户任务覆盖"
                                % (task["id"], dep))
    missions = kept + tasks
    summary = {"mode": mode, "side": side,
               "user_tasks": len(tasks),
               "kept_default_missions": len(kept),
               "dropped_default_missions": list(dropped),
               "trimmed_default_missions": list(trimmed),
               "request_hash": request_digest(request)}
    return missions, summary


def load_request(path: str, default_side: int = 0) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return parse_request(raw, default_side=default_side)
